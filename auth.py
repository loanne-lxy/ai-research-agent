"""JWT auth + research read endpoints for the multi-user web.

Design (the lazy version, the ceiling is documented):
  - Identity: AgentScope's ``get_current_user_id`` reads the ``X-User-ID``
    header (its docstring: "Temporary header-based identity; will be
    replaced by JWT auth"). We do that replacement at the ASGI edge with a
    single middleware: ``Authorization: Bearer <jwt>`` -> user_id -> header.
    Every official router (sessions/agents/credentials/knowledge/...) keeps
    working untouched, and the old header-based flow still works when no
    Authorization header is present (migration window).
  - Users: one JSON file (data/users.json): {username: {salt, hash}}.
    stdlib pbkdf2_hmac — no passlib/bcrypt (not installed, not needed).
    ponytail: file-based user store, sqlite table if multi-user writes
    become frequent.
  - /api/auth/* lives OUTSIDE the JWT middleware (login can't be a
    chicken-and-egg); the middleware skips /api/auth/.

Read endpoints (all require a valid token):
  GET /api/research/sessions          recent research, newest first
  GET /api/research/sessions/{sid}    full research sidecar (may 404)
  GET /api/memory                     ReMe daily notes + per-topic files
  GET /api/knowledge/stats            corpus counts (Knowledge Overview)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import config as C
from session import SESSION_DIR

JWT_SECRET = os.getenv("JWT_SECRET") or secrets.token_hex(32)  # ponytail: per-process secret, restart invalidates tokens; fixed via JWT_SECRET in .env when that matters
TOKEN_TTL = 7 * 86400  # 7 days; ponytail: no refresh flow, re-login when expired
MEMORY_ROOT = C.PROJECT_ROOT / "data" / "memory"

router = APIRouter(prefix="/api")


# ------------------------------------------------------------- users
USERS_FILE = C.PROJECT_ROOT / "data" / "users.json"


def _load_users() -> dict:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=1),
                          encoding="utf-8")


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(),
                               salt.encode(), 120_000).hex()


def create_user(username: str, password: str) -> bool:
    """Create a user. Returns False if the name is taken."""
    users = _load_users()
    if username in users or not password:
        return False
    salt = secrets.token_hex(16)
    users[username] = {"salt": salt, "hash": _hash(password, salt)}
    _save_users(users)
    return True


def verify_user(username: str, password: str) -> bool:
    u = _load_users().get(username)
    if not u:
        return False
    return hmac.compare_digest(_hash(password, u["salt"]), u["hash"])


# ------------------------------------------------------------- jwt
class LoginRequest(BaseModel):
    username: str
    password: str


def _token(username: str) -> str:
    return jwt.encode(
        {"sub": username, "exp": int(time.time()) + TOKEN_TTL},
        JWT_SECRET, algorithm="HS256")


@router.post("/auth/login")
def login(body: LoginRequest) -> dict:
    if not verify_user(body.username, body.password):
        raise HTTPException(401, "bad username or password")
    return {"token": _token(body.username), "username": body.username}


def require_user(request: Request) -> str:
    """FastAPI dependency for /api/* (non-auth) endpoints.

    Reads the ``jwt_user`` scope marker set by :class:`JwtAuthMiddleware`
    — NOT the raw X-User-ID header, so a client that forges the header
    without a valid token gets 401 on our endpoints (the official routers
    keep their legacy header behaviour during migration).
    """
    user_id = request.scope.get("jwt_user")
    if not user_id:
        raise HTTPException(401, "authentication required")
    return user_id


@router.post("/auth/me", summary="Validate the current token")
def auth_me(user_id: str = Depends(require_user)):
    return {"username": user_id, "authenticated": True}


class CreateUserRequest(BaseModel):
    username: str
    password: str


@router.post("/users", summary="Create a new login user (any authenticated user)")
def create_user_endpoint(body: CreateUserRequest, user_id: str = Depends(require_user)):
    """LAN-internal account creation: any logged-in user may provision
    colleagues. No self-registration endpoint — deliberate (ponytail: trust
    boundary is the LAN, add per-user approval when that changes)."""
    username = body.username.strip()
    if not username or not body.password or len(body.password) < 4:
        raise HTTPException(400, "username required, password >= 4 chars")
    if not create_user(username, body.password):
        raise HTTPException(409, f"username {username!r} already exists")
    return {"username": username, "created": True}


@router.post("/bootstrap", summary="Ensure the caller's research setup")
async def bootstrap(request: Request, user_id: str = Depends(require_user)):
    """Idempotently create the caller's own credential + Research Expert
    agent + default session (each user gets their own — that is the whole
    multi-user story, no official router touched). Returns ids plus the
    model binding so the UI can create further sessions."""
    # Late binding: the server runs as `python web_service.py` (module
    # `__main__`), so `import web_service` would re-execute the whole file.
    # Grab ensure_user from whichever module object already holds it.
    import sys
    ws = sys.modules.get("web_service")
    if ws is None:
        ws = sys.modules["__main__"]
    return await ws.ensure_user(
        request.app.state.storage,
        request.app.state.workspace_manager,
        user_id,
    )


# ------------------------------------------------------------- ASGI edge
class JwtAuthMiddleware:
    """Pure-ASGI (streaming-safe) edge auth.

    ``Authorization: Bearer <jwt>`` -> user_id -> rewrites the ``X-User-ID``
    header in the scope, so AgentScope's ``get_current_user_id`` and EVERY
    official router (sessions/agents/credentials/knowledge/chat/...) see the
    authenticated identity with zero changes. Sets ``scope["jwt_user"]`` for
    our own /api/* endpoints. ``/api/auth/login`` is exempt (chicken-and-egg).
    No Authorization header -> legacy X-User-ID pass-through (migration
    window for existing clients).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] != "/api/auth/login":
            auth = dict(scope["headers"]).get(b"authorization", b"").decode()
            if auth.startswith("Bearer "):
                try:
                    payload = jwt.decode(auth[7:], JWT_SECRET, algorithms=["HS256"])
                    uid = str(payload["sub"])
                except (jwt.PyJWTError, KeyError):
                    await self._401(send)
                    return
                scope["headers"] = [
                    *[(k, v) for k, v in scope["headers"] if k != b"x-user-id"],
                    (b"x-user-id", uid.encode()),
                ]
                scope["jwt_user"] = uid
        await self.app(scope, receive, send)

    @staticmethod
    async def _401(send) -> None:
        body = json.dumps({"detail": "invalid or expired token"}).encode()
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


# ------------------------------------------------------------- research reads
def _sidecar(session_id: str) -> Path | None:
    p = SESSION_DIR / f"{session_id}.research.json"
    return p if p.exists() else None


@router.get("/research/sessions")
def list_research(_: str = Depends(require_user)) -> dict:
    items = []
    for p in sorted(SESSION_DIR.glob("*.research.json"),
                    key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        items.append({
            "session_id": p.name.removesuffix(".research.json"),
            "query": d.get("query", ""),
            "verdict": d.get("verdict"),
            "iterations": d.get("research_iterations", 0),
            "citations": d.get("citations", {}),
            "updated_at": int(p.stat().st_mtime),
        })
    return {"sessions": items}


@router.get("/research/sessions/{session_id}")
def get_research(session_id: str, _: str = Depends(require_user)) -> dict:
    p = _sidecar(session_id)
    if p is None:
        raise HTTPException(404, "no research state for this session")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------- memory / knowledge
@router.get("/memory")
def list_memory(_: str = Depends(require_user)) -> dict:
    daily, topics = [], []
    ddir = MEMORY_ROOT / "daily"
    if ddir.exists():
        for f in sorted(ddir.iterdir(), reverse=True):
            if f.is_dir():
                topics.append({"path": str(f.relative_to(MEMORY_ROOT)),
                               "size": sum(g.stat().st_size for g in f.glob("*.md"))})
            elif f.suffix == ".md":
                daily.append({"path": str(f.relative_to(MEMORY_ROOT)),
                              "size": f.stat().st_size})
    return {"daily": daily[:30], "topics": topics[:30]}


class _MemFileRequest(BaseModel):
    path: str  # relative to data/memory, e.g. "daily/2026-09-17/x.md"


@router.post("/memory/file", summary="Read one memory .md (path-traversal guarded)")
def memory_file(body: _MemFileRequest, _: str = Depends(require_user)) -> dict:
    p = (MEMORY_ROOT / body.path.lstrip("/")).resolve()
    # ponytail: substring check on the resolved path, fine for an intranet tool
    if not str(p).startswith(str(MEMORY_ROOT.resolve())) or not p.is_file():
        raise HTTPException(404, "no such memory file")
    return {"path": body.path, "content": p.read_text(encoding="utf-8")}


class _CiteRequest(BaseModel):
    ids: list[str]


@router.post("/citations", summary="Resolve citation ids to corpus objects")
def resolve_citations(body: _CiteRequest, _: str = Depends(require_user)) -> dict:
    from corpus.store import resolve
    out = {}
    for cid in body.ids[:200]:
        r = resolve(cid)
        out[cid] = r
    return {"resolved": out}


@router.get("/knowledge/stats")
def knowledge_stats(_: str = Depends(require_user)) -> dict:
    from tools.knowledge import corpus_stats
    return corpus_stats()