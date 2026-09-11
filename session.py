"""Cross-process session persistence (save / list / load).

The in-process REPL keeps conversation state in AgentState for the
lifetime of one run. This module adds durability: serialize the whole
state (session_id, context messages, tool/permission state) to disk so
a *new* process can resume the same conversation.

Mechanism: AgentState is a pydantic v2 model — persistence is a plain
JSON round-trip. Zero new dependencies, zero schema of our own.

Files live in <project root>/sessions/<session_id>.json (gitignored).
"""
from __future__ import annotations

import json
from pathlib import Path

from agentscope.state import AgentState

SESSION_DIR = Path(__file__).resolve().parent / "sessions"


def save_state(state: AgentState) -> Path:
    """Write the full agent state to disk; returns the file path."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    sid = state.session_id or "session"
    # ponytail: json (not jsonl/append) — a session is rewritten whole on
    # each /save; append-style logs matter only if we need per-turn replay.
    p = SESSION_DIR / f"{sid}.json"
    p.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_state(path: Path | None = None) -> AgentState | None:
    """Load a session state; path=None means the most recent file."""
    p = path
    if p is None:
        files = sorted(SESSION_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime)
        p = files[-1] if files else None
    if p is None or not p.exists():
        return None
    return AgentState.model_validate_json(p.read_text(encoding="utf-8"))


def list_sessions() -> list[str]:
    """Newest-first one-liners: session_id — N msgs — mtime."""
    out = []
    for f in sorted(SESSION_DIR.glob("*.json"),
                    key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            n = len(json.loads(f.read_text(encoding="utf-8")).get("context", []))
            stamp = f.stat().st_mtime
            out.append(f"{f.stem} — {n} msgs — "
                       f"{__import__('datetime').datetime.fromtimestamp(stamp):%m-%d %H:%M}")
        except Exception:  # noqa: BLE001 - a corrupt file must not kill listing
            out.append(f"{f.stem} — (unreadable)")
    return out


def save_research_memory(session_id: str, mem: dict) -> Path:
    """Persist the research working memory next to the session file.
    AgentState (external pydantic model) can't carry an extra field, so a
    sibling <session_id>.research.json is the zero-coupling home."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    p = SESSION_DIR / f"{session_id}.research.json"
    p.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_research_memory(session_id: str) -> dict | None:
    p = SESSION_DIR / f"{session_id}.research.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - corrupt sidecar must not kill load
        return None


if __name__ == "__main__":
    # self-check: round-trip a real state through disk
    from agentscope.message import UserMsg

    st = AgentState(context=[UserMsg(name="user", content="hi"),
                             UserMsg(name="user", content="again")])
    p = save_state(st)
    st2 = load_state(p)
    assert st2.session_id == st.session_id, "session_id lost in round-trip"
    assert len(st2.context) == 2, "context length mismatch"
    p.unlink()
    # research working memory round-trips as a sidecar next to the session
    mem = {"query": "q", "intent": "research", "research_iterations": 2,
           "gaps": ["评测"], "citations": {"verified": 68, "total": 68}}
    mp = save_research_memory(st.session_id, mem)
    got = load_research_memory(st.session_id)
    assert got == mem, "research memory lost in round-trip"
    mp.unlink()
    assert load_research_memory(st.session_id) is None, "gone after unlink"
    print(f"self-check ok: round-trip {p.name} (2 msgs) + research memory, cleaned")