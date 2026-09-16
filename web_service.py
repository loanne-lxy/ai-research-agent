"""AgentScope 2.0 Agent Service for the Research Agent (web entry point).

Run:  .venv/bin/python web_service.py            # http://127.0.0.1:8000
      .venv/bin/python web_service.py --port 9000

Wires the *existing* ResearchAgent (agent/builder + research loop) into the
official Agent Service (agentscope.app.create_app) so the official Web UI
can drive it over HTTP + SSE. No hand-rolled WebSocket / streaming / chat UI.

Backends (this box has NO redis-server):
  storage   AsyncSQLAlchemyStorage (sqlite + aiosqlite, file-backed)
  bus       InMemoryMessageBus       (single-process)
  workspace LocalWorkspaceManager

``seed()`` idempotently registers, for user "loanne":
  * the Qwen OpenAI-compatible credential (endpoint/key from .env),
  * the Research Expert agent (system_prompt.md),
  * one session bound to that agent + the Qwen model config,
so the Web UI has something to talk to immediately (they can equally be
created from the UI; seed just skips the manual steps).
"""
from __future__ import annotations

import argparse
import json

import uvicorn

import config as C
from agent.builder import load_system_prompt
from agent.research_agent import ResearchAgent

USER_ID = "loanne"
STORAGE_DB = C.PROJECT_ROOT / "data" / "agentscope_storage.db"
WORKSPACE_ROOT = C.PROJECT_ROOT / "workspaces"


# ------------------------------------------------------------- knowledge tools
async def _extra_agent_tools(user_id: str, agent_id: str, session_id: str) -> list:
    """My V2 Knowledge Tools, appended to the service toolkit per agent.

    ``create_app(extra_agent_tools=...)`` is the seam where non-workspace
    tools join (signature: user_id, agent_id, session_id -> list[ToolBase],
    awaited once per turn). Reuses the exact toolset the CLI builds via
    ``knowledge_toolkit()`` — no re-listing.
    """
    from tools.knowledge import knowledge_toolkit
    tk = knowledge_toolkit()
    tools = []
    for group in tk.tool_groups:
        tools.extend(group.tools)
    return tools


# ------------------------------------------------------------- the app
def build_app():
    from agentscope.app import create_app
    from agentscope.app.message_bus import InMemoryMessageBus
    from agentscope.app.storage import AsyncSQLAlchemyStorage
    from agentscope.app.workspace_manager import LocalWorkspaceManager

    STORAGE_DB.parent.mkdir(parents=True, exist_ok=True)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    storage = AsyncSQLAlchemyStorage(f"sqlite+aiosqlite:///{STORAGE_DB}")
    bus = InMemoryMessageBus()
    workspace = LocalWorkspaceManager(basedir=str(WORKSPACE_ROOT))

    app = create_app(
        storage,
        bus,
        workspace_manager=workspace,
        custom_agent_cls=ResearchAgent,
        extra_agent_tools=_extra_agent_tools,
    )
    return app, storage, workspace


# ------------------------------------------------------------- seed
async def seed(storage, workspace, user_id: str = USER_ID) -> dict:
    """Idempotently register credential + research agent + session."""
    from agentscope.agent import ContextConfig, ReActConfig
    from agentscope.app.storage import (
        AgentData, AgentRecord,
        ChatModelConfig, SessionConfig,
    )
    from agentscope.credential import OpenAICredential

    llm = C.load_llm_config()

    # --- credential (idempotent by name) ---
    creds = await storage.list_credentials(user_id)
    qwen = next((c for c in creds if c.data.get("name") == "qwen"), None)
    if qwen is None:
        cred_id = await storage.upsert_credential(
            user_id,
            OpenAICredential(
                name="qwen",
                api_key=llm.api_key,
                base_url=llm.base_url,
            ),
        )
        qwen = await storage.get_credential(user_id, cred_id)
        print(f"[seed] created Qwen credential id={cred_id}")
    else:
        cred_id = qwen.id

    # --- agent (idempotent by name) ---
    agents = await storage.list_agents(user_id)
    agent = next((a for a in agents if a.data.name == "Research Expert"), None)
    if agent is None:
        agent_id = await storage.upsert_agent(
            user_id,
            AgentRecord(
                user_id=user_id,
                data=AgentData(
                    name="Research Expert",
                    system_prompt=load_system_prompt(),
                    context_config=ContextConfig(),
                    react_config=ReActConfig(),
                ),
            ),
        )
        agent = await storage.get_agent(user_id, agent_id)
        print(f"[seed] created agent id={agent_id}")
    else:
        agent_id = agent.id

    # --- session (idempotent by name) ---
    sessions = await storage.list_sessions(user_id, agent_id)
    session = next((s for s in sessions if s.config.name == "research-default"), None)
    if session is None:
        workspace_id = await workspace.assign_workspace_id(
            user_id=user_id, agent_id=agent_id, session_id="")
        session = await storage.upsert_session(
            user_id, agent_id,
            config=SessionConfig(
                workspace_id=workspace_id,
                name="research-default",
                chat_model_config=ChatModelConfig(
                    type="openai_credential",
                    credential_id=cred_id,
                    model=llm.model,
                    # match the CLI model exactly (temperature + a large
                    # max_tokens: Qwen3.8 is a reasoning model, its thinking
                    # tokens count against the budget)
                    parameters={"temperature": llm.temperature,
                                "max_tokens": llm.max_tokens},
                ),
            ),
        )
        print(f"[seed] created session id={session.id} (model={llm.model})")

    return {"credential": cred_id, "agent": agent_id, "session": session.id}


# ------------------------------------------------------------- entry
def main():
    ap = argparse.ArgumentParser(
        description="Research Agent — AgentScope 2.0 Agent Service (web)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app, storage, workspace = build_app()

    # Seed once at startup (idempotent). The service's own lifespan enters
    # storage / bus / workspace as async contexts (initialising them), so
    # seed *inside* that context, not before.
    from contextlib import asynccontextmanager

    orig_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _lifespan(asgi_app):
        async with orig_lifespan(asgi_app):
            ids = await seed(storage, workspace)
            print(f"[service] seeded ids={json.dumps(ids)}")
            print(f"[service] ready — UI server_url: http://{args.host}:{args.port} "
                  f"username: {USER_ID}")
            yield

    app.router.lifespan_context = _lifespan
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()