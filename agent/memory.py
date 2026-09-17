"""Long-term memory — the process-wide ReMeMiddleware singleton.

Four states, four homes (keep them apart):
  1. Conversation State   -> AgentState.context (AgentScope session
     mechanism; SQLite session record on the web service, AgentState +
     sessions/*.json in the CLI). Per-session, automatic.
  2. Research Working     -> ResearchState + build_research_memory() in
     research.py, sidecar sessions/<id>.research.json. Task-level, lives
     only for one research run. Never feeds ReMe.
  3. Long-term Experience -> THIS module. ReMe workspace under data/memory/
     (daily cards + digest + index). The official ReMeMiddleware writes it
     back automatically after every reply (LLM extraction over this turn's
     increment) and consolidates nightly (dream). Cross-session.
  4. Knowledge Base       -> corpus/ (KAL: Event/Article/Source). Domain
     corpus snapshot from weekly-ai-report; reached only through the
     knowledge tools. It is not agent memory and ReMe never touches it.

ReMe is embedded in-process by the middleware (no separate service); the
embedded app starts lazily on first use. One middleware instance per
process, shared by the main agent and the stateless revision agent (ReMe
keys per-conversation state by session_id read live from the agent).
"""
from __future__ import annotations

from agentscope.credential import OpenAICredential
from agentscope.middleware import ReMeMiddleware
from agentscope.model import OpenAIChatModel

import config as C

MEMORY_DIR = C.PROJECT_ROOT / "data" / "memory"

_middleware: ReMeMiddleware | None = None


def get_memory_middleware() -> ReMeMiddleware:
    """The process-wide ReMeMiddleware (lazy)."""
    global _middleware
    if _middleware is None:
        llm = C.load_llm_config()
        credential = OpenAICredential(api_key=llm.api_key, base_url=llm.base_url)
        # Separate model instance for ReMe's background jobs (auto_memory
        # extraction, dream). Same endpoint, no thinking: extraction is a
        # short-horizon background job and must not burn reasoning budget.
        extraction_model = OpenAIChatModel(
            credential=credential,
            model=llm.model,
            parameters=OpenAIChatModel.Parameters(
                temperature=llm.temperature,
                max_tokens=llm.max_tokens,
                thinking_enable=False,
            ),
        )
        _middleware = ReMeMiddleware(
            workspace_dir=str(MEMORY_DIR),
            parameters=ReMeMiddleware.Parameters(
                chat_model=extraction_model,
                mode="both",  # auto-retrieve/inject + memory_search tool
                top_k=5,
            ),
        )
    return _middleware