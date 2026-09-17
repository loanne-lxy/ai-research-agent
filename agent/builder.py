"""Assembly of the research agent: model + knowledge tools + system prompt
+ long-term memory (ReMe).

build_agent() is the single factory every entry point uses, so model
wiring, permissions, and the toolset stay consistent:
  model      — config.load_llm_config (.env, same credentials as the
               weekly-ai-report pipeline)
  tools      — tools.knowledge.knowledge_toolkit (read-only KAL tools)
  memory     — agent.memory.get_memory_middleware (official ReMe
               middleware; async because ReMe's memory_search tool is
               built async)
  prompt     — agent/system_prompt.md (editable without code changes)
  permission — DONT_ASK: all knowledge tools are is_read_only=True,
               so they auto-approve and the CLI never blocks on a
               confirmation prompt.
"""
from __future__ import annotations

from pathlib import Path

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState

from config import load_llm_config
from tools.knowledge import knowledge_toolkit

PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.md"


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


async def build_agent() -> Agent:
    from agent.memory import get_memory_middleware

    llm = load_llm_config()
    credential = OpenAICredential(api_key=llm.api_key, base_url=llm.base_url)
    params = OpenAIChatModel.Parameters(
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
    )
    model = OpenAIChatModel(credential=credential, model=llm.model,
                            parameters=params)
    memory = get_memory_middleware()
    toolkit = knowledge_toolkit()
    # ReMe's search tool joins the agent's basic tool group (async build).
    await toolkit.add_tool(await memory.list_tools())
    return Agent(
        name="ResearchExpert",
        system_prompt=load_system_prompt(),
        model=model,
        toolkit=toolkit,
        middlewares=[memory],
        state=AgentState(
            permission_context=PermissionContext(mode=PermissionMode.DONT_ASK),
        ),
    )


_research_model: "OpenAIChatModel | None" = None


def get_research_model() -> OpenAIChatModel:
    """Unified model for the research controller's structured-JSON LLM calls
    (planner / evidence evaluator / follow-up generator).

    Same endpoint + credential as build_agent — one ``load_llm_config()``
    source, so there is no second .env read and no second base_url/api_key/
    model to keep in sync. Parameters are left open (no fixed budget or
    temperature) and thinking is off, so each call passes its own
    ``temperature`` / ``max_tokens`` via generate_kwargs (these flow into the
    OpenAI request without colliding with the model's fixed params). One
    instance per process, like the ReMe middleware.
    """
    global _research_model
    if _research_model is None:
        llm = load_llm_config()
        credential = OpenAICredential(api_key=llm.api_key, base_url=llm.base_url)
        _research_model = OpenAIChatModel(
            credential=credential,
            model=llm.model,
            # open params: per-call temperature/max_tokens, thinking off
            parameters=OpenAIChatModel.Parameters(),
            stream=False,
            max_retries=0,  # research._llm owns the retry loop
            extra_body={"enable_thinking": False},
        )
    return _research_model