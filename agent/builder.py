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