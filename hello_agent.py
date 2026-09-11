"""Hello Agent — end-to-end smoke check of the LLM chain.

Minimal AgentScope 2.0 agent: one model call plus one trivial tool
(get_current_time). Use it to verify the Qwen endpoint and the
AgentScope wiring before working on the research agent itself.

Chain: Python -> AgentScope (ReAct) -> LLM (Qwen3.8-27B) -> Answer

Run:  .venv/bin/python hello_agent.py
"""
from __future__ import annotations

import asyncio
import datetime

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, Toolkit

from config import load_llm_config


def get_current_time() -> str:
    """Return the current date and time (UTC+8, Beijing).

    Use this tool whenever the user asks about the current date or time.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def build_agent() -> Agent:
    llm = load_llm_config()
    credential = OpenAICredential(api_key=llm.api_key, base_url=llm.base_url)
    params = OpenAIChatModel.Parameters(
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
    )
    model = OpenAIChatModel(credential=credential, model=llm.model, parameters=params)
    toolkit = Toolkit(tools=[FunctionTool(get_current_time, is_read_only=True)])
    # Unattended CLI: DONT_ASK auto-allows read-only tools, refuses anything
    # that would otherwise prompt a (absent) user — never blocks.
    state = AgentState(
        permission_context=PermissionContext(mode=PermissionMode.DONT_ASK),
    )
    agent = Agent(
        name="ResearchExpert",
        system_prompt=(
            "You are an AI domain research expert. Answer concisely in Chinese. "
            "If the user asks about the current date or time, call the "
            "get_current_time tool first and use its result."
        ),
        model=model,
        toolkit=toolkit,
        state=state,
    )
    return agent


def extract_text(msg) -> str:
    """Render a Msg (text block list or plain string) to plain text."""
    c = msg.content
    if isinstance(c, str):
        return c
    parts = []
    for block in c:
        if isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_call":
                parts.append(f"  [tool call] {block.get('name') or block.get('input')}")
            # thinking / tool_result blocks: skip (debug only)
        else:
            btype = getattr(block, "type", "text")
            if btype == "text":
                parts.append(getattr(block, "text", str(block)))
    return "\n".join(p for p in parts if p)


async def main() -> None:
    agent = build_agent()
    questions = [
        "你好，介绍一下你自己，一两句话就行。",
        "现在几点了？",
    ]
    for q in questions:
        print(f"\n> {q}")
        resp = await agent.reply(UserMsg(name="user", content=q))
        print(f"< {extract_text(resp)}")


if __name__ == "__main__":
    asyncio.run(main())