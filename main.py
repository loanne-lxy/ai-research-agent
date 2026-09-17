"""CLI REPL — the user-facing entry point of the research agent.

Run:  .venv/bin/python main.py

Multi-turn: one agent instance is kept for the whole session;
AgentState.context accumulates history automatically, so follow-ups
like "刚才提到的那个展开讲讲" resolve. /save and /load persist that
state across processes (see session.py).

The research turn itself is delegated to the shared V2 loop
(agent.research_agent.run_research_loop) — the same engine the AgentScope
web service drives. This file only handles the REPL (commands, I/O,
persistence); the loop logic lives in one place.

Commands:  /quit exit   /reset start a fresh session   /tools list tools
            /save persist session to disk   /load resume (latest or /load <id>)
            /sessions list saved sessions   /state show research memory
"""
from __future__ import annotations

import asyncio

from agentscope.event import HintBlockEvent

from agent.builder import build_agent
from agent.research_agent import run_research_loop
from intent import classify_intent
from research import format_research_memory
from session import (SESSION_DIR, list_sessions, load_research_memory,
                     load_state, save_research_memory, save_state)


async def run_repl() -> None:
    agent = await build_agent()
    active_memory = None  # last research working memory in this session
    print("AI 研究专家（知识库快照，/tools 看工具，/reset 清会话，/quit 退出）")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/reset":
            agent = await build_agent()
            active_memory = None
            print("（新会话已建立）")
            continue
        if q == "/tools":
            schemas = await agent.toolkit.get_tool_schemas()
            for s in schemas:
                f = s["function"]
                print(f"  {f['name']}: {f['description'].splitlines()[0]}")
            continue
        if q == "/save":
            p = save_state(agent.state)
            if active_memory is not None:
                save_research_memory(agent.state.session_id, active_memory)
            print(f"（已保存 {p}，重启后 /load 恢复）")
            continue
        if q.startswith("/load"):
            arg = q.split(maxsplit=1)[1].strip() if " " in q else None
            st = load_state(SESSION_DIR / f"{arg}.json" if arg else None)
            if st is None:
                print("（没有可恢复的会话）")
            else:
                agent.state = st
                active_memory = load_research_memory(st.session_id)
                print(f"（已恢复 {st.session_id}，{len(st.context)} 条历史消息"
                      f"{' + 研究状态' if active_memory else ''}）")
            continue
        if q == "/sessions":
            rows = list_sessions()
            print("\n".join(f"  {r}" for r in rows) if rows else "（无已保存会话）")
            continue
        if q == "/state":
            if active_memory is None:
                print("（本会话尚无研究状态；问一个研究类问题后会出现）")
            else:
                print(format_research_memory(active_memory))
            continue
        it = classify_intent(q)
        print(f"（intent: {it.intent} — {it.reason}）")
        # Delegate the turn to the shared V2 research loop (also drives the
        # web service): intent → plan → research → evaluate → gap → follow-up
        # + citation validation. Progress streams out as HintBlockEvents; the
        # final answer and the research working memory come back via `out`.
        out: dict = {}
        try:
            async for ev in run_research_loop(q, agent, build_agent, out):
                if isinstance(ev, HintBlockEvent):
                    print(ev.hint)
        except Exception as e:  # noqa: BLE001 - REPL must survive model hiccups
            print(f"[error] {type(e).__name__}: {e}（可重试，知识库未受影响）")
            continue
        print(f"< {out.get('answer', '')}")
        if out.get("result") is not None:
            active_memory = out["result"]
    print("bye")


if __name__ == "__main__":
    asyncio.run(run_repl())