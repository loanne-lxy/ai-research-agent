"""v1 阶段验收测试 — 3-question acceptance against the real model, no mocks.

Purpose: prove the v1 research agent actually works end to end, so the
version can be signed off. Re-runnable at any time to confirm the three
v1 guarantees still hold (regression gate before moving to v2):

Q1 检索准确  — ask about W36 Agent trends; agent must actually call the
               search tools, and every event it cites must exist in the DB
               with week == 2026-W36.
Q2 引用真实  — every citation in the answer resolves (0 fabricated).
Q3 多轮上下文 — a follow-up referring to "刚才提到的那条" must reuse Q1's
               event id (proof context carried over), not re-guess.

Run:  .venv/bin/python acceptance.py
Exit 0 = all pass.
"""
from __future__ import annotations

import re
import sys

from agentscope.message import UserMsg

from agent.builder import build_agent
from citations import validate
from corpus import resolve

EVENT_RE = re.compile(r"evt_[0-9a-f]{12}")


def answer_text(resp) -> str:
    if isinstance(resp.content, str):
        return resp.content
    return "\n".join(
        b.text for b in resp.content
        if getattr(b, "type", None) == "text" and getattr(b, "text", None))


def tool_calls_in_state(agent) -> list[str]:
    """Tool calls executed so far, read back from agent.state.context
    (the final reply Msg only carries text; the ReAct loop's tool
    calls live in the context history)."""
    names = []
    for msg in agent.state.context:
        c = msg.content
        if not isinstance(c, list):
            continue
        for b in c:
            if getattr(b, "type", None) == "tool_call" and getattr(b, "name", None):
                names.append(b.name)
    return names


async def ask(agent, q: str, tries: int = 3) -> object:
    """Ask with retry: the LLM endpoint occasionally returns empty."""
    for i in range(tries):
        resp = await agent.reply(UserMsg(name="user", content=q))
        if answer_text(resp).strip():
            return resp
        print(f"  (第 {i + 1} 次返回空，重试…)")
    raise RuntimeError("model returned empty after retries")


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}  {detail}")
    return ok


async def main() -> int:
    agent = build_agent()
    results = []

    print("Q1 检索准确 — W36 Agent 方向重要进展（3 条）")
    r1 = await ask(agent, "W36 这周 AI Agent 方向有什么重要进展？给出 3 条。")
    a1 = answer_text(r1)
    print(a1[:400] + ("…" if len(a1) > 400 else ""))
    ev_ids = EVENT_RE.findall(a1)
    tools = tool_calls_in_state(agent)
    results.append(check("调用了检索工具", "search_events" in tools or "get_event" in tools,
                         f"tools={tools}"))
    bad = [e for e in ev_ids if resolve(e) is None]
    wrong_week = [
        e for e in ev_ids
        if (m := resolve(e)) and m.get("object", {}).get("week") != "2026-W36"]
    results.append(check("引用事件全部真实存在", not bad and len(ev_ids) >= 1,
                         f"{len(ev_ids)} 个事件ID, 假ID={bad}"))
    results.append(check("全部属于 W36", not wrong_week, f"周不符={wrong_week}"))

    print("\nQ2 引用真实 — Q1 回答中每个 citation 可反查")
    v1 = validate(a1)
    print(f"  citations: {v1['total']}, missing: {v1['missing']}")
    results.append(check("0 个编造引用", v1["total"] > 0 and not v1["missing"],
                         f"unverified={v1['missing']}"))

    print("\nQ3 多轮上下文 — 追问“刚才提到的第 1 条事件”")
    top_evt = ev_ids[0] if ev_ids else ""
    r3 = await ask(agent, "刚才提到的第 1 条事件，标题完整是什么？不需要重新查库，直接回答。")
    a3 = answer_text(r3)
    print(a3[:400] + ("…" if len(a3) > 400 else ""))
    results.append(check("复用了 Q1 的事件ID（上下文生效）",
                         bool(top_evt) and top_evt in a3,
                         f"期望出现 {top_evt}"))
    v3 = validate(a3)
    results.append(check("追问回答引用仍可反查", not v3["missing"],
                         f"unverified={v3['missing']}"))

    print("\n" + "=" * 52)
    print(f"验收结果: {sum(results)}/{len(results)} 通过")
    if not all(results):
        print("FAIL 项存在，见上。")
    return 0 if all(results) else 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))