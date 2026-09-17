"""Targeted check for the Step-7 architecture: does the revision agent
search on its own (Evaluator gap -> Agent tool_call -> observation)?

E2E runs depend on the LLM evaluator returning 'insufficient' AND not
hitting the endpoint's intermittent empty-200s, which is flaky. This
script drives the revision path directly (same call main.py makes) and
inspects agent.state.context — the ground truth for whether the agent
issued its own search_events/search_articles tool_calls.

Pass: the revision agent made >=1 search tool_call on its own.
"""
import asyncio
import json

from agentscope.message import UserMsg

from agent.builder import build_agent
from main import extract_text
from research import build_revision_query


def _tool_calls(state):
    """Return the list of tool-call names the agent actually made, by
    scanning every context message for tool_call blocks."""
    calls = []
    for msg in state.context:
        c = getattr(msg, "content", None)
        blocks = c if isinstance(c, list) else []
        for b in blocks:
            btype = (getattr(b, "type", None)
                     or (b.get("type") if isinstance(b, dict) else None))
            if btype == "tool_call":
                name = (getattr(b, "name", None)
                        or (b.get("name") if isinstance(b, dict) else None))
                if name:
                    calls.append(name)
    return calls


async def main() -> int:
    issues = {
        "verdict": "insufficient",
        "sufficient": False,
        "ok": [],
        "issues": "Agent 推理优化/评测成本方面证据不足，"
                  "缺少 2026 Q2 后的最新季度评测基准数据",
    }
    answer = "初版：Agent 工具调用与 MCP 生态统一在加速（趋势：基础设施标准化）。"
    suggested = ["agent 评测基准 2026 Q2", "推理成本优化 最新数据"]

    msg = build_revision_query("2026 Q2 以来 Agent 领域的发展趋势是什么？",
                               issues, suggested_queries=suggested,
                               answer=answer)
    print("=== 修订消息（交给全新 agent）===")
    print(msg[:400], "...\n")

    # same call main.py makes for the revision round
    agent = await build_agent()
    resp = await agent.reply(UserMsg(name="user", content=msg))

    print("=== agent 自主发起的 tool_call ===")
    calls = _tool_calls(agent.state)
    for i, n in enumerate(calls, 1):
        print(f"  {i}. {n}")

    search_calls = [c for c in calls if c in ("search_events", "search_articles")]
    text = extract_text(resp)
    print("\n=== 最终回答开头 ===")
    print(text[:300], "...\n")

    ok = len(search_calls) >= 1
    print(f"RESULT: {'PASS' if ok else 'FAIL'} — agent 自主搜索 tool_call "
          f"{len(search_calls)} 次 (全部 tool_call {len(calls)} 次)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))