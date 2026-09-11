"""CLI REPL — the user-facing entry point of the research agent.

Run:  .venv/bin/python main.py

Multi-turn: one agent instance is kept for the whole session;
AgentState.context accumulates history automatically, so follow-ups
like "刚才提到的那个展开讲讲" resolve. /save and /load persist that
state across processes (see session.py).

Commands:  /quit exit   /reset start a fresh session   /tools list tools
            /save persist session to disk   /load resume (latest or /load <id>)
            /sessions list saved sessions   /state show research memory
"""
from __future__ import annotations

import asyncio

from agentscope.message import UserMsg

from agent.builder import build_agent
from citations import extract_claims, format_report, validate
from intent import FACT, classify_intent
from research import (MAX_ROUNDS, ResearchState, build_research_memory,
                      build_research_query, build_revision_query,
                      detect_gaps, evaluate_evidence, format_research_memory,
                      new_followups, research_plan_from_query,
                      should_continue)
from session import (SESSION_DIR, list_sessions, load_research_memory,
                     load_state, save_research_memory, save_state)


def format_evidence_report(issues: dict) -> str:
    """Per-direction evidence counts + gaps/conflicts, so the user sees
    the evaluation layer worked (one line per direction, ✗ = 明显不足)."""
    lines = ["【证据评估】"]
    for d in issues.get("directions", []):
        mark = "✓" if d.get("sufficient") else "✗"
        conflict = "⚠ 有矛盾" if d.get("conflict") else ""
        note = f" — {d['note']}" if d.get("note") else ""
        lines.append(
            f"  {mark} {d.get('name', '?')}：{d.get('events', 0)} events / "
            f"{d.get('articles', 0)} sources {conflict}{note}".rstrip()
        )
    for g in issues.get("gaps", []):
        lines.append(f"  缺口：{g}")
    for c in issues.get("conflicts", []):
        lines.append(f"  矛盾：{c}")
    return "\n".join(lines)


def extract_text(msg) -> str:
    """Render a reply Msg to plain text: text blocks + a one-line marker
    per tool call (so the user sees the agent actually searched).
    Thinking / raw tool-result blocks are skipped."""
    c = msg.content
    if isinstance(c, str):
        return c
    parts = []
    for b in c:
        btype = (getattr(b, "type", None)
                 or (b.get("type") if isinstance(b, dict) else None))
        if btype == "text":
            t = getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
            if t:
                parts.append(t)
        elif btype == "tool_call":
            name = getattr(b, "name", None) or (b.get("name") if isinstance(b, dict) else None)
            parts.append(f"  [检索] {name}")
    return "\n".join(parts)


async def run_repl() -> None:
    agent = build_agent()
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
            agent = build_agent()
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
        # research path (v2): plan first — the plan becomes the agent's
        # working state in context; fact questions run the v1 path as-is.
        research = it.intent != FACT
        query = build_research_query(q) if research else q
        issues = {}  # last evidence-evaluation dict (drives working memory)
        rstate = ResearchState()
        try:
            resp = await agent.reply(UserMsg(name="user", content=query))
        except Exception as e:  # noqa: BLE001 - REPL must survive model hiccups
            print(f"[error] {type(e).__name__}: {e}（可重试，知识库未受影响）")
            continue
        answer = extract_text(resp)
        if research:  # v2 research loop: Research → Evaluate → Gap →
            # Follow-up, repeated until evidence is sufficient, the round
            # budget is exhausted, or no new follow-up queries remain.
            while True:
                issues = evaluate_evidence(q, research_plan_from_query(query),
                                           answer)
                if not issues:  # evaluator failed after retries -> non-blocking stop
                    print("（证据评估失败，按当前版本定稿）")
                    break
                if issues.get("directions"):
                    print(format_evidence_report(issues))
                if (issues.get("verdict") != "needs_work"
                        or not (issues.get("gaps") or issues.get("conflicts"))):
                    break
                # STOP_CONDITION pre-gate: don't spend an LLM call once the
                # round budget is reached (rstate.round = rounds completed).
                if rstate.round >= MAX_ROUNDS:
                    print(f"（停止：轮次预算用尽（{MAX_ROUNDS}轮），按当前版本定稿）")
                    break
                new_queries = new_followups(rstate, detect_gaps(q, issues))
                if not should_continue(rstate, issues, new_queries):
                    print(f"（停止：无新增补充检索词，按当前版本定稿）")
                    break
                rstate.round += 1  # = completed supplementary rounds
                print(f"（第{rstate.round}轮：证据不足/存在矛盾，由Agent按缺口自行补充检索…）")
                for qq in new_queries:
                    print(f"  建议检索: {qq}")
                try:
                    # stateless revision: a fresh agent, so the 3-round
                    # research history never stacks in one context (E2E
                    # blew 103k tokens and the framework truncation wiped
                    # the task + citations block). The agent does the
                    # supplementary research ITSELF (search_events/
                    # search_articles); the loop only suggests queries.
                    rev_agent = build_agent()
                    rev = await rev_agent.reply(UserMsg(
                        name="user",
                        content=build_revision_query(q, issues,
                                                     suggested_queries=new_queries,
                                                     answer=answer)))
                    answer = extract_text(rev)
                except Exception as e:  # noqa: BLE001 - keep the draft
                    print(f"[error] 修订失败，沿用上一版：{type(e).__name__}")
                    break
        print(f"< {answer}")
        try:
            cit = validate(answer)
            claims = extract_claims(answer)  # claim -> evidence ids (generated, not post-hoc)
            print(format_report(cit))
        except Exception as e:  # noqa: BLE001 - validation must never kill the REPL
            print(f"[citations] check failed: {type(e).__name__}: {e}")
            cit = {"ok": [], "total": 0, "missing": []}
            claims = []
        if research:  # assemble the working memory from what actually ran
            active_memory = build_research_memory(
                q, it.intent, research_plan_from_query(query), rstate,
                issues, cit, claims)
    print("bye")


if __name__ == "__main__":
    asyncio.run(run_repl())