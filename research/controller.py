"""Research controller — the only place that orchestrates the loop.

    query -> intent -> planning -> research -> evaluation -> gap detection
          -> follow-up -> synthesis -> citation validation

Every round is a base ``Agent.reply_stream`` (ReAct) turn; the controller
only decides *when* to evaluate, *whether* to keep going (state.should_continue),
and *what* to hand the next agent (planner / evidence / followup prompts).
It streams the round's AgentEvents to the caller and populates ``out`` with
the final answer + the research working memory dict.

The structured-JSON LLM calls (planner / evaluator / follow-up) use the
unified AgentScope model injected via ``model`` (DI); when omitted they fall
back to the shared factory singleton — the controller never builds a model
of its own.
"""
from __future__ import annotations

import uuid
from typing import AsyncGenerator, Awaitable, Callable

from agentscope.agent import Agent
from agentscope.event import AgentEvent, HintBlockEvent
from agentscope.message import Msg, TextBlock

from citations import extract_claims, format_report, validate
from intent import FACT, classify_intent
from research.evidence import (
    build_revision_query,
    claim_sources,
    evaluate_evidence,
)
from research.followup import build_followup_queries
from research.llm import set_research_model
from research.memory import build_research_memory
from research.planner import build_research_query, research_plan_from_query
from research.state import (
    MAX_ROUNDS,
    ResearchState,
    new_followups,
    should_continue,
)


def _user_msg(text: str) -> Msg:
    """A well-formed user Msg: AgentScope Msg needs role + block-list
    content (a bare string is rejected by pydantic)."""
    return Msg(name="user", role="user", content=[TextBlock(text=text)])


def _hint(reply_id: str, text: str, source: str = "ResearchAgent") -> HintBlockEvent:
    """A one-shot progress hint for the UI (rendered as a system-hint block)."""
    return HintBlockEvent(
        reply_id=reply_id, block_id=str(uuid.uuid4()), source=source, hint=text,
    )


def _current_reply_id(agent: Agent, fallback: str) -> str:
    """reply_id a hint block should tag with: the live round's reply_id if
    the base round has started, else a stable placeholder for the turn."""
    rid = getattr(agent.state, "reply_id", None)
    return rid if rid else fallback


async def run_research_loop(
    question: str,
    main_agent: Agent,
    make_agent: Callable[[], Awaitable[Agent]],
    out: dict | None = None,
    model=None,
) -> AsyncGenerator[AgentEvent, None]:
    """Run the V2 research loop, streaming AgentEvents to the caller.

    ``main_agent`` handles round 1 in its own (persistent) state; each
    follow-up round runs on a *fresh* agent from ``make_agent()`` (stateless
    revision — the same context-blowup fix the CLI uses). Every round's
    events are yielded so the service can stream + persist them; progress
    markers ride HintBlockEvents. ``make_agent`` is async because building
    the agent awaits the ReMe memory-tool registration.

    ``model`` is the unified AgentScope model injected by the caller (DI)
    for the structured-JSON LLM calls (planner / evidence evaluator /
    follow-up generator). When omitted (CLI self-checks) it falls back to the
    shared factory singleton — research never builds its own model.

    ``out`` (optional dict) is populated on completion:
      answer      final answer text
      final_msg   the final round's Msg
      result      the research working memory dict (build_research_memory)
      intent      the classified intent string

    Yields:
      AgentEvent — model / tool / text / hint events for the whole turn.
    """
    # Lazy: agent.research_agent imports this module at top level, so pulling
    # its text helpers here would be a circular import at module load.
    from agent.research_agent import extract_text, format_evidence_report

    if model is not None:
        set_research_model(model)
    out = out if out is not None else {}
    it = classify_intent(question)
    out["intent"] = it.intent
    if it.intent == FACT:
        # Simple QA: single ReAct round, no research loop (v1 path as-is).
        agen = Agent.reply_stream(main_agent, _user_msg(question),
                                  yield_final_msg=True)
        final_msg = None
        async for ev in agen:
            if isinstance(ev, Msg):
                final_msg = ev
            else:
                yield ev
        if final_msg is None:
            raise RuntimeError("Agent did not produce a final message.")
        out["answer"] = extract_text(final_msg)
        out["final_msg"] = final_msg
        out["result"] = {"query": question, "intent": FACT}
        return

    research = True
    query = await build_research_query(question)
    issues: dict = {}
    rstate = ResearchState()

    # ---- round 1 (main agent, persistent state) ----
    agen = Agent.reply_stream(main_agent, _user_msg(query),
                              yield_final_msg=True)
    final_msg: Msg | None = None
    async for ev in agen:
        if isinstance(ev, Msg):
            final_msg = ev
        else:
            yield ev
    if final_msg is None:
        raise RuntimeError("Agent did not produce a final message.")
    answer = extract_text(final_msg)

    # ---- bounded evaluate -> gap -> follow-up loop ----
    while True:
        issues = await evaluate_evidence(
                    question, research_plan_from_query(query), answer, claim_sources(answer))
        if issues.get("verdict") == "audit_failed":  # fail-closed
            yield _hint(_current_reply_id(main_agent, "audit"),
                        "⚠️ 证据审核不可用（audit_failed）：无法证明充分性，降级定稿")
            break
        if issues.get("directions"):
            yield _hint(_current_reply_id(main_agent, "eval"),
                        format_evidence_report(issues))
        if (issues.get("verdict") != "needs_work"
                or not (issues.get("gaps") or issues.get("conflicts"))):
            break
        if rstate.round >= MAX_ROUNDS:
            yield _hint(_current_reply_id(main_agent, "stop"),
                        f"（停止：轮次预算用尽（{MAX_ROUNDS}轮），按当前版本定稿）")
            break
        new_queries = new_followups(
            rstate, await build_followup_queries(question, issues))
        if not should_continue(rstate, issues, new_queries):
            yield _hint(_current_reply_id(main_agent, "stop"),
                        "（停止：无新增补充检索词，按当前版本定稿）")
            break
        rstate.round += 1
        yield _hint(_current_reply_id(main_agent, "round"),
                    f"（第{rstate.round}轮：证据不足/存在矛盾，由Agent按缺口自行补充检索… "
                    + "；".join(new_queries) + "）")
        try:
            # stateless revision on a fresh agent (see module docstring).
            rev_agent = await make_agent()
            rev_agen = Agent.reply_stream(
                rev_agent,
                _user_msg(build_revision_query(
                    question, issues, suggested_queries=new_queries, answer=answer)),
                yield_final_msg=True,
            )
            rev_msg: Msg | None = None
            async for ev in rev_agen:
                if isinstance(ev, Msg):
                    rev_msg = ev
                else:
                    yield ev
            if rev_msg is None:
                raise RuntimeError("revision produced no message")
            answer = extract_text(rev_msg)
            final_msg = rev_msg
        except Exception as e:  # noqa: BLE001 - keep the draft, like the CLI
            yield _hint(_current_reply_id(main_agent, "err"),
                        f"[error] 修订失败，沿用上一版：{type(e).__name__}")
            break

    if research and issues.get("verdict") == "audit_failed":
        yield _hint(_current_reply_id(main_agent, "warn"),
                    "⚠️ 证据审核不可用（audit_failed）：本轮未完成充分性验证，"
                    "以下答案未经证据审核")

    # ---- citation validation + working memory ----
    try:
        cit = validate(answer)
        claims = extract_claims(answer)
        yield _hint(_current_reply_id(main_agent, "cite"),
                    format_report(cit))
    except Exception as e:  # noqa: BLE001 - validation must never kill the turn
        yield _hint(_current_reply_id(main_agent, "cite_err"),
                    f"[citations] check failed: {type(e).__name__}: {e}")
        cit = {"ok": [], "total": 0, "missing": []}
        claims = []

    out["answer"] = answer
    out["final_msg"] = final_msg
    out["result"] = build_research_memory(
        question, it.intent, research_plan_from_query(query), rstate,
        issues, cit, claims)
    return


__all__ = ["run_research_loop", "MAX_ROUNDS", "ResearchState",
           "new_followups", "should_continue", "build_research_memory",
           "format_research_memory", "make_plan", "evaluate_evidence",
           "build_followup_queries", "build_research_query",
           "build_revision_query", "claim_sources", "evidence_map",
           "research_plan_from_query", "set_research_model"]