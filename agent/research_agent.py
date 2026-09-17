"""Research agent for the AgentScope Agent Service (web) — reuses the V2 loop.

Wraps the *existing* Research Loop (intent → plan → research → evidence
eval → gap → follow-up, bounded by MAX_ROUNDS) into an AgentScope
``Agent`` subclass whose ``reply_stream`` drives the service's Web UI.

Why a subclass (not a middleware, not a new orchestrator):
  The Agent Service assembles each turn from stored records — model from
  the session's credential, toolkit from the workspace + extra_agent_tools,
  system_prompt from the agent record — then calls
  ``agent.reply_stream(inputs=msg)`` and persists the turn by *applying the
  events* it streams (see agentscope/app/_service/_chat.py). So the only
  hook the service exposes for custom turn behaviour is a custom
  ``agent_cls`` + its ``reply_stream``. We override exactly that and run
  the V2 loop inside it, delegating each research round to the *base*
  ``Agent.reply_stream`` (the ReAct search loop) so every model/tool/text
  event streams to the UI unchanged.

Deliberately NOT done (ponytail):
  - No multi-agent / swarm. The V2 loop is one agent + a *stateless*
    revision round (a fresh AgentState) to keep 3-round research from
    stacking in one context — the same fix the CLI uses (E2E blew 103k
    tokens). The revision agent is tagged with the session's id so its
    events still land in the same UI conversation.
  - Research-round progress is surfaced as ``HintBlockEvent`` (the
    framework's ``HintBlock``), which the official Web UI renders as a
    system-hint block. ``CustomEvent`` is *skipped* by the UI for unknown
    names, so a hint block is the renderable channel for progress.
"""
from __future__ import annotations

import uuid
from typing import AsyncGenerator, Awaitable, Callable

from agentscope.agent import Agent
from agentscope.event import (
    AgentEvent,
    HintBlockEvent,
    ReplyStartEvent,
)
from agentscope.message import Msg, TextBlock
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState

from citations import extract_claims, format_report, validate
from intent import FACT, classify_intent
from research import (
    MAX_ROUNDS,
    ResearchState,
    build_followup_queries,
    build_research_memory,
    build_research_query,
    build_revision_query,
    claim_sources,
    evaluate_evidence,
    new_followups,
    research_plan_from_query,
    set_research_model,
    should_continue,
)


# ------------------------------------------------------------- text helpers
def extract_text(msg: Msg) -> str:
    """Render a reply Msg to plain text: text blocks + a one-line marker
    per tool call (so the user sees the agent actually searched). Thinking /
    raw tool-result blocks are skipped. (Moved from main.py — shared by the
    CLI REPL and the web research loop.)"""
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


def format_evidence_report(issues: dict) -> str:
    """Per-direction evidence counts + gaps/conflicts (moved from main.py)."""
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


def _user_msg(text: str) -> Msg:
    """A well-formed user Msg: AgentScope Msg needs role + block-list
    content (a bare string is rejected by pydantic)."""
    return Msg(name="user", role="user", content=[TextBlock(text=text)])


def _hint(reply_id: str, text: str, source: str = "ResearchAgent") -> HintBlockEvent:
    """A one-shot progress hint for the UI (rendered as a system-hint block)."""
    return HintBlockEvent(
        reply_id=reply_id, block_id=str(uuid.uuid4()), source=source, hint=text,
    )


# ------------------------------------------------------- shared research loop
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

    # ---- bounded evaluate → gap → follow-up loop ----
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


# ------------------------------------------------------------- the Agent
class ResearchAgent(Agent):
    """The V2 research agent, driven by the AgentScope Agent Service.

    ``__init__`` is the *exact* signature the service's ChatService passes
    (name, system_prompt, model, toolkit, model_config, context_config,
    react_config, state, middlewares, offloader). ``reply_stream`` runs the
    research loop and yields the events the service streams to the Web UI.
    """

    async def reply_stream(self, inputs=None, structured_schema=None,
                           yield_final_msg: bool = False):
        # Continuation inputs (confirm / external-exec results) resume a
        # parked base reply — delegate straight to the base loop, no
        # research fork.
        if inputs is not None and not isinstance(inputs, (Msg, list)):
            async for ev in super().reply_stream(
                    inputs=inputs, structured_schema=structured_schema,
                    yield_final_msg=yield_final_msg):
                yield ev
            return

        question = _first_user_text(inputs)
        out: dict = {}
        from agent.builder import get_research_model
        async for ev in run_research_loop(
                question, self, self._make_revision_agent, out,
                model=get_research_model()):
            yield ev

    async def _make_revision_agent(self) -> "ResearchAgent":
        """A stateless revision agent sharing this agent's model/prompt/
        tools/config but with a fresh context (DONT_ASK, same session id so
        its events stream to this session). The ReMe middleware is re-passed
        (process-wide singleton, same instance the parent was built with) so
        the revision round's exchange is written back to long-term memory
        like any other turn. ``async`` to match ``make_agent()`` in
        ``run_research_loop`` (the CLI's ``build_agent`` is async)."""
        from agent.memory import get_memory_middleware

        return ResearchAgent(
            name=self.name,
            system_prompt=self._system_prompt,
            model=self.model,
            toolkit=self.toolkit,
            model_config=self.model_config,
            context_config=self.context_config,
            react_config=self.react_config,
            injection_config=self.injection_config,
            middlewares=[get_memory_middleware()],
            state=AgentState(
                session_id=self.state.session_id,
                permission_context=PermissionContext(mode=PermissionMode.DONT_ASK),
            ),
            offloader=self.offloader,
        )


def _first_user_text(inputs) -> str:
    """Pull the question text out of the service's input (Msg / list[Msg])."""
    if inputs is None:
        return ""
    msgs = inputs if isinstance(inputs, list) else [inputs]
    for m in reversed(msgs):
        c = getattr(m, "content", None)
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            for b in c:
                t = getattr(b, "type", None)
                txt = getattr(b, "text", None)
                if t == "text" and txt:
                    return txt
    return ""


__all__ = ["ResearchAgent", "run_research_loop",
           "extract_text", "format_evidence_report"]