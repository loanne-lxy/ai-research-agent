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

from agentscope.agent import Agent
from agentscope.event import AgentEvent
from agentscope.message import Msg
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState

from research.controller import run_research_loop


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


__all__ = ["ResearchAgent", "extract_text", "format_evidence_report"]
