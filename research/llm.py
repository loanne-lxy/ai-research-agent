"""Unified LLM client for the research pipeline's structured-JSON calls.

The ONE model the planner / evidence evaluator / follow-up generator use.
It's the AgentScope ``OpenAIChatModel`` injected by the controller
(``set_research_model``) — research never builds its own client, re-reads
.env, or keeps a second base_url/api_key/model. The self-check and any
standalone call fall back to the shared factory singleton when not yet
injected (``_resolve_model``), so the whole pipeline has a single model
source.
"""
from __future__ import annotations

from agentscope.message import Msg, TextBlock

# The injected unified AgentScope model (set by run_research_loop via
# set_research_model). None until the controller injects it.
_research_model = None


def set_research_model(model) -> None:
    """Inject the process-wide AgentScope model (DI from the controller)."""
    global _research_model
    _research_model = model


def _resolve_model():
    """The injected model, or the shared factory singleton as a fallback."""
    if _research_model is not None:
        return _research_model
    from agent.builder import get_research_model
    return get_research_model()


async def _llm(prompt: str, max_tokens: int = 2000) -> str:
    """One LLM call via the unified AgentScope model, 3 attempts; returns
    raw content or "" on failure.

    enable_thinking=False on purpose: Qwen3.8 is a reasoning model, and for
    these *structured-JSON* tasks its thinking is pure overhead that eats the
    completion budget (observed: 6.3k thinking tokens, empty content,
    finish_reason=length — on both a 3.6k-token full draft and a 2.5k-token
    evidence map). With thinking off the same call is ~2s and returns the
    JSON. max_tokens is still generous; the endpoint also flaps with empty
    200s, so the retry loop below (the SDK's own retries are off) covers it.
    """
    import time

    model = _resolve_model()
    budget = max(max_tokens, 2 * len(prompt))
    msg = Msg(name="user", role="user", content=[TextBlock(text=prompt)])
    for attempt in range(3):
        try:
            resp = await model(
                [msg],
                temperature=0,
                max_tokens=budget,
            )
            # resp is a ChatResponse: content is a list of blocks. Join the
            # TextBlock texts (thinking blocks are filtered out — see above).
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            if text and text.strip():
                return text
        except Exception:  # noqa: BLE001 - the REPL must never die here
            pass
        if attempt < 2:
            time.sleep(2)
    return ""