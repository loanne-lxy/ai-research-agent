"""Typed state for the bounded research loop.

The loop's working state (round counter + already-tried follow-up queries),
the per-iteration stop decision, and the round-budget cap. ``ResearchState``
is task-scoped working state — it is *not* long-term memory and must not be
written to ReMe (see agent/memory.py).

The loop's full outputs are assembled into the research working memory dict
by ``research.memory.build_research_memory`` — that dict is the ``out`` the
controller populates (it is a plain JSON-sidecar shape, so it stays a dict,
not a class).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# max supplementary-search rounds (initial research excluded)
MAX_ROUNDS = 3


@dataclass
class ResearchState:
    """Loop state carried across rounds: round number + follow-up queries
    already tried. Dedup is the loop's real stop condition — a recurring
    gap rephrases to the same query the agent already saw, so it cannot
    re-trigger the same search forever."""
    round: int = 0
    seen: set = field(default_factory=set)


def new_followups(state: ResearchState, queries: list) -> list:
    """Filter already-tried follow-up queries (case/whitespace-insensitive);
    mutates state.seen. Returns only the new ones, in order."""
    out = []
    for q in queries:
        k = str(q).strip().lower()
        if k and k not in state.seen:
            state.seen.add(k)
            out.append(str(q).strip())
    return out


def should_continue(state: ResearchState, issues: dict,
                    new_queries: list) -> bool:
    """Stop condition: evidence judged sufficient, round budget reached
    (state.round = supplementary rounds already completed; MAX_ROUNDS is
    the cap), or no new follow-up queries to try."""
    if issues.get("verdict") != "needs_work":
        return False
    if state.round >= MAX_ROUNDS:
        return False
    return bool(new_queries)