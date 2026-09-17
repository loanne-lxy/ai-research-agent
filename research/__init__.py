"""Research pipeline (V2) — module boundaries for the bounded research loop.

The old single research.py split by responsibility:

    prompts.py   prompt templates + static fallback (DEFAULT_PLAN)
    llm.py       unified AgentScope model client (_llm) + DI (set_research_model)
    state.py     ResearchState / new_followups / should_continue / MAX_ROUNDS
    planner.py   make_plan / build_research_query / research_plan_from_query
    evidence.py  evidence_map / claim_sources / evaluate_evidence /
                 build_revision_query
    followup.py  build_followup_queries (gap detection)
    memory.py    build_research_memory / format_research_memory
    controller.py run_research_loop — the only orchestrator

    query -> intent -> planning -> research -> evaluation -> gap detection
          -> follow-up -> synthesis -> citation validation

The controller (run_research_loop) lives here, NOT in agent/research_agent.py;
that module keeps only the AgentScope Agent subclass (ResearchAgent) that
drives this loop from the service's Web UI.
"""
from __future__ import annotations

from research.controller import (
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
    run_research_loop,
    set_research_model,
    should_continue,
)
from research.evidence import evidence_map
from research.llm import _llm, _resolve_model
from research.memory import format_research_memory
from research.planner import _parse, make_plan
from research.prompts import (
    DEFAULT_PLAN,
    EVAL_PROMPT,
    FOLLOWUP_PROMPT,
    PLAN_PROMPT,
)

__all__ = [
    "MAX_ROUNDS",
    "ResearchState",
    "new_followups",
    "should_continue",
    "make_plan",
    "build_research_query",
    "research_plan_from_query",
    "evidence_map",
    "claim_sources",
    "evaluate_evidence",
    "build_revision_query",
    "build_followup_queries",
    "build_research_memory",
    "format_research_memory",
    "set_research_model",
    "run_research_loop",
    "DEFAULT_PLAN",
    "PLAN_PROMPT",
    "EVAL_PROMPT",
    "FOLLOWUP_PROMPT",
    "_llm",
    "_resolve_model",
    "_parse",
]