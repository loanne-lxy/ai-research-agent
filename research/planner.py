"""Research planning — one LLM call turns a question into an ordered plan.

The plan is the agent's *working state* for the follow-up research (injected
into the conversation), not a document to show the user. ``make_plan`` never
raises — on any failure it falls back to the generic DEFAULT_PLAN.
"""
from __future__ import annotations

import json
import re

from research import llm
from research.prompts import DEFAULT_PLAN, PLAN_PROMPT


def _parse(text: str) -> list:
    """Extract the JSON array of steps; anything malformed -> []."""
    m = re.search(r"\[.*\]", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else []
        steps = [str(s).strip() for s in data
                 if isinstance(s, (str, int, float)) and str(s).strip()]
        return steps[:8]
    except (json.JSONDecodeError, TypeError):
        return []


async def make_plan(question: str) -> list:
    """Research plan via one LLM call; never raises — falls back to
    DEFAULT_PLAN."""
    steps = _parse(await llm._llm(PLAN_PROMPT.replace("{question}", question)))
    return steps or list(DEFAULT_PLAN)


async def build_research_query(question: str) -> str:
    """Compose the working-state message: plan + directive, so the agent
    executes against the plan instead of re-planning or narrating it."""
    plan = await make_plan(question)
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1))
    return (
        f"[研究任务] 研究计划（你的工作状态，按此推进，不要向用户重复计划全文）：\n"
        f"{numbered}\n\n研究问题：{question}"
    )


def research_plan_from_query(query: str) -> list:
    """Recover the numbered plan lines from a build_research_query output
    (the plan is the working state in context; the evaluator sees the
    same steps). Malformed/absent -> [] (evaluator uses defaults)."""
    m = re.search(r"研究计划（.*?）：\n((?:\d+\. .*\n?)+)", query, re.S)
    if not m:
        return []
    return [re.sub(r"^\d+\. ", "", line).strip()
            for line in m.group(1).splitlines() if line.strip()]