"""Gap detection & follow-up query generation.

Turn the evaluator's gaps AND conflicts into concrete follow-up search
queries (one LLM call). On any failure, fall back to the raw
gap/conflict strings (still valid search phrases). Conflicts must produce
queries too — the loop's stop gate keys on gaps-or-conflicts, so a
conflict-only round would otherwise exit with nothing to search.
"""
from __future__ import annotations

import json
import re

from research import llm
from research.prompts import FOLLOWUP_PROMPT


async def build_followup_queries(question: str, issues: dict) -> list:
    """Turn the evaluator's gaps AND conflicts into concrete follow-up
    search queries. One LLM call; on any failure fall back to the raw
    strings (still valid search phrases). Conflicts must produce queries
    too — the loop's stop gate keys on gaps-or-conflicts, so a
    conflict-only round would otherwise exit with nothing to search."""
    gaps = [str(g).strip() for g in issues.get("gaps", []) if str(g).strip()]
    conflicts = [str(c).strip() for c in issues.get("conflicts", [])
                 if str(c).strip()]
    if not gaps and not conflicts:
        return []
    lines = [f"- 缺口: {g}" for g in gaps] + \
            [f"- 矛盾: {c}" for c in conflicts]
    raw = await llm._llm(FOLLOWUP_PROMPT
               .replace("{question}", question)
               .replace("{items}", "\n".join(lines)))
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
        items = data.get("follow_ups") if isinstance(data, dict) else None
        queries = [str(x).strip() for x in (items or [])
                   if isinstance(x, (str, int)) and str(x).strip()]
        return queries[:6] or list(gaps + conflicts)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return list(gaps + conflicts)