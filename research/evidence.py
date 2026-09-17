"""Evidence evaluation — judge a draft's grounding, then compose the
revision message.

Two Python-computed inputs anchor the LLM (so the model checks real counts
instead of counting rows):
  * ``evidence_map``    — the draft's lines as the evaluator sees them
  * ``claim_sources``   — per-claim DISTINCT independent-source counts

One LLM call then produces the per-direction verdict / gaps / conflicts.
On any failure the evaluation is *fail-closed* (``verdict=audit_failed``),
so the caller ships a clearly-marked degraded answer rather than passing an
unvetted draft as if it were sufficient. ``build_revision_query`` turns a
``needs_work`` verdict into the message a *fresh* (stateless) agent searches
with on its own.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ValidationError

from corpus import get_event, resolve
from research import llm
from research.prompts import DEFAULT_PLAN, EVAL_PROMPT


def evidence_map(draft: str, cap: int = 2500) -> str:
    """Draft -> what an evidence evaluation needs: every non-empty line —
    headings, cited evidence lines, AND uncited prose. The evaluator's
    job is to flag factual lines that carry no citation ID, so uncited
    lines must survive into the map (dropping them made the 'facts need
    citations' check impossible to run). Per-line truncation + the
    whole-map cap keep the evaluator call cheap.
    ponytail: the cap cuts the tail, so a >2500-char draft loses its
    '## Citations' block from the map; raise the cap (budget allows
    3000) if that shows up in E2E.
    """
    return "\n".join(l.strip()[:200] for l in draft.splitlines() if l.strip())[:cap]


def claim_sources(draft: str, max_claims: int = 8) -> list:
    """Per-claim evidence profile: [{claim, support, sources}], where
    sources = count of DISTINCT independent sources (event -> its
    articles' sources; bare article -> its own source) backing the
    claim's cited ids, via the same resolve chain citations.validate
    uses. This is real data, not the LLM counting rows: N cited lines can
    all trace to ONE source, and that is the failure sufficiency should
    catch. An unresolvable citation contributes no source (a claim whose
    ids are all fabricated shows 0 -> flagged insufficient, never silently
    dropped); a draft with no '## Citations' claims yields [].
    ponytail: claim labels are the model's short tags (趋势1/趋势2) —
    finer claim identity needs sentence-level claims in the answer format.
    """
    from citations import extract_claims
    profiles = []
    for item in extract_claims(draft):
        sources = set()
        for cid in item["evidence"]:
            # same split as citations.validate: events via get_event (the
            # resolve path returns an event WITHOUT its articles), articles
            # via resolve.
            if cid.startswith("evt_"):
                ev = get_event(cid)
                if ev is None:
                    continue
                sources.update(a.get("source") for a in ev.get("articles", [])
                               if a.get("source"))
            else:
                r = resolve(cid)
                if r is None:
                    continue
                src = r["object"].get("source")
                if src:
                    sources.add(src)
        profiles.append({"claim": item["claim"], "support": len(item["evidence"]),
                         "sources": len(sources)})
    profiles.sort(key=lambda p: p["sources"], reverse=True)
    return profiles[:max_claims]


class DirectionEvaluation(BaseModel):
    name: str
    events: int
    articles: int
    sufficient: bool
    conflict: bool
    note: str = ""


class EvidenceEvaluation(BaseModel):
    directions: list[DirectionEvaluation]
    gaps: list[str]
    conflicts: list[str]
    verdict: Literal["sufficient", "needs_work"]


def _parse_eval(text: str) -> dict:
    """Extract + schema-validate the evaluation JSON. Malformed or
    schema-invalid -> {} (evaluate_evidence's fail-closed guard maps that
    to audit_failed — a schema violation means the audit can't be trusted,
    not that the draft is fine)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return EvidenceEvaluation.model_validate(data).model_dump()
    except (json.JSONDecodeError, TypeError, ValidationError):
        return {}


async def evaluate_evidence(question: str, plan: list, draft: str,
                      claim_srcs: list | None = None) -> dict:
    """One LLM call judging the draft's evidence (per-direction counts,
    sufficiency, conflicts, gaps). Sufficiency is anchored on
    claim_srcs — per-claim independent-source counts computed in Python
    (claim_sources), so the model checks 'claim -> how many independent
    sources' instead of counting rows itself. Never raises. On failure (empty
    response or unusable verdict) it returns {"verdict": "audit_failed"}
    — fail-closed, so the caller ships a *degraded, clearly-marked*
    answer instead of passing an unvetted draft as if it were sufficient.
    """
    if not plan:
        plan = list(DEFAULT_PLAN)
    plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1))
    claim_text = "\n".join(f"- {p['claim']}: {p['sources']} 个独立来源（{p['support']} 条引用）"
                           for p in (claim_srcs or [])) or "(未解析出带引用的结论)"
    # Only the evidence map + a lean prompt reach the evaluator: this
    # Qwen3.8 deployment over-thinks verbose prompts (11k+ thinking
    # tokens on the heavy version, empty content) but stays reliable on
    # a short one. Budget 8000 so even ~13k-style thinking can't clip
    # the JSON (observed flaky, so keep slack).
    raw = await llm._llm(
        EVAL_PROMPT
        .replace("{question}", question)
        .replace("{plan}", plan_text)
        .replace("{claims}", claim_text)
        .replace("{draft}", evidence_map(draft)),
        max_tokens=8000,
    )
    issues = _parse_eval(raw)
    # fail-closed: an audit that can't run is NOT "no issues" — a missing
    # or unusable verdict means we can't prove the draft is grounded, so
    # it downgrades to audit_failed rather than shipping as if it passed.
    if issues.get("verdict") not in ("sufficient", "needs_work"):
        return {"verdict": "audit_failed"}
    return issues


def build_revision_query(question: str, issues: dict,
                         suggested_queries: list | None = None,
                         answer: str = "") -> str:
    """Compose the follow-up message sent to a FRESH agent after a
    needs_work evaluation (revisions run stateless — carrying the whole
    research history into one context is what blew past the compression
    threshold in E2E). A revision only needs the previous answer + the
    evaluation findings, so the prior answer is inlined in full. Gaps to
    fill, conflicts to resolve, with the directive that the revised
    answer is final and must restate all citations.
    `suggested_queries` are follow-up search terms the loop derived from
    the gaps — handed to the agent as STARTING POINTS; it decides what to
    actually search and calls search_events/search_articles itself
    (ReAct: the loop must never do the agent's research for it)."""
    gaps = [str(g) for g in issues.get("gaps", []) if str(g).strip()]
    conflicts = [str(c) for c in issues.get("conflicts", []) if str(c).strip()]
    parts = [
        f"[证据评估] 上一版回答的评估发现以下问题，请针对性补强后给出最终版：\n"
        f"研究问题：{question}\n"
    ]
    if answer.strip():
        parts.append(
            "上一版回答（在它的基础上修订，保留仍成立的引用和 '## Citations' 块）：\n"
            f"---\n{answer.strip()}\n---"
        )
    if gaps:
        parts.append("证据不足、需补充检索的方向：")
        parts.extend(f"  - {g}" for g in gaps)
    if conflicts:
        parts.append("存在矛盾、需澄清或消解的点：")
        parts.extend(f"  - {c}" for c in conflicts)
    if suggested_queries:
        parts.append("建议检索词（起点，可改写/拆分/换语言）：")
        parts.extend(f"  - {q}" for q in suggested_queries)
    parts.append(
        "要求：不要重新执行研究计划或重做已完成的检索，只针对上述缺口/矛盾，"
        "自己调用 search_events / search_articles 做补充检索"
        "（可参考建议检索词，换关键词、中英切换、看事件背后的文章），"
        "根据检索结果继续推理，"
        "然后直接给出修订后的完整最终回答（不要只给增量）。"
        "新证据的引用 ID 必须来自工具返回值；保留仍成立的原有引用。"
        "若某缺口在知识库中确实无补充来源，在最终回答中明确标注为证据缺口。"
    )
    return "\n".join(parts)