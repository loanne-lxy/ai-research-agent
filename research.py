"""Research path (v2) — working state for research questions.

Planning: a research question gets a research plan (ordered steps) that
becomes the agent's working state for the follow-up research — not a
document to show the user. The plan is injected into the conversation,
so it lives in AgentState.context: the agent sees it on every later
turn, and /save persists it with the session.

Evidence evaluation: between the agent's draft answer and the final
answer, one LLM call judges per-direction evidence sufficiency and
contradictions (Tool Result → Evidence Evaluation → Answer). Gaps are
turned into suggested follow-up search queries that the REVISION AGENT
is told to search itself (Gap Detection -> Suggested Query -> Agent ->
search_events/search_articles -> Observation); the loop never searches
on the agent's behalf. The whole thing is a bounded research loop
(ResearchState, MAX_ROUNDS, stop = sufficient / budget / no new
follow-ups).

Research working memory: the loop's real outputs (query, intent, plan,
per-direction evidence, gaps, follow-ups, round count, citation check)
are assembled into one dict (build_research_memory) — the agent's
Research Working Memory. main.py exposes it via /state and /save writes
it as a session sidecar (session.save_research_memory).

Run:  .venv/bin/python research.py "2026 Q2 以来 Agent 领域的发展趋势是什么？"
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ValidationError

from config import load_llm_config
from corpus import get_event, resolve

# ponytail: static generic fallback — used when the model call or JSON
# parse fails. If a domain-specific default measurably plans better,
# switch on corpus_stats categories.
DEFAULT_PLAN = [
    "收集与研究问题直接相关的事件与文章",
    "按方向/主题归类",
    "评估各方向的证据强度",
    "识别证据缺口",
    "针对缺口补充检索",
    "形成带引用的综合结论",
]

_PROMPT = """为以下研究问题制定一份研究计划。只返回一个 JSON 数组（不要其他文字），
数组元素是 3~7 个简短中文步骤（字符串），按执行顺序排列，覆盖：
收集证据 → 归类 → 评估证据 → 识别缺口 → 补充检索 → 综合结论。
步骤要针对该问题具体化，不要空话。

问题：{question}"""


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


def _llm(prompt: str, max_tokens: int = 2000) -> str:
    """One LLM call, 3 attempts; returns raw content or "" on failure.

    enable_thinking=False on purpose: Qwen3.8 is a reasoning model,
    and for these *structured-JSON* tasks its thinking is pure
    overhead that eats the completion budget (observed: 6.3k thinking
    tokens, empty content, finish_reason=length — on both a 3.6k-token
    full draft and a 2.5k-token evidence map). With thinking off the
    same call is ~2s and returns the JSON. max_tokens is still
    generous; the endpoint also flaps with empty 200s, which the SDK
    does not retry.
    """
    budget = max(max_tokens, 2 * len(prompt))
    import time
    from openai import OpenAI
    llm = load_llm_config()
    client = OpenAI(base_url=llm.base_url, api_key=llm.api_key, timeout=120)
    for attempt in range(3):
        try:
            out = client.chat.completions.create(
                model=llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=budget,
                extra_body={"enable_thinking": False},
            )
            content = out.choices[0].message.content or ""
            if content.strip():
                return content
        except Exception:  # noqa: BLE001 - the REPL must never die here
            pass
        if attempt < 2:
            time.sleep(2)
    return ""


def make_plan(question: str) -> list:
    """Research plan via one LLM call; never raises — falls back to
    DEFAULT_PLAN."""
    steps = _parse(_llm(_PROMPT.replace("{question}", question)))
    return steps or list(DEFAULT_PLAN)


def build_research_query(question: str) -> str:
    """Compose the working-state message: plan + directive, so the agent
    executes against the plan instead of re-planning or narrating it."""
    plan = make_plan(question)
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


_EVAL_PROMPT = """你是证据审核员。下面是研究回答的证据地图（章节标题+全部正文行；不带引用ID的正文行即未引用陈述）。只返回一个JSON对象（不要其他文字），note不超过15字：
{{"directions":[{{"name":"方向","events":数,"articles":数,"sufficient":true或false,"conflict":true或false,"note":"简"}}],"gaps":["需补充检索的缺口"],"conflicts":["相互矛盾的点"],"verdict":"sufficient或needs_work"}}
标准：某方向证据行<3 → sufficient=false；单来源（独立source仅1个）支撑的结论 → sufficient=false 且列入 gaps（写"单来源: <结论>，需补充独立来源"）；证据互相矛盾 → conflict=true；未带引用ID的事实性陈述 → 列入 gaps（写"未引用: <该事实>"，寒暄/方法论解释不算）；存在 sufficient=false / conflict=true / gaps → verdict=needs_work。只数地图里实际出现的引用，草稿已声明知识库未覆盖的缺口要列入 gaps。

研究问题：{question}
研究计划：
{plan}

结论-独立来源数：
{claims}

证据地图：
{draft}"""


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


def evaluate_evidence(question: str, plan: list, draft: str,
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
    raw = _llm(
        _EVAL_PROMPT
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


# ------------------------------------------------- gap detection & follow-up
_FOLLOWUP_PROMPT = """下面是研究回答被评估后发现的证据缺口与矛盾。为每个缺口/矛盾生成一条检索关键词（15字以内，直接给检索词，可中英混合，可含年份/季度）；矛盾的检索词优先用于裁决哪方属实。只返回一个JSON对象：
{"follow_ups":["检索词1","检索词2"]}
不要解释。

研究问题：{question}
缺口/矛盾：
{items}"""


def build_followup_queries(question: str, issues: dict) -> list:
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
    raw = _llm(_FOLLOWUP_PROMPT
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


# ------------------------------------------------------- research loop
MAX_ROUNDS = 3  # max supplementary-search rounds (initial research excluded)


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


def build_research_memory(question: str, intent: str, plan: list,
                          state: "ResearchState", issues: dict,
                          citations: dict,
                          claims: list | None = None) -> dict:
    """Assemble the research working memory from what the loop already
    computed — the single source of truth for /state and /save sidecar.
    (fields we don't actually produce — domain, per-call retrieved_*,
    separate final_claims — are deliberately absent, not empty stubs;
    `claims` is the generated claim->evidence-ids map, which we do have.)"""
    return {
        "query": question,
        "intent": intent,
        "research_plan": plan,
        "evidence": issues.get("directions", []),
        "gaps": issues.get("gaps", []),
        "conflicts": issues.get("conflicts", []),
        "followup_queries": sorted(state.seen),
        "research_iterations": state.round,
        "verdict": issues.get("verdict"),
        "claims": claims or [],
        "citations": {
            "verified": len(citations.get("ok", [])),
            "total": citations.get("total", 0),
            "missing": citations.get("missing", []),
        },
    }


def format_research_memory(mem: dict) -> str:
    """One human-readable block of the working memory for /state."""
    c = mem.get("citations", {})
    lines = [
        "【研究状态 Research Working Memory】",
        f"  问题: {mem.get('query', '?')}",
        f"  意图: {mem.get('intent', '?')}",
        f"  计划: {' | '.join(mem.get('research_plan', [])) or '(无)'}",
        f"  迭代: {mem.get('research_iterations', 0)} 轮   判定: {mem.get('verdict') or '(未评估)'}",
    ]
    for d in mem.get("evidence", []):
        mark = "✓" if d.get("sufficient") else "✗"
        lines.append(f"  {mark} {d.get('name', '?')}: "
                     f"{d.get('events', 0)} ev / {d.get('articles', 0)} src")
    gaps = mem.get("gaps", [])
    lines.append(f"  缺口: {gaps if gaps else '(无)'}")
    fu = mem.get("followup_queries", [])
    lines.append(f"  已试补充检索: {fu if fu else '(无)'}")
    cl = mem.get("claims", [])
    if cl:
        for item in cl:
            lines.append(f"  结论: {item['claim']}  ← 证据 {item['evidence']}")
    else:
        lines.append("  结论-证据映射: (无)")
    lines.append(f"  引用: {c.get('verified', 0)}/{c.get('total', 0)} verified"
                 + (f" — 未解析: {c['missing']}" if c.get("missing") else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    # self-check: parse logic + composition, no LLM call
    assert _parse('["收集事件","归类"]') == ["收集事件", "归类"]
    assert _parse('计划如下：\n```json\n["a","b"]\n```\n') == ["a", "b"]
    assert _parse("no json here") == []
    assert _parse('[{"a": 1}]') == []  # non-string elements dropped
    assert len(_parse(json.dumps([f"s{i}" for i in range(20)]))) == 8
    # eval parsing + revision composition
    em = evidence_map("# 一、A\n没有引用的论述行。\n- 证据 [evt_307d86c5e35e] 展开\n## 二、B\n- 另一条 [2d5ee61a05111f0a] 与 evt_abc123def456")
    # uncited prose stays in the map — the evaluator must be able to flag it
    assert "没有引用的论述行" in em and "evt_307d86c5e35e" in em and "## 二、B" in em
    assert len(evidence_map("x" * 5000)) <= 2500  # cap still bounds the map
    # per-claim independent-source counts (sufficiency anchor): distinct
    # sources across a claim's evidence, real corpus id + a fabricated one
    from corpus import list_events
    _real_ev = list_events(limit=1)[0]["citation"]
    _cs = claim_sources(f"某趋势。\n\n## Citations\n- {_real_ev} — 事件标题（趋势1）\n- evt_{'0' * 12} — 假引用（趋势1）\n")
    assert _cs and _cs[0]["claim"] == "趋势1" and _cs[0]["support"] == 2, _cs
    assert _cs[0]["sources"] >= 1, _cs  # real event -> its articles' sources
    assert claim_sources("无引用的纯文本") == []  # no tracked claims -> []
    # fail-closed: an audit that can't run must NOT masquerade as "no issues"
    _real_llm = _llm
    _llm = lambda *a, **k: ""  # endpoint down / empty 200s
    try:
        assert evaluate_evidence("q", ["p"], "d") == {"verdict": "audit_failed"}
        _llm = lambda *a, **k: '{"verdict":"sufficient","directions":[],"gaps":[],"conflicts":[]}'
        assert evaluate_evidence("q", ["p"], "d").get("verdict") == "sufficient"
        _llm = lambda *a, **k: '{"foo":1}'  # JSON but no usable verdict
        assert evaluate_evidence("q", ["p"], "d") == {"verdict": "audit_failed"}
    finally:
        _llm = _real_llm
    ok = _parse_eval('```json\n{"directions":[{"name":"Memory","events":15,"articles":5,"sufficient":true,"conflict":false,"note":"ok"}],'
                     '"gaps":["评测方向仅2条事件"],"conflicts":[],"verdict":"needs_work"}\n```')
    assert ok["verdict"] == "needs_work"
    assert ok["directions"][0]["name"] == "Memory"
    assert "评测方向仅2条事件" in ok["gaps"]
    # schema validation: wrong enum / wrong types / missing keys are
    # rejected -> {} (not silently accepted as a valid evaluation)
    assert _parse_eval('{"verdict":"banana"}') == {}
    assert _parse_eval('{"directions":"hello","gaps":[],"conflicts":[],"verdict":"sufficient"}') == {}
    assert _parse_eval('{"directions":[],"gaps":"x","conflicts":[],"verdict":"sufficient"}') == {}
    assert _parse_eval('{"directions":[{}],"gaps":[],"conflicts":[],"verdict":"sufficient"}') == {}
    assert _parse_eval("no json") == {}
    rq = build_revision_query("测试问题", ok)
    assert "评测方向仅2条事件" in rq and "最终回答" in rq
    assert "不要重新执行研究计划" in rq  # 受控：只补检，不重做整个计划
    # stateless revision: the prior answer must be inlined so a fresh agent
    # has everything it needs (E2E: history-stacked context blew 103k tokens)
    rq_a = build_revision_query("测试问题", ok, answer="## 趋势\n- evt_a\n## Citations\n- evt_a — x（趋势1）")
    assert "## Citations" in rq_a and "evt_a" in rq_a and "上一版回答" in rq_a
    # agent does its own research: suggested queries are handed over as
    # starting points, and the directive makes it call the tools itself
    rq_s = build_revision_query("测试问题", ok,
                                suggested_queries=["评测基准 2026"],
                                answer="旧版")
    assert "建议检索词" in rq_s and "评测基准 2026" in rq_s
    assert "自己调用 search_events / search_articles" in rq_s
    # follow-up queries: LLM rephrases gaps+conflicts -> search terms
    # (fallback = raw gap/conflict strings); the agent searches with them itself
    _llm_orig = _llm
    _llm = lambda p: '{"follow_ups": ["Agent 评测基准 2026", "企业部署 案例 2026"]}'
    assert build_followup_queries("测试问题", ok) == ["Agent 评测基准 2026", "企业部署 案例 2026"]
    _llm = lambda p: "not json"  # parse failure -> raw fallback
    assert build_followup_queries("测试问题", ok) == ["评测方向仅2条事件"]
    _llm = lambda p: ""  # empty response -> fallback
    assert build_followup_queries("测试问题", ok) == ["评测方向仅2条事件"]
    assert build_followup_queries("测试问题", {"gaps": []}) == []
    # conflict-only round must still yield a query (old detect_gaps returned []
    # here and the loop stopped on 'no new follow-ups' despite needs_work)
    cf = {"verdict": "needs_work", "gaps": [], "conflicts": ["厂商自报成绩与独立评测矛盾"]}
    _llm = lambda p: '{"follow_ups": ["该厂商 成绩 独立复现"]}'
    assert build_followup_queries("测试问题", cf) == ["该厂商 成绩 独立复现"]
    _llm = lambda p: "not json"
    assert build_followup_queries("测试问题", cf) == ["厂商自报成绩与独立评测矛盾"]
    _llm = _llm_orig
    # research loop: stop condition = sufficient verdict / round budget /
    # no NEW follow-ups (dedup is what prevents spinning on recurring gaps)
    rs = ResearchState()
    need = {"verdict": "needs_work", "gaps": ["评测不足"], "conflicts": []}
    # production order: new_followups() filters + marks seen, then
    # should_continue() reads the already-filtered list (never sees dupes)
    assert new_followups(rs, ["A 评测"]) == ["A 评测"]   # first time -> new
    assert new_followups(rs, ["A 评测"]) == []           # duplicate -> dropped
    assert should_continue(rs, need, ["A 评测"]) is True  # fresh query -> continue
    assert should_continue(rs, need, []) is False         # nothing new -> stop
    assert new_followups(rs, ["b 评测"]) == ["b 评测"]
    assert should_continue(rs, need, ["b 评测"]) is True
    rs.round = MAX_ROUNDS - 1  # 2 completed -> the 3rd still permitted
    assert should_continue(rs, need, ["b 评测"]) is True
    rs.round = MAX_ROUNDS  # 3 completed -> budget reached
    assert should_continue(rs, need, ["b 评测"]) is False
    assert should_continue(ResearchState(), {"verdict": "sufficient"}, ["x"]) is False
    # working memory: assembled from real loop outputs, not stubbed
    rs2 = ResearchState()
    rs2.round = 2
    rs2.seen = {"b 评测", "A 评测"}
    cit = {"ok": ["evt_a", "b"], "total": 3, "missing": ["evt_z"], "chain": {}}
    claims = [{"claim": "Memory 长期化", "evidence": ["evt_a"]}]
    mem = build_research_memory("问题Q", "research", ["s1", "s2"], rs2,
                                need, cit, claims)
    assert mem["query"] == "问题Q" and mem["intent"] == "research"
    assert mem["research_plan"] == ["s1", "s2"]
    assert mem["gaps"] == ["评测不足"]
    assert mem["followup_queries"] == ["A 评测", "b 评测"]  # sorted, deduped
    assert mem["research_iterations"] == 2
    assert mem["claims"] == claims
    assert mem["citations"] == {"verified": 2, "total": 3, "missing": ["evt_z"]}
    assert "评测不足" in format_research_memory(mem)
    assert "问题Q" in format_research_memory(mem)
    assert "结论: Memory 长期化  ← 证据 ['evt_a']" in format_research_memory(mem)
    # empty issues -> nothing to fix, caller ships draft
    assert build_revision_query("q", {"gaps": [], "conflicts": []})
    orig = make_plan
    make_plan = lambda q: ["步骤A", "步骤B"]  # monkeypatch, no LLM
    q = build_research_query("测试问题")
    assert "1. 步骤A" in q and "2. 步骤B" in q and "测试问题" in q
    make_plan = orig
    print("self-check ok: parse + composition + eval + revision")
    if len(sys.argv) > 1:  # optional live probe against the real model
        for s in make_plan(sys.argv[1]):
            print(f"  {s}")
        print("---")
        print(build_research_query(sys.argv[1]))