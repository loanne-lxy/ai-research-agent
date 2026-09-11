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

from config import load_llm_config

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


_EVAL_PROMPT = """你是证据审核员。下面是研究回答的证据地图（章节标题+带引用ID的证据行）。只返回一个JSON对象（不要其他文字），note不超过15字：
{{"directions":[{{"name":"方向","events":数,"articles":数,"sufficient":true或false,"conflict":true或false,"note":"简"}}],"gaps":["需补充检索的缺口"],"conflicts":["相互矛盾的点"],"verdict":"sufficient或needs_work"}}
标准：某方向证据行<3 → sufficient=false；证据互相矛盾 → conflict=true；存在 sufficient=false / conflict=true / gaps → verdict=needs_work。只数地图里实际出现的引用，草稿已声明知识库未覆盖的缺口要列入 gaps。

研究问题：{question}
研究计划：
{plan}

证据地图：
{draft}"""


_CITATION_RE = re.compile(r"evt_[0-9a-f]{12}|[0-9a-f]{16}")


def evidence_map(draft: str, cap: int = 2500) -> str:
    """Compress a draft to what an evidence evaluation actually needs:
    section headings + every line carrying a citation ID. Prose without
    citations drops out — it makes no evidentiary claim. Keeps the map
    short enough that the evaluator's own reasoning stays cheap.
    """
    lines = []
    for line in draft.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or _CITATION_RE.search(s):
            lines.append(s[:200])
    return "\n".join(lines)[:cap]


def _parse_eval(text: str) -> dict:
    """Extract the JSON object; malformed -> {} (treat as no issues)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def evaluate_evidence(question: str, plan: list, draft: str) -> dict:
    """One LLM call judging the draft's evidence (per-direction counts,
    sufficiency, conflicts, gaps). Never raises — returns {} on failure,
    which the caller treats as 'no issues' and ships the draft as final.
    """
    if not plan:
        plan = list(DEFAULT_PLAN)
    plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1))
    # Only the evidence map + a lean prompt reach the evaluator: this
    # Qwen3.8 deployment over-thinks verbose prompts (11k+ thinking
    # tokens on the heavy version, empty content) but stays reliable on
    # a short one. Budget 8000 so even ~13k-style thinking can't clip
    # the JSON (observed flaky, so keep slack).
    raw = _llm(
        _EVAL_PROMPT
        .replace("{question}", question)
        .replace("{plan}", plan_text)
        .replace("{draft}", evidence_map(draft)),
        max_tokens=8000,
    )
    return _parse_eval(raw)


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
_FOLLOWUP_PROMPT = """下面是研究回答被评估后发现的证据缺口。为每个缺口生成一条检索关键词（15字以内，直接给检索词，可中英混合，可含年份/季度），只返回一个JSON对象：
{"follow_ups":["检索词1","检索词2"]}
不要解释。

研究问题：{question}
缺口：
{gaps}"""


def detect_gaps(question: str, issues: dict) -> list:
    """Turn the evaluator's free-text gaps into concrete follow-up
    search queries. One LLM call; on any failure fall back to the raw
    gap strings (still valid search phrases)."""
    gaps = [str(g).strip() for g in issues.get("gaps", []) if str(g).strip()]
    if not gaps:
        return []
    raw = _llm(_FOLLOWUP_PROMPT
               .replace("{question}", question)
               .replace("{gaps}", "\n".join(f"- {g}" for g in gaps)))
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
        items = data.get("follow_ups") if isinstance(data, dict) else None
        queries = [str(x).strip() for x in (items or [])
                   if isinstance(x, (str, int)) and str(x).strip()]
        return queries[:6] or list(gaps)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return list(gaps)


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
    assert em.count("没有引用") == 0 and "evt_307d86c5e35e" in em and "## 二、B" in em
    ok = _parse_eval('```json\n{"directions":[{"name":"Memory","events":15,"articles":5,"sufficient":true,"conflict":false,"note":"ok"}],'
                     '"gaps":["评测方向仅2条事件"],"conflicts":[],"verdict":"needs_work"}\n```')
    assert ok["verdict"] == "needs_work"
    assert ok["directions"][0]["name"] == "Memory"
    assert "评测方向仅2条事件" in ok["gaps"]
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
    # gap detection: LLM rephrases -> suggested follow-up queries
    # (fallback = raw gaps); the agent searches with them itself
    _llm_orig = _llm
    _llm = lambda p: '{"follow_ups": ["Agent 评测基准 2026", "企业部署 案例 2026"]}'
    assert detect_gaps("测试问题", ok) == ["Agent 评测基准 2026", "企业部署 案例 2026"]
    _llm = lambda p: "not json"  # parse failure -> raw gap fallback
    assert detect_gaps("测试问题", ok) == ["评测方向仅2条事件"]
    _llm = lambda p: ""  # empty response -> fallback
    assert detect_gaps("测试问题", ok) == ["评测方向仅2条事件"]
    assert detect_gaps("测试问题", {"gaps": []}) == []
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