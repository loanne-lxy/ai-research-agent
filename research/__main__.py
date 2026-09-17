"""Self-check for the research package: parse logic + composition, no model.

Run:  .venv/bin/python -m research
      .venv/bin/python -m research "<问题>"   # + live plan probe (real model)
"""
from __future__ import annotations

import json
import sys

from corpus import list_events
from research import llm
from research.evidence import (
    build_revision_query,
    claim_sources,
    evaluate_evidence,
    evidence_map,
    _parse_eval,
)
from research.followup import build_followup_queries
from research.memory import build_research_memory, format_research_memory
from research.planner import (
    _parse,
    build_research_query,
    make_plan,
)
from research.state import (
    MAX_ROUNDS,
    ResearchState,
    new_followups,
    should_continue,
)


async def _const(value: str) -> str:
    """Coroutine-returning constant: stand-in for the async _llm so the
    self-check exercises the parse/fallback paths without a model."""
    return value


def _main() -> None:
    import asyncio

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
    _real_ev = list_events(limit=1)[0]["citation"]
    _cs = claim_sources(f"某趋势。\n\n## Citations\n- {_real_ev} — 事件标题（趋势1）\n- evt_{'0' * 12} — 假引用（趋势1）\n")
    assert _cs and _cs[0]["claim"] == "趋势1" and _cs[0]["support"] == 2, _cs
    assert _cs[0]["sources"] >= 1, _cs  # real event -> its articles' sources
    assert claim_sources("无引用的纯文本") == []  # no tracked claims -> []
    # fail-closed: an audit that can't run must NOT masquerade as "no issues"
    # (llm._llm is the single model seam — patch the module attribute so the
    # async fakes reach planner/evidence/followup without touching the model)
    async def _eval_checks():
        real = llm._llm
        llm._llm = lambda *a, **k: _const("")  # endpoint down / empty 200s
        try:
            assert await evaluate_evidence("q", ["p"], "d") == {"verdict": "audit_failed"}
            llm._llm = lambda *a, **k: _const('{"verdict":"sufficient","directions":[],"gaps":[],"conflicts":[]}')
            assert (await evaluate_evidence("q", ["p"], "d")).get("verdict") == "sufficient"
            llm._llm = lambda *a, **k: _const('{"foo":1}')  # JSON but no usable verdict
            assert await evaluate_evidence("q", ["p"], "d") == {"verdict": "audit_failed"}
        finally:
            llm._llm = real
    asyncio.run(_eval_checks())
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
    cf = {"verdict": "needs_work", "gaps": [], "conflicts": ["厂商自报成绩与独立评测矛盾"]}

    async def _fu_checks():
        real = llm._llm
        llm._llm = lambda *a, **k: _const('{"follow_ups": ["Agent 评测基准 2026", "企业部署 案例 2026"]}')
        assert await build_followup_queries("测试问题", ok) == ["Agent 评测基准 2026", "企业部署 案例 2026"]
        llm._llm = lambda *a, **k: _const("not json")  # parse failure -> raw fallback
        assert await build_followup_queries("测试问题", ok) == ["评测方向仅2条事件"]
        llm._llm = lambda *a, **k: _const("")  # empty response -> fallback
        assert await build_followup_queries("测试问题", ok) == ["评测方向仅2条事件"]
        assert await build_followup_queries("测试问题", {"gaps": []}) == []
        # conflict-only round must still yield a query (old detect_gaps
        # returned [] here and the loop stopped on 'no new follow-ups')
        llm._llm = lambda *a, **k: _const('{"follow_ups": ["该厂商 成绩 独立复现"]}')
        assert await build_followup_queries("测试问题", cf) == ["该厂商 成绩 独立复现"]
        llm._llm = lambda *a, **k: _const("not json")
        assert await build_followup_queries("测试问题", cf) == ["厂商自报成绩与独立评测矛盾"]
        llm._llm = real
    asyncio.run(_fu_checks())
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

    # build_research_query / make_plan are async; verify composition with an
    # async fake make_plan (patched as a module attribute on research.planner
    # so build_research_query sees it).
    from research import planner

    async def _rq_checks():
        real = planner.make_plan
        async def fake_plan(q):
            return ["步骤A", "步骤B"]
        planner.make_plan = fake_plan
        try:
            q = await build_research_query("测试问题")
            assert "1. 步骤A" in q and "2. 步骤B" in q and "测试问题" in q
        finally:
            planner.make_plan = real
    asyncio.run(_rq_checks())
    print("self-check ok: parse + composition + eval + revision")
    if len(sys.argv) > 1:  # optional live probe against the real model
        for s in asyncio.run(make_plan(sys.argv[1])):
            print(f"  {s}")
        print("---")
        print(asyncio.run(build_research_query(sys.argv[1])))


if __name__ == "__main__":
    _main()