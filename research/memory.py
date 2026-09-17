"""Research working memory — the loop's real outputs as one dict.

``build_research_memory`` assembles query / intent / plan / per-direction
evidence / gaps / follow-ups / round count / citation check into the single
dict that is the source of truth for /state and the /save session sidecar
(session.save_research_memory). ``format_research_memory`` renders that dict
for /state.
"""
from __future__ import annotations


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