"""Citation validator — the project's answer-trustability guarantee.

Every factual answer must cite corpus ids. This module checks the chain
Answer -> Event/Article -> Source: it takes a finished answer, pulls out
every citation id the model claimed, resolves each against the corpus
(read-only, never writes), and builds the traceable chain. A citation
that does not resolve is flagged UNVERIFIED — that is how a fabricated
id is caught.

Citation ids (see corpus.store contract):
  event   evt_ + 12 hex   e.g. evt_815fcf6e2a6f
  article 16 hex          e.g. 07c0edc88a897d1f

Citations are read from the '## Citations' block when present (the system
prompt mandates it) and fall back to the full answer otherwise — scanning
only that block also keeps bare hex in URLs from being misread as cites.

On top of id resolution this module tracks *which claim each piece of
evidence supports*: the per-line '（支撑了哪条结论）' note is parsed into a
claim -> [evidence ids] map (extract_claims / trace_claims), so a final
answer's [1][2][3] carry their claim->evidence linkage at generation time,
not just a post-hoc check.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from corpus import get_event, resolve

# modifier suffixes models tack onto claim labels; stripped so label variants
# ("趋势1核心" / "趋势1 MCP") pool under one claim
_CLAIM_SUFFIXES = ("核心", "关键", "主要", "补充", "MCP", "细节")

_EVENT_RE = re.compile(r"\bevt_[0-9a-f]{12}\b")
_ARTICLE_RE = re.compile(r"\b[0-9a-f]{16}\b")
_SECTION_RE = re.compile(r"(?im)^\s*##\s*citations\s*$")


def _citation_section(text: str) -> str:
    """Text after the last '## Citations' header, else the whole answer."""
    parts = _SECTION_RE.split(text)
    return parts[-1].strip() if len(parts) > 1 else text


def extract_citations(text: str) -> List[str]:
    """Unique citation ids, first-seen order. Events and articles are
    disjoint shapes (evt_ prefix vs. bare 16-hex), so the two regexes
    never collide and cover every id the KAL can hand back."""
    seen: List[str] = []
    for rx in (_EVENT_RE, _ARTICLE_RE):
        for m in rx.findall(text):
            if m not in seen:
                seen.append(m)
    return seen


def validate(text: str) -> Dict[str, Any]:
    """Resolve every citation in an answer.

    Returns {total, ok, missing, chain} where chain maps citation ->
    {type, title, week, ...} plus (for events) its articles, each with
    its source name — the full Answer->Event->Article->Source trace.
    """
    ids = extract_citations(_citation_section(text))
    ok: List[str] = []
    missing: List[str] = []
    chain: Dict[str, Any] = {}
    for c in ids:
        if c.startswith("evt_"):
            ev = get_event(c)
            if ev is None:
                missing.append(c)
                continue
            ok.append(c)
            chain[c] = {
                "type": "event",
                "title": ev["title"],
                "week": ev["week"],
                "articles": [
                    {"citation": a["citation"], "title": a["title"],
                     "source": a["source"]}
                    for a in ev.get("articles", [])
                ],
            }
        else:
            r = resolve(c)
            if r is None or r["type"] != "article":
                missing.append(c)
                continue
            a = r["object"]
            ok.append(c)
            chain[c] = {
                "type": "article",
                "title": a["title"],
                "source": a["source"],
                "url": a["url"],
                "week": a["week"],
            }
    return {"total": len(ids), "ok": ok, "missing": missing, "chain": chain}


def _claim_key(label: str) -> str:
    """Normalize a claim label so modifier variants pool under one claim."""
    for suf in _CLAIM_SUFFIXES:
        if label.endswith(suf) and len(label) > len(suf):
            label = label[: -len(suf)].rstrip()
    return label


def extract_claims(text: str) -> List[Dict[str, Any]]:
    """Invert the '## Citations' block into claim -> evidence IDs.

    Each line already reads '- <id> — <title>（<claim label>）'; the
    trailing parenthetical is the claim the evidence supports. Labels are
    short tags the model writes (趋势1 / 趋势2 MCP / 趋势1核心), not full
    sentences — _claim_key strips modifier suffixes so variants pool.
    No real sentence-per-claim unless the prompt asks for one. Falls back
    to the title when a line carries no parenthetical.
    """
    section = _citation_section(text)
    claims: Dict[str, List[str]] = {}
    order: List[str] = []
    for raw in section.splitlines():
        line = raw.strip().lstrip("-•* \t")
        m = re.match(r"(evt_[0-9a-f]{12}|[0-9a-f]{16})\s*[—–-]+\s*(.+)$", line)
        if not m:
            continue
        cid, rest = m.group(1), m.group(2).strip()
        pm = re.search(r"[（(]([^（）()]*)[)）]\s*$", rest)
        claim = _claim_key((pm.group(1).strip() if pm else rest) or rest)
        if claim not in claims:
            claims[claim] = []
            order.append(claim)
        if cid not in claims[claim]:
            claims[claim].append(cid)
    return [{"claim": c, "evidence": claims[c]} for c in order]


def trace_claims(text: str) -> List[Dict[str, Any]]:
    """Full chain claim -> evidence id -> {event/article -> article/source}.

    Pairs extract_claims with validate(): each evidence id is annotated
    with the resolved record already built by validate (or None when the
    id is fabricated), giving Claim -> Evidence -> Event -> Article ->
    Source in one structure.
    """
    chain = validate(text)["chain"]
    return [
        {
            "claim": item["claim"],
            "evidence": [{"id": cid, "resolved": chain.get(cid)}
                         for cid in item["evidence"]],
        }
        for item in extract_claims(text)
    ]


def format_report(result: Dict[str, Any]) -> str:
    total = result["total"]
    ok = result["ok"]
    missing = result["missing"]
    if total == 0:
        return "[citations] (none cited)"
    n_ev = sum(1 for v in result["chain"].values() if v["type"] == "event")
    n_ar = sum(1 for v in result["chain"].values() if v["type"] == "article")
    line = f"[citations] {len(ok)}/{total} verified ({n_ev} events, {n_ar} articles)"
    if missing:
        line += " — UNVERIFIED: " + ", ".join(missing)
    return line


if __name__ == "__main__":
    # runnable self-check: one real id (resolved) + two bogus (flagged)
    from corpus import list_events

    real = list_events(limit=1)[0]["citation"]
    bogus_ev = "evt_" + "0" * 12
    bogus_ar = "f" * 16
    ans = f"See {real} for detail, also {bogus_ev} and {bogus_ar}.\n\n## Citations\n- {real} — x\n- {bogus_ev} — y\n- {bogus_ar} — z\n"
    r = validate(ans)
    assert r["total"] == 3, r
    assert real in r["ok"] and real in r["chain"], r
    assert r["chain"][real]["type"] == "event"
    assert r["chain"][real]["articles"], "event chain must carry its articles"
    assert r["missing"] == [bogus_ev, bogus_ar], r
    # claim -> evidence tracking: short label tags (model format); 趋势1核心
    # and 趋势1 MCP must pool under 趋势1 via _claim_key
    ans2 = (f"多智能体框架密集发布，工具链也在演进。\n\n"
            f"## Citations\n"
            f"- {real} — 多智能体事件（趋势1核心）\n"
            f"- 2d5ee61a05111f0a — 多智能体研究（趋势1）\n"
            f"- {bogus_ar} — 工具链文章（趋势2 MCP）\n")
    cl = extract_claims(ans2)
    assert [c["claim"] for c in cl] == ["趋势1", "趋势2"], cl
    assert cl[0]["evidence"] == [real, "2d5ee61a05111f0a"], cl  # 趋势1核心 pooled
    assert cl[1]["evidence"] == [bogus_ar], cl                   # 趋势2 MCP -> 趋势2
    tr = trace_claims(ans2)
    assert tr[0]["evidence"][0]["resolved"] is not None, "real id must resolve"
    assert tr[1]["evidence"][0]["resolved"] is None, "fabricated id must be None"
    print("PASS  citations self-check")
    print("  real:", real, "->", len(r["chain"][real]["articles"]), "articles chained")
    print("  missing (correctly flagged):", r["missing"])
    print("  claim tracking:", [f'{c["claim"]} -> {c["evidence"]}' for c in cl])