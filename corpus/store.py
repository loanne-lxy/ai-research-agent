"""Knowledge Access Layer (KAL) — stable, schema-independent read API.

The ONLY entry point the agent tools use. Contract (do not break):

  search_events(query, week?, category?, k=10) -> [Event]
  get_event(event_id)          -> Event | None      # includes its Articles
  list_events(week?, category?, limit=50)        -> [Event]
  search_articles(query, week?, category?, k=10) -> [Article]
  get_article(article_id)      -> Article | None
  get_sources(active_only=False)                 -> [Source]
  resolve(citation)            -> {"type", "object"} | None
  corpus_stats()               -> dict

Citation contract
  Every Event/Article carries a `citation` field. For now it equals the
  underlying stable content-hash id (event: evt_xxxxx, article: 16-hex).
  Answers must cite that value verbatim; `resolve()` is the inverse
  operation used by the citation validator. The KAL keeps `citation` as a
  first-class field so the id scheme can change later without touching
  the agent — tools only ever read `citation`.

Return stability
  - fixed key set per object type (below); missing/None DB values become
    "" / 0 / None, never raise
  - summaries truncated to SUMMARY_MAX (300) chars
  - ordering is deterministic (score desc, then id)
  - search results carry a `score` (0..1); non-search results do not

Object schemas
  Event    {id, citation, title, week, category, importance, novelty,
            impact, source_count, summary, [score]}
  Event.articles (only via get_event) : [Article]
  Article  {id, citation, title, url, week, category, source, published,
            summary, [score]}
  Source   {citation, name, connector, category, status, enabled,
            articles_this_week, eval_score, last_success}

Under the hood: corpus._sqlite (the only schema-aware code) + corpus.search
(rank/keyword/semantic). This module never touches SQL.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional

from . import _sqlite as db
from .search import rank_records

SUMMARY_MAX = 300


def _clamp(v, default):
    return default if v is None else v


def _trunc(s: Optional[str], n: int = SUMMARY_MAX) -> str:
    s = _clamp(s, "") or ""
    return s if len(s) <= n else s[:n] + "…"


# ------------------------------------------------------------- mappers
def _event(row: Dict[str, Any], score: Optional[float] = None) -> Dict[str, Any]:
    e = {
        "id": row["id"],
        "citation": row["id"],  # stable citation = content-hash id
        "title": _clamp(row.get("title"), ""),
        "week": _clamp(row.get("week_label"), ""),
        "category": _clamp(row.get("category"), ""),
        "importance": _clamp(row.get("importance"), 0.0),
        "novelty": _clamp(row.get("novelty"), 0.0),
        "impact": _clamp(row.get("impact"), 0.0),
        "source_count": _clamp(row.get("source_count"), 0),
        "summary": _trunc(row.get("summary")),
    }
    if score is not None:
        e["score"] = score
    return e


def _article(row: Dict[str, Any], score: Optional[float] = None) -> Dict[str, Any]:
    a = {
        "id": row["id"],
        "citation": row["id"],
        "title": _clamp(row.get("title"), ""),
        "url": _clamp(row.get("url"), ""),
        "week": _clamp(row.get("week_bucket"), ""),
        "category": _clamp(row.get("category"), ""),
        "source": _clamp(row.get("source_name"), ""),
        "published": _clamp(row.get("published"), ""),
        "summary": _trunc(row.get("summary")),
    }
    if score is not None:
        a["score"] = score
    return a


def _source(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "citation": _clamp(row.get("name"), ""),  # source name is unique per pool
        "name": _clamp(row.get("name"), ""),
        "connector": _clamp(row.get("connector"), ""),
        "category": _clamp(row.get("category"), ""),
        "status": _clamp(row.get("status"), ""),
        "enabled": _clamp(row.get("enabled"), 1) == 1,
        "articles_this_week": _clamp(row.get("articles_this_week"), 0),
        "eval_score": row.get("eval_score"),
        "last_success": _clamp(row.get("last_success"), ""),
    }


# ------------------------------------------------------------- events
def search_events(
    query: str,
    week: Optional[str] = None,
    category: Optional[str] = None,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """Hybrid (keyword + optional semantic) search over event titles/summaries."""
    rows = db.fetch_events(week=week, category=category)
    return [_event(r, r.get("score")) for r in rank_records(rows, query, k=k)]


def list_events(
    week: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Deterministic listing: importance desc, novelty desc, id asc."""
    return [_event(r) for r in db.fetch_events(week=week, category=category, limit=limit)]


def get_event(event_id: str) -> Optional[Dict[str, Any]]:
    """Full event with its attached articles (the Event -> Article evidence
    chain). Returns None for unknown ids."""
    row = db.fetch_event_by_id(event_id)
    if row is None:
        return None
    e = _event(row)
    urls = db.fetch_event_article_urls(event_id)
    by_url = {r["url"]: r for r in db.fetch_articles_by_urls(urls)}
    arts = []
    for u in urls:
        r = by_url.get(u)
        if r:
            arts.append(_article(r))
    arts.sort(key=lambda a: (a["published"] or ""), reverse=True)
    e["articles"] = arts
    return e


def resolve_article_ref(article_id: str) -> Optional[Dict[str, Any]]:
    row = db.fetch_article_by_id(article_id)
    return _article(row) if row else None


# ----------------------------------------------------------- articles
def search_articles(
    query: str,
    week: Optional[str] = None,
    category: Optional[str] = None,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """Hybrid search over article titles/summaries."""
    rows = db.fetch_articles(week=week, category=category)
    return [_article(r, r.get("score")) for r in rank_records(rows, query, k=k)]


def get_article(article_id: str) -> Optional[Dict[str, Any]]:
    return resolve_article_ref(article_id)


# ------------------------------------------------------------- sources
def get_sources(active_only: bool = False) -> List[Dict[str, Any]]:
    """Source pool state (which feeds are live, their health)."""
    return [_source(r) for r in db.fetch_sources(active_only=active_only)]


# -------------------------------------------------------------- misc
def resolve(citation: str) -> Optional[Dict[str, Any]]:
    """Inverse of the citation contract: given a citation value from an
    answer, return the object it points to (or None if not found / bogus).
    Used by the citation validator. Prefers events (evt_* prefix / known
    event ids), then articles."""
    if not citation:
        return None
    row = db.fetch_event_by_id(citation)
    if row is not None:
        return {"type": "event", "object": _event(row)}
    row = db.fetch_article_by_id(citation)
    if row is not None:
        return {"type": "article", "object": _article(row)}
    return None


def corpus_stats() -> Dict[str, Any]:
    """Corpus shape — for orienting the agent / sanity checks."""
    stats = db.corpus_counts()
    stats["weeks"] = db.distinct_weeks()
    stats["categories"] = db.distinct_event_categories()
    stats["semantic_search"] = _semantic_status()
    return stats


def _semantic_status() -> str:
    from .search import semantic_available
    return "on" if semantic_available() else "keyword-only (fastembed not installed)"


__all__ = [
    "search_events", "get_event", "list_events",
    "search_articles", "get_article",
    "get_sources", "resolve", "corpus_stats",
]