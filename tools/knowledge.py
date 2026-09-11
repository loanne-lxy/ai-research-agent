"""Knowledge tools — the agent's only window into the knowledge base.

AgentScope FunctionTools over the KAL facade (corpus.*): search_events /
get_event / search_articles / get_sources / corpus_stats. No SQL, no
schema, no direct DB access here — if the KAL contract changes, only
this file and the system prompt need to follow.

Design rules
  - every tool is read-only (FunctionTool(is_read_only=True)); the agent
    runs in DONT_ASK mode, so these are auto-approved and never block
  - every tool returns a plain dict -> AgentScope JSON-serializes it for
    the model (ensure_ascii=False, so Chinese stays readable)
  - every record carries `citation`; the docstrings tell the model to
    quote citations verbatim in answers (citations.py validates them
    via corpus.resolve())
  - tools never raise into the agent loop: failures become an
    {"error": ...} result so ReAct can continue/rephrase
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentscope.tool import FunctionTool

from corpus import (
    corpus_stats as _kal_stats,
    get_event as _kal_get_event,
    get_sources as _kal_get_sources,
    search_articles as _kal_search_articles,
    search_events as _kal_search_events,
)


def _guarded(fn, **kwargs):
    """Run a KAL call without ever raising into the agent loop."""
    try:
        return fn(**kwargs)
    except Exception as e:  # noqa: BLE001 - boundary by design
        return {"error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------ tools
def search_events(
    query: str,
    week: Optional[str] = None,
    category: Optional[str] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """Search the weekly AI-research event knowledge base by relevance.

    Use this first for any question about recent AI developments: it
    finds consolidated events (each event groups related articles).

    Args:
        query: Search keywords or a short question, Chinese or English
            (e.g. "视觉语言模型", "agent 工具调用", "PDE 神经网络").
        week: Optional week filter, format "2026-W36". Omit for all weeks.
        category: Optional category filter (see corpus_stats for values).
        k: Max results to return, default 10.

    Returns:
        dict with "results": list of events. Each event has: id,
        citation (USE THIS ID when citing in answers), title, week,
        category, importance, novelty, impact, source_count, summary
        (<=300 chars), score. To read an event's full detail and its
        source articles, call get_event with the event's id.
    """
    rows = _guarded(_kal_search_events, query=query, week=week,
                    category=category, k=min(k, 25))
    if isinstance(rows, dict):
        return rows  # error dict
    return {"count": len(rows), "results": rows}


def get_event(event_id: str) -> Dict[str, Any]:
    """Get one event in full, including all its source articles.

    Call this after search_events / list to inspect the evidence behind an
    event: it returns the event plus its article list (the
    Event -> Article -> Source chain).

    Args:
        event_id: The event id (equals its citation field, e.g.
            "evt_307d86c5e35e"), as returned by search_events.

    Returns:
        dict: the event's full record plus "articles": list of
        {id, citation, title, url, week, category, source, published,
        summary}. Cite the event by its citation; cite individual
        findings by the article citation. Returns {"error": ...} for an
        unknown id.
    """
    ev = _guarded(_kal_get_event, event_id=event_id)
    if ev is None:
        return {"error": f"no event with id {event_id!r}"}
    if isinstance(ev, dict) and "error" in ev:
        return ev
    return ev


def search_articles(
    query: str,
    week: Optional[str] = None,
    category: Optional[str] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """Search individual articles (papers, releases, posts) by relevance.

    Use when the user wants the primary sources rather than the
    consolidated event view, or when search_events misses the topic.

    Args:
        query: Search keywords or short question, Chinese or English.
        week: Optional week filter, format "2026-W36".
        category: Optional category filter (see corpus_stats for values).
        k: Max results to return, default 10.

    Returns:
        dict with "results": list of articles. Each has: id, citation
        (USE THIS when citing), title, url, week, category, source,
        published, summary (<=300 chars), score.
    """
    rows = _guarded(_kal_search_articles, query=query, week=week,
                    category=category, k=min(k, 25))
    if isinstance(rows, dict):
        return rows
    return {"count": len(rows), "results": rows}


def get_sources(active_only: bool = False) -> Dict[str, Any]:
    """List the monitored source pool and its health.

    Use when the user asks which sites/feeds are tracked, whether a
    source is working, or to ground a claim like "according to <source>".

    Args:
        active_only: If True, only list sources with status "active".

    Returns:
        dict with "count" and "results": list of sources, each with:
        citation (the source name), name, connector (arxiv/github/exa/...),
        category, status, enabled, articles_this_week, eval_score,
        last_success.
    """
    rows = _guarded(_kal_get_sources, active_only=active_only)
    if isinstance(rows, dict):
        return rows
    return {"count": len(rows), "results": rows}


def corpus_stats() -> Dict[str, Any]:
    """Describe the knowledge base: size, available weeks, categories.

    Call this first when the user's question references "最近/本周/上周"
    or when you need to know what week labels and categories exist to
    pass as filters.

    Returns:
        dict: articles, events, sources, sources_active counts;
        weeks (newest first), categories; semantic_search status
        ("on" or "keyword-only").
    """
    return _guarded(_kal_stats)


# --------------------------------------------------------------- assembly
def knowledge_toolkit():
    """Build the AgentScope Toolkit with all knowledge tools wired in."""
    from agentscope.tool import Toolkit

    tools = [
        FunctionTool(search_events, is_read_only=True),
        FunctionTool(get_event, is_read_only=True),
        FunctionTool(search_articles, is_read_only=True),
        FunctionTool(get_sources, is_read_only=True),
        FunctionTool(corpus_stats, is_read_only=True),
    ]
    return Toolkit(tools=tools)


__all__ = [
    "search_events", "get_event", "search_articles",
    "get_sources", "corpus_stats", "knowledge_toolkit",
]