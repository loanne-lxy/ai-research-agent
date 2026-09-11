"""Knowledge Access Layer (KAL) for the weekly-ai-report corpus.

Public surface is the stable API re-exported below. Import from the
package root, never from submodules:

    from corpus import search_events, get_event, search_articles, get_sources
"""
from .store import (
    corpus_stats,
    get_article,
    get_event,
    get_sources,
    list_events,
    resolve,
    search_articles,
    search_events,
)

__all__ = [
    "search_events", "get_event", "list_events",
    "search_articles", "get_article",
    "get_sources", "resolve", "corpus_stats",
]