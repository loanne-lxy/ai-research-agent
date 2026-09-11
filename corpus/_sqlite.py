"""SQLite adapter — the ONLY module that knows the DB schema.

Everything else in this package talks to domain objects, not tables.
If weekly-ai-report adds/renames/drops a column, the breakage is
contained here: mappers read fields via dict.get() so a missing column
degrades to a default instead of crashing (stable-returns contract).

All connections open read-only (URI mode=ro). This package never writes.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

# Knowledge base lives in the sibling project. Override via env if needed.
KB_DIR = Path(os.environ.get("WEEKLY_KB_DIR", "/home/loanne/weekly-ai-report/data"))
KNOWLEDGE_DB = KB_DIR / "knowledge.db"
SOURCE_DB = KB_DIR / "source.db"


def _conn(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"knowledge db not found: {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _row(r: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """Row -> plain dict (only the columns that exist). None -> None."""
    return dict(r) if r is not None else None


# ---------------------------------------------------------------- events
def fetch_events(
    week: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    sql = (
        "SELECT id, week_label, title, summary, category, importance, novelty, "
        "impact, source_count, created_at FROM events"
    )
    where, args = _where([
        ("week_label", week),
        ("category", category),
    ])
    sql += where + " ORDER BY importance DESC, novelty DESC, id LIMIT ?"
    args = list(args) + [limit]
    with _conn(KNOWLEDGE_DB) as con:
        return [d for d in (_row(r) for r in con.execute(sql, args)) if d]


def fetch_event_by_id(event_id: str) -> Optional[Dict[str, Any]]:
    sql = (
        "SELECT id, week_label, title, summary, category, importance, novelty, "
        "impact, source_count, created_at FROM events WHERE id = ?"
    )
    with _conn(KNOWLEDGE_DB) as con:
        return _row(con.execute(sql, (event_id,)).fetchone())


def fetch_event_article_urls(event_id: str) -> List[str]:
    with _conn(KNOWLEDGE_DB) as con:
        rows = con.execute(
            "SELECT article_url FROM event_articles WHERE event_id = ? "
            "ORDER BY article_url", (event_id,),
        )
        return [r[0] for r in rows]


def count_event_articles(event_id: str) -> int:
    with _conn(KNOWLEDGE_DB) as con:
        return con.execute(
            "SELECT COUNT(*) FROM event_articles WHERE event_id = ?", (event_id,),
        ).fetchone()[0]


# -------------------------------------------------------------- articles
def fetch_articles(
    week: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    sql = (
        "SELECT id, url, title, summary, published, source_name, category, "
        "week_bucket FROM articles"
    )
    where, args = _where([
        ("week_bucket", week),
        ("category", category),
    ])
    sql += where + " ORDER BY published DESC, id LIMIT ?"
    args = list(args) + [limit]
    with _conn(KNOWLEDGE_DB) as con:
        return [d for d in (_row(r) for r in con.execute(sql, args)) if d]


def fetch_articles_by_urls(urls: List[str]) -> List[Dict[str, Any]]:
    if not urls:
        return []
    sql = (
        "SELECT id, url, title, summary, published, source_name, category, "
        "week_bucket FROM articles WHERE url IN ({}) "
        "ORDER BY published DESC, id".format(",".join("?" * len(urls)))
    )
    with _conn(KNOWLEDGE_DB) as con:
        return [d for d in (_row(r) for r in con.execute(sql, list(urls))) if d]


def fetch_article_by_id(article_id: str) -> Optional[Dict[str, Any]]:
    sql = (
        "SELECT id, url, title, summary, published, source_name, category, "
        "week_bucket FROM articles WHERE id = ?"
    )
    with _conn(KNOWLEDGE_DB) as con:
        return _row(con.execute(sql, (article_id,)).fetchone())


# --------------------------------------------------------------- sources
def fetch_sources(active_only: bool = False) -> List[Dict[str, Any]]:
    sql = (
        "SELECT name, connector, category, status, enabled, articles_this_week, "
        "eval_score, last_success FROM sources"
    )
    where, args = _where([
        ("status", "active") if active_only else None,
    ])
    sql += where + " ORDER BY connector, name"
    with _conn(SOURCE_DB) as con:
        return [d for d in (_row(r) for r in con.execute(sql, args)) if d]


# ------------------------------------------------------------------ meta
def corpus_counts() -> Dict[str, int]:
    with _conn(KNOWLEDGE_DB) as con:
        out = {
            "articles": con.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
            "events": con.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        }
    with _conn(SOURCE_DB) as con:
        out["sources"] = con.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        out["sources_active"] = con.execute(
            "SELECT COUNT(*) FROM sources WHERE status = 'active'").fetchone()[0]
    return out


def distinct_weeks() -> List[str]:
    with _conn(KNOWLEDGE_DB) as con:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT week_label FROM events WHERE week_label IS NOT NULL "
            "ORDER BY week_label DESC")]


def distinct_event_categories() -> List[str]:
    with _conn(KNOWLEDGE_DB) as con:
        return [r[0] for r in con.execute(
            "SELECT category FROM events WHERE category IS NOT NULL AND "
            "category != 'Uncategorized' GROUP BY category ORDER BY COUNT(*) DESC")]


def _where(pairs) -> tuple[str, list]:
    """Build WHERE from (column, value) pairs; None entries are skipped."""
    clauses, args = [], []
    for pair in pairs:
        if pair is None:
            continue
        col, val = pair
        if val is not None:
            clauses.append(f"{col} = ?")
            args.append(val)
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), args