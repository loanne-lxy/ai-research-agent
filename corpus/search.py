"""Ranking — turns (records, query) into scored, ordered results.

Two signals, combined:
  keyword  — token/substring match, title weighted 3x summary. CJK-aware:
             long CJK tokens fall back to character bigrams so partial
             overlap still scores ("视觉模型" ~ "视觉语言模型").
  semantic — fastembed cosine, only when fastembed + the cached model are
             available (paraphrase-multilingual-MiniLM-L12-v2, same model
             the weekly pipeline uses, served from local cache,
             HF_HUB_OFFLINE). Degrades to keyword-only if not installed.

This module is pure computation over record dicts — no SQL, no schema.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_CACHE = Path.home() / ".cache" / "fastembed"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / "embed_cache.npz"

W_KW, W_SEM = 0.4, 0.6
# semantic must beat a weak keyword hit to be meaningful
SEM_FLOOR = 0.25

_embedder = None
_embed_available: Optional[bool] = None


# ------------------------------------------------------------ tokenizing
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]{1,}")


def tokenize(query: str) -> List[str]:
    """ASCII words + CJK spans. CJK spans of len>=3 also contribute
    character bigrams (weaker signal) so partial matches still rank."""
    q = query.lower()
    toks: List[str] = []
    for m in _WORD_RE.findall(q):
        if m not in ("http", "https", "www", "com", "org", "arxiv"):
            toks.append(m)
    for span in _CJK_RE.findall(q):
        toks.append(span)
        if len(span) >= 3:
            toks.extend(span[i:i + 2] for i in range(len(span) - 1))
    # de-dup, keep order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def keyword_score(query_tokens: Sequence[str], title: str, summary: str) -> float:
    """0..1 — title hits weigh 3x, summary 1x, bigrams (len==2 CJK) half."""
    t = (title or "").lower()
    s = (summary or "").lower()
    if not query_tokens:
        return 0.0
    raw = 0.0
    for tok in query_tokens:
        cjk2 = len(tok) == 2 and _CJK_RE.fullmatch(tok)
        w = 0.5 if cjk2 else 1.0
        if tok in t:
            raw += 3.0 * w
        if tok in s:
            raw += 1.0 * w
    return min(raw / 10.0, 1.0)


# ------------------------------------------------------------ embeddings
def _get_embedder():
    global _embedder
    if _embedder is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from fastembed import TextEmbedding
        # local_files_only: load from the shared HF cache (same model +
        # cache dir as the weekly-ai-report pipeline); no network. If the
        # cache is absent, semantic_available() degrades to keyword-only.
        _embedder = TextEmbedding(model_name=MODEL,
                                  cache_dir=str(MODEL_CACHE),
                                  local_files_only=True)
    return _embedder


def semantic_available() -> bool:
    global _embed_available
    if _embed_available is None:
        try:
            _get_embedder()
            _embed_available = True
        except Exception:
            _embed_available = False
    return _embed_available


def _embed(texts: List[str]):
    import numpy as np
    vecs = np.array(list(_get_embedder().embed(texts)), dtype=np.float32)
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vecs / norm


def _load_cache(record_ids: List[str], cache_path: Optional[Path] = None
                ) -> Optional[Dict[str, "np.ndarray"]]:
    """Map id -> unit vector, or None when cache missing/stale."""
    import numpy as np
    cache_path = cache_path or CACHE_PATH
    if not cache_path.exists():
        return None
    try:
        z = np.load(cache_path, allow_pickle=False)
        cached = list(z["ids"])
        if set(cached) != set(record_ids) or len(cached) != len(record_ids):
            return None  # corpus changed -> rebuild
        return {rid: z["vecs"][i] for i, rid in enumerate(cached)}
    except Exception:
        return None


def _save_cache(record_ids: List[str], vecs,
                cache_path: Optional[Path] = None) -> None:
    import numpy as np
    cache_path = cache_path or CACHE_PATH
    cache_path.parent.mkdir(exist_ok=True)
    # str dtype (not object): object arrays need allow_pickle=True to read
    # back, which _load_cache refuses.
    np.savez_compressed(cache_path, vecs=vecs, ids=np.array(record_ids))


def semantic_scores(records: List[Dict[str, Any]], query: str,
                    title: str = "title", summary: str = "summary"
                    ) -> Dict[str, float]:
    """id -> cosine similarity. {} when embeddings unavailable.

    Cache is sharded by record-set size: events (147) and articles (1692)
    live in separate files, so alternating event/article searches no
    longer evict each other's cache (the old single slot re-embedded the
    whole corpus on every other call). Id-set validation in _load_cache
    still guards against stale content of the same size."""
    import numpy as np
    if not records or not semantic_available():
        return {}
    cache_path = CACHE_PATH.with_name(f"embed_cache_{len(records)}.npz")
    cache = _load_cache([r["id"] for r in records], cache_path)
    if cache is None:
        texts = [f"{r.get(title) or ''} . {r.get(summary) or ''}" for r in records]
        t0 = time.time()
        vecs = _embed(texts)
        _save_cache([r["id"] for r in records], vecs, cache_path)
        cache = {r["id"]: vecs[i] for i, r in enumerate(records)}
        print(f"[corpus] embedded {len(texts)} records in {time.time()-t0:.1f}s "
              f"-> {cache_path}")
    qv = _embed([f"{query}"])[0]
    out = {}
    for rid, v in cache.items():
        c = float(np.dot(v, qv))
        if c > SEM_FLOOR:
            out[rid] = c
    return out


# -------------------------------------------------------------- combine
def rank_records(
    records: List[Dict[str, Any]],
    query: str,
    k: int = 10,
    title: str = "title",
    summary: str = "summary",
) -> List[Dict[str, Any]]:
    """Score + order record dicts. Returns new dicts with a 'score' field
    (0..1, 3 decimals), best first. Records with score 0 are dropped."""
    toks = tokenize(query)
    use_sem = semantic_available()
    sem = semantic_scores(records, query, title, summary) if use_sem else {}
    out = []
    for r in records:
        kw = keyword_score(toks, r.get(title) or "", r.get(summary) or "")
        s = sem.get(r["id"], 0.0)
        if use_sem:
            score = W_KW * kw + W_SEM * s
        else:
            score = kw
        if score <= 0.0:
            continue
        item = dict(r)
        item["score"] = round(float(score), 3)
        out.append(item)
    out.sort(key=lambda r: (-r["score"], r["id"]))
    return out[:k]