"""Cosine-similarity scoring: flat-centroid profile, recommend, and similar.

SPECTER2 vectors are not unit-normalized, so we L2-normalize everything before
taking dot products (dot of unit vectors == cosine similarity).
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from . import db
from .s2 import EMBEDDING_DIM, S2Client


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_centroid(conn: sqlite3.Connection) -> tuple[np.ndarray | None, int]:
    """Flat centroid = mean of L2-normalized library embeddings, re-normalized.

    Returns (centroid_unit_vector, n_papers); (None, 0) if no embeddings.
    """
    rows = conn.execute(
        "SELECT embedding FROM papers WHERE in_library = 1 AND embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        return None, 0
    mat = np.stack([db.deserialize_embedding(r["embedding"]) for r in rows])
    centroid = _unit_rows(mat).mean(axis=0)
    return _unit(centroid), len(rows)


def _load_matrix(
    conn: sqlite3.Connection, extra_sql: str = "", params: tuple = ()
) -> tuple[list[str], np.ndarray]:
    rows = conn.execute(
        f"SELECT arxiv_id, embedding FROM papers "
        f"WHERE embedding IS NOT NULL {extra_sql}",
        params,
    ).fetchall()
    ids = [r["arxiv_id"] for r in rows]
    if not ids:
        return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    mat = np.stack([db.deserialize_embedding(r["embedding"]) for r in rows])
    return ids, mat


def _hydrate(conn: sqlite3.Connection, scored: list[tuple[str, float]]) -> list[dict]:
    """Attach metadata to (arxiv_id, score) pairs, preserving order."""
    if not scored:
        return []
    id_to_score = dict(scored)
    placeholders = ", ".join("?" for _ in scored)
    rows = conn.execute(
        f"SELECT arxiv_id, title, authors, categories, published_date, in_library "
        f"FROM papers WHERE arxiv_id IN ({placeholders})",
        [aid for aid, _ in scored],
    ).fetchall()
    by_id = {r["arxiv_id"]: r for r in rows}
    out = []
    for aid, score in scored:  # scored is already sorted
        r = by_id[aid]
        out.append({
            "arxiv_id": aid,
            "score": round(float(score), 4),
            "title": r["title"],
            "authors": json.loads(r["authors"] or "[]"),
            "categories": json.loads(r["categories"] or "[]"),
            "published_date": r["published_date"],
            "in_library": bool(r["in_library"]),
        })
    return out


def _rank(ids: list[str], scores: np.ndarray, top: int, min_score: float) -> list[tuple[str, float]]:
    order = np.argsort(-scores)
    ranked = []
    for i in order:
        s = float(scores[i])
        if s < min_score:
            break
        ranked.append((ids[i], s))
        if len(ranked) >= top:
            break
    return ranked


def recommend(
    conn: sqlite3.Connection,
    top: int = 15,
    min_score: float = 0.0,
    categories: list[str] | None = None,
    days: int | None = None,
) -> list[dict]:
    """Top non-library papers scored against the library centroid.

    Optional filters: `categories` (arXiv codes, matched against the stored
    category list) and `days` (only papers published within the window).
    """
    centroid, n = build_centroid(conn)
    if centroid is None:
        return []

    extra = "AND in_library = 0"
    params: list = []
    if categories:
        clauses = " OR ".join("categories LIKE ?" for _ in categories)
        extra += f" AND ({clauses})"
        params += [f'%"{c}"%' for c in categories]
    if days:
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        extra += " AND published_date >= ?"
        params.append(cutoff)

    ids, mat = _load_matrix(conn, extra, tuple(params))
    if not ids:
        return []
    scores = _unit_rows(mat) @ centroid
    return _hydrate(conn, _rank(ids, scores, top, min_score))


def recommend_hot(conn: sqlite3.Connection, top: int = 15) -> list[dict]:
    """Top non-library papers ranked by citation rate (citations / days since published)."""
    from datetime import date

    rows = conn.execute(
        "SELECT arxiv_id, citation_count, published_date FROM papers "
        "WHERE in_library = 0 AND citation_count IS NOT NULL AND published_date IS NOT NULL"
    ).fetchall()
    if not rows:
        return []

    today = date.today()
    scored: list[tuple[str, float]] = []
    for r in rows:
        try:
            pub = date.fromisoformat(r["published_date"])
        except ValueError:
            continue
        age = max(1, (today - pub).days)
        rate = r["citation_count"] / age
        scored.append((r["arxiv_id"], rate))

    scored.sort(key=lambda x: x[1], reverse=True)
    return _hydrate(conn, scored[:top])


def recommend_hot_similar(
    conn: sqlite3.Connection,
    top: int = 15,
    min_score: float = 0.0,
    alpha: float = 0.5,
) -> list[dict]:
    """Top non-library papers by 50/50 blend of cosine similarity and citation rate."""
    from datetime import date

    centroid, _ = build_centroid(conn)
    if centroid is None:
        return []

    rows = conn.execute(
        "SELECT arxiv_id, embedding, citation_count, published_date FROM papers "
        "WHERE in_library = 0 AND embedding IS NOT NULL "
        "AND citation_count IS NOT NULL AND published_date IS NOT NULL"
    ).fetchall()
    if not rows:
        return []

    today = date.today()
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    rates: list[float] = []

    for r in rows:
        try:
            pub = date.fromisoformat(r["published_date"])
        except ValueError:
            continue
        age = max(1, (today - pub).days)
        ids.append(r["arxiv_id"])
        vectors.append(db.deserialize_embedding(r["embedding"]))
        rates.append(r["citation_count"] / age)

    if not ids:
        return []

    mat = np.stack(vectors)
    cosine_scores = _unit_rows(mat) @ centroid  # shape (N,)
    rate_arr = np.array(rates, dtype=np.float64)

    # Normalize both to [0, 1]
    def _norm01(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    combined = alpha * _norm01(cosine_scores) + (1 - alpha) * _norm01(rate_arr)
    scored = list(zip(ids, combined.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = [(aid, s) for aid, s in scored if s >= min_score]
    return _hydrate(conn, scored[:top])


def similar(
    conn: sqlite3.Connection,
    arxiv_id: str,
    client: S2Client,
    top: int = 15,
    exclude_library: bool = True,
) -> list[dict] | None:
    """Papers most similar to a given arXiv ID.

    Uses the stored embedding if present, else fetches it from S2. Returns None
    if no embedding can be obtained (paper unknown to S2 / not yet embedded).
    """
    row = conn.execute(
        "SELECT embedding FROM papers WHERE arxiv_id = ? AND embedding IS NOT NULL",
        (arxiv_id,),
    ).fetchone()
    if row:
        query_vec = db.deserialize_embedding(row["embedding"])
    else:
        results, _ = client.fetch_embeddings([arxiv_id])
        if not results:
            return None
        query_vec = results[0].vector

    extra = "AND arxiv_id != ?"
    params: tuple = (arxiv_id,)
    if exclude_library:
        extra += " AND in_library = 0"
    ids, mat = _load_matrix(conn, extra, params)
    if not ids:
        return []
    scores = _unit_rows(mat) @ _unit(query_vec)
    return _hydrate(conn, _rank(ids, scores, top, min_score=-1.0))
