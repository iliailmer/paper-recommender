"""Speed benchmarks for the scoring and bulk-write hot paths.

Run `uv run pytest tests/test_benchmarks.py --benchmark-only -v` locally for
full timing stats (mean, stddev, rounds). CI runs the whole suite with
`--benchmark-disable`, which executes each benchmarked function once as a
plain correctness check without gathering timing statistics — shared CI
runners are too noisy for meaningful absolute-timing comparisons.

N_PAPERS is set well beyond the current production DB size (~3000 papers)
to leave headroom for judging whether the numpy-based scoring still holds
up as the library grows over time.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from arxiv_recommender import db, recommend

N_PAPERS = 5000
EMBEDDING_DIM = 768


def _random_vector(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=EMBEDDING_DIM).astype(np.float32)


def _populate(conn, n_papers: int, rng: np.random.Generator) -> None:
    """Insert n_papers synthetic non-library papers plus a few library papers
    (needed to build a centroid) with random embeddings and citation counts."""
    for i in range(5):
        conn.execute(
            "INSERT INTO papers (arxiv_id, in_library, embedding) VALUES (?, 1, ?)",
            (f"lib.{i}", db.serialize_embedding(_random_vector(rng))),
        )
    rows = [
        (
            f"paper.{i}",
            "Title",
            json.dumps([]),
            json.dumps([]),
            "2026-01-01",
            db.serialize_embedding(_random_vector(rng)),
            int(rng.integers(0, 200)),
        )
        for i in range(n_papers)
    ]
    conn.executemany(
        "INSERT INTO papers "
        "(arxiv_id, title, authors, categories, published_date, embedding, citation_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


@pytest.fixture
def populated_conn(conn):
    rng = np.random.default_rng(42)
    _populate(conn, N_PAPERS, rng)
    return conn


def test_recommend_speed(populated_conn, benchmark):
    result = benchmark(recommend.recommend, populated_conn, top=15, min_score=0.0)
    assert len(result) == 15


def test_recommend_hot_speed(populated_conn, benchmark):
    result = benchmark(recommend.recommend_hot, populated_conn, top=15)
    assert len(result) == 15


def test_recommend_hot_similar_speed(populated_conn, benchmark):
    result = benchmark(recommend.recommend_hot_similar, populated_conn, top=15)
    assert len(result) == 15


def test_set_embeddings_bulk_speed(populated_conn, benchmark):
    rng = np.random.default_rng(1)
    updates = [
        (f"paper.{i}", _random_vector(rng), None, int(rng.integers(0, 200)))
        for i in range(N_PAPERS)
    ]
    benchmark(db.set_embeddings, populated_conn, updates)


def test_insert_fetched_papers_cold_speed(conn, benchmark):
    """Bulk-insert N_PAPERS brand-new rows; the table is truncated before each
    round so this measures a cold insert, not the steady-state conflict-skip
    path (most real runs re-fetch mostly-already-seen papers)."""
    papers = [
        {
            "arxiv_id": f"paper.{i}",
            "title": "Title",
            "authors": json.dumps([]),
            "abstract": "",
            "categories": json.dumps([]),
            "published_date": "2026-01-01",
            "fetched_date": "2026-01-01T00:00:00+00:00",
        }
        for i in range(N_PAPERS)
    ]

    def setup():
        conn.execute("DELETE FROM papers")
        conn.commit()
        return (conn, papers), {}

    result = benchmark.pedantic(db.insert_fetched_papers, setup=setup, rounds=5)
    assert result == N_PAPERS
