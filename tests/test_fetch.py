from __future__ import annotations

import threading

from arxiv_recommender import db as db_module
from arxiv_recommender import fetch

FAKE_PAPERS = [
    {
        "arxiv_id": "2210.00001",
        "title": "Paper A",
        "authors": ["One, A"],
        "abstract": "",
        "categories": ["cs.LG"],
        "published_date": "2026-01-01",
    },
    {
        "arxiv_id": "2210.00002",
        "title": "Paper B",
        "authors": ["Two, B"],
        "abstract": "",
        "categories": ["cs.LG"],
        "published_date": "2026-01-01",
    },
]


def _init_db(db_path):
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    conn.close()


def test_run_fetch_without_cancel_event_saves_digest(tmp_path, monkeypatch):
    db_path = tmp_path / "papers.db"
    _init_db(db_path)

    monkeypatch.setattr(
        "arxiv_recommender.fetch.arxiv_api.search_recent", lambda *a, **k: FAKE_PAPERS
    )
    monkeypatch.setattr(
        "arxiv_recommender.s2.S2Client.fetch_embeddings",
        lambda self, ids, progress_cb=None, cancel_event=None: ([], ids),
    )

    summary = fetch.run_fetch(
        db_path=db_path, categories=["cs.LG"], days_back=7, max_fetch=10,
        top_k=5, min_score=-1.0,
    )

    assert summary["new"] == 2
    assert summary["cancelled"] is False
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    assert db_module.latest_digest(conn) is not None
    conn.close()


def test_run_fetch_skips_scoring_when_cancelled_during_embed(tmp_path, monkeypatch):
    db_path = tmp_path / "papers.db"
    _init_db(db_path)

    monkeypatch.setattr(
        "arxiv_recommender.fetch.arxiv_api.search_recent", lambda *a, **k: FAKE_PAPERS
    )

    def fake_fetch_embeddings(self, ids, progress_cb=None, cancel_event=None):
        cancel_event.set()  # simulate a cancel request landing mid-embed
        return [], ids

    monkeypatch.setattr(
        "arxiv_recommender.s2.S2Client.fetch_embeddings", fake_fetch_embeddings
    )

    cancel_event = threading.Event()
    summary = fetch.run_fetch(
        db_path=db_path, categories=["cs.LG"], days_back=7, max_fetch=10,
        top_k=5, min_score=-1.0, cancel_event=cancel_event,
    )

    assert summary["new"] == 2  # step 1 (fetch+store) already committed, kept
    assert summary["cancelled"] is True
    assert summary["recommended"] == 0
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    assert db_module.latest_digest(conn) is None  # step 3 was skipped
    conn.close()


def test_run_fetch_already_cancelled_still_fetches_but_skips_rest(tmp_path, monkeypatch):
    db_path = tmp_path / "papers.db"
    _init_db(db_path)

    monkeypatch.setattr(
        "arxiv_recommender.fetch.arxiv_api.search_recent", lambda *a, **k: FAKE_PAPERS
    )
    embed_calls = []
    monkeypatch.setattr(
        "arxiv_recommender.s2.S2Client.fetch_embeddings",
        lambda self, ids, progress_cb=None, cancel_event=None: embed_calls.append(1),
    )

    cancel_event = threading.Event()
    cancel_event.set()  # already cancelled before run_fetch is even called
    summary = fetch.run_fetch(
        db_path=db_path, categories=["cs.LG"], days_back=7, max_fetch=10,
        top_k=5, min_score=-1.0, cancel_event=cancel_event,
    )

    # Step 1 (arXiv search + store) isn't itself interruptible mid-call, so
    # it still runs; step 2 (embed) is skipped entirely since the event was
    # already set before it would have started.
    assert summary["new"] == 2
    assert summary["cancelled"] is True
    assert embed_calls == []
