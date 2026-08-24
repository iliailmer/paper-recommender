from __future__ import annotations

import threading

import numpy as np

from arxiv_recommender import db as db_module
from arxiv_recommender import embed
from arxiv_recommender.s2 import EmbeddingResult
from tests.conftest import insert_paper


def test_embed_reports_progress(tmp_path, monkeypatch):
    db_path = tmp_path / "papers.db"
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    insert_paper(conn, "lib.1", in_library=True, embedding=None)
    conn.close()

    def fake_fetch_embeddings(self, arxiv_ids, progress_cb=None, cancel_event=None):
        if progress_cb is not None:
            progress_cb(1, 1)
        vec = np.zeros(768, dtype=np.float32)
        return [EmbeddingResult("lib.1", vec, "s2id", citation_count=0)], []

    monkeypatch.setattr(
        "arxiv_recommender.s2.S2Client.fetch_embeddings", fake_fetch_embeddings
    )

    calls: list[tuple[int, int]] = []
    summary = embed.embed(
        db_path,
        api_key="",
        batch_size=500,
        library_only=True,
        progress_cb=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(1, 1)]
    assert summary["stored"] == 1
    assert summary["cancelled"] is False


def test_embed_without_progress_cb_still_works(tmp_path, monkeypatch):
    db_path = tmp_path / "papers.db"
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    insert_paper(conn, "lib.1", in_library=True, embedding=None)
    conn.close()

    def fake_fetch_embeddings(self, arxiv_ids, progress_cb=None, cancel_event=None):
        vec = np.zeros(768, dtype=np.float32)
        return [EmbeddingResult("lib.1", vec, "s2id", citation_count=0)], []

    monkeypatch.setattr(
        "arxiv_recommender.s2.S2Client.fetch_embeddings", fake_fetch_embeddings
    )

    summary = embed.embed(db_path, api_key="", batch_size=500, library_only=True)
    assert summary["stored"] == 1


def test_embed_no_missing_ids_skips_fetch(tmp_path):
    db_path = tmp_path / "papers.db"
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    conn.close()

    summary = embed.embed(db_path, api_key="", batch_size=500, library_only=True)
    assert summary == {
        "requested": 0, "stored": 0, "missing": [], "coverage": (0, 0), "cancelled": False,
    }


def test_embed_reports_cancelled_when_event_is_set(tmp_path, monkeypatch):
    db_path = tmp_path / "papers.db"
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    insert_paper(conn, "lib.1", in_library=True, embedding=None)
    conn.close()

    def fake_fetch_embeddings(self, arxiv_ids, progress_cb=None, cancel_event=None):
        cancel_event.set()  # simulate S2Client stopping partway through
        return [], []

    monkeypatch.setattr(
        "arxiv_recommender.s2.S2Client.fetch_embeddings", fake_fetch_embeddings
    )

    cancel_event = threading.Event()
    summary = embed.embed(
        db_path, api_key="", batch_size=500, library_only=True, cancel_event=cancel_event
    )

    assert summary["cancelled"] is True
    assert summary["stored"] == 0
