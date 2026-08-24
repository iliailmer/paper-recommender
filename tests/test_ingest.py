from __future__ import annotations

import threading

from arxiv_recommender import db as db_module
from arxiv_recommender import ingest

BIB_TWO_PAPERS = """
@article{a,
  title = {Paper A},
  author = {One, A},
  year = {2022},
  url = {http://arxiv.org/abs/2210.00001},
}
@article{b,
  title = {Paper B},
  author = {Two, B},
  year = {2022},
  url = {http://arxiv.org/abs/2210.00002},
}
"""


def test_ingest_reports_progress_per_paper(tmp_path):
    bib_path = tmp_path / "library.bib"
    bib_path.write_text(BIB_TWO_PAPERS, encoding="utf-8")
    db_path = tmp_path / "papers.db"

    calls: list[tuple[int, int]] = []
    summary = ingest.ingest(
        bib_path, db_path, progress_cb=lambda done, total: calls.append((done, total))
    )

    assert calls == [(1, 2), (2, 2)]
    assert summary["ingested"] == 2


def test_ingest_without_progress_cb_still_works(tmp_path):
    bib_path = tmp_path / "library.bib"
    bib_path.write_text(BIB_TWO_PAPERS, encoding="utf-8")
    db_path = tmp_path / "papers.db"

    summary = ingest.ingest(bib_path, db_path)
    assert summary["ingested"] == 2
    assert summary["cancelled"] is False


def test_ingest_stops_early_when_cancelled_mid_run(tmp_path):
    bib_path = tmp_path / "library.bib"
    bib_path.write_text(BIB_TWO_PAPERS, encoding="utf-8")
    db_path = tmp_path / "papers.db"

    cancel_event = threading.Event()

    def progress_cb(done, total):
        if done == 1:
            cancel_event.set()  # simulate a cancel request landing after paper 1

    summary = ingest.ingest(
        bib_path, db_path, progress_cb=progress_cb, cancel_event=cancel_event
    )

    assert summary["ingested"] == 1
    assert summary["cancelled"] is True

    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    assert db_module.library_count(conn) == 1  # the first paper's upsert was committed
    conn.close()


def test_ingest_already_cancelled_ingests_nothing(tmp_path):
    bib_path = tmp_path / "library.bib"
    bib_path.write_text(BIB_TWO_PAPERS, encoding="utf-8")
    db_path = tmp_path / "papers.db"

    cancel_event = threading.Event()
    cancel_event.set()

    summary = ingest.ingest(bib_path, db_path, cancel_event=cancel_event)

    assert summary["ingested"] == 0
    assert summary["cancelled"] is True
