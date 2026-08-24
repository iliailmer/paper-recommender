from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from arxiv_recommender import db as db_module
from arxiv_recommender.config import Config
from arxiv_recommender.server import create_app
from tests.conftest import insert_paper


def _make_client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "test.db"
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    conn.close()

    config = Config(
        raw={"s2": {}},
        config_path=tmp_path / "config.toml",
        db_path=db_path,
        bib_path=tmp_path / "library.bib",
    )
    app = create_app(config)
    return TestClient(app)


def test_root_renders_empty_state_with_no_library(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "No papers matched" in resp.text


def test_root_renders_recommendations(tmp_path):
    db_path = tmp_path / "test.db"
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    insert_paper(
        conn, "lib.1", in_library=True,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    insert_paper(
        conn, "close.1",
        embedding=np.array([2.0, 0.0], dtype=np.float32),
        published_date="2026-01-01", title="Close Paper",
    )
    conn.close()

    config = Config(
        raw={"s2": {}},
        config_path=tmp_path / "config.toml",
        db_path=db_path,
        bib_path=tmp_path / "library.bib",
    )
    app = create_app(config)
    client = TestClient(app)

    resp = client.get("/", params={"min_score": -1.0})
    assert resp.status_code == 200
    assert "Close Paper" in resp.text
    assert "arxiv.org/abs/close.1" in resp.text


def test_root_accepts_filter_params(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/", params={"top": 5, "min_score": 0.1, "cat": "cs.LG,cs.AI", "days": 7})
    assert resp.status_code == 200


def test_root_handles_blank_days_from_empty_form(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/", params={"top": 15, "min_score": 0.0, "cat": "", "days": ""})
    assert resp.status_code == 200


def test_root_handles_all_blank_fields_from_cleared_form(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/", params={"top": "", "min_score": "", "cat": "", "days": ""})
    assert resp.status_code == 200


def test_root_rejects_invalid_top_and_min_score(tmp_path):
    client = _make_client(tmp_path)
    assert client.get("/", params={"top": "abc"}).status_code == 422
    assert client.get("/", params={"top": 0}).status_code == 422
    assert client.get("/", params={"min_score": 9}).status_code == 422


def test_root_omits_meta_refresh_when_no_job_running(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/")
    assert '<meta http-equiv="refresh"' not in resp.text


def test_root_shows_meta_refresh_and_progress_while_job_running(tmp_path):
    client = _make_client(tmp_path)
    client.app.state.jobs["ingest"] = {
        "state": "running",
        "started_at": "x",
        "finished_at": None,
        "progress": {"done": 1, "total": 2},
        "result": None,
        "error": None,
    }

    resp = client.get("/")

    assert resp.status_code == 200
    assert '<meta http-equiv="refresh" content="2">' in resp.text
    assert "1/2" in resp.text


def test_post_jobs_ingest_runs_and_records_result(tmp_path):
    db_path = tmp_path / "test.db"
    conn = db_module.connect(db_path)
    db_module.init_db(conn)
    conn.close()

    bib_path = tmp_path / "library.bib"
    bib_path.write_text(
        "@article{a,\n"
        " title = {Paper A},\n"
        " author = {One, A},\n"
        " year = {2022},\n"
        " url = {http://arxiv.org/abs/2210.00001},\n"
        "}\n",
        encoding="utf-8",
    )

    config = Config(
        raw={"s2": {}}, config_path=tmp_path / "config.toml", db_path=db_path, bib_path=bib_path
    )
    app = create_app(config)
    client = TestClient(app)

    resp = client.post("/jobs/ingest", follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == "/"
    job = app.state.jobs["ingest"]
    assert job["state"] == "done"
    assert job["result"]["ingested"] == 1


def test_post_jobs_embed_records_error_from_underlying_failure(tmp_path, monkeypatch):
    client = _make_client(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("s2 unavailable")

    monkeypatch.setattr("arxiv_recommender.server.run_embed", boom)

    resp = client.post("/jobs/embed", follow_redirects=False)

    assert resp.status_code == 303
    job = client.app.state.jobs["embed"]
    assert job["state"] == "error"
    assert "s2 unavailable" in job["error"]


def test_post_jobs_fetch_runs_and_records_result(tmp_path, monkeypatch):
    client = _make_client(tmp_path)

    def fake_run_fetch(**kwargs):
        assert kwargs["categories"] == []
        return {
            "fetched": 10, "new": 3, "embedded": 3, "no_embedding": 0,
            "recommended": 3, "recs": [], "coverage": (3, 3),
            "timings": {"fetch": 0.1, "embed": 0.1, "score": 0.1, "total": 0.3},
        }

    monkeypatch.setattr("arxiv_recommender.server.run_fetch", fake_run_fetch)

    resp = client.post("/jobs/fetch", follow_redirects=False)

    assert resp.status_code == 303
    job = client.app.state.jobs["fetch"]
    assert job["state"] == "done"
    assert job["result"]["new"] == 3


def test_root_renders_fetch_result_banner(tmp_path):
    client = _make_client(tmp_path)
    client.app.state.jobs["fetch"] = {
        "state": "done",
        "started_at": "x",
        "finished_at": "y",
        "progress": None,
        "result": {"fetched": 10, "new": 3, "embedded": 3},
        "error": None,
    }

    resp = client.get("/")

    assert resp.status_code == 200
    assert "3 new papers" in resp.text


def test_post_jobs_ingest_is_a_noop_while_lock_held(tmp_path, monkeypatch):
    client = _make_client(tmp_path)
    calls = []
    monkeypatch.setattr(
        "arxiv_recommender.server.run_ingest",
        lambda *a, **k: calls.append(1) or {"ingested": 0, "skipped": [], "library_total": 0},
    )

    client.app.state.job_locks["ingest"].acquire()
    try:
        resp = client.post("/jobs/ingest", follow_redirects=False)
        assert resp.status_code == 303
        assert calls == []
    finally:
        client.app.state.job_locks["ingest"].release()
