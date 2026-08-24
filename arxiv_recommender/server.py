"""FastAPI server exposing the recommender.

    uv run arxiv-serve          # reads [server] host/port from config.toml

Endpoints:
    GET  /                     browsable, filterable recommendations page
    GET  /status               DB + profile stats
    GET  /recommend            top non-library papers vs library centroid
    GET  /digest, /digest.md   latest saved digest
    GET  /similar/{arxiv_id}   papers most similar to a given arXiv ID
    POST /jobs/ingest          re-ingest the Zotero library (background)
    POST /jobs/embed           fetch missing embeddings from S2 (background)
    POST /jobs/fetch           pull new arXiv papers + rescore (background)
    POST /jobs/{name}/cancel   ask a running job to stop at its next checkpoint
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BeforeValidator, Field

from . import db, recommend
from .config import Config, load_config
from .embed import embed as run_embed
from .fetch import run_fetch
from .format import format_recommendations
from .ingest import ingest as run_ingest
from .s2 import S2Client

PACKAGE_ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))

# A browser submits a cleared form field as e.g. top= (empty string), which
# Pydantic can't parse as int/float. Coerce blank to a fallback before
# validation so range checks (ge=1, le=200, etc.) still reject real bad
# input (top=abc, min_score=9).
def _blank_as(default: object) -> BeforeValidator:
    return BeforeValidator(lambda v: default if v == "" else v)


BlankAsNone = Annotated[Annotated[int, Field(ge=1)] | None, _blank_as(None)]
BlankTop = Annotated[Annotated[int, Field(ge=1, le=200)], _blank_as(15)]
BlankMinScore = Annotated[Annotated[float, Field(ge=-1.0, le=1.0)], _blank_as(0.0)]


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="arXiv Recommender", version="0.1.0")
    app.state.config = config
    s2 = config.section("s2")
    app.state.s2_client = S2Client(
        api_key=s2.get("api_key", ""), batch_size=s2.get("batch_size", 500)
    )

    # In-memory only — single-process local tool, no need to survive a restart.
    # Keyed by job name ("ingest" / "embed"); None until that job has run once.
    app.state.jobs: dict[str, dict | None] = {"ingest": None, "embed": None, "fetch": None}
    job_locks = {
        "ingest": threading.Lock(), "embed": threading.Lock(), "fetch": threading.Lock(),
    }
    app.state.job_locks = job_locks  # exposed for tests to simulate a run in progress
    # Persistent, reused across runs: cleared at the start of each run, set by
    # POST /jobs/{name}/cancel. A stray set from a finished run can't leak
    # into the next one since it's cleared before that next run starts.
    cancel_events = {
        "ingest": threading.Event(), "embed": threading.Event(), "fetch": threading.Event(),
    }
    app.state.cancel_events = cancel_events

    def get_conn():
        # New connection per request: sqlite connections aren't thread-safe to share.
        return db.connect(config.db_path)

    def run_job(
        name: str,
        target: Callable[[Callable[[int, int], None], threading.Event], dict],
    ) -> None:
        # Runs in Starlette's background-task worker thread. A second POST
        # while this job is already running just fails to acquire the lock
        # and returns without starting a duplicate run.
        lock = job_locks[name]
        if not lock.acquire(blocking=False):
            return
        cancel_event = cancel_events[name]
        cancel_event.clear()
        app.state.jobs[name] = {
            "state": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "progress": None,
            "result": None,
            "error": None,
        }

        def on_progress(done: int, total: int) -> None:
            app.state.jobs[name]["progress"] = {"done": done, "total": total}

        try:
            result = target(on_progress, cancel_event)
            app.state.jobs[name]["state"] = "cancelled" if cancel_event.is_set() else "done"
            app.state.jobs[name]["result"] = result
        except Exception as exc:  # noqa: BLE001 — surface any failure in the UI banner
            app.state.jobs[name]["state"] = "error"
            app.state.jobs[name]["error"] = str(exc)
        finally:
            app.state.jobs[name]["finished_at"] = datetime.now(timezone.utc).isoformat()
            lock.release()

    app.mount("/static", StaticFiles(directory=str(PACKAGE_ROOT / "static")), name="static")

    # `cat` is a single comma-separated string here (not /recommend's repeated
    # list param) because a plain HTML text input can't submit a repeated
    # query key — same filter, form-friendly wire format.
    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        top: Annotated[BlankTop, Query()] = 15,
        min_score: Annotated[BlankMinScore, Query()] = 0.0,
        cat: str = Query("", description="Comma-separated category codes"),
        days: Annotated[BlankAsNone, Query()] = None,
    ):
        categories = [c.strip() for c in cat.split(",") if c.strip()] or None
        conn = get_conn()
        try:
            results = recommend.recommend(
                conn, top=top, min_score=min_score, categories=categories, days=days
            )
        finally:
            conn.close()
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "recs": results,
                "top": top,
                "min_score": min_score,
                "cat": categories or [],
                "days": days,
                "error": None,
                "jobs": app.state.jobs,
            },
        )

    @app.post("/jobs/ingest")
    def trigger_ingest(background_tasks: BackgroundTasks):
        def target(on_progress: Callable[[int, int], None], cancel_event: threading.Event) -> dict:
            return run_ingest(
                config.bib_path, config.db_path,
                progress_cb=on_progress, cancel_event=cancel_event,
            )

        background_tasks.add_task(run_job, "ingest", target)
        return RedirectResponse("/", status_code=303)

    @app.post("/jobs/embed")
    def trigger_embed(background_tasks: BackgroundTasks):
        def target(on_progress: Callable[[int, int], None], cancel_event: threading.Event) -> dict:
            s2cfg = config.section("s2")
            return run_embed(
                config.db_path,
                s2cfg.get("api_key", ""),
                s2cfg.get("batch_size", 500),
                library_only=True,
                progress_cb=on_progress,
                cancel_event=cancel_event,
            )

        background_tasks.add_task(run_job, "embed", target)
        return RedirectResponse("/", status_code=303)

    @app.post("/jobs/fetch")
    def trigger_fetch(background_tasks: BackgroundTasks):
        def target(on_progress: Callable[[int, int], None], cancel_event: threading.Event) -> dict:
            fetch_cfg = config.section("fetch")
            s2cfg = config.section("s2")
            return run_fetch(
                db_path=config.db_path,
                categories=fetch_cfg.get("categories", []),
                days_back=fetch_cfg.get("days_back", 7),
                max_fetch=fetch_cfg.get("max_fetch", 500),
                top_k=fetch_cfg.get("top_k", 15),
                min_score=fetch_cfg.get("min_score", 0.0),
                api_key=s2cfg.get("api_key", ""),
                batch_size=s2cfg.get("batch_size", 500),
                progress_cb=on_progress,
                cancel_event=cancel_event,
            )

        background_tasks.add_task(run_job, "fetch", target)
        return RedirectResponse("/", status_code=303)

    @app.post("/jobs/{name}/cancel")
    def cancel_job(name: str):
        if name not in cancel_events:
            raise HTTPException(404, f"Unknown job: {name}")
        cancel_events[name].set()
        return RedirectResponse("/", status_code=303)

    @app.get("/status")
    def status():
        conn = get_conn()
        try:
            with_emb, total = db.embedding_coverage(conn)
            last_fetch = conn.execute(
                "SELECT MAX(fetched_date) AS d FROM papers"
            ).fetchone()["d"]
            centroid, n = recommend.build_centroid(conn)
            return {
                "papers_total": total,
                "library_size": db.library_count(conn),
                "embedding_coverage": f"{with_emb}/{total}",
                "centroid_papers": n,
                "centroid_ready": centroid is not None,
                "last_fetched": last_fetch,
            }
        finally:
            conn.close()

    @app.get("/recommend")
    def recommend_endpoint(
        top: int = Query(15, ge=1, le=200),
        min_score: float = Query(0.0, ge=-1.0, le=1.0),
        cat: list[str] | None = Query(None, description="Filter by arXiv category code (repeatable)"),
        days: int | None = Query(None, ge=1, description="Only papers published within N days"),
    ):
        conn = get_conn()
        try:
            results = recommend.recommend(
                conn, top=top, min_score=min_score, categories=cat, days=days
            )
            return {"count": len(results), "results": results}
        finally:
            conn.close()

    @app.get("/digest")
    def digest_endpoint():
        conn = get_conn()
        try:
            digest = db.latest_digest(conn)
            if digest is None:
                raise HTTPException(404, "No digest yet — run `arxiv-fetch` first.")
            return digest
        finally:
            conn.close()

    @app.get("/digest.md", response_class=PlainTextResponse)
    def digest_md_endpoint():
        conn = get_conn()
        try:
            digest = db.latest_digest(conn)
            if digest is None:
                raise HTTPException(404, "No digest yet — run `arxiv-fetch` first.")
            return format_recommendations(
                digest["results"],
                header=f"arXiv digest — {digest['created_date'][:10]}",
            )
        finally:
            conn.close()

    @app.get("/similar/{arxiv_id}")
    def similar_endpoint(
        arxiv_id: str,
        top: int = Query(15, ge=1, le=200),
        exclude_library: bool = Query(True),
    ):
        conn = get_conn()
        try:
            results = recommend.similar(
                conn, arxiv_id, app.state.s2_client,
                top=top, exclude_library=exclude_library,
            )
            if results is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No embedding available for {arxiv_id} (unknown to S2 "
                           f"or not yet embedded).",
                )
            return {"query": arxiv_id, "count": len(results), "results": results}
        finally:
            conn.close()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the arXiv recommender API.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    import uvicorn

    config = load_config(args.config)
    server = config.section("server")
    host = args.host or server.get("host", "127.0.0.1")
    port = args.port or server.get("port", 8000)

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
