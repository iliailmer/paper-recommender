"""FastAPI server exposing the recommender.

    uv run arxiv-serve          # reads [server] host/port from config.toml

Endpoints:
    GET /status               DB + profile stats
    GET /recommend            top non-library papers vs library centroid
    GET /similar/{arxiv_id}   papers most similar to a given arXiv ID
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse

from . import db, recommend
from .config import Config, load_config
from .format import format_recommendations
from .s2 import S2Client


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="arXiv Recommender", version="0.1.0")
    app.state.config = config
    s2 = config.section("s2")
    app.state.s2_client = S2Client(
        api_key=s2.get("api_key", ""), batch_size=s2.get("batch_size", 500)
    )

    def get_conn():
        # New connection per request: sqlite connections aren't thread-safe to share.
        return db.connect(config.db_path)

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
