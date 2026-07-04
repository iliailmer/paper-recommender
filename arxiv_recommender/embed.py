"""Fetch SPECTER2 embeddings for papers that lack them, and store them.

Run with:
    uv run arxiv-embed         # embed library papers missing vectors
    uv run arxiv-embed --all   # include non-library papers too

Papers S2 has no embedding for stay NULL, so simply re-running this later
re-tries them (e.g. once S2 indexes a very new paper).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from . import db
from .config import load_config
from .s2 import S2Client


def embed(db_path: Path, api_key: str, batch_size: int, library_only: bool = True) -> dict:
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        ids = db.ids_missing_embeddings(conn, library_only=library_only)
        if not ids:
            with_emb, total = db.embedding_coverage(conn)
            return {"requested": 0, "stored": 0, "missing": [], "coverage": (with_emb, total)}

        client = S2Client(api_key=api_key, batch_size=batch_size)
        results, missing = client.fetch_embeddings(ids)

        db.set_embeddings(conn, [
            (r.arxiv_id, r.vector, r.s2_paper_id, r.citation_count) for r in results
        ])
        conn.commit()

        with_emb, total = db.embedding_coverage(conn)
        return {
            "requested": len(ids),
            "stored": len(results),
            "missing": missing,
            "coverage": (with_emb, total),
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SPECTER2 embeddings from Semantic Scholar.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.toml")
    parser.add_argument("--db", type=Path, default=None, help="Override db_path")
    parser.add_argument("--all", action="store_true", help="Embed non-library papers too")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show backoff/retry logs")
    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    cfg = load_config(args.config)
    db_path = args.db or cfg.db_path
    s2 = cfg.section("s2")
    api_key = s2.get("api_key", "")
    batch_size = s2.get("batch_size", 500)

    if not api_key:
        print("No S2 API key set — using the throttled keyless tier (expect retries).")

    summary = embed(db_path, api_key, batch_size, library_only=not args.all)

    with_emb, total = summary["coverage"]
    if summary["requested"] == 0:
        print(f"Nothing to do. Embedding coverage: {with_emb}/{total}.")
        return
    print(f"\nStored {summary['stored']}/{summary['requested']} embeddings. "
          f"Coverage: {with_emb}/{total}.")
    if summary["missing"]:
        print(f"No embedding available from S2 for {len(summary['missing'])}:")
        for aid in summary["missing"]:
            print(f"  - {aid}")


if __name__ == "__main__":
    main()
