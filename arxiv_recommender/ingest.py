"""Ingest the Zotero library into SQLite.

Parses the configured .bib file, extracts arXiv papers, and upserts them into
the papers table with in_library = 1. Run with:

    uv run arxiv-ingest
    uv run arxiv-ingest --bib path/to/library.bib --db path/to/papers.db
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import bibtex, db
from .config import load_config


def ingest(bib_path: Path, db_path: Path) -> dict:
    """Parse the library and upsert it. Returns a summary dict."""
    papers, skipped = bibtex.parse_library(bib_path)
    fetched_date = datetime.now(timezone.utc).isoformat()

    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        for paper in papers:
            db.upsert_paper(
                conn,
                {
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "authors": json.dumps(paper["authors"]),
                    "abstract": paper["abstract"],
                    "categories": json.dumps(paper["categories"]),
                    "published_date": paper["published_date"],
                    "fetched_date": fetched_date,
                    "in_library": 1,
                    "collection": None,  # not present in this export
                    "date_added": None,  # not present in this export
                },
            )
        conn.commit()
        total_in_library = db.library_count(conn)
    finally:
        conn.close()

    return {
        "ingested": len(papers),
        "skipped": skipped,
        "library_total": total_in_library,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Zotero library into SQLite.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.toml")
    parser.add_argument("--bib", type=Path, default=None, help="Override bib_path")
    parser.add_argument("--db", type=Path, default=None, help="Override db_path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bib_path = args.bib or cfg.bib_path
    db_path = args.db or cfg.db_path

    print(f"Ingesting {bib_path}")
    print(f"      ->  {db_path}")
    summary = ingest(bib_path, db_path)

    print(f"\nIngested {summary['ingested']} arXiv papers "
          f"(library now holds {summary['library_total']}).")
    if summary["skipped"]:
        print(f"Skipped {len(summary['skipped'])} entries with no arXiv ID:")
        for key in summary["skipped"]:
            print(f"  - {key}")


if __name__ == "__main__":
    main()
