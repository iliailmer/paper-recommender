"""Nightly job: fetch recent arXiv papers, embed them, score, save a digest.

    uv run arxiv-fetch                      # uses [fetch] config defaults
    uv run arxiv-fetch --days 3 --max 100   # override window / cap

Pipeline:
    arXiv API search  ->  insert new (in_library=0)
                      ->  fetch SPECTER2 embeddings for anything missing
                      ->  score non-library papers against the library centroid
                      ->  persist the ranked list as a digest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from loguru import logger

from . import arxiv_api, db, recommend
from .config import load_config
from .format import format_recommendations
from .s2 import S2Client


def run_fetch(
    db_path: Path,
    categories: list[str],
    days_back: int,
    max_fetch: int,
    top_k: int,
    min_score: float,
    api_key: str = "",
    batch_size: int = 500,
    mode: str = "default",
) -> dict:
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        t_start = time.perf_counter()

        # 1. Fetch + store new papers.
        fetched = arxiv_api.search_recent(categories, days_back, max_fetch)
        now = datetime.now(timezone.utc).isoformat()
        new_count = db.insert_fetched_papers(conn, [
            {
                "arxiv_id": p["arxiv_id"],
                "title": p["title"],
                "authors": json.dumps(p["authors"]),
                "abstract": p["abstract"],
                "categories": json.dumps(p["categories"]),
                "published_date": p["published_date"],
                "fetched_date": now,
            }
            for p in fetched
        ])
        conn.commit()
        t_fetch = time.perf_counter()
        logger.info(
            "arXiv fetch + store: {:.2f}s ({} fetched, {} new)",
            t_fetch - t_start, len(fetched), new_count,
        )

        # 2. Embed anything still missing a vector (library + new).
        missing_ids = db.ids_missing_embeddings(conn, library_only=False)
        embedded = 0
        no_embedding: list[str] = []
        if missing_ids:
            client = S2Client(api_key=api_key, batch_size=batch_size)
            results, no_embedding = client.fetch_embeddings(missing_ids)
            db.set_embeddings(conn, [
                (r.arxiv_id, r.vector, r.s2_paper_id, r.citation_count) for r in results
            ])
            embedded = len(results)
            conn.commit()
        t_embed = time.perf_counter()
        logger.info(
            "S2 embed: {:.2f}s ({} embedded, {} unavailable)",
            t_embed - t_fetch, embedded, len(no_embedding),
        )

        # 3. Score + 4. persist digest.
        if mode == "hot":
            recs = recommend.recommend_hot(conn, top=top_k)
        elif mode == "hot_similar":
            recs = recommend.recommend_hot_similar(conn, top=top_k)
        else:
            recs = recommend.recommend(conn, top=top_k, min_score=min_score)
        params = {
            "categories": categories,
            "days_back": days_back,
            "top_k": top_k,
            "min_score": min_score,
        }
        db.save_digest(conn, params, recs)
        conn.commit()
        t_score = time.perf_counter()
        logger.info("Score + save digest: {:.2f}s ({} recs)", t_score - t_embed, len(recs))
        logger.info("Total: {:.2f}s", t_score - t_start)

        with_emb, total = db.embedding_coverage(conn)
        return {
            "fetched": len(fetched),
            "new": new_count,
            "embedded": embedded,
            "no_embedding": len(no_embedding),
            "recommended": len(recs),
            "recs": recs,
            "coverage": (with_emb, total),
            "timings": {
                "fetch": t_fetch - t_start,
                "embed": t_embed - t_fetch,
                "score": t_score - t_embed,
                "total": t_score - t_start,
            },
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly arXiv fetch + score + digest.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--days", type=int, default=None, help="Look-back window")
    parser.add_argument("--max", type=int, default=None, help="Max papers to fetch")
    parser.add_argument("--top", type=int, default=None, help="Digest size")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only the digest markdown.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--hot", action="store_true",
                            help="Rank by citation rate (citations / days since published).")
    mode_group.add_argument("--hot-similar", action="store_true",
                            help="50/50 blend of citation rate and cosine similarity.")
    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    cfg = load_config(args.config)
    fetch = cfg.section("fetch")
    s2 = cfg.section("s2")

    if args.hot:
        mode = "hot"
    elif args.hot_similar:
        mode = "hot_similar"
    else:
        mode = "default"

    summary = run_fetch(
        db_path=args.db or cfg.db_path,
        categories=fetch.get("categories", []),
        days_back=args.days or fetch.get("days_back", 7),
        max_fetch=args.max or fetch.get("max_fetch", 500),
        top_k=args.top or fetch.get("top_k", 15),
        min_score=fetch.get("min_score", 0.0),
        api_key=s2.get("api_key", ""),
        batch_size=s2.get("batch_size", 500),
        mode=mode,
    )

    if not args.quiet:
        with_emb, total = summary["coverage"]
        print(f"Fetched {summary['fetched']} papers ({summary['new']} new). "
              f"Embedded {summary['embedded']} (+{summary['no_embedding']} unavailable). "
              f"Coverage {with_emb}/{total}.\n")
    mode_label = {"hot": "hot", "hot_similar": "hot+similar"}.get(mode, "")
    header_parts = ["arXiv digest"]
    if mode_label:
        header_parts.append(mode_label)
    header_parts.append(date.today().isoformat())
    print(format_recommendations(
        summary["recs"],
        header=" · ".join(header_parts),
    ))


if __name__ == "__main__":
    main()
