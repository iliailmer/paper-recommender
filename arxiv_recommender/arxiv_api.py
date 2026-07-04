"""Query the arXiv API for recent submissions in target categories.

Unlike the BibTeX import, papers from here carry proper arXiv category *codes*
(e.g. cs.LG) rather than long-form Zotero keywords.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import arxiv
from loguru import logger

_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def _canonical_id(short_id: str) -> str:
    return _VERSION_RE.sub("", short_id.strip())


def search_recent(
    categories: list[str],
    days_back: int = 7,
    max_fetch: int = 500,
) -> list[dict]:
    """Fetch papers submitted in the last `days_back` days in `categories`.

    Results are sorted by submission date (newest first); iteration stops once
    a paper older than the cutoff is reached or `max_fetch` is hit.
    """
    if not categories:
        return []

    query = " OR ".join(f"cat:{c}" for c in categories)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_fetch,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[dict] = []
    for result in client.results(search):
        # Window check uses `updated` (the SubmittedDate sort key) so a recently
        # revised old paper doesn't trigger an early break on its old v1 date.
        if result.updated < cutoff:
            break  # sorted newest-first, so everything after is older too
        papers.append({
            "arxiv_id": _canonical_id(result.get_short_id()),
            "title": (result.title or "").strip(),
            "authors": [a.name for a in result.authors],
            "abstract": (result.summary or "").strip(),
            "categories": list(result.categories),
            "published_date": result.published.date().isoformat(),
        })
        if len(papers) >= max_fetch:
            break

    logger.info("arXiv returned {} papers in window", len(papers))
    return papers
