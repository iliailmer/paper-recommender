"""Parse a Zotero BibTeX export and extract arXiv papers.

This export (standard Zotero, not Better BibTeX) carries the arXiv ID in
three places, in order of preference:
    url  = {http://arxiv.org/abs/2210.02747}
    doi  = {10.48550/arXiv.2210.02747}
    note = {arXiv:2210.02747}
There is no `eprint`/`archiveprefix` field, and no collection or date_added
field, so those paper columns are left empty by ingest.
"""

from __future__ import annotations

import re
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser

# New-style IDs: 2210.02747 (4-digit YYMM + 4-5 digit number), optional vN.
# Old-style IDs: hep-th/9901001 or math.NA/0123456.
_ID_NEW = r"\d{4}\.\d{4,5}"
_ID_OLD = r"[a-z][a-z.-]+/\d{7}"
_ID = rf"(?:{_ID_NEW}|{_ID_OLD})(?:v\d+)?"

_ABS_RE = re.compile(rf"arxiv\.org/abs/({_ID})", re.IGNORECASE)
_DOI_RE = re.compile(rf"10\.48550/arxiv\.({_ID})", re.IGNORECASE)
_NOTE_RE = re.compile(rf"arxiv:\s*({_ID})", re.IGNORECASE)

_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _canonical_id(raw: str) -> str:
    """Strip a trailing version suffix; lowercase old-style prefixes."""
    return _VERSION_RE.sub("", raw.strip())


def extract_arxiv_id(entry: dict) -> str | None:
    """Pull a canonical arXiv ID from a bib entry's url/doi/note fields."""
    for field, pattern in (
        ("url", _ABS_RE),
        ("doi", _DOI_RE),
        ("note", _NOTE_RE),
    ):
        value = entry.get(field)
        if value and (m := pattern.search(value)):
            return _canonical_id(m.group(1))
    return None


def _clean(text: str | None) -> str | None:
    """Strip BibTeX capitalization braces and collapse whitespace."""
    if text is None:
        return None
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _parse_authors(raw: str | None) -> list[str]:
    """Split a BibTeX author string ('Last, First and ...') into names."""
    if not raw:
        return []
    cleaned = _clean(raw) or ""
    return [name.strip() for name in cleaned.split(" and ") if name.strip()]


def _published_date(entry: dict) -> str | None:
    """Build a YYYY or YYYY-MM date string from year/month fields."""
    year = (entry.get("year") or "").strip()
    if not year:
        return None
    month_raw = (entry.get("month") or "").strip().lower()
    month = None
    if month_raw[:3] in _MONTHS:
        month = _MONTHS[month_raw[:3]]
    elif month_raw.isdigit():
        month = int(month_raw)
    return f"{year}-{month:02d}" if month else year


def _entry_to_paper(entry: dict) -> dict | None:
    """Map a bib entry to a paper dict, or None if it has no arXiv ID."""
    arxiv_id = extract_arxiv_id(entry)
    if arxiv_id is None:
        return None
    keywords = entry.get("keywords") or ""
    categories = [k.strip() for k in _clean(keywords).split(",")] if keywords else []
    return {
        "arxiv_id": arxiv_id,
        "title": _clean(entry.get("title")),
        "authors": _parse_authors(entry.get("author")),
        "abstract": _clean(entry.get("abstract")),
        "categories": categories,
        "published_date": _published_date(entry),
    }


def parse_library(bib_path: Path | str) -> tuple[list[dict], list[str]]:
    """Parse a .bib file into paper dicts.

    Returns (papers, skipped_keys) where skipped_keys are the citation keys
    of entries that had no recoverable arXiv ID.
    """
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    with open(bib_path, encoding="utf-8") as f:
        db = bibtexparser.load(f, parser=parser)

    papers: list[dict] = []
    skipped: list[str] = []
    for entry in db.entries:
        # BibTeX field names are case-insensitive; normalize to lowercase.
        entry = {k.lower(): v for k, v in entry.items()}
        paper = _entry_to_paper(entry)
        if paper is None:
            skipped.append(entry.get("id", "<unknown>"))
        else:
            papers.append(paper)
    return papers, skipped
