"""SQLite layer: schema, connection, and paper upserts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id        TEXT PRIMARY KEY,
    title           TEXT,
    authors         TEXT,          -- JSON array
    abstract        TEXT,
    categories      TEXT,          -- JSON array
    published_date  TEXT,
    fetched_date    TEXT,
    in_library      INTEGER DEFAULT 0,
    collection      TEXT,          -- Zotero collection name if in_library
    date_added      TEXT,          -- Zotero date_added if in_library
    embedding       BLOB,          -- float32 numpy array, 768-dim
    s2_paper_id     TEXT           -- Semantic Scholar ID for citation graph
);

CREATE TABLE IF NOT EXISTS digests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_date    TEXT,
    params          TEXT,          -- JSON: categories, days, top_k
    results         TEXT           -- JSON array of {arxiv_id, score}
);

CREATE INDEX IF NOT EXISTS idx_papers_in_library ON papers(in_library);
"""

# Columns that ingest owns and may overwrite on re-ingest. Notably absent:
# embedding and s2_paper_id, which are populated by later pipeline stages and
# must survive a re-ingest of the library.
_LIBRARY_COLUMNS = (
    "title",
    "authors",
    "abstract",
    "categories",
    "published_date",
    "fetched_date",
    "in_library",
    "collection",
    "date_added",
)


def connect(db_path: Path | str) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_paper(conn: sqlite3.Connection, paper: dict) -> None:
    """Insert a paper, or update its library-owned fields on conflict.

    Preserves embedding and s2_paper_id across re-ingests.
    """
    cols = ["arxiv_id", *_LIBRARY_COLUMNS]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in _LIBRARY_COLUMNS)
    sql = (
        f"INSERT INTO papers ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(arxiv_id) DO UPDATE SET {updates}"
    )
    conn.execute(sql, [paper.get(c) for c in cols])


def library_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM papers WHERE in_library = 1"
    ).fetchone()
    return row["n"]
