"""SQLite layer: schema, connection, and paper upserts."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EMBEDDING_DTYPE = np.float32

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
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Run schema migrations, handling columns that may already exist."""
    for col, typedef in [
        ("citation_count", "INTEGER"),
        ("citation_count_updated", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
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


# --- embeddings -----------------------------------------------------------

def serialize_embedding(vector: np.ndarray) -> bytes:
    """Pack a 1-D float32 array into raw bytes for BLOB storage."""
    return np.asarray(vector, dtype=EMBEDDING_DTYPE).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Unpack a BLOB back into a 1-D float32 array."""
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE)


def ids_missing_embeddings(conn: sqlite3.Connection, library_only: bool = True) -> list[str]:
    """arXiv IDs of papers that have no embedding yet."""
    sql = "SELECT arxiv_id FROM papers WHERE embedding IS NULL"
    if library_only:
        sql += " AND in_library = 1"
    return [row["arxiv_id"] for row in conn.execute(sql).fetchall()]


def set_embeddings(
    conn: sqlite3.Connection,
    results: list[tuple[str, np.ndarray, str | None, int | None]],
) -> None:
    """Bulk-store embeddings (and citation counts) for many papers in one round-trip.

    Each tuple is (arxiv_id, vector, s2_paper_id, citation_count).
    """
    if not results:
        return
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "UPDATE papers SET embedding = ?, s2_paper_id = ?, "
        "citation_count = COALESCE(?, citation_count), "
        "citation_count_updated = CASE WHEN ? IS NOT NULL THEN ? ELSE citation_count_updated END "
        "WHERE arxiv_id = ?",
        [
            (serialize_embedding(vector), s2_paper_id, citation_count, citation_count, now, arxiv_id)
            for arxiv_id, vector, s2_paper_id, citation_count in results
        ],
    )


def insert_fetched_papers(conn: sqlite3.Connection, papers: list[dict]) -> int:
    """Bulk-insert newly-fetched (non-library) papers in one round-trip.

    Uses ON CONFLICT DO NOTHING so it never overwrites an existing row —
    importantly, it won't downgrade a paper already marked in_library.
    Returns the number of rows actually inserted.
    """
    if not papers:
        return 0
    cur = conn.executemany(
        "INSERT INTO papers "
        "(arxiv_id, title, authors, abstract, categories, published_date, "
        " fetched_date, in_library) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 0) "
        "ON CONFLICT(arxiv_id) DO NOTHING",
        [
            (
                p["arxiv_id"],
                p["title"],
                p["authors"],
                p["abstract"],
                p["categories"],
                p["published_date"],
                p["fetched_date"],
            )
            for p in papers
        ],
    )
    return cur.rowcount


def save_digest(conn: sqlite3.Connection, params: dict, results: list[dict]) -> None:
    conn.execute(
        "INSERT INTO digests (created_date, params, results) VALUES (?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(params),
            json.dumps(results),
        ),
    )


def latest_digest(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        "SELECT created_date, params, results FROM digests ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {
        "created_date": row["created_date"],
        "params": json.loads(row["params"]),
        "results": json.loads(row["results"]),
    }


def embedding_coverage(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (papers_with_embedding, total_papers)."""
    with_emb = conn.execute(
        "SELECT COUNT(*) AS n FROM papers WHERE embedding IS NOT NULL"
    ).fetchone()["n"]
    total = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
    return with_emb, total
