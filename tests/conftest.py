from __future__ import annotations

import numpy as np
import pytest

from arxiv_recommender import db as db_module


@pytest.fixture
def conn(tmp_path):
    connection = db_module.connect(tmp_path / "test.db")
    db_module.init_db(connection)
    yield connection
    connection.close()


def insert_paper(
    connection,
    arxiv_id: str,
    *,
    in_library: bool = False,
    embedding: np.ndarray | None = None,
    citation_count: int | None = None,
    published_date: str | None = None,
    title: str = "",
    authors: str = "[]",
    categories: str = "[]",
) -> None:
    """Insert a row directly, bypassing the ingest/fetch pipelines, for test setup."""
    connection.execute(
        "INSERT INTO papers "
        "(arxiv_id, title, authors, categories, published_date, in_library, "
        " embedding, citation_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            arxiv_id,
            title,
            authors,
            categories,
            published_date,
            1 if in_library else 0,
            db_module.serialize_embedding(embedding) if embedding is not None else None,
            citation_count,
        ),
    )
    connection.commit()
