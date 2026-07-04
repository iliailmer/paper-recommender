from __future__ import annotations

import json

import numpy as np

from arxiv_recommender import db


class TestMigration:
    def test_init_db_is_idempotent(self, conn):
        # conn fixture already ran init_db once; running it again must not error.
        db.init_db(conn)
        db.init_db(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()]
        assert "citation_count" in cols
        assert "citation_count_updated" in cols


class TestUpsertPaper:
    def _base_paper(self, **overrides):
        paper = {
            "arxiv_id": "1000.00001",
            "title": "Original Title",
            "authors": json.dumps(["A. Author"]),
            "abstract": "Original abstract",
            "categories": json.dumps(["cs.LG"]),
            "published_date": "2020-01-01",
            "fetched_date": "2020-01-01T00:00:00+00:00",
            "in_library": 1,
            "collection": None,
            "date_added": None,
        }
        paper.update(overrides)
        return paper

    def test_insert_new_paper(self, conn):
        db.upsert_paper(conn, self._base_paper())
        conn.commit()
        row = conn.execute(
            "SELECT title FROM papers WHERE arxiv_id = ?", ("1000.00001",)
        ).fetchone()
        assert row["title"] == "Original Title"

    def test_reingest_updates_library_fields(self, conn):
        db.upsert_paper(conn, self._base_paper())
        conn.commit()
        db.upsert_paper(conn, self._base_paper(title="Updated Title"))
        conn.commit()
        row = conn.execute(
            "SELECT title FROM papers WHERE arxiv_id = ?", ("1000.00001",)
        ).fetchone()
        assert row["title"] == "Updated Title"

    def test_reingest_preserves_embedding_and_s2_id(self, conn):
        db.upsert_paper(conn, self._base_paper())
        conn.commit()
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        db.set_embeddings(conn, [("1000.00001", vec, "s2:123", 42)])
        conn.commit()

        # Re-ingest the same paper (e.g. Zotero re-export).
        db.upsert_paper(conn, self._base_paper(title="Retitled"))
        conn.commit()

        row = conn.execute(
            "SELECT title, embedding, s2_paper_id FROM papers WHERE arxiv_id = ?",
            ("1000.00001",),
        ).fetchone()
        assert row["title"] == "Retitled"
        assert row["s2_paper_id"] == "s2:123"
        np.testing.assert_array_equal(db.deserialize_embedding(row["embedding"]), vec)


class TestInsertFetchedPapers:
    def _paper(self, arxiv_id):
        return {
            "arxiv_id": arxiv_id,
            "title": "Title",
            "authors": json.dumps([]),
            "abstract": "Abstract",
            "categories": json.dumps([]),
            "published_date": "2026-01-01",
            "fetched_date": "2026-01-01T00:00:00+00:00",
        }

    def test_inserts_new_papers_and_returns_count(self, conn):
        count = db.insert_fetched_papers(conn, [self._paper("1000.00001"), self._paper("1000.00002")])
        conn.commit()
        assert count == 2
        total = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
        assert total == 2

    def test_empty_list_returns_zero(self, conn):
        assert db.insert_fetched_papers(conn, []) == 0

    def test_does_not_overwrite_existing_library_paper(self, conn):
        db.upsert_paper(conn, {
            "arxiv_id": "1000.00001",
            "title": "Library Title",
            "authors": json.dumps([]),
            "abstract": "",
            "categories": json.dumps([]),
            "published_date": "2020-01-01",
            "fetched_date": "2020-01-01T00:00:00+00:00",
            "in_library": 1,
            "collection": None,
            "date_added": None,
        })
        conn.commit()

        count = db.insert_fetched_papers(conn, [self._paper("1000.00001")])
        conn.commit()

        assert count == 0  # conflict -> DO NOTHING
        row = conn.execute(
            "SELECT title, in_library FROM papers WHERE arxiv_id = ?", ("1000.00001",)
        ).fetchone()
        assert row["title"] == "Library Title"
        assert row["in_library"] == 1


class TestSetEmbeddings:
    def _insert_bare_paper(self, conn, arxiv_id):
        conn.execute(
            "INSERT INTO papers (arxiv_id, in_library) VALUES (?, 0)", (arxiv_id,)
        )
        conn.commit()

    def test_bulk_sets_embedding_and_citation_count(self, conn):
        self._insert_bare_paper(conn, "1000.00001")
        self._insert_bare_paper(conn, "1000.00002")
        vec1 = np.array([1.0, 2.0], dtype=np.float32)
        vec2 = np.array([3.0, 4.0], dtype=np.float32)

        db.set_embeddings(conn, [
            ("1000.00001", vec1, "s2:1", 10),
            ("1000.00002", vec2, "s2:2", 20),
        ])
        conn.commit()

        row1 = conn.execute(
            "SELECT embedding, s2_paper_id, citation_count FROM papers WHERE arxiv_id = ?",
            ("1000.00001",),
        ).fetchone()
        np.testing.assert_array_equal(db.deserialize_embedding(row1["embedding"]), vec1)
        assert row1["s2_paper_id"] == "s2:1"
        assert row1["citation_count"] == 10

    def test_empty_list_is_a_noop(self, conn):
        db.set_embeddings(conn, [])  # must not raise

    def test_null_citation_count_preserves_existing_value(self, conn):
        self._insert_bare_paper(conn, "1000.00001")
        vec = np.array([1.0, 2.0], dtype=np.float32)
        db.set_embeddings(conn, [("1000.00001", vec, "s2:1", 10)])
        conn.commit()

        # Re-embed without a citation count (e.g. S2 omitted it this time).
        db.set_embeddings(conn, [("1000.00001", vec, "s2:1", None)])
        conn.commit()

        row = conn.execute(
            "SELECT citation_count FROM papers WHERE arxiv_id = ?", ("1000.00001",)
        ).fetchone()
        assert row["citation_count"] == 10


class TestEmbeddingCoverage:
    def test_counts_papers_with_and_without_embeddings(self, conn):
        conn.execute("INSERT INTO papers (arxiv_id, in_library) VALUES ('a.1', 0)")
        conn.execute("INSERT INTO papers (arxiv_id, in_library) VALUES ('a.2', 0)")
        conn.commit()
        vec = np.array([1.0], dtype=np.float32)
        db.set_embeddings(conn, [("a.1", vec, None, None)])
        conn.commit()

        with_emb, total = db.embedding_coverage(conn)
        assert with_emb == 1
        assert total == 2


class TestDigests:
    def test_save_and_load_latest_digest(self, conn):
        params = {"top_k": 5}
        results = [{"arxiv_id": "1000.00001", "score": 0.9}]
        db.save_digest(conn, params, results)
        conn.commit()

        digest = db.latest_digest(conn)
        assert digest["params"] == params
        assert digest["results"] == results

    def test_latest_digest_returns_none_when_empty(self, conn):
        assert db.latest_digest(conn) is None

    def test_latest_digest_returns_most_recent(self, conn):
        db.save_digest(conn, {"top_k": 1}, [])
        conn.commit()
        db.save_digest(conn, {"top_k": 2}, [])
        conn.commit()

        digest = db.latest_digest(conn)
        assert digest["params"] == {"top_k": 2}
