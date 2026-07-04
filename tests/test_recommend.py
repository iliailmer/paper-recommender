from __future__ import annotations

import numpy as np

from arxiv_recommender import recommend
from tests.conftest import insert_paper


class TestBuildCentroid:
    def test_no_library_embeddings_returns_none(self, conn):
        centroid, n = recommend.build_centroid(conn)
        assert centroid is None
        assert n == 0

    def test_single_library_paper(self, conn):
        vec = np.array([3.0, 4.0], dtype=np.float32)  # norm 5
        insert_paper(conn, "1000.00001", in_library=True, embedding=vec)
        centroid, n = recommend.build_centroid(conn)
        assert n == 1
        np.testing.assert_allclose(centroid, [0.6, 0.8], atol=1e-6)

    def test_ignores_papers_without_embedding(self, conn):
        insert_paper(conn, "1000.00001", in_library=True, embedding=None)
        centroid, n = recommend.build_centroid(conn)
        assert centroid is None
        assert n == 0

    def test_ignores_non_library_papers(self, conn):
        vec = np.array([1.0, 0.0], dtype=np.float32)
        insert_paper(conn, "1000.00001", in_library=False, embedding=vec)
        centroid, n = recommend.build_centroid(conn)
        assert centroid is None
        assert n == 0


class TestRecommend:
    def test_ranks_by_cosine_similarity(self, conn):
        insert_paper(conn, "lib.1", in_library=True, embedding=np.array([1.0, 0.0], dtype=np.float32))
        # Same direction as library -> similarity 1.0
        insert_paper(
            conn, "close.1", embedding=np.array([2.0, 0.0], dtype=np.float32),
            published_date="2026-01-01", title="Close Paper",
        )
        # Orthogonal -> similarity 0.0
        insert_paper(
            conn, "far.1", embedding=np.array([0.0, 1.0], dtype=np.float32),
            published_date="2026-01-01", title="Far Paper",
        )

        results = recommend.recommend(conn, top=10, min_score=-1.0)

        assert [r["arxiv_id"] for r in results] == ["close.1", "far.1"]
        assert results[0]["score"] == 1.0
        assert results[1]["score"] == 0.0

    def test_excludes_library_papers(self, conn):
        vec = np.array([1.0, 0.0], dtype=np.float32)
        insert_paper(conn, "lib.1", in_library=True, embedding=vec)
        insert_paper(conn, "lib.2", in_library=True, embedding=vec, published_date="2026-01-01")

        results = recommend.recommend(conn, top=10, min_score=-1.0)
        assert results == []

    def test_min_score_filters_low_scores(self, conn):
        insert_paper(conn, "lib.1", in_library=True, embedding=np.array([1.0, 0.0], dtype=np.float32))
        insert_paper(
            conn, "far.1", embedding=np.array([0.0, 1.0], dtype=np.float32),
            published_date="2026-01-01",
        )

        results = recommend.recommend(conn, top=10, min_score=0.5)
        assert results == []

    def test_category_filter(self, conn):
        insert_paper(conn, "lib.1", in_library=True, embedding=np.array([1.0, 0.0], dtype=np.float32))
        insert_paper(
            conn, "cs.1", embedding=np.array([1.0, 0.0], dtype=np.float32),
            published_date="2026-01-01", categories='["cs.LG"]',
        )
        insert_paper(
            conn, "math.1", embedding=np.array([1.0, 0.0], dtype=np.float32),
            published_date="2026-01-01", categories='["math.NA"]',
        )

        results = recommend.recommend(conn, top=10, min_score=-1.0, categories=["cs.LG"])
        assert [r["arxiv_id"] for r in results] == ["cs.1"]

    def test_days_filter(self, conn):
        import datetime
        insert_paper(conn, "lib.1", in_library=True, embedding=np.array([1.0, 0.0], dtype=np.float32))
        today = datetime.date.today().isoformat()
        old_date = (datetime.date.today() - datetime.timedelta(days=100)).isoformat()
        insert_paper(
            conn, "recent.1", embedding=np.array([1.0, 0.0], dtype=np.float32), published_date=today,
        )
        insert_paper(
            conn, "old.1", embedding=np.array([1.0, 0.0], dtype=np.float32), published_date=old_date,
        )

        results = recommend.recommend(conn, top=10, min_score=-1.0, days=7)
        assert [r["arxiv_id"] for r in results] == ["recent.1"]


class TestRecommendHot:
    def test_ranks_by_citation_rate(self, conn):
        import datetime
        ten_days_ago = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
        insert_paper(conn, "high.1", citation_count=100, published_date=ten_days_ago)
        insert_paper(conn, "low.1", citation_count=10, published_date=ten_days_ago)

        results = recommend.recommend_hot(conn, top=10)

        assert [r["arxiv_id"] for r in results] == ["high.1", "low.1"]
        assert results[0]["score"] == 10.0  # 100 citations / 10 days
        assert results[1]["score"] == 1.0   # 10 citations / 10 days

    def test_excludes_library_papers(self, conn):
        insert_paper(conn, "lib.1", in_library=True, citation_count=1000, published_date="2026-01-01")
        results = recommend.recommend_hot(conn, top=10)
        assert results == []

    def test_excludes_papers_without_citation_count(self, conn):
        insert_paper(conn, "no_citations.1", citation_count=None, published_date="2026-01-01")
        results = recommend.recommend_hot(conn, top=10)
        assert results == []

    def test_age_clamped_to_at_least_one_day(self, conn):
        import datetime
        today = datetime.date.today().isoformat()
        insert_paper(conn, "brand_new.1", citation_count=5, published_date=today)
        results = recommend.recommend_hot(conn, top=10)
        assert results[0]["score"] == 5.0  # 5 citations / max(1, 0) days


class TestRecommendHotSimilar:
    def test_ranks_by_cosine_when_citation_rates_are_flat(self, conn):
        """Equal citation rates should not swamp the cosine signal (regression
        test for the flat-normalization bug where equal-or-zero citation
        counts collapsed the blended score)."""
        insert_paper(conn, "lib.1", in_library=True, embedding=np.array([1.0, 0.0], dtype=np.float32))
        insert_paper(
            conn, "close.1",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            citation_count=0, published_date="2026-01-01",
        )
        insert_paper(
            conn, "far.1",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            citation_count=0, published_date="2026-01-01",
        )

        results = recommend.recommend_hot_similar(conn, top=10)

        assert [r["arxiv_id"] for r in results] == ["close.1", "far.1"]

    def test_ranks_by_citation_rate_when_cosine_is_flat(self, conn):
        import datetime
        insert_paper(conn, "lib.1", in_library=True, embedding=np.array([1.0, 0.0], dtype=np.float32))
        ten_days_ago = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
        insert_paper(
            conn, "hot.1",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            citation_count=100, published_date=ten_days_ago,
        )
        insert_paper(
            conn, "cold.1",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            citation_count=1, published_date=ten_days_ago,
        )

        results = recommend.recommend_hot_similar(conn, top=10)

        assert [r["arxiv_id"] for r in results] == ["hot.1", "cold.1"]

    def test_excludes_papers_missing_embedding_or_citation_count(self, conn):
        insert_paper(conn, "lib.1", in_library=True, embedding=np.array([1.0, 0.0], dtype=np.float32))
        insert_paper(
            conn, "no_embedding.1", embedding=None, citation_count=5, published_date="2026-01-01",
        )
        insert_paper(
            conn, "no_citations.1",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            citation_count=None, published_date="2026-01-01",
        )

        results = recommend.recommend_hot_similar(conn, top=10)
        assert results == []

    def test_no_library_centroid_returns_empty(self, conn):
        insert_paper(
            conn, "some.1",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            citation_count=5, published_date="2026-01-01",
        )
        results = recommend.recommend_hot_similar(conn, top=10)
        assert results == []
