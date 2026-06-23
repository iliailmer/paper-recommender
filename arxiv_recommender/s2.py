"""Semantic Scholar API client for fetching SPECTER2 embeddings.

Designed for the keyless free tier, which is heavily throttled: requests share
a global pool and return 429 with *no* Retry-After header, so we apply our own
exponential backoff with jitter. To minimize request count we send arXiv IDs in
batches (up to 500 per request) and sleep politely between batches.

An API key (config [s2] api_key) is optional; when present it is sent as the
`x-api-key` header and raises the rate limit, but the code path is identical.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

import numpy as np
import requests

logger = logging.getLogger(__name__)

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "paperId,embedding.specter_v2"
EMBEDDING_DIM = 768


@dataclass
class EmbeddingResult:
    arxiv_id: str
    vector: np.ndarray  # float32, shape (768,)
    s2_paper_id: str | None


class S2Client:
    def __init__(
        self,
        api_key: str = "",
        batch_size: int = 500,
        timeout: float = 30.0,
        max_retries: int = 8,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
        polite_interval: float = 1.1,
    ):
        self.batch_size = max(1, min(batch_size, 500))  # API caps at 500
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.polite_interval = polite_interval
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def _post_with_retry(self, ids: list[str]) -> list[dict | None]:
        """POST one batch, retrying on 429 / 5xx with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            resp = self.session.post(
                BATCH_URL,
                params={"fields": FIELDS},
                json={"ids": ids},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            # 400 "No valid paper ids given" means none of the IDs resolved;
            # treat every requested ID as missing rather than erroring.
            if resp.status_code == 400 and "no valid paper ids" in resp.text.lower():
                return [None] * len(ids)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = min(self.base_backoff * (2 ** attempt), self.max_backoff)
                    wait += random.uniform(0, wait * 0.25)  # jitter
                logger.warning(
                    "S2 %s (attempt %d/%d); backing off %.1fs",
                    resp.status_code, attempt + 1, self.max_retries, wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"S2 batch failed after {self.max_retries} retries")

    def fetch_embeddings(self, arxiv_ids: list[str]) -> tuple[list[EmbeddingResult], list[str]]:
        """Fetch SPECTER2 embeddings for arXiv IDs.

        Returns (results, missing) where `missing` are IDs S2 had no embedding
        for (unknown paper, or no embedding computed yet for a very new one).
        Result order matches the API's per-item alignment with the request.
        """
        results: list[EmbeddingResult] = []
        missing: list[str] = []

        batches = [
            arxiv_ids[i : i + self.batch_size]
            for i in range(0, len(arxiv_ids), self.batch_size)
        ]
        for n, chunk in enumerate(batches):
            if n > 0:
                time.sleep(self.polite_interval)
            ids = [f"ARXIV:{a}" for a in chunk]
            items = self._post_with_retry(ids)
            for arxiv_id, item in zip(chunk, items):
                vec = self._extract_vector(item)
                if vec is None:
                    missing.append(arxiv_id)
                else:
                    results.append(
                        EmbeddingResult(arxiv_id, vec, item.get("paperId"))
                    )
        return results, missing

    @staticmethod
    def _extract_vector(item: dict | None) -> np.ndarray | None:
        if not item:
            return None
        emb = item.get("embedding")
        if not emb or not emb.get("vector"):
            return None
        vec = np.asarray(emb["vector"], dtype=np.float32)
        if vec.shape != (EMBEDDING_DIM,):
            logger.warning("unexpected embedding dim %s; skipping", vec.shape)
            return None
        return vec
