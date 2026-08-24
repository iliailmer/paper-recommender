from __future__ import annotations

import threading

from arxiv_recommender.s2 import S2Client


def test_fetch_embeddings_reports_progress_per_batch(monkeypatch):
    client = S2Client(api_key="", batch_size=1, polite_interval=0)
    monkeypatch.setattr(client, "_post_with_retry", lambda ids: [None for _ in ids])

    calls: list[tuple[int, int]] = []
    client.fetch_embeddings(
        ["a.1", "a.2"], progress_cb=lambda done, total: calls.append((done, total))
    )

    assert calls == [(1, 2), (2, 2)]


def test_fetch_embeddings_without_progress_cb_still_works(monkeypatch):
    client = S2Client(api_key="", batch_size=500)
    monkeypatch.setattr(client, "_post_with_retry", lambda ids: [None for _ in ids])

    results, missing = client.fetch_embeddings(["a.1"])
    assert results == []
    assert missing == ["a.1"]


def test_fetch_embeddings_stops_at_next_batch_when_cancelled(monkeypatch):
    client = S2Client(api_key="", batch_size=1, polite_interval=0)
    calls = []
    monkeypatch.setattr(
        client, "_post_with_retry", lambda ids: calls.append(ids) or [None for _ in ids]
    )

    cancel_event = threading.Event()

    def progress_cb(done, total):
        if done == 1:
            cancel_event.set()  # simulate a cancel request landing after batch 1

    client.fetch_embeddings(
        ["a.1", "a.2", "a.3"], progress_cb=progress_cb, cancel_event=cancel_event
    )

    assert len(calls) == 1  # batch 2/3 never started


def test_fetch_embeddings_already_cancelled_does_nothing(monkeypatch):
    client = S2Client(api_key="", batch_size=1, polite_interval=0)
    calls = []
    monkeypatch.setattr(
        client, "_post_with_retry", lambda ids: calls.append(ids) or [None for _ in ids]
    )

    cancel_event = threading.Event()
    cancel_event.set()

    results, missing = client.fetch_embeddings(["a.1"], cancel_event=cancel_event)

    assert calls == []
    assert results == []
    assert missing == []
