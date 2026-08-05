# Web UI — Design Spec

## Summary

Add a server-rendered HTML page to `arxiv-serve` for browsing live recommendations, replacing the need to read raw JSON or `/digest.md` in a terminal. Local-network use only — no auth, no write actions (read-only browsing).

## Invocation

No new command. Visiting `arxiv-serve`'s root in a browser is the new entry point:

```bash
uv run arxiv-serve
# open http://127.0.0.1:8000/
```

## Route (`server.py`)

New `GET /` route added to `create_app`, alongside the existing JSON/markdown endpoints (`/status`, `/recommend`, `/digest`, `/digest.md`, `/similar/{id}`), which are unchanged.

`GET /` accepts the same query params as `/recommend`:

- `top` (default 15)
- `min_score` (default 0.0)
- `cat` (repeatable, arXiv category codes)
- `days` (optional, look-back window)

On load it calls `recommend.recommend()` directly (same function `/recommend` uses) and renders the results into an HTML template. Submitting the filter form re-issues a `GET /` with the new query string — full page reload, no JS.

## Templates

New `arxiv_recommender/templates/` directory, wired up via FastAPI's `Jinja2Templates`.

- `index.html` — filter form (top/min_score/cat/days) + results table
- Table columns: title (links to `arxiv.org/abs/{arxiv_id}`), authors (joined), score, published date, category tags
- No pagination, no client-side JS — `top` param controls result count
- Empty state: "No papers matched — try widening days or lowering min_score" when results are empty

## Static assets

A single small `style.css` (no framework, no CDN) served via FastAPI's `StaticFiles`, mounted at `/static`. Keeps the page readable without pulling in Tailwind/Bootstrap.

## Dependencies

- `jinja2` added to `pyproject.toml` main dependencies (FastAPI's `Jinja2Templates` requires it; not currently pulled in transitively as a hard dependency)
- No other new dependencies

## Error Handling

Same failure modes as `/recommend` today (e.g. no centroid yet because the library hasn't been ingested/embedded) — render the empty state rather than a 500, with a message pointing at `arxiv-ingest` / `arxiv-embed`.

## Testing

- New test in `tests/` (alongside existing `test_db.py`/`test_recommend.py` patterns) hitting `GET /` via FastAPI's `TestClient`, asserting 200 and that known paper titles from a seeded DB appear in the response body
- Existing JSON endpoint tests unchanged

## What's Not In Scope

- Mark-as-seen/dismiss actions, or any DB writes from the UI
- "Similar papers" browsing or fetch-on-demand triggers from the browser
- Auth, HTTPS, or remote/away-from-home access
- htmx or any partial-reload interactivity (clean upgrade path later if full-page reloads become annoying)
- Styling beyond plain readable CSS (no design system)
