# arXiv Recommender

Personal arXiv paper recommender. Give it your Zotero library, it fetches SPECTER2 embeddings from Semantic Scholar and ranks new arXiv papers by similarity to what you already read. No local models — just embeddings, cosine similarity, and SQLite.

## Quickstart

```bash
uv sync
cp /path/to/your/library.bib .    # Zotero BibTeX export
uv run arxiv-ingest                # load your library
uv run arxiv-embed                 # embed it
uv run arxiv-fetch --top 5         # get today's recommendations
```

## Commands

- `arxiv-ingest` — load `library.bib` into the DB
- `arxiv-embed` — embed library papers
- `arxiv-fetch [--hot | --hot-similar] [--top N]` — fetch new papers, score, print digest
- `arxiv-serve` — optional FastAPI server (`/recommend`, `/similar/{id}`, `/digest.md`)

Run any command with `--help` for flags. Settings live in `config.toml`.

## Note

Semantic Scholar indexes new arXiv papers with a ~1-2 week lag — very recent papers won't score until S2 catches up. Each `arxiv-fetch` run retries automatically.

## License

MIT
