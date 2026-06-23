"""Shared markdown formatter for recommendation lists."""

from __future__ import annotations


def format_recommendations(recs: list[dict], header: str | None = None) -> str:
    lines: list[str] = []
    if header:
        lines += [f"# {header}", ""]
    lines += [f"_{len(recs)} recommendations_", ""]
    for i, r in enumerate(recs, 1):
        authors = ", ".join(r["authors"][:3]) + ("…" if len(r["authors"]) > 3 else "")
        cats = ", ".join(r["categories"])
        lines += [
            f"## {i}. {r['title']}  ·  `{r['score']:.3f}`",
            f"- **arXiv:** [{r['arxiv_id']}](https://arxiv.org/abs/{r['arxiv_id']})"
            f"  ·  **{r['published_date']}**  ·  {cats}",
            f"- {authors}",
            "",
        ]
    return "\n".join(lines)
