"""Load and resolve config.toml.

Paths in config.toml are relative to the project root (the directory that
contains config.toml). This module resolves them to absolute paths so the
rest of the code never has to care about the current working directory.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# Project root = parent of this package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class Config:
    raw: dict
    config_path: Path
    db_path: Path
    bib_path: Path

    def section(self, name: str) -> dict:
        """Return a config section, or an empty dict if absent."""
        return self.raw.get(name, {})


def _resolve(path_str: str, root: Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (root / p)


def load_config(config_path: Path | str | None = None) -> Config:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    root = path.resolve().parent
    db_path = _resolve(raw["storage"]["db_path"], root)
    bib_path = _resolve(raw["profile"]["bib_path"], root)

    return Config(raw=raw, config_path=path, db_path=db_path, bib_path=bib_path)
