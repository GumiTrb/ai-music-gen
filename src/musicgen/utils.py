"""Общие утилиты: конфиги и пути."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def resolve(path: str | Path) -> Path:
    """Относительные пути считаются от корня репозитория."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
