"""Configuration: data and model locations (configs/paths.yaml) and method settings (configs/cadre.yaml)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """configs/<name>.yaml; the path file can be replaced with the environment variable CADRE_PATHS."""
    file = Path(os.environ["CADRE_PATHS"]) if name == "paths" and os.environ.get("CADRE_PATHS") else ROOT / "configs" / f"{name}.yaml"
    return yaml.safe_load(file.read_text())


def path(key: str) -> Path:
    """A location from configs/paths.yaml; relative paths are resolved against the repository root."""
    value = Path(str(load("paths")[key])).expanduser()
    return value if value.is_absolute() else ROOT / value


def work(*parts: str) -> Path:
    """A file or directory under the working directory, created on demand."""
    out = path("work").joinpath(*parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def settings() -> dict[str, Any]:
    return load("cadre")
