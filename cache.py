"""Shared file-based cache for expensive, deterministic build steps.

RAG concept: chunking (and later, eval-set generation) can be slow or cost
money to redo, but their output only depends on their inputs — so once
built, the result belongs on disk, not recomputed on every run.
"""

from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


def load_cached_or_build(
    path: Path,
    build_fn: Callable[[], T],
    loader: Callable[[Path], T],
    saver: Callable[[T, Path], None],
) -> T:
    """Return the cached result at `path`, building and saving it if missing."""
    if path.exists():
        return loader(path)

    result = build_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    saver(result, path)
    return result
