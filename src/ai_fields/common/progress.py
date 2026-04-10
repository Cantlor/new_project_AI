"""Shared progress-reporting helpers for ai_fields modules.

Goals:
- one consistent progress-control policy across modules;
- TTY-aware interactive bars for local runs;
- clean no-bar behavior for CI/non-interactive logs;
- explicit user control via env/arg without changing core logic.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    from tqdm.auto import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _tqdm = None  # type: ignore[assignment]
    _TQDM_AVAILABLE = False

_ENV_NAME = "AI_FIELDS_PROGRESS"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def resolve_progress_enabled(
    progress_enabled: bool | None = None,
    *,
    stream: Any | None = None,
) -> bool:
    """Resolve final progress-enabled flag.

    Priority:
    1) explicit function arg;
    2) env var AI_FIELDS_PROGRESS (0/1/true/false/...);
    3) auto mode: True only for interactive TTY streams.
    """
    if isinstance(progress_enabled, bool):
        return progress_enabled
    if progress_enabled is not None:
        # Defensive: unexpected non-bool values must not silently enable progress.
        return False

    raw = os.getenv(_ENV_NAME)
    if raw is not None:
        normalized = raw.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False

    target = stream if stream is not None else sys.stderr
    isatty = getattr(target, "isatty", None)
    if callable(isatty):
        try:
            return bool(isatty())
        except OSError:
            return False
    return False


def _resolve_total(iterable: Any, total: int | None) -> int | None:
    if total is not None:
        return int(total)
    try:
        return len(iterable)  # type: ignore[arg-type]
    except TypeError:
        return None


def iter_progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    progress_enabled: bool | None = None,
    leave: bool = False,
) -> Iterator[Any]:
    """Wrap iterable with tqdm when progress is enabled, else return plain iterator."""
    enabled = resolve_progress_enabled(progress_enabled)
    if enabled and _TQDM_AVAILABLE:
        resolved_total = _resolve_total(iterable, total=total)
        return _tqdm(
            iterable,
            total=resolved_total,
            desc=desc,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
        )
    return iter(iterable)


@dataclass
class _NullProgress:
    total: int | None = None

    def update(self, _: int = 1) -> None:
        return None

    def set_postfix(self, *_: Any, **__: Any) -> None:
        return None

    def close(self) -> None:
        return None


@contextmanager
def progress_bar(
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    progress_enabled: bool | None = None,
    leave: bool = False,
) -> Iterator[Any]:
    """Context manager returning tqdm progress bar or a no-op stub."""
    enabled = resolve_progress_enabled(progress_enabled)
    if enabled and _TQDM_AVAILABLE:
        bar = _tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
        )
        try:
            yield bar
        finally:
            bar.close()
        return

    yield _NullProgress(total=total)


__all__ = [
    "iter_progress",
    "progress_bar",
    "resolve_progress_enabled",
]

