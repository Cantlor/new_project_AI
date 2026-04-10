"""Unit tests for ai_fields.common.progress.

Covers:
  - resolve_progress_enabled() priority logic
  - iter_progress() wrapping behavior
  - progress_bar() context manager behavior
  - _NullProgress stub correctness

These tests never require tqdm — they exercise the policy layer in isolation.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from ai_fields.common.progress import (
    _NullProgress,
    iter_progress,
    progress_bar,
    resolve_progress_enabled,
)

# ---------------------------------------------------------------------------
# resolve_progress_enabled
# ---------------------------------------------------------------------------


class TestResolveProgressEnabled:
    def test_explicit_true_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("AI_FIELDS_PROGRESS", "0")
        assert resolve_progress_enabled(True) is True

    def test_explicit_false_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("AI_FIELDS_PROGRESS", "1")
        assert resolve_progress_enabled(False) is False

    def test_env_true_values(self, monkeypatch):
        for val in ("1", "true", "yes", "on", "TRUE", "YES"):
            monkeypatch.setenv("AI_FIELDS_PROGRESS", val)
            assert resolve_progress_enabled(None) is True, f"expected True for {val!r}"

    def test_env_false_values(self, monkeypatch):
        for val in ("0", "false", "no", "off", "FALSE", "NO"):
            monkeypatch.setenv("AI_FIELDS_PROGRESS", val)
            assert resolve_progress_enabled(None) is False, f"expected False for {val!r}"

    def test_non_bool_non_none_returns_false(self, monkeypatch):
        monkeypatch.delenv("AI_FIELDS_PROGRESS", raising=False)
        # Any non-bool non-None value should defensively return False
        assert resolve_progress_enabled(0) is False  # type: ignore[arg-type]
        assert resolve_progress_enabled("yes") is False  # type: ignore[arg-type]

    def test_auto_tty_detection_non_tty_stream(self, monkeypatch):
        monkeypatch.delenv("AI_FIELDS_PROGRESS", raising=False)

        class _FakeStream:
            def isatty(self):
                return False

        assert resolve_progress_enabled(None, stream=_FakeStream()) is False

    def test_auto_tty_detection_tty_stream(self, monkeypatch):
        monkeypatch.delenv("AI_FIELDS_PROGRESS", raising=False)

        class _FakeTTY:
            def isatty(self):
                return True

        assert resolve_progress_enabled(None, stream=_FakeTTY()) is True

    def test_auto_no_isatty_attribute_returns_false(self, monkeypatch):
        monkeypatch.delenv("AI_FIELDS_PROGRESS", raising=False)

        class _NoIsatty:
            pass

        assert resolve_progress_enabled(None, stream=_NoIsatty()) is False

    def test_auto_isatty_raises_oserror_returns_false(self, monkeypatch):
        monkeypatch.delenv("AI_FIELDS_PROGRESS", raising=False)

        class _ErrorIsatty:
            def isatty(self):
                raise OSError("device unavailable")

        assert resolve_progress_enabled(None, stream=_ErrorIsatty()) is False


# ---------------------------------------------------------------------------
# _NullProgress stub
# ---------------------------------------------------------------------------


class TestNullProgress:
    def test_update_returns_none(self):
        null = _NullProgress(total=10)
        assert null.update(1) is None
        assert null.update() is None

    def test_set_postfix_returns_none(self):
        null = _NullProgress()
        assert null.set_postfix(loss=0.5) is None
        assert null.set_postfix("ignored") is None

    def test_close_returns_none(self):
        null = _NullProgress()
        assert null.close() is None

    def test_total_stored(self):
        null = _NullProgress(total=42)
        assert null.total == 42

    def test_no_total(self):
        null = _NullProgress()
        assert null.total is None


# ---------------------------------------------------------------------------
# iter_progress
# ---------------------------------------------------------------------------


class TestIterProgress:
    def test_disabled_returns_plain_iteration(self):
        items = [1, 2, 3]
        result = list(iter_progress(items, progress_enabled=False))
        assert result == items

    def test_disabled_generator_iteration(self):
        gen = (x * 2 for x in range(4))
        result = list(iter_progress(gen, progress_enabled=False, total=4))
        assert result == [0, 2, 4, 6]

    def test_enabled_without_tqdm_falls_back_gracefully(self, monkeypatch):
        # Simulate tqdm unavailable
        import ai_fields.common.progress as prog_mod
        monkeypatch.setattr(prog_mod, "_TQDM_AVAILABLE", False)
        items = [10, 20, 30]
        result = list(iter_progress(items, progress_enabled=True))
        assert result == items

    def test_full_iteration_completes(self):
        items = list(range(50))
        result = list(iter_progress(items, desc="test", unit="item", progress_enabled=False))
        assert result == items


# ---------------------------------------------------------------------------
# progress_bar context manager
# ---------------------------------------------------------------------------


class TestProgressBar:
    def test_disabled_yields_null_progress(self):
        with progress_bar(total=10, progress_enabled=False) as bar:
            assert isinstance(bar, _NullProgress)

    def test_disabled_null_progress_update_and_postfix(self):
        # Verify the null stub is safe to call in the standard pattern
        collected = []
        with progress_bar(total=3, progress_enabled=False) as bar:
            for i in range(3):
                bar.update(1)
                bar.set_postfix(i=i)
                collected.append(i)
        assert collected == [0, 1, 2]

    def test_enabled_without_tqdm_falls_back_to_null(self, monkeypatch):
        import ai_fields.common.progress as prog_mod
        monkeypatch.setattr(prog_mod, "_TQDM_AVAILABLE", False)
        with progress_bar(total=5, progress_enabled=True) as bar:
            assert isinstance(bar, _NullProgress)

    def test_context_manager_always_exits_cleanly(self):
        # Even if an exception is raised, bar must be closed
        with pytest.raises(ValueError):
            with progress_bar(total=10, progress_enabled=False) as bar:
                bar.update(1)
                raise ValueError("intentional")
