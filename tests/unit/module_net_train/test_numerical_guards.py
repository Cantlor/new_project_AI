"""Focused numerical stability guards for module_net_train runtime path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ai_fields.common.errors import ContractError
from ai_fields.module_net_train.run_train import _build_runtime_input_normalizer
from ai_fields.module_net_train.trainer import _prepare_batch


def _write_norm_stats(path: Path, *, hi: float = 1000.0) -> Path:
    payload = {
        "band_stats": [{"band_idx": idx, "p_lo": 0.0, "p_hi": float(hi)} for idx in range(8)],
        "clip_percentiles": [0.5, 99.5],
        "n_valid_pixels": 1,
        "computed_on": "unit_test",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_runtime_normalizer_produces_finite_scaled_tensor(tmp_path: Path) -> None:
    stats_path = _write_norm_stats(tmp_path / "norm_stats.json", hi=1200.0)
    normalizer = _build_runtime_input_normalizer(
        normalization={
            "normalization_name": "per_band_robust_percentile",
            "stats_source": str(stats_path),
            "clip_percentiles": [0.5, 99.5],
            "scaling_range": [0.0, 1.0],
        },
        feature_mode="raw8",
    )

    image = torch.full((2, 9, 32, 32), fill_value=65536.0, dtype=torch.float32)
    image[:, 8] = 0.0  # valid channel stays unchanged
    out = normalizer(image)
    assert torch.isfinite(out).all()
    assert float(out[:, :8].min()) >= 0.0
    assert float(out[:, :8].max()) <= 1.0
    assert float(out[:, 8].max()) == 0.0


def test_runtime_normalizer_rejects_degenerate_stats(tmp_path: Path) -> None:
    stats_path = _write_norm_stats(tmp_path / "norm_stats_bad.json", hi=0.0)
    with pytest.raises(ContractError, match="p_lo < p_hi"):
        _build_runtime_input_normalizer(
            normalization={
                "normalization_name": "per_band_robust_percentile",
                "stats_source": str(stats_path),
                "clip_percentiles": [0.5, 99.5],
                "scaling_range": [0.0, 1.0],
            },
            feature_mode="raw8",
        )


def test_prepare_batch_fails_fast_on_nonfinite_after_normalization() -> None:
    batch = {
        "image": torch.ones((1, 9, 8, 8), dtype=torch.float32),
        "extent": torch.zeros((1, 8, 8), dtype=torch.int64),
        "boundary": torch.zeros((1, 8, 8), dtype=torch.int64),
        "distance": torch.zeros((1, 8, 8), dtype=torch.float32),
        "valid": torch.ones((1, 8, 8), dtype=torch.bool),
    }

    def _bad_normalizer(x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y[:, 0, 0, 0] = torch.nan
        return y

    with pytest.raises(ContractError, match="after normalization"):
        _prepare_batch(
            batch,
            device=torch.device("cpu"),
            input_normalizer=_bad_normalizer,
        )
