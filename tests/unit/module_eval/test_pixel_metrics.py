"""Unit tests for module_eval Stage B minimal global/pixel metrics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.errors import ContractError
import ai_fields.module_eval.pixel_metrics as pixel_metrics_module
from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract
from ai_fields.module_eval.pixel_metrics import (
    PixelBinarizationPolicy,
    build_pixel_metrics_summary,
    compute_global_pixel_metrics,
)

rasterio = pytest.importorskip("rasterio")


def _write_raster(
    path: Path,
    *,
    array: np.ndarray,
    crs: str = "EPSG:32637",
    transform: Any | None = None,
) -> Path:
    from rasterio.transform import from_origin

    if transform is None:
        transform = from_origin(100.0, 200.0, 1.0, 1.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(array.dtype),
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(array)
    return path


def _build_eval_arrays(
    *,
    invalid_zero_at_bottom_right: bool = True,
    ignore_top_left: bool = True,
) -> dict[str, np.ndarray]:
    h, w = 4, 4

    gt_extent = np.zeros((1, h, w), dtype=np.uint8)
    gt_extent[0, 1, 1] = 1
    gt_extent[0, 1, 2] = 1
    gt_extent[0, 2, 1] = 1
    gt_extent[0, 2, 2] = 1
    if ignore_top_left:
        gt_extent[0, 0, 0] = 255

    gt_boundary = np.zeros((1, h, w), dtype=np.uint8)
    gt_boundary[0, 1:3, 1] = 1
    gt_boundary[0, 1:3, 2] = 2

    gt_valid = np.ones((1, h, w), dtype=np.uint8)
    if invalid_zero_at_bottom_right:
        gt_valid[0, 3, 3] = 0

    pred_extent_prob = np.full((1, h, w), 0.1, dtype=np.float32)
    pred_extent_prob[0, 1, 1] = 0.9   # TP
    pred_extent_prob[0, 1, 2] = 0.9   # TP
    pred_extent_prob[0, 2, 1] = 0.4   # FN for thr=0.5
    pred_extent_prob[0, 2, 2] = 0.9   # TP
    pred_extent_prob[0, 0, 1] = 0.8   # FP
    pred_extent_prob[0, 0, 0] = 0.9   # ignore-area positive
    pred_extent_prob[0, 3, 3] = 0.9   # invalid-area positive

    pred_boundary_prob = np.zeros((3, h, w), dtype=np.float32)
    pred_boundary_prob[0] = 0.8
    pred_boundary_prob[1] = 0.1
    pred_boundary_prob[2] = 0.1

    pred_distance_pred = np.full((1, h, w), 1.0, dtype=np.float32)

    pred_valid = gt_valid.copy()

    return {
        "gt_extent": gt_extent,
        "gt_boundary": gt_boundary,
        "gt_valid": gt_valid,
        "pred_extent_prob": pred_extent_prob,
        "pred_boundary_prob": pred_boundary_prob,
        "pred_distance_pred": pred_distance_pred,
        "pred_valid": pred_valid,
    }


def _resolve_contract(tmp_path: Path, arrays: dict[str, np.ndarray]):
    paths = {
        "gt_extent_path": _write_raster(tmp_path / "gt_extent.tif", array=arrays["gt_extent"]),
        "gt_boundary_path": _write_raster(tmp_path / "gt_boundary.tif", array=arrays["gt_boundary"]),
        "gt_valid_path": _write_raster(tmp_path / "gt_valid.tif", array=arrays["gt_valid"]),
        "pred_extent_prob_path": _write_raster(
            tmp_path / "extent_prob.tif",
            array=arrays["pred_extent_prob"],
        ),
        "pred_boundary_prob_path": _write_raster(
            tmp_path / "boundary_prob.tif",
            array=arrays["pred_boundary_prob"],
        ),
        "pred_distance_pred_path": _write_raster(
            tmp_path / "distance_pred.tif",
            array=arrays["pred_distance_pred"],
        ),
        "pred_valid_path": _write_raster(tmp_path / "valid.tif", array=arrays["pred_valid"]),
    }
    return resolve_evaluation_input_contract(**paths)


def test_compute_global_pixel_metrics_happy_path(tmp_path: Path) -> None:
    contract = _resolve_contract(tmp_path, _build_eval_arrays())
    policy = PixelBinarizationPolicy(
        extent_prob_threshold=0.5,
        threshold_provenance="eval_baseline_threshold_v1",
    )

    result = compute_global_pixel_metrics(input_contract=contract, policy=policy)

    assert result.tp == 3
    assert result.fp == 1
    assert result.fn == 1
    assert result.tn == 9

    assert result.precision == pytest.approx(0.75)
    assert result.recall == pytest.approx(0.75)
    assert result.f1 == pytest.approx(0.75)
    assert result.iou == pytest.approx(0.6)
    assert result.stage_scope == "stage_b_global_pixel_metrics_only"

    summary = build_pixel_metrics_summary(result)
    assert summary["metrics"]["extent_iou"] == pytest.approx(0.6)
    assert summary["threshold_policy"]["extent_prob_threshold"] == pytest.approx(0.5)


def test_compute_global_pixel_metrics_valid_aware(tmp_path: Path) -> None:
    contract_invalid = _resolve_contract(
        tmp_path / "invalid_excluded",
        _build_eval_arrays(invalid_zero_at_bottom_right=True),
    )
    contract_all_valid = _resolve_contract(
        tmp_path / "all_valid",
        _build_eval_arrays(invalid_zero_at_bottom_right=False),
    )
    policy = PixelBinarizationPolicy(
        extent_prob_threshold=0.5,
        threshold_provenance="eval_baseline_threshold_v1",
    )

    metrics_invalid = compute_global_pixel_metrics(
        input_contract=contract_invalid,
        policy=policy,
    )
    metrics_all_valid = compute_global_pixel_metrics(
        input_contract=contract_all_valid,
        policy=policy,
    )

    # Bottom-right positive prediction contributes FP only when valid=1.
    assert metrics_invalid.fp + 1 == metrics_all_valid.fp
    assert metrics_invalid.precision > metrics_all_valid.precision


def test_compute_global_pixel_metrics_ignore_aware(tmp_path: Path) -> None:
    contract_with_ignore = _resolve_contract(
        tmp_path / "with_ignore",
        _build_eval_arrays(ignore_top_left=True),
    )
    contract_without_ignore = _resolve_contract(
        tmp_path / "without_ignore",
        _build_eval_arrays(ignore_top_left=False),
    )
    policy = PixelBinarizationPolicy(
        extent_prob_threshold=0.5,
        threshold_provenance="eval_baseline_threshold_v1",
    )

    m_ignore = compute_global_pixel_metrics(input_contract=contract_with_ignore, policy=policy)
    m_no_ignore = compute_global_pixel_metrics(input_contract=contract_without_ignore, policy=policy)

    # Top-left positive prediction contributes FP only when ignore=0.
    assert m_ignore.fp + 1 == m_no_ignore.fp
    assert m_ignore.ignored_pixels == 1
    assert m_no_ignore.ignored_pixels == 0


def test_compute_global_pixel_metrics_threshold_semantics(tmp_path: Path) -> None:
    contract = _resolve_contract(tmp_path, _build_eval_arrays())

    low_thr = PixelBinarizationPolicy(
        extent_prob_threshold=0.5,
        threshold_provenance="explicit_policy_v1",
    )
    high_thr = PixelBinarizationPolicy(
        extent_prob_threshold=0.95,
        threshold_provenance="explicit_policy_v1",
    )

    m_low = compute_global_pixel_metrics(input_contract=contract, policy=low_thr)
    m_high = compute_global_pixel_metrics(input_contract=contract, policy=high_thr)

    assert m_low.recall > m_high.recall
    assert m_low.iou > m_high.iou
    assert m_low.threshold == pytest.approx(0.5)
    assert m_high.threshold == pytest.approx(0.95)


def test_compute_global_pixel_metrics_fails_when_pixel_track_not_ready(tmp_path: Path) -> None:
    contract = _resolve_contract(tmp_path, _build_eval_arrays())
    not_ready_contract = replace(
        contract,
        track_readiness=replace(
            contract.track_readiness,
            pixel_ready=False,
            pixel_reason="synthetic_not_ready_for_test",
        ),
    )
    policy = PixelBinarizationPolicy(
        extent_prob_threshold=0.5,
        threshold_provenance="eval_baseline_threshold_v1",
    )

    with pytest.raises(ContractError, match="pixel track is not ready"):
        compute_global_pixel_metrics(input_contract=not_ready_contract, policy=policy)


def test_compute_global_pixel_metrics_fails_on_invalid_threshold_policy(tmp_path: Path) -> None:
    contract = _resolve_contract(tmp_path, _build_eval_arrays())
    bad_policy = PixelBinarizationPolicy(
        extent_prob_threshold=1.5,
        threshold_provenance="eval_baseline_threshold_v1",
    )

    with pytest.raises(ContractError, match="extent_prob_threshold must be in \\[0,1\\]"):
        compute_global_pixel_metrics(input_contract=contract, policy=bad_policy)


def test_compute_global_pixel_metrics_windowed_matches_full_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _build_eval_arrays()
    contract = _resolve_contract(tmp_path, arrays)
    policy = PixelBinarizationPolicy(
        extent_prob_threshold=0.5,
        threshold_provenance="eval_baseline_threshold_v1",
    )
    monkeypatch.setattr(pixel_metrics_module, "_STREAM_WINDOW_SIZE", 2)

    result = compute_global_pixel_metrics(input_contract=contract, policy=policy)

    gt_extent = arrays["gt_extent"][0]
    gt_valid = arrays["gt_valid"][0]
    pred_extent = arrays["pred_extent_prob"][0]
    valid01 = gt_valid > 0
    eval_mask = valid01 & (gt_extent != policy.ignore_gt_label)
    gt_positive = (gt_extent == policy.positive_gt_label) & eval_mask
    pred_positive = (pred_extent >= policy.extent_prob_threshold) & eval_mask

    tp = int(np.logical_and(gt_positive, pred_positive).sum())
    fp = int(np.logical_and(~gt_positive & eval_mask, pred_positive).sum())
    fn = int(np.logical_and(gt_positive, ~pred_positive & eval_mask).sum())
    tn = int(np.logical_and(~gt_positive & eval_mask, ~pred_positive & eval_mask).sum())

    assert result.tp == tp
    assert result.fp == fp
    assert result.fn == fn
    assert result.tn == tn
