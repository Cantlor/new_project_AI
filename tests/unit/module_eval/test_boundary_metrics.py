"""Unit tests for module_eval Stage C minimal boundary metrics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.ndimage as ndi

from ai_fields.common.errors import ContractError
import ai_fields.module_eval.boundary_metrics as boundary_metrics_module
from ai_fields.module_eval.boundary_metrics import (
    BoundaryEvaluationPolicy,
    build_boundary_metrics_summary,
    compute_boundary_metrics,
)
from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("scipy")


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
    include_invalid_fp: bool = False,
    shifted_prediction: bool = False,
    class2_dominant_on_gt: bool = False,
) -> dict[str, np.ndarray]:
    h, w = 5, 6
    gt_extent = np.ones((1, h, w), dtype=np.uint8)
    gt_boundary = np.zeros((1, h, w), dtype=np.uint8)
    # Simple vertical boundary (skeleton class=1) at x=2
    gt_boundary[0, 1:4, 2] = 1

    gt_valid = np.ones((1, h, w), dtype=np.uint8)
    if include_invalid_fp:
        gt_valid[0, 4, 5] = 0

    pred_extent_prob = np.full((1, h, w), 0.7, dtype=np.float32)
    pred_distance_pred = np.full((1, h, w), 1.0, dtype=np.float32)
    pred_valid = gt_valid.copy()

    pred_boundary_prob = np.zeros((3, h, w), dtype=np.float32)
    pred_boundary_prob[0] = 0.9
    pred_boundary_prob[1] = 0.05
    pred_boundary_prob[2] = 0.05

    pred_x = 3 if shifted_prediction else 2
    if class2_dominant_on_gt:
        pred_boundary_prob[0, 1:4, pred_x] = 0.2
        pred_boundary_prob[1, 1:4, pred_x] = 0.1
        pred_boundary_prob[2, 1:4, pred_x] = 0.7  # non-background argmax true, skeleton-only false
    else:
        pred_boundary_prob[0, 1:4, pred_x] = 0.2
        pred_boundary_prob[1, 1:4, pred_x] = 0.7
        pred_boundary_prob[2, 1:4, pred_x] = 0.1

    if include_invalid_fp:
        pred_boundary_prob[0, 4, 5] = 0.1
        pred_boundary_prob[1, 4, 5] = 0.8
        pred_boundary_prob[2, 4, 5] = 0.1

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


def _default_policy() -> BoundaryEvaluationPolicy:
    return BoundaryEvaluationPolicy(
        prediction_interpretation="argmax_non_background",
        gt_interpretation="non_background",
        threshold_provenance="eval_boundary_baseline_v1",
        bde_enabled=True,
    )


def test_compute_boundary_metrics_happy_path(tmp_path: Path) -> None:
    contract = _resolve_contract(tmp_path, _build_eval_arrays())
    result = compute_boundary_metrics(input_contract=contract, policy=_default_policy())

    assert result.boundary_precision == pytest.approx(1.0)
    assert result.boundary_recall == pytest.approx(1.0)
    assert result.boundary_f1 == pytest.approx(1.0)
    assert result.boundary_bde == pytest.approx(0.0)
    assert result.boundary_bde_units == "meters"

    summary = build_boundary_metrics_summary(result)
    assert summary["metrics"]["boundary_f1"] == pytest.approx(1.0)
    assert summary["metrics"]["boundary_bde_units"] == "meters"


def test_compute_boundary_metrics_valid_aware(tmp_path: Path) -> None:
    contract_invalid = _resolve_contract(
        tmp_path / "invalid_excluded",
        _build_eval_arrays(include_invalid_fp=True),
    )
    contract_all_valid = _resolve_contract(
        tmp_path / "all_valid",
        _build_eval_arrays(include_invalid_fp=False),
    )

    policy = _default_policy()
    m_invalid = compute_boundary_metrics(input_contract=contract_invalid, policy=policy)
    m_all_valid = compute_boundary_metrics(input_contract=contract_all_valid, policy=policy)

    # Invalid-added FP must not affect metrics when valid mask excludes it.
    assert m_invalid.fp == m_all_valid.fp
    assert m_invalid.boundary_precision == pytest.approx(m_all_valid.boundary_precision)


def test_compute_boundary_metrics_policy_semantics_are_explicit(tmp_path: Path) -> None:
    contract = _resolve_contract(
        tmp_path,
        _build_eval_arrays(class2_dominant_on_gt=True),
    )

    non_background_policy = BoundaryEvaluationPolicy(
        prediction_interpretation="argmax_non_background",
        gt_interpretation="non_background",
        threshold_provenance="policy_non_background",
        bde_enabled=False,
    )
    skeleton_only_policy = BoundaryEvaluationPolicy(
        prediction_interpretation="argmax_skeleton_only",
        gt_interpretation="non_background",
        threshold_provenance="policy_skeleton_only",
        bde_enabled=False,
    )

    res_non_bg = compute_boundary_metrics(input_contract=contract, policy=non_background_policy)
    res_skel = compute_boundary_metrics(input_contract=contract, policy=skeleton_only_policy)

    assert res_non_bg.boundary_recall > res_skel.boundary_recall
    assert res_non_bg.prediction_interpretation == "argmax_non_background"
    assert res_skel.prediction_interpretation == "argmax_skeleton_only"


def test_compute_boundary_metrics_not_ready_failure(tmp_path: Path) -> None:
    contract = _resolve_contract(tmp_path, _build_eval_arrays())
    not_ready = replace(
        contract,
        track_readiness=replace(
            contract.track_readiness,
            boundary_ready=False,
            boundary_reason="synthetic_boundary_not_ready",
        ),
    )

    with pytest.raises(ContractError, match="boundary track is not ready"):
        compute_boundary_metrics(input_contract=not_ready, policy=_default_policy())


def test_compute_boundary_metrics_bde_shift_case(tmp_path: Path) -> None:
    contract = _resolve_contract(
        tmp_path,
        _build_eval_arrays(shifted_prediction=True),
    )
    result = compute_boundary_metrics(input_contract=contract, policy=_default_policy())

    # One-pixel horizontal shift with 1m pixel size in EPSG:32637 -> BDE ~= 1 meter.
    assert result.boundary_bde is not None
    assert result.boundary_bde_units == "meters"
    assert result.boundary_bde == pytest.approx(1.0, abs=1e-6)


def test_compute_boundary_metrics_invalid_policy_failure(tmp_path: Path) -> None:
    contract = _resolve_contract(tmp_path, _build_eval_arrays())
    bad_policy = BoundaryEvaluationPolicy(
        prediction_interpretation="argmax_non_background",
        gt_interpretation="non_background",
        threshold_provenance="x",
        non_background_prob_threshold=2.0,
    )
    with pytest.raises(ContractError, match="non_background_prob_threshold must be in \\[0,1\\]"):
        compute_boundary_metrics(input_contract=contract, policy=bad_policy)


def test_compute_boundary_metrics_empty_boundaries_zero_if_both_empty(tmp_path: Path) -> None:
    arrays = _build_eval_arrays()
    arrays["gt_boundary"][:] = 0
    arrays["pred_boundary_prob"][:] = 0.0
    arrays["pred_boundary_prob"][0] = 1.0
    contract = _resolve_contract(tmp_path, arrays)

    policy = BoundaryEvaluationPolicy(
        prediction_interpretation="argmax_non_background",
        gt_interpretation="non_background",
        threshold_provenance="eval_boundary_baseline_v1",
        bde_enabled=True,
        empty_boundary_handling="zero_if_both_empty",
    )
    result = compute_boundary_metrics(input_contract=contract, policy=policy)

    assert result.boundary_bde == pytest.approx(0.0)
    assert result.boundary_bde_units == "meters"
    assert result.gt_boundary_positive_pixels == 0
    assert result.pred_boundary_positive_pixels == 0


def test_compute_boundary_metrics_empty_boundaries_explicit_error(tmp_path: Path) -> None:
    arrays = _build_eval_arrays()
    arrays["gt_boundary"][:] = 0
    arrays["pred_boundary_prob"][:] = 0.0
    arrays["pred_boundary_prob"][0] = 1.0
    contract = _resolve_contract(tmp_path, arrays)

    with pytest.raises(ContractError, match="BDE cannot be computed with empty boundary sets"):
        compute_boundary_metrics(input_contract=contract, policy=_default_policy())


def test_compute_boundary_metrics_windowed_matches_full_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _build_eval_arrays(include_invalid_fp=True, class2_dominant_on_gt=True)
    contract = _resolve_contract(tmp_path, arrays)
    policy = BoundaryEvaluationPolicy(
        prediction_interpretation="argmax_non_background",
        gt_interpretation="non_background",
        threshold_provenance="eval_boundary_baseline_v1",
        bde_enabled=False,
    )
    monkeypatch.setattr(boundary_metrics_module, "_STREAM_WINDOW_SIZE", 2)

    result = compute_boundary_metrics(input_contract=contract, policy=policy)

    gt_boundary = arrays["gt_boundary"][0]
    gt_valid = arrays["gt_valid"][0]
    pred_boundary_prob = arrays["pred_boundary_prob"]
    eval_mask = gt_valid > 0

    gt_positive = (gt_boundary != 0) & eval_mask
    pred_positive = (pred_boundary_prob.argmax(axis=0) != 0) & eval_mask

    tp = int(np.logical_and(gt_positive, pred_positive).sum())
    fp = int(np.logical_and(~gt_positive & eval_mask, pred_positive).sum())
    fn = int(np.logical_and(gt_positive, ~pred_positive & eval_mask).sum())

    assert result.tp == tp
    assert result.fp == fp
    assert result.fn == fn


def test_bde_coordinate_matches_edt_reference(tmp_path: Path) -> None:
    contract = _resolve_contract(tmp_path, _build_eval_arrays(shifted_prediction=True))
    policy = _default_policy()

    with rasterio.open(contract.gt_boundary.path) as gt_ds, rasterio.open(
        contract.gt_valid.path
    ) as valid_ds, rasterio.open(contract.pred_boundary_prob.path) as pred_ds:
        gt_boundary = gt_ds.read(1)
        valid = valid_ds.read(1) > 0
        pred_prob = pred_ds.read((1, 2, 3)).astype(np.float32)

    gt_positive = (gt_boundary != 0) & valid
    pred_positive = (pred_prob.argmax(axis=0) != 0) & valid

    sampling, units_ref = boundary_metrics_module._resolve_bde_spacing_and_units(
        input_contract=contract,
        policy=policy,
    )
    pred_to_gt = float(ndi.distance_transform_edt(~gt_positive, sampling=sampling)[pred_positive].mean())
    gt_to_pred = float(ndi.distance_transform_edt(~pred_positive, sampling=sampling)[gt_positive].mean())
    bde_edt = 0.5 * (pred_to_gt + gt_to_pred)

    pred_coords = np.argwhere(pred_positive).astype(np.int32)
    gt_coords = np.argwhere(gt_positive).astype(np.int32)
    bde_coords, units_coords = boundary_metrics_module._compute_bde_from_coords(
        pred_coords_rc=pred_coords,
        gt_coords_rc=gt_coords,
        input_contract=contract,
        policy=policy,
    )

    assert units_coords == units_ref
    assert bde_coords == pytest.approx(bde_edt, abs=1e-9)
