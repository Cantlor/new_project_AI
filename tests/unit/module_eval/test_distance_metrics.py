"""Unit tests for module_eval Stage B.5 distance auxiliary metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.errors import ContractError, ValidPolicyError
import ai_fields.module_eval.distance_metrics as distance_metrics_module
from ai_fields.module_eval.distance_metrics import (
    DistanceEvaluationPolicy,
    DistanceMetricsResult,
    build_distance_metrics_summary,
    compute_distance_metrics,
)

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")


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


def _write_polygons(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {"geometry": "Polygon", "properties": {"instance_id": "int"}}
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs="EPSG:32637") as dst:
        dst.write({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(101.0, 199.0), (104.0, 199.0), (104.0, 196.0), (101.0, 196.0), (101.0, 199.0)]],
            },
            "properties": {"instance_id": 1},
        })
    return path


def _make_contract(tmp_path: Path, *, include_gt_distance: bool):
    from ai_fields.common.constants import DATA_CONTRACT_VERSION
    from ai_fields.common.manifests import write_manifest
    from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract

    h, w = 6, 8
    gt_extent = np.zeros((1, h, w), dtype=np.uint8)
    gt_extent[0, 1:5, 1:7] = 1
    gt_boundary = np.zeros((1, h, w), dtype=np.uint8)
    gt_valid = np.ones((1, h, w), dtype=np.uint8)
    gt_valid[0, 0, 0] = 0
    gt_distance = np.full((1, h, w), 2.0, dtype=np.float32)
    pred_extent = np.full((1, h, w), 0.8, dtype=np.float32)
    pred_boundary = np.zeros((3, h, w), dtype=np.float32)
    pred_boundary[0] = 1.0
    pred_distance = np.full((1, h, w), 1.5, dtype=np.float32)
    pred_valid = gt_valid.copy()

    paths = {
        "gt_extent_path": _write_raster(tmp_path / "gt_extent.tif", array=gt_extent),
        "gt_boundary_path": _write_raster(tmp_path / "gt_boundary.tif", array=gt_boundary),
        "gt_valid_path": _write_raster(tmp_path / "gt_valid.tif", array=gt_valid),
        "pred_extent_prob_path": _write_raster(tmp_path / "extent_prob.tif", array=pred_extent),
        "pred_boundary_prob_path": _write_raster(tmp_path / "boundary_prob.tif", array=pred_boundary),
        "pred_distance_pred_path": _write_raster(tmp_path / "distance_pred.tif", array=pred_distance),
        "pred_valid_path": _write_raster(tmp_path / "valid.tif", array=pred_valid),
    }
    if include_gt_distance:
        paths["gt_distance_path"] = _write_raster(tmp_path / "gt_distance.tif", array=gt_distance)
    else:
        paths["gt_distance_path"] = None

    return resolve_evaluation_input_contract(**paths)


def _policy() -> DistanceEvaluationPolicy:
    return DistanceEvaluationPolicy(threshold_provenance="eval_distance_v1")


class TestDistanceMetricsHappyPath:
    def test_mae_and_rmse_computed_on_valid_pixels(self, tmp_path: Path) -> None:
        contract = _make_contract(tmp_path, include_gt_distance=True)
        result = compute_distance_metrics(input_contract=contract, policy=_policy())

        assert not result.skipped
        assert result.skip_reason is None
        assert math.isfinite(result.mae)
        assert math.isfinite(result.rmse)
        # pred=1.5, gt=2.0 → diff=-0.5 everywhere on valid pixels
        # MAE = 0.5, RMSE = 0.5
        assert result.mae == pytest.approx(0.5, abs=1e-5)
        assert result.rmse == pytest.approx(0.5, abs=1e-5)

    def test_zero_diff_gives_zero_metrics(self, tmp_path: Path) -> None:
        """Perfect prediction → MAE=RMSE=0."""
        from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract

        h, w = 4, 4
        arr = np.full((1, h, w), 1.5, dtype=np.float32)
        valid = np.ones((1, h, w), dtype=np.uint8)
        gt_boundary = np.zeros((1, h, w), dtype=np.uint8)
        pred_boundary = np.zeros((3, h, w), dtype=np.float32)
        pred_boundary[0] = 1.0
        gt_extent = np.zeros((1, h, w), dtype=np.uint8)
        pred_extent = np.zeros((1, h, w), dtype=np.float32)

        paths = {
            "gt_extent_path": _write_raster(tmp_path / "e.tif", array=gt_extent),
            "gt_boundary_path": _write_raster(tmp_path / "b.tif", array=gt_boundary),
            "gt_valid_path": _write_raster(tmp_path / "v.tif", array=valid),
            "gt_distance_path": _write_raster(tmp_path / "gd.tif", array=arr),
            "pred_extent_prob_path": _write_raster(tmp_path / "pe.tif", array=pred_extent),
            "pred_boundary_prob_path": _write_raster(tmp_path / "pb.tif", array=pred_boundary),
            "pred_distance_pred_path": _write_raster(tmp_path / "pd.tif", array=arr),
            "pred_valid_path": _write_raster(tmp_path / "pv.tif", array=valid),
        }
        contract = resolve_evaluation_input_contract(**paths)
        result = compute_distance_metrics(input_contract=contract, policy=_policy())

        assert result.mae == pytest.approx(0.0, abs=1e-9)
        assert result.rmse == pytest.approx(0.0, abs=1e-9)

    def test_valid_aware_excludes_invalid_pixels(self, tmp_path: Path) -> None:
        """Invalid pixels (valid==0) must be excluded from MAE/RMSE."""
        from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract

        h, w = 4, 4
        gt_dist = np.full((1, h, w), 0.0, dtype=np.float32)
        pred_dist = np.full((1, h, w), 0.0, dtype=np.float32)
        # Set large error in invalid region — should not affect result
        pred_dist[0, 0, 0] = 999.0
        valid = np.ones((1, h, w), dtype=np.uint8)
        valid[0, 0, 0] = 0  # mark that pixel invalid

        gt_boundary = np.zeros((1, h, w), dtype=np.uint8)
        pred_boundary = np.zeros((3, h, w), dtype=np.float32)
        pred_boundary[0] = 1.0
        gt_extent = np.zeros((1, h, w), dtype=np.uint8)
        pred_extent = np.zeros((1, h, w), dtype=np.float32)

        paths = {
            "gt_extent_path": _write_raster(tmp_path / "e.tif", array=gt_extent),
            "gt_boundary_path": _write_raster(tmp_path / "b.tif", array=gt_boundary),
            "gt_valid_path": _write_raster(tmp_path / "v.tif", array=valid),
            "gt_distance_path": _write_raster(tmp_path / "gd.tif", array=gt_dist),
            "pred_extent_prob_path": _write_raster(tmp_path / "pe.tif", array=pred_extent),
            "pred_boundary_prob_path": _write_raster(tmp_path / "pb.tif", array=pred_boundary),
            "pred_distance_pred_path": _write_raster(tmp_path / "pd.tif", array=pred_dist),
            "pred_valid_path": _write_raster(tmp_path / "pv.tif", array=valid),
        }
        contract = resolve_evaluation_input_contract(**paths)
        result = compute_distance_metrics(input_contract=contract, policy=_policy())

        assert result.mae == pytest.approx(0.0, abs=1e-9)
        assert result.rmse == pytest.approx(0.0, abs=1e-9)
        assert result.valid_pixels < h * w  # some pixels excluded


class TestDistanceMetricsSkipBehavior:
    def test_skipped_when_gt_distance_absent_and_policy_skip(self, tmp_path: Path) -> None:
        contract = _make_contract(tmp_path, include_gt_distance=False)
        policy = DistanceEvaluationPolicy(
            threshold_provenance="eval_v1",
            absent_gt_policy="skip",
        )
        result = compute_distance_metrics(input_contract=contract, policy=policy)

        assert result.skipped is True
        assert result.skip_reason is not None
        assert math.isnan(result.mae)
        assert math.isnan(result.rmse)

    def test_raises_when_gt_distance_absent_and_policy_error(self, tmp_path: Path) -> None:
        contract = _make_contract(tmp_path, include_gt_distance=False)
        policy = DistanceEvaluationPolicy(
            threshold_provenance="eval_v1",
            absent_gt_policy="error",
        )
        with pytest.raises(ContractError, match="gt_distance is absent"):
            compute_distance_metrics(input_contract=contract, policy=policy)


class TestDistanceMetricsPolicyValidation:
    def test_invalid_absent_gt_policy_raises(self, tmp_path: Path) -> None:
        contract = _make_contract(tmp_path, include_gt_distance=True)
        policy = DistanceEvaluationPolicy(
            threshold_provenance="eval_v1",
            absent_gt_policy="unknown_mode",  # type: ignore[arg-type]
        )
        with pytest.raises(ContractError, match="absent_gt_policy"):
            compute_distance_metrics(input_contract=contract, policy=policy)

    def test_empty_threshold_provenance_raises(self, tmp_path: Path) -> None:
        contract = _make_contract(tmp_path, include_gt_distance=True)
        policy = DistanceEvaluationPolicy(threshold_provenance="  ")
        with pytest.raises(ContractError, match="threshold_provenance"):
            compute_distance_metrics(input_contract=contract, policy=policy)


class TestBuildDistanceMetricsSummary:
    def test_summary_has_required_keys(self, tmp_path: Path) -> None:
        contract = _make_contract(tmp_path, include_gt_distance=True)
        result = compute_distance_metrics(input_contract=contract, policy=_policy())
        summary = build_distance_metrics_summary(result)

        assert "stage_scope" in summary
        assert "skipped" in summary
        assert "metrics" in summary
        assert "mae" in summary["metrics"]
        assert "rmse" in summary["metrics"]
        assert "counts" in summary

    def test_skipped_summary_has_none_metrics(self, tmp_path: Path) -> None:
        contract = _make_contract(tmp_path, include_gt_distance=False)
        policy = DistanceEvaluationPolicy(threshold_provenance="v1", absent_gt_policy="skip")
        result = compute_distance_metrics(input_contract=contract, policy=policy)
        summary = build_distance_metrics_summary(result)

        assert summary["skipped"] is True
        assert summary["metrics"]["mae"] is None
        assert summary["metrics"]["rmse"] is None


def test_distance_metrics_windowed_matches_full_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _make_contract(tmp_path, include_gt_distance=True)
    monkeypatch.setattr(distance_metrics_module, "_STREAM_WINDOW_SIZE", 2)
    result = compute_distance_metrics(
        input_contract=contract,
        policy=DistanceEvaluationPolicy(threshold_provenance="eval_distance_v1"),
    )

    with rasterio.open(contract.gt_distance.path) as gt_ds, rasterio.open(
        contract.pred_distance_pred.path
    ) as pred_ds, rasterio.open(contract.gt_valid.path) as valid_ds:
        gt = gt_ds.read(1).astype(np.float32)
        pred = pred_ds.read(1).astype(np.float32)
        valid = valid_ds.read(1) > 0
    diff = (pred - gt)[valid].astype(np.float64)

    assert result.mae == pytest.approx(float(np.abs(diff).mean()))
    assert result.rmse == pytest.approx(float(np.sqrt((diff ** 2).mean())))
