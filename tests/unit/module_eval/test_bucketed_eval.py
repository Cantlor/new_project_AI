"""Unit tests for module_eval Stage D.5 bucketed object/structure metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ai_fields.module_eval.bucketed_eval as bucketed_eval_module
from ai_fields.common.errors import ContractError
from ai_fields.module_eval.bucketed_eval import (
    BucketSizePolicy,
    build_bucketed_eval_summary,
    compute_bucketed_object_metrics,
    write_bucketed_eval_artifact,
)
from ai_fields.module_eval.object_metrics import ObjectMatchingPolicy

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


def _write_polygon_gpkg(path: Path, polygons: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {"geometry": "Polygon", "properties": {"instance_id": "int"}}
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs="EPSG:32637") as dst:
        for poly in polygons:
            dst.write(poly)
    return path


def _make_two_parcels_contract(tmp_path: Path):
    """Two GT parcels, two predicted parcels (1:1 match), EPSG:32637 projected."""
    from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract

    h, w = 8, 12
    gt_extent = np.zeros((1, h, w), dtype=np.uint8)
    gt_extent[0, 1:7, 1:5] = 1   # parcel A
    gt_extent[0, 1:7, 6:11] = 1  # parcel B

    gt_boundary = np.zeros((1, h, w), dtype=np.uint8)
    gt_valid = np.ones((1, h, w), dtype=np.uint8)
    gt_valid[0, 0, 0] = 0

    pred_extent = np.zeros((1, h, w), dtype=np.float32)
    pred_extent[0, 1:7, 1:5] = 0.9
    pred_extent[0, 1:7, 6:11] = 0.9

    pred_boundary = np.zeros((3, h, w), dtype=np.float32)
    pred_boundary[0] = 1.0
    pred_distance = np.ones((1, h, w), dtype=np.float32)
    pred_valid = gt_valid.copy()

    # GT parcels: A is 4*6=24 px, B is 5*6=30 px (both "small" by pixel count at default thresholds)
    gt_parcels = _write_polygon_gpkg(tmp_path / "gt_parcels.gpkg", [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(101.0, 198.0), (105.0, 198.0), (105.0, 192.0), (101.0, 192.0), (101.0, 198.0)]],
            },
            "properties": {"instance_id": 1},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(106.0, 198.0), (111.0, 198.0), (111.0, 192.0), (106.0, 192.0), (106.0, 198.0)]],
            },
            "properties": {"instance_id": 2},
        },
    ])
    pred_parcels_path = tmp_path / "pred_parcels.gpkg"
    pred_parcels_path.parent.mkdir(parents=True, exist_ok=True)
    pred_schema = {"geometry": "Polygon", "properties": {"instance_id": "int", "polygon_confidence": "float"}}
    with fiona.open(pred_parcels_path, "w", driver="GPKG", schema=pred_schema, crs="EPSG:32637") as dst:
        dst.write({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(101.0, 198.0), (105.0, 198.0), (105.0, 192.0), (101.0, 192.0), (101.0, 198.0)]],
            },
            "properties": {"instance_id": 1, "polygon_confidence": 0.9},
        })
        dst.write({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(106.0, 198.0), (111.0, 198.0), (111.0, 192.0), (106.0, 192.0), (106.0, 198.0)]],
            },
            "properties": {"instance_id": 2, "polygon_confidence": 0.85},
        })
    pred_parcels = pred_parcels_path

    paths = {
        "gt_extent_path": _write_raster(tmp_path / "gt_extent.tif", array=gt_extent),
        "gt_boundary_path": _write_raster(tmp_path / "gt_boundary.tif", array=gt_boundary),
        "gt_valid_path": _write_raster(tmp_path / "gt_valid.tif", array=gt_valid),
        "pred_extent_prob_path": _write_raster(tmp_path / "extent_prob.tif", array=pred_extent),
        "pred_boundary_prob_path": _write_raster(tmp_path / "boundary_prob.tif", array=pred_boundary),
        "pred_distance_pred_path": _write_raster(tmp_path / "distance_pred.tif", array=pred_distance),
        "pred_valid_path": _write_raster(tmp_path / "valid.tif", array=pred_valid),
        "gt_parcels_path": gt_parcels,
        "post_parcels_gpkg_path": pred_parcels,
        "post_parcel_instance_path": _write_raster(
            tmp_path / "parcel_instance.tif",
            array=gt_extent.astype(np.int32),
        ),
    }
    return resolve_evaluation_input_contract(**paths)


def _default_object_policy() -> ObjectMatchingPolicy:
    return ObjectMatchingPolicy(
        threshold_provenance="eval_object_v1",
        min_iou_threshold=0.2,
        min_overlap_gt_threshold=0.2,
        min_overlap_pred_threshold=0.2,
        match_rule="iou_or_overlap",
    )


def _default_bucket_policy() -> BucketSizePolicy:
    return BucketSizePolicy(
        threshold_provenance="eval_bucket_v1_baseline",
        use_projected_area=True,
    )


class TestComputeBucketedObjectMetrics:
    def test_happy_path_returns_three_buckets(self, tmp_path: Path) -> None:
        contract = _make_two_parcels_contract(tmp_path)
        result = compute_bucketed_object_metrics(
            input_contract=contract,
            object_policy=_default_object_policy(),
            bucket_policy=_default_bucket_policy(),
        )
        assert len(result.buckets) == 3
        bucket_names = {b.bucket_name for b in result.buckets}
        assert bucket_names == {"small", "medium", "large"}

    def test_gt_object_counts_sum_to_total(self, tmp_path: Path) -> None:
        contract = _make_two_parcels_contract(tmp_path)
        result = compute_bucketed_object_metrics(
            input_contract=contract,
            object_policy=_default_object_policy(),
            bucket_policy=_default_bucket_policy(),
        )
        total = sum(b.gt_object_count for b in result.buckets)
        assert total == 2  # two GT parcels

    def test_goc_guc_gtc_are_floats_in_valid_range(self, tmp_path: Path) -> None:
        contract = _make_two_parcels_contract(tmp_path)
        result = compute_bucketed_object_metrics(
            input_contract=contract,
            object_policy=_default_object_policy(),
            bucket_policy=_default_bucket_policy(),
        )
        for b in result.buckets:
            assert 0.0 <= b.goc <= 1.0
            assert 0.0 <= b.guc <= 1.0
            assert b.gtc >= 0.0
            assert 0.0 <= b.normalized_gtc <= 1.0

    def test_area_unit_is_m2_for_projected_crs(self, tmp_path: Path) -> None:
        contract = _make_two_parcels_contract(tmp_path)
        result = compute_bucketed_object_metrics(
            input_contract=contract,
            object_policy=_default_object_policy(),
            bucket_policy=_default_bucket_policy(),
        )
        assert result.area_unit == "m2"
        assert result.pixel_size_m2 is not None
        assert result.pixel_size_m2 > 0.0

    def test_pixel_unit_when_use_projected_false(self, tmp_path: Path) -> None:
        contract = _make_two_parcels_contract(tmp_path)
        policy = BucketSizePolicy(
            threshold_provenance="eval_v1",
            use_projected_area=False,
        )
        result = compute_bucketed_object_metrics(
            input_contract=contract,
            object_policy=_default_object_policy(),
            bucket_policy=policy,
        )
        assert result.area_unit == "pixels"
        assert result.pixel_size_m2 is None

    def test_bucket_thresholds_stored_in_result(self, tmp_path: Path) -> None:
        contract = _make_two_parcels_contract(tmp_path)
        result = compute_bucketed_object_metrics(
            input_contract=contract,
            object_policy=_default_object_policy(),
            bucket_policy=_default_bucket_policy(),
        )
        assert "small_max" in result.bucket_thresholds
        assert "medium_max" in result.bucket_thresholds

    def test_sparse_relation_memory_guard_fails_fast(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contract = _make_two_parcels_contract(tmp_path)
        monkeypatch.setattr(bucketed_eval_module, "_RELATION_EDGE_MAX_COUNT", 0)

        with pytest.raises(ContractError, match="sparse relation edges exceed memory safety guard"):
            compute_bucketed_object_metrics(
                input_contract=contract,
                object_policy=_default_object_policy(),
                bucket_policy=_default_bucket_policy(),
            )


class TestWriteBucketedEvalArtifact:
    def test_file_is_written_with_schema_name(self, tmp_path: Path) -> None:
        contract = _make_two_parcels_contract(tmp_path)
        result = compute_bucketed_object_metrics(
            input_contract=contract,
            object_policy=_default_object_policy(),
            bucket_policy=_default_bucket_policy(),
        )
        path = tmp_path / "metrics_by_bucket.json"
        write_bucketed_eval_artifact(path, run_id="run_001", eval_mode="end_to_end", result=result)

        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_name"] == "eval.metrics_by_bucket"
        assert len(payload["buckets"]) == 3
        for bucket in payload["buckets"]:
            assert "bucket_name" in bucket
            assert "goc" in bucket
            assert "guc" in bucket
            assert "gtc" in bucket


class TestBuildBucketedEvalSummary:
    def test_summary_has_required_keys(self, tmp_path: Path) -> None:
        contract = _make_two_parcels_contract(tmp_path)
        result = compute_bucketed_object_metrics(
            input_contract=contract,
            object_policy=_default_object_policy(),
            bucket_policy=_default_bucket_policy(),
        )
        summary = build_bucketed_eval_summary(result)
        assert "stage_scope" in summary
        assert "buckets" in summary
        for name in ("small", "medium", "large"):
            assert name in summary["buckets"]
