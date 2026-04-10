"""Smoke tests for canonical shell runner tools/run_module_eval.sh."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.manifests import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "tools" / "run_module_eval.sh"

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")


def _run_runner(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


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


def _write_polygons(path: Path, *, include_confidence: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    props: dict[str, str] = {"instance_id": "int"}
    if include_confidence:
        props["polygon_confidence"] = "float"

    schema = {"geometry": "Polygon", "properties": props}
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs="EPSG:32637") as dst:
        feats = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        (101.0, 199.0),
                        (104.0, 199.0),
                        (104.0, 196.0),
                        (101.0, 196.0),
                        (101.0, 199.0),
                    ]],
                },
                "properties": {
                    "instance_id": 1,
                    **({"polygon_confidence": 0.82} if include_confidence else {}),
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        (104.0, 199.0),
                        (107.0, 199.0),
                        (107.0, 196.0),
                        (104.0, 196.0),
                        (104.0, 199.0),
                    ]],
                },
                "properties": {
                    "instance_id": 2,
                    **({"polygon_confidence": 0.74} if include_confidence else {}),
                },
            },
        ]
        for feat in feats:
            dst.write(feat)
    return path


def _build_arrays(height: int = 6, width: int = 8) -> dict[str, np.ndarray]:
    gt_extent = np.zeros((1, height, width), dtype=np.uint8)
    gt_extent[0, 1:5, 1:7] = 1
    gt_extent[0, 0, 0] = 255

    gt_boundary = np.zeros((1, height, width), dtype=np.uint8)
    gt_boundary[0, 1:5, 3] = 1
    gt_boundary[0, 1:5, 2] = 2
    gt_boundary[0, 1:5, 4] = 2

    gt_valid = np.ones((1, height, width), dtype=np.uint8)
    gt_valid[0, 0, 0] = 0

    gt_distance = np.full((1, height, width), 2.0, dtype=np.float32)
    gt_distance[0, 1:5, 3] = 0.0

    pred_extent = np.full((1, height, width), 0.2, dtype=np.float32)
    pred_extent[0, 1:5, 1:7] = 0.85

    pred_boundary = np.zeros((3, height, width), dtype=np.float32)
    pred_boundary[0] = 0.8
    pred_boundary[1] = 0.1
    pred_boundary[2] = 0.1
    pred_boundary[1, 1:5, 3] = 0.65
    pred_boundary[2, 1:5, 3] = 0.2
    pred_boundary[0, 1:5, 3] = 0.15

    pred_distance = np.full((1, height, width), 1.2, dtype=np.float32)
    pred_distance[0, 1:5, 3] = 0.1

    pred_valid = gt_valid.copy()

    post_instance = np.zeros((1, height, width), dtype=np.int32)
    post_instance[0, 1:5, 1:4] = 1
    post_instance[0, 1:5, 4:7] = 2
    post_instance[0, 0, 0] = -1

    return {
        "gt_extent": gt_extent,
        "gt_boundary": gt_boundary,
        "gt_valid": gt_valid,
        "gt_distance": gt_distance,
        "pred_extent": pred_extent,
        "pred_boundary": pred_boundary,
        "pred_distance": pred_distance,
        "pred_valid": pred_valid,
        "post_instance": post_instance,
    }


class TestRunModuleEvalScript:
    def test_dry_run_resolves_run_dirs(self, tmp_path: Path) -> None:
        prep = tmp_path / "prep"
        predict = tmp_path / "predict"
        post = tmp_path / "post"

        _touch(prep / "03_prepare_features" / "valid.tif")
        _touch(prep / "04_prepare_targets" / "extent.tif")
        _touch(prep / "04_prepare_targets" / "boundary.tif")
        _touch(prep / "04_prepare_targets" / "distance.tif")
        _touch(prep / "02_prepare_spatial_context" / "vector_in_raster_crs.gpkg")

        _touch(predict / "extent_prob.tif")
        _touch(predict / "boundary_prob.tif")
        _touch(predict / "distance_pred.tif")
        _touch(predict / "valid.tif")
        _touch(predict / "predict_manifest.json")

        _touch(post / "parcel_instance.tif")
        _touch(post / "parcels.gpkg")
        _touch(post / "postprocess_manifest.json")

        result = _run_runner(
            "--prep-run-dir",
            str(prep),
            "--predict-run-dir",
            str(predict),
            "--postprocess-run-dir",
            str(post),
            "--output-dir",
            str(tmp_path / "eval_runs"),
            "--run-id",
            "eval-run-test",
            "--dry-run",
        )

        assert result.returncode == 0, result.stderr
        assert "[INFO] Entry point: ai_fields.module_eval.run_eval.run_eval" in result.stdout
        assert f"[INFO] Prep run dir: {prep}" in result.stdout
        assert f"[INFO] Predict run dir: {predict}" in result.stdout
        assert f"[INFO] Postprocess run dir: {post}" in result.stdout

    def test_missing_required_postprocess_vector_fails_clearly(self, tmp_path: Path) -> None:
        prep = tmp_path / "prep"
        predict = tmp_path / "predict"
        post = tmp_path / "post"

        _touch(prep / "03_prepare_features" / "valid.tif")
        _touch(prep / "04_prepare_targets" / "extent.tif")
        _touch(prep / "04_prepare_targets" / "boundary.tif")
        _touch(prep / "02_prepare_spatial_context" / "vector_in_raster_crs.gpkg")

        _touch(predict / "extent_prob.tif")
        _touch(predict / "boundary_prob.tif")
        _touch(predict / "distance_pred.tif")
        _touch(predict / "valid.tif")

        # Intentionally missing parcels.gpkg
        _touch(post / "parcel_instance.tif")

        result = _run_runner(
            "--prep-run-dir",
            str(prep),
            "--predict-run-dir",
            str(predict),
            "--postprocess-run-dir",
            str(post),
            "--dry-run",
        )

        assert result.returncode != 0
        assert "Missing required input artifact" in result.stderr
        assert "parcels.gpkg" in result.stderr

    def test_smoke_run_creates_eval_artifacts(self, tmp_path: Path) -> None:
        arrays = _build_arrays()

        prep = tmp_path / "prep"
        predict = tmp_path / "predict"
        post = tmp_path / "post"

        gt_extent_path = _write_raster(prep / "04_prepare_targets" / "extent.tif", array=arrays["gt_extent"])
        gt_boundary_path = _write_raster(prep / "04_prepare_targets" / "boundary.tif", array=arrays["gt_boundary"])
        gt_valid_path = _write_raster(prep / "03_prepare_features" / "valid.tif", array=arrays["gt_valid"])
        gt_distance_path = _write_raster(prep / "04_prepare_targets" / "distance.tif", array=arrays["gt_distance"])
        gt_parcels_path = _write_polygons(prep / "02_prepare_spatial_context" / "vector_in_raster_crs.gpkg", include_confidence=False)

        pred_extent_path = _write_raster(predict / "extent_prob.tif", array=arrays["pred_extent"])
        pred_boundary_path = _write_raster(predict / "boundary_prob.tif", array=arrays["pred_boundary"])
        pred_distance_path = _write_raster(predict / "distance_pred.tif", array=arrays["pred_distance"])
        pred_valid_path = _write_raster(predict / "valid.tif", array=arrays["pred_valid"])

        post_instance_path = _write_raster(post / "parcel_instance.tif", array=arrays["post_instance"])
        post_parcels_path = _write_polygons(post / "parcels.gpkg", include_confidence=True)

        write_manifest(
            predict / "predict_manifest.json",
            {
                "schema_name": "target_predict.predict_manifest",
                "schema_version": "v1",
                "module_name": "module_target_predict",
                "module_version": None,
                "data_contract_version": DATA_CONTRACT_VERSION,
                "run_id": "predict_smoke_001",
                "stage_name": "predict_scene",
                "created_at_utc": "2026-04-09T00:00:00Z",
                "status": "success",
                "output_paths": {
                    "extent_prob": str(pred_extent_path),
                    "boundary_prob": str(pred_boundary_path),
                    "distance_pred": str(pred_distance_path),
                    "valid": str(pred_valid_path),
                },
            },
        )

        write_manifest(
            post / "postprocess_manifest.json",
            {
                "schema_name": "postprocess_vectorize.postprocess_manifest",
                "schema_version": "v1",
                "module_name": "module_postprocess_vectorize",
                "module_version": None,
                "data_contract_version": DATA_CONTRACT_VERSION,
                "run_id": "post_smoke_001",
                "stage_name": "export_postprocess_artifacts",
                "created_at_utc": "2026-04-09T00:00:00Z",
                "status": "success",
                "inputs": {
                    "extent_prob_path": str(pred_extent_path),
                    "boundary_prob_path": str(pred_boundary_path),
                    "distance_pred_path": str(pred_distance_path),
                    "valid_path": str(pred_valid_path),
                    "aoi_path": None,
                },
                "outputs": {
                    "parcel_instance_path": str(post_instance_path),
                    "parcels_gpkg_path": str(post_parcels_path),
                    "optional_exports": [],
                },
            },
        )

        output_root = tmp_path / "eval_runs"
        run_id = "eval-smoke-001"
        result = _run_runner(
            "--prep-run-dir",
            str(prep),
            "--predict-run-dir",
            str(predict),
            "--postprocess-run-dir",
            str(post),
            "--output-dir",
            str(output_root),
            "--run-id",
            run_id,
            "--no-progress",
        )

        assert result.returncode == 0, result.stderr
        run_dir = output_root / run_id
        assert (run_dir / "eval_manifest.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "config_used.yaml").exists()
        assert (run_dir / "error_taxonomy.json").exists()

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["schema_name"] == "eval.summary"
        assert "pixel" in summary["metric_summary"]
        assert "boundary" in summary["metric_summary"]
        assert "object_structure" in summary["metric_summary"]
