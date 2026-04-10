"""Unit tests for module_eval minimal run-level orchestration path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import read_manifest, write_manifest
from ai_fields.module_eval.boundary_metrics import BoundaryEvaluationPolicy
from ai_fields.module_eval.pixel_metrics import PixelBinarizationPolicy
from ai_fields.module_eval.object_metrics import ObjectMatchingPolicy
from ai_fields.module_eval.run_eval import EvalRunInputs, EvalRunPolicies, run_eval

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
yaml = pytest.importorskip("yaml")


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
                "properties": {"instance_id": 1, **({"polygon_confidence": 0.82} if include_confidence else {})},
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
                "properties": {"instance_id": 2, **({"polygon_confidence": 0.74} if include_confidence else {})},
            },
        ]
        for feat in feats:
            dst.write(feat)
    return path


def _build_base_arrays(height: int = 6, width: int = 8) -> dict[str, np.ndarray]:
    gt_extent = np.zeros((1, height, width), dtype=np.uint8)
    gt_extent[0, 1:5, 1:7] = 1
    gt_extent[0, 0, 0] = 255

    gt_boundary = np.zeros((1, height, width), dtype=np.uint8)
    gt_boundary[0, 1:5, 3] = 1
    gt_boundary[0, 1:5, 2] = 2
    gt_boundary[0, 1:5, 4] = 2

    gt_valid = np.ones((1, height, width), dtype=np.uint8)
    gt_valid[0, 0, 0] = 0
    gt_valid[0, 5, 7] = 0

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


def _write_predict_manifest(tmp_path: Path, paths: dict[str, Path]) -> Path:
    p = tmp_path / "predict_manifest.json"
    write_manifest(
        p,
        {
            "schema_name": "target_predict.predict_manifest",
            "schema_version": "v1",
            "module_name": "module_target_predict",
            "module_version": None,
            "data_contract_version": DATA_CONTRACT_VERSION,
            "run_id": "predict_run_001",
            "stage_name": "predict_scene",
            "created_at_utc": "2026-04-06T00:00:00Z",
            "status": "success",
            "output_paths": {
                "extent_prob": str(paths["pred_extent_prob_path"]),
                "boundary_prob": str(paths["pred_boundary_prob_path"]),
                "distance_pred": str(paths["pred_distance_pred_path"]),
                "valid": str(paths["pred_valid_path"]),
            },
        },
    )
    return p


def _write_postprocess_manifest(tmp_path: Path, paths: dict[str, Path]) -> Path:
    p = tmp_path / "postprocess_manifest.json"
    write_manifest(
        p,
        {
            "schema_name": "postprocess_vectorize.postprocess_manifest",
            "schema_version": "v1",
            "module_name": "module_postprocess_vectorize",
            "module_version": None,
            "data_contract_version": DATA_CONTRACT_VERSION,
            "run_id": "postprocess_run_001",
            "stage_name": "export_postprocess_artifacts",
            "created_at_utc": "2026-04-06T00:00:00Z",
            "status": "success",
            "inputs": {
                "extent_prob_path": str(paths["pred_extent_prob_path"]),
                "boundary_prob_path": str(paths["pred_boundary_prob_path"]),
                "distance_pred_path": str(paths["pred_distance_pred_path"]),
                "valid_path": str(paths["pred_valid_path"]),
                "aoi_path": None,
            },
            "outputs": {
                "parcel_instance_path": str(paths["post_parcel_instance_path"]),
                "parcels_gpkg_path": str(paths["post_parcels_gpkg_path"]),
                "optional_exports": [],
            },
        },
    )
    return p


def _build_happy_bundle(tmp_path: Path) -> tuple[EvalRunInputs, EvalRunPolicies]:
    arr = _build_base_arrays()
    paths = {
        "gt_extent_path": _write_raster(tmp_path / "gt_extent.tif", array=arr["gt_extent"]),
        "gt_boundary_path": _write_raster(tmp_path / "gt_boundary.tif", array=arr["gt_boundary"]),
        "gt_valid_path": _write_raster(tmp_path / "gt_valid.tif", array=arr["gt_valid"]),
        "gt_distance_path": _write_raster(tmp_path / "gt_distance.tif", array=arr["gt_distance"]),
        "pred_extent_prob_path": _write_raster(tmp_path / "extent_prob.tif", array=arr["pred_extent"]),
        "pred_boundary_prob_path": _write_raster(tmp_path / "boundary_prob.tif", array=arr["pred_boundary"]),
        "pred_distance_pred_path": _write_raster(tmp_path / "distance_pred.tif", array=arr["pred_distance"]),
        "pred_valid_path": _write_raster(tmp_path / "valid.tif", array=arr["pred_valid"]),
        "post_parcel_instance_path": _write_raster(tmp_path / "parcel_instance.tif", array=arr["post_instance"]),
        "gt_parcels_path": _write_polygons(tmp_path / "gt_parcels.gpkg", include_confidence=False),
        "post_parcels_gpkg_path": _write_polygons(tmp_path / "parcels.gpkg", include_confidence=True),
    }
    predict_manifest_path = _write_predict_manifest(tmp_path, paths)
    postprocess_manifest_path = _write_postprocess_manifest(tmp_path, paths)

    inputs = EvalRunInputs(
        **paths,
        predict_manifest_path=predict_manifest_path,
        postprocess_manifest_path=postprocess_manifest_path,
    )
    policies = EvalRunPolicies(
        eval_mode="end_to_end_single_scene",
        pixel_policy=PixelBinarizationPolicy(
            extent_prob_threshold=0.5,
            threshold_provenance="eval_pixel_threshold_v1",
        ),
        boundary_policy=BoundaryEvaluationPolicy(
            prediction_interpretation="argmax_non_background",
            gt_interpretation="non_background",
            threshold_provenance="eval_boundary_policy_v1",
            bde_enabled=False,
        ),
        object_policy=ObjectMatchingPolicy(
            threshold_provenance="eval_object_matching_v1",
            min_iou_threshold=0.2,
            min_overlap_gt_threshold=0.2,
            min_overlap_pred_threshold=0.2,
            match_rule="iou_or_overlap",
        ),
    )
    return inputs, policies


def test_run_eval_happy_path(tmp_path: Path) -> None:
    inputs, policies = _build_happy_bundle(tmp_path / "inputs")
    result = run_eval(
        run_id="eval_run_001",
        inputs=inputs,
        policies=policies,
        output_dir=tmp_path / "runs" / "module_eval" / "eval_run_001",
    )

    assert result.ready_for_next_stage is True
    assert result.eval_manifest_path.exists()
    assert result.summary_path.exists()
    assert result.config_used_path.exists()
    assert result.run_dir.name == "eval_run_001"

    assert "metrics" in result.pixel_metrics_summary
    assert "metrics" in result.boundary_metrics_summary
    assert "metrics" in result.object_metrics_summary

    manifest = read_manifest(result.eval_manifest_path)
    assert manifest["schema_name"] == "eval.eval_manifest"
    assert manifest["stage_coverage"]["stage_e"] is True
    assert sorted(manifest["source_run_ids"]) == ["postprocess_run_001", "predict_run_001"]

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["schema_name"] == "eval.summary"
    assert "pixel" in summary["metric_summary"]
    assert "boundary" in summary["metric_summary"]
    assert "object_structure" in summary["metric_summary"]

    cfg = yaml.safe_load(result.config_used_path.read_text(encoding="utf-8"))
    assert cfg["effective_policy_contract"]["pixel_policy"]["extent_prob_threshold"] == pytest.approx(0.5)


def test_run_eval_fail_fast_on_broken_input_contract(tmp_path: Path) -> None:
    inputs, policies = _build_happy_bundle(tmp_path / "inputs")
    broken_inputs = replace(inputs, gt_extent_path=tmp_path / "missing_gt_extent.tif")

    with pytest.raises(ContractError):
        run_eval(
            run_id="eval_run_fail",
            inputs=broken_inputs,
            policies=policies,
            output_dir=tmp_path / "runs" / "module_eval" / "eval_run_fail",
        )


def test_run_eval_fail_fast_when_object_track_not_ready(tmp_path: Path) -> None:
    inputs, policies = _build_happy_bundle(tmp_path / "inputs")
    no_object_inputs = replace(
        inputs,
        gt_parcels_path=None,
        post_parcels_gpkg_path=None,
    )

    with pytest.raises(ContractError, match="object track is not ready"):
        run_eval(
            run_id="eval_run_not_ready",
            inputs=no_object_inputs,
            policies=policies,
            output_dir=tmp_path / "runs" / "module_eval" / "eval_run_not_ready",
        )


def test_run_eval_stable_artifact_layout(tmp_path: Path) -> None:
    inputs, policies = _build_happy_bundle(tmp_path / "inputs")
    run_dir = tmp_path / "stable_layout" / "eval_run_010"
    result = run_eval(
        run_id="eval_run_010",
        inputs=inputs,
        policies=policies,
        output_dir=run_dir,
    )

    assert result.eval_manifest_path == run_dir / "eval_manifest.json"
    assert result.summary_path == run_dir / "summary.json"
    assert result.config_used_path == run_dir / "config_used.yaml"


def test_run_eval_fails_on_unknown_eval_mode(tmp_path: Path) -> None:
    inputs, policies = _build_happy_bundle(tmp_path / "inputs")
    bad_policies = replace(policies, eval_mode="unknown_mode")

    with pytest.raises(ContractError, match="Unsupported eval_mode"):
        run_eval(
            run_id="eval_run_bad_mode",
            inputs=inputs,
            policies=bad_policies,
            output_dir=tmp_path / "runs" / "module_eval" / "eval_run_bad_mode",
        )


def test_run_eval_rejects_comparison_mode(tmp_path: Path) -> None:
    inputs, policies = _build_happy_bundle(tmp_path / "inputs")
    bad_policies = replace(policies, eval_mode="comparison")

    with pytest.raises(ContractError, match="does not support eval_mode='comparison'"):
        run_eval(
            run_id="eval_run_comparison_mode",
            inputs=inputs,
            policies=bad_policies,
            output_dir=tmp_path / "runs" / "module_eval" / "eval_run_comparison_mode",
        )
