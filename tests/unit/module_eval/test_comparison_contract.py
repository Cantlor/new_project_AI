"""Unit tests for module_eval minimal comparison input-contract layer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import write_manifest
from ai_fields.module_eval.boundary_metrics import BoundaryEvaluationPolicy
from ai_fields.module_eval.comparison_contract import (
    build_comparison_readiness_summary,
    resolve_eval_comparison_contract,
)
from ai_fields.module_eval.object_metrics import ObjectMatchingPolicy
from ai_fields.module_eval.pixel_metrics import PixelBinarizationPolicy
from ai_fields.module_eval.run_eval import EvalRunInputs, EvalRunPolicies, run_eval

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


def _build_inputs(tmp_path: Path) -> EvalRunInputs:
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
    return EvalRunInputs(
        **paths,
        predict_manifest_path=predict_manifest_path,
        postprocess_manifest_path=postprocess_manifest_path,
    )


def _build_policies(
    *,
    pixel_threshold: float = 0.5,
    object_match_rule: str = "iou_or_overlap",
    eval_mode: str = "end_to_end_single_scene",
) -> EvalRunPolicies:
    return EvalRunPolicies(
        eval_mode=eval_mode,
        pixel_policy=PixelBinarizationPolicy(
            extent_prob_threshold=float(pixel_threshold),
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
            match_rule=object_match_rule,
        ),
    )


def test_resolve_eval_comparison_contract_happy_path(tmp_path: Path) -> None:
    left = run_eval(
        run_id="left_eval_run",
        inputs=_build_inputs(tmp_path / "left_inputs"),
        policies=_build_policies(),
        output_dir=tmp_path / "runs" / "module_eval" / "left_eval_run",
    )
    right = run_eval(
        run_id="right_eval_run",
        inputs=_build_inputs(tmp_path / "right_inputs"),
        policies=_build_policies(),
        output_dir=tmp_path / "runs" / "module_eval" / "right_eval_run",
    )

    result = resolve_eval_comparison_contract(
        left_run_dir=left.run_dir,
        right_run_dir=right.run_dir,
    )
    assert result.ready_for_pairwise_compare is True
    assert result.partially_ready is False
    assert set(result.comparable_metric_groups) == {"pixel", "boundary", "object_structure"}
    assert result.non_comparable_metric_groups == {}

    summary = build_comparison_readiness_summary(result)
    assert summary["ready_for_pairwise_compare"] is True


def test_resolve_eval_comparison_contract_policy_mismatch(tmp_path: Path) -> None:
    left = run_eval(
        run_id="left_eval_run",
        inputs=_build_inputs(tmp_path / "left_inputs"),
        policies=_build_policies(pixel_threshold=0.5),
        output_dir=tmp_path / "runs" / "module_eval" / "left_eval_run",
    )
    right = run_eval(
        run_id="right_eval_run",
        inputs=_build_inputs(tmp_path / "right_inputs"),
        policies=_build_policies(pixel_threshold=0.7),
        output_dir=tmp_path / "runs" / "module_eval" / "right_eval_run",
    )

    result = resolve_eval_comparison_contract(
        left_run_dir=left.run_dir,
        right_run_dir=right.run_dir,
    )
    assert result.ready_for_pairwise_compare is False
    assert result.partially_ready is True
    assert "pixel" not in result.comparable_metric_groups
    assert "pixel threshold mismatch" in result.non_comparable_metric_groups["pixel"]


def test_resolve_eval_comparison_contract_partial_comparability(tmp_path: Path) -> None:
    left = run_eval(
        run_id="left_eval_run",
        inputs=_build_inputs(tmp_path / "left_inputs"),
        policies=_build_policies(object_match_rule="iou_or_overlap"),
        output_dir=tmp_path / "runs" / "module_eval" / "left_eval_run",
    )
    right = run_eval(
        run_id="right_eval_run",
        inputs=_build_inputs(tmp_path / "right_inputs"),
        policies=_build_policies(object_match_rule="iou_only"),
        output_dir=tmp_path / "runs" / "module_eval" / "right_eval_run",
    )

    result = resolve_eval_comparison_contract(
        left_run_dir=left.run_dir,
        right_run_dir=right.run_dir,
    )
    assert result.ready_for_pairwise_compare is False
    assert result.partially_ready is True
    assert set(result.comparable_metric_groups) == {"pixel", "boundary"}
    assert "object_structure" in result.non_comparable_metric_groups
    assert "object match_rule mismatch" in result.non_comparable_metric_groups["object_structure"]


def test_resolve_eval_comparison_contract_missing_artifact_fails(tmp_path: Path) -> None:
    left = run_eval(
        run_id="left_eval_run",
        inputs=_build_inputs(tmp_path / "left_inputs"),
        policies=_build_policies(),
        output_dir=tmp_path / "runs" / "module_eval" / "left_eval_run",
    )
    right = run_eval(
        run_id="right_eval_run",
        inputs=_build_inputs(tmp_path / "right_inputs"),
        policies=_build_policies(),
        output_dir=tmp_path / "runs" / "module_eval" / "right_eval_run",
    )

    right.summary_path.unlink()
    with pytest.raises(ContractError, match="right_summary_path does not exist"):
        resolve_eval_comparison_contract(
            left_run_dir=left.run_dir,
            right_run_dir=right.run_dir,
        )


def test_resolve_eval_comparison_contract_stage_coverage_mismatch(tmp_path: Path) -> None:
    left = run_eval(
        run_id="left_eval_run",
        inputs=_build_inputs(tmp_path / "left_inputs"),
        policies=_build_policies(),
        output_dir=tmp_path / "runs" / "module_eval" / "left_eval_run",
    )
    right = run_eval(
        run_id="right_eval_run",
        inputs=_build_inputs(tmp_path / "right_inputs"),
        policies=_build_policies(),
        output_dir=tmp_path / "runs" / "module_eval" / "right_eval_run",
    )

    # Simulate an artifact set where object/structure metric group is unavailable.
    payload = json.loads(right.summary_path.read_text(encoding="utf-8"))
    metric_summary = payload["metric_summary"]
    metric_summary.pop("object_structure", None)
    right.summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = resolve_eval_comparison_contract(
        left_run_dir=left.run_dir,
        right_run_dir=right.run_dir,
    )
    assert result.ready_for_pairwise_compare is False
    assert result.partially_ready is True
    assert "object_structure" not in result.comparable_metric_groups
    assert "right:object_structure missing in summary.metric_summary" in result.non_comparable_metric_groups["object_structure"]

