"""Unit tests for module_eval Stage E artifact export layer."""

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
from ai_fields.module_eval.boundary_metrics import (
    BoundaryEvaluationPolicy,
    compute_boundary_metrics,
)
from ai_fields.module_eval.export import export_eval_artifacts
from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract
from ai_fields.module_eval.object_metrics import ObjectMatchingPolicy, compute_object_structure_metrics
from ai_fields.module_eval.pixel_metrics import (
    PixelBinarizationPolicy,
    compute_global_pixel_metrics,
)
from ai_fields.module_eval.visual_diagnostics import VisualDiagnosticsResult

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


def _write_polygons(
    path: Path,
    *,
    include_confidence: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    properties: dict[str, str] = {"instance_id": "int"}
    if include_confidence:
        properties["polygon_confidence"] = "float"

    schema = {"geometry": "Polygon", "properties": properties}
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs="EPSG:32637") as dst:
        features = [
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
        for feat in features:
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
    manifest_path = tmp_path / "predict_manifest.json"
    write_manifest(
        manifest_path,
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
    return manifest_path


def _write_postprocess_manifest(tmp_path: Path, paths: dict[str, Path]) -> Path:
    manifest_path = tmp_path / "postprocess_manifest.json"
    write_manifest(
        manifest_path,
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
    return manifest_path


def _build_eval_bundle(tmp_path: Path):
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
    predict_manifest = _write_predict_manifest(tmp_path, paths)
    postprocess_manifest = _write_postprocess_manifest(tmp_path, paths)

    contract = resolve_evaluation_input_contract(
        **paths,
        predict_manifest_path=predict_manifest,
        postprocess_manifest_path=postprocess_manifest,
    )

    pixel_policy = PixelBinarizationPolicy(
        extent_prob_threshold=0.5,
        threshold_provenance="eval_pixel_threshold_v1",
    )
    boundary_policy = BoundaryEvaluationPolicy(
        prediction_interpretation="argmax_non_background",
        gt_interpretation="non_background",
        threshold_provenance="eval_boundary_policy_v1",
        bde_enabled=False,
    )
    object_policy = ObjectMatchingPolicy(
        threshold_provenance="eval_object_matching_v1",
        min_iou_threshold=0.2,
        min_overlap_gt_threshold=0.2,
        min_overlap_pred_threshold=0.2,
        match_rule="iou_or_overlap",
    )

    pixel_result = compute_global_pixel_metrics(
        input_contract=contract,
        policy=pixel_policy,
    )
    boundary_result = compute_boundary_metrics(
        input_contract=contract,
        policy=boundary_policy,
    )
    object_result = compute_object_structure_metrics(
        input_contract=contract,
        policy=object_policy,
    )
    return (
        contract,
        pixel_policy,
        boundary_policy,
        object_policy,
        pixel_result,
        boundary_result,
        object_result,
    )


def test_export_eval_artifacts_happy_path(tmp_path: Path) -> None:
    (
        contract,
        pixel_policy,
        boundary_policy,
        object_policy,
        pixel_result,
        boundary_result,
        object_result,
    ) = _build_eval_bundle(tmp_path / "bundle")

    artifacts = export_eval_artifacts(
        output_dir=tmp_path / "eval_run",
        run_id="eval_run_001",
        eval_mode="end_to_end_single_scene",
        input_contract=contract,
        pixel_result=pixel_result,
        boundary_result=boundary_result,
        object_result=object_result,
        pixel_policy=pixel_policy,
        boundary_policy=boundary_policy,
        object_policy=object_policy,
    )

    assert artifacts.eval_manifest_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.config_used_path.exists()

    manifest = read_manifest(artifacts.eval_manifest_path)
    assert manifest["schema_name"] == "eval.eval_manifest"
    assert manifest["stage_coverage"]["stage_e"] is True
    assert manifest["metrics_enabled"]["pixel"] == [
        "extent_iou",
        "extent_f1",
        "extent_precision",
        "extent_recall",
    ]
    assert len(manifest["source_manifest_paths"]) == 2
    assert sorted(manifest["source_run_ids"]) == ["postprocess_run_001", "predict_run_001"]

    with artifacts.summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["schema_name"] == "eval.summary"
    assert "pixel" in summary["metric_summary"]
    assert "boundary" in summary["metric_summary"]
    assert "object_structure" in summary["metric_summary"]
    assert summary["metric_summary"]["object_structure"]["metrics"]["gtc"] == pytest.approx(
        object_result.gtc
    )

    payload = yaml.safe_load(artifacts.config_used_path.read_text(encoding="utf-8"))
    assert payload["effective_policy_contract"]["pixel_policy"]["extent_prob_threshold"] == pytest.approx(0.5)
    assert (
        payload["effective_policy_contract"]["object_matching_policy"]["match_rule"]
        == "iou_or_overlap"
    )


def test_export_eval_artifacts_fails_on_missing_required_metric_result(tmp_path: Path) -> None:
    (
        contract,
        pixel_policy,
        boundary_policy,
        object_policy,
        pixel_result,
        boundary_result,
        object_result,
    ) = _build_eval_bundle(tmp_path / "bundle")

    with pytest.raises(ContractError, match="object_result must be ObjectStructureMetricsResult"):
        export_eval_artifacts(
            output_dir=tmp_path / "eval_run",
            run_id="eval_run_001",
            eval_mode="end_to_end_single_scene",
            input_contract=contract,
            pixel_result=pixel_result,
            boundary_result=boundary_result,
            object_result=None,  # type: ignore[arg-type]
            pixel_policy=pixel_policy,
            boundary_policy=boundary_policy,
            object_policy=object_policy,
        )

    assert object_result is not None


def test_error_taxonomy_artifact_exists_and_has_all_classes(tmp_path: Path) -> None:
    (
        contract,
        pixel_policy,
        boundary_policy,
        object_policy,
        pixel_result,
        boundary_result,
        object_result,
    ) = _build_eval_bundle(tmp_path / "bundle")

    artifacts = export_eval_artifacts(
        output_dir=tmp_path / "eval_run",
        run_id="eval_run_taxonomy",
        eval_mode="end_to_end_single_scene",
        input_contract=contract,
        pixel_result=pixel_result,
        boundary_result=boundary_result,
        object_result=object_result,
        pixel_policy=pixel_policy,
        boundary_policy=boundary_policy,
        object_policy=object_policy,
    )

    assert artifacts.error_taxonomy_path.exists()
    assert artifacts.error_taxonomy_path.name == "error_taxonomy.json"

    taxonomy = json.loads(artifacts.error_taxonomy_path.read_text(encoding="utf-8"))
    assert taxonomy["schema_name"] == "eval.error_taxonomy"
    assert taxonomy["run_id"] == "eval_run_taxonomy"

    required_classes = {
        "split_error", "merge_error", "missed_parcel",
        "spurious_parcel", "invalid_area_artifact", "boundary_shift",
    }
    assert required_classes == set(taxonomy["taxonomy"].keys())

    for cls in ("split_error", "merge_error", "missed_parcel", "spurious_parcel", "invalid_area_artifact"):
        assert "count" in taxonomy["taxonomy"][cls]
        assert isinstance(taxonomy["taxonomy"][cls]["count"], int)

    assert "value" in taxonomy["taxonomy"]["boundary_shift"]
    assert "units" in taxonomy["taxonomy"]["boundary_shift"]

    counts = taxonomy["object_counts"]
    assert counts["gt_total"] == object_result.gt_object_count
    assert counts["pred_total"] == object_result.pred_object_count

    assert taxonomy["taxonomy"]["split_error"]["count"] == object_result.split_gt_count
    assert taxonomy["taxonomy"]["merge_error"]["count"] == object_result.merged_gt_count
    assert taxonomy["taxonomy"]["missed_parcel"]["count"] == object_result.unmatched_gt_count
    assert taxonomy["taxonomy"]["spurious_parcel"]["count"] == object_result.spurious_pred_count
    assert taxonomy["taxonomy"]["invalid_area_artifact"]["count"] == object_result.gt_excluded_zero_valid_count

    # error_taxonomy_path must also be referenced in the manifest
    manifest = read_manifest(artifacts.eval_manifest_path)
    assert str(artifacts.error_taxonomy_path) == manifest["artifacts"]["error_taxonomy_path"]
    assert manifest["stage_coverage"]["error_taxonomy"] is True


def test_export_eval_artifacts_fails_on_policy_result_mismatch(tmp_path: Path) -> None:
    (
        contract,
        pixel_policy,
        boundary_policy,
        object_policy,
        pixel_result,
        boundary_result,
        object_result,
    ) = _build_eval_bundle(tmp_path / "bundle")
    broken_pixel_result = replace(pixel_result, threshold=0.9)

    with pytest.raises(ContractError, match="pixel_result threshold/policy mismatch"):
        export_eval_artifacts(
            output_dir=tmp_path / "eval_run",
            run_id="eval_run_001",
            eval_mode="end_to_end_single_scene",
            input_contract=contract,
            pixel_result=broken_pixel_result,
            boundary_result=boundary_result,
            object_result=object_result,
            pixel_policy=pixel_policy,
            boundary_policy=boundary_policy,
            object_policy=object_policy,
        )


def test_export_eval_artifacts_truthful_when_visual_diagnostics_disabled(tmp_path: Path) -> None:
    (
        contract,
        pixel_policy,
        boundary_policy,
        object_policy,
        pixel_result,
        boundary_result,
        object_result,
    ) = _build_eval_bundle(tmp_path / "bundle")

    artifacts = export_eval_artifacts(
        output_dir=tmp_path / "eval_run",
        run_id="eval_run_no_visuals",
        eval_mode="end_to_end_single_scene",
        input_contract=contract,
        pixel_result=pixel_result,
        boundary_result=boundary_result,
        object_result=object_result,
        generate_visual_diagnostics=False,
        pixel_policy=pixel_policy,
        boundary_policy=boundary_policy,
        object_policy=object_policy,
    )

    manifest = read_manifest(artifacts.eval_manifest_path)
    assert manifest["stage_coverage"]["stage_e5_visual_diagnostics"] is False

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert "E5_visual_diagnostics" not in set(summary["stage_coverage"]["implemented"])
    assert summary["warnings"] == []


def test_export_eval_artifacts_truthful_when_visual_diagnostics_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        pixel_policy,
        boundary_policy,
        object_policy,
        pixel_result,
        boundary_result,
        object_result,
    ) = _build_eval_bundle(tmp_path / "bundle")

    def _fake_visual(*_: Any, **__: Any) -> VisualDiagnosticsResult:
        return VisualDiagnosticsResult(
            extent_overlay_path=None,
            boundary_heatmap_path=None,
            diagnostics_index_path=None,
            skipped=True,
            skip_reason="matplotlib not available in test",
        )

    monkeypatch.setattr("ai_fields.module_eval.export.write_visual_diagnostics", _fake_visual)

    artifacts = export_eval_artifacts(
        output_dir=tmp_path / "eval_run",
        run_id="eval_run_visual_skipped",
        eval_mode="end_to_end_single_scene",
        input_contract=contract,
        pixel_result=pixel_result,
        boundary_result=boundary_result,
        object_result=object_result,
        generate_visual_diagnostics=True,
        pixel_policy=pixel_policy,
        boundary_policy=boundary_policy,
        object_policy=object_policy,
    )

    manifest = read_manifest(artifacts.eval_manifest_path)
    assert manifest["stage_coverage"]["stage_e5_visual_diagnostics"] is False

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert "E5_visual_diagnostics" not in set(summary["stage_coverage"]["implemented"])
    assert any("visual diagnostics skipped" in warning for warning in summary["warnings"])
