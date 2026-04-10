"""Unit tests for module_eval Stage A input contract layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError, SpatialContractError
from ai_fields.common.manifests import write_manifest
import ai_fields.module_eval.input_contract as input_contract_module
from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract

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


def _write_polygons(
    path: Path,
    *,
    include_confidence: bool,
    schema_geometry: str = "Polygon",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    properties: dict[str, str] = {"instance_id": "int"}
    if include_confidence:
        properties["polygon_confidence"] = "float"

    schema = {
        "geometry": schema_geometry,
        "properties": properties,
    }
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


def _write_happy_paths(tmp_path: Path) -> dict[str, Path]:
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
    return paths


def _write_predict_manifest(tmp_path: Path, paths: dict[str, Path], *, schema_name: str) -> Path:
    manifest_path = tmp_path / "predict_manifest.json"
    write_manifest(
        manifest_path,
        {
            "schema_name": schema_name,
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


def test_resolve_evaluation_input_contract_happy_path(tmp_path: Path) -> None:
    paths = _write_happy_paths(tmp_path)
    predict_manifest_path = _write_predict_manifest(
        tmp_path, paths, schema_name="target_predict.predict_manifest"
    )
    postprocess_manifest_path = _write_postprocess_manifest(tmp_path, paths)

    resolved = resolve_evaluation_input_contract(
        **paths,
        predict_manifest_path=predict_manifest_path,
        postprocess_manifest_path=postprocess_manifest_path,
    )

    assert resolved.compatible is True
    assert resolved.common_width == 8
    assert resolved.common_height == 6
    assert resolved.common_crs == "EPSG:32637"

    assert resolved.track_readiness.pixel_ready is True
    assert resolved.track_readiness.boundary_ready is True
    assert resolved.track_readiness.object_structure_ready is True

    assert resolved.output_contract.pixel_metrics == (
        "extent_iou",
        "extent_f1",
        "extent_precision",
        "extent_recall",
    )
    assert resolved.output_contract.boundary_metrics == (
        "boundary_f1",
        "boundary_precision",
        "boundary_recall",
        "bde",
    )
    assert resolved.output_contract.object_structure_metrics == ("goc", "guc", "gtc")

    assert set(resolved.source_run_ids) == {"predict_run_001", "postprocess_run_001"}


def test_resolve_evaluation_input_contract_fails_on_spatial_mismatch(tmp_path: Path) -> None:
    paths = _write_happy_paths(tmp_path)
    arr = _build_base_arrays(width=9)
    bad_pred_extent = _write_raster(
        tmp_path / "extent_prob_bad.tif",
        array=arr["pred_extent"],
    )

    with pytest.raises(SpatialContractError, match="Spatial mismatch"):
        resolve_evaluation_input_contract(
            gt_extent_path=paths["gt_extent_path"],
            gt_boundary_path=paths["gt_boundary_path"],
            gt_valid_path=paths["gt_valid_path"],
            pred_extent_prob_path=bad_pred_extent,
            pred_boundary_prob_path=paths["pred_boundary_prob_path"],
            pred_distance_pred_path=paths["pred_distance_pred_path"],
            pred_valid_path=paths["pred_valid_path"],
        )


def test_resolve_evaluation_input_contract_fails_on_semantic_mismatch(tmp_path: Path) -> None:
    paths = _write_happy_paths(tmp_path)
    bad_boundary = _build_base_arrays()["gt_boundary"].copy()
    bad_boundary[0, 2, 2] = 7
    bad_boundary_path = _write_raster(tmp_path / "gt_boundary_bad.tif", array=bad_boundary)

    with pytest.raises(ContractError, match="unsupported labels"):
        resolve_evaluation_input_contract(
            gt_extent_path=paths["gt_extent_path"],
            gt_boundary_path=bad_boundary_path,
            gt_valid_path=paths["gt_valid_path"],
            pred_extent_prob_path=paths["pred_extent_prob_path"],
            pred_boundary_prob_path=paths["pred_boundary_prob_path"],
            pred_distance_pred_path=paths["pred_distance_pred_path"],
            pred_valid_path=paths["pred_valid_path"],
        )


def test_resolve_evaluation_input_contract_partial_readiness_without_vectors(tmp_path: Path) -> None:
    paths = _write_happy_paths(tmp_path)

    resolved = resolve_evaluation_input_contract(
        gt_extent_path=paths["gt_extent_path"],
        gt_boundary_path=paths["gt_boundary_path"],
        gt_valid_path=paths["gt_valid_path"],
        pred_extent_prob_path=paths["pred_extent_prob_path"],
        pred_boundary_prob_path=paths["pred_boundary_prob_path"],
        pred_distance_pred_path=paths["pred_distance_pred_path"],
        pred_valid_path=paths["pred_valid_path"],
    )

    assert resolved.track_readiness.pixel_ready is True
    assert resolved.track_readiness.boundary_ready is True
    assert resolved.track_readiness.object_structure_ready is False
    assert resolved.track_readiness.object_structure_reason is not None
    assert "missing required vector sources" in resolved.track_readiness.object_structure_reason


def test_resolve_evaluation_input_contract_streamed_window_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_happy_paths(tmp_path)
    monkeypatch.setattr(input_contract_module, "_STREAM_WINDOW_SIZE", 2)

    resolved = resolve_evaluation_input_contract(
        gt_extent_path=paths["gt_extent_path"],
        gt_boundary_path=paths["gt_boundary_path"],
        gt_valid_path=paths["gt_valid_path"],
        pred_extent_prob_path=paths["pred_extent_prob_path"],
        pred_boundary_prob_path=paths["pred_boundary_prob_path"],
        pred_distance_pred_path=paths["pred_distance_pred_path"],
        pred_valid_path=paths["pred_valid_path"],
        gt_distance_path=paths["gt_distance_path"],
        post_parcel_instance_path=paths["post_parcel_instance_path"],
    )

    summary = resolved.semantic_compatibility_summary
    assert summary["gt_extent_unique_values"] == (0, 1, 255)
    assert summary["gt_boundary_unique_values"] == (0, 1, 2)
    assert summary["gt_valid_unique_values"] == (0, 1)
    assert summary["pred_valid_unique_values"] == (0, 1)
    assert summary["pred_extent_prob_range"] == pytest.approx((0.2, 0.85))
    assert summary["pred_distance_pred_range"] == pytest.approx((0.1, 1.2))
    assert summary["gt_distance_range"] == pytest.approx((0.0, 2.0))


def test_resolve_evaluation_input_contract_fails_on_missing_required_input(tmp_path: Path) -> None:
    paths = _write_happy_paths(tmp_path)

    with pytest.raises(ContractError, match="pred_valid_path does not exist"):
        resolve_evaluation_input_contract(
            gt_extent_path=paths["gt_extent_path"],
            gt_boundary_path=paths["gt_boundary_path"],
            gt_valid_path=paths["gt_valid_path"],
            pred_extent_prob_path=paths["pred_extent_prob_path"],
            pred_boundary_prob_path=paths["pred_boundary_prob_path"],
            pred_distance_pred_path=paths["pred_distance_pred_path"],
            pred_valid_path=tmp_path / "missing_valid.tif",
        )


def test_resolve_evaluation_input_contract_fails_on_malformed_manifest(tmp_path: Path) -> None:
    paths = _write_happy_paths(tmp_path)
    bad_manifest_path = _write_predict_manifest(
        tmp_path, paths, schema_name="wrong.schema_name"
    )

    with pytest.raises(ContractError, match="predict_manifest schema_name"):
        resolve_evaluation_input_contract(
            gt_extent_path=paths["gt_extent_path"],
            gt_boundary_path=paths["gt_boundary_path"],
            gt_valid_path=paths["gt_valid_path"],
            pred_extent_prob_path=paths["pred_extent_prob_path"],
            pred_boundary_prob_path=paths["pred_boundary_prob_path"],
            pred_distance_pred_path=paths["pred_distance_pred_path"],
            pred_valid_path=paths["pred_valid_path"],
            predict_manifest_path=bad_manifest_path,
        )


def test_resolve_evaluation_input_contract_allows_invalid_label_minus_one_in_post_instance(
    tmp_path: Path,
) -> None:
    paths = _write_happy_paths(tmp_path)
    arr = _build_base_arrays()
    arr["post_instance"][0, 0, 0] = -1
    arr["post_instance"][0, 0, 1] = -1
    paths["post_parcel_instance_path"] = _write_raster(
        tmp_path / "parcel_instance_with_invalid.tif",
        array=arr["post_instance"],
    )

    resolved = resolve_evaluation_input_contract(
        **paths,
    )
    unique_vals = resolved.semantic_compatibility_summary["post_parcel_instance_unique_values"]
    assert unique_vals is not None
    assert -1 in unique_vals


def test_resolve_evaluation_input_contract_fails_on_post_instance_labels_below_minus_one(
    tmp_path: Path,
) -> None:
    paths = _write_happy_paths(tmp_path)
    arr = _build_base_arrays()
    arr["post_instance"][0, 0, 0] = -2
    paths["post_parcel_instance_path"] = _write_raster(
        tmp_path / "parcel_instance_bad_negative.tif",
        array=arr["post_instance"],
    )

    with pytest.raises(ContractError, match="labels below -1"):
        resolve_evaluation_input_contract(
            **paths,
        )


def test_resolve_evaluation_input_contract_accepts_repo_relative_manifest_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workspace)

    paths = _write_happy_paths(workspace / "artifacts")

    manifest_dir = workspace / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    predict_manifest_path = manifest_dir / "predict_manifest.json"
    write_manifest(
        predict_manifest_path,
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
                "extent_prob": str(paths["pred_extent_prob_path"].relative_to(workspace)),
                "boundary_prob": str(paths["pred_boundary_prob_path"].relative_to(workspace)),
                "distance_pred": str(paths["pred_distance_pred_path"].relative_to(workspace)),
                "valid": str(paths["pred_valid_path"].relative_to(workspace)),
            },
        },
    )

    postprocess_manifest_path = manifest_dir / "postprocess_manifest.json"
    write_manifest(
        postprocess_manifest_path,
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
                "extent_prob_path": str(paths["pred_extent_prob_path"].relative_to(workspace)),
                "boundary_prob_path": str(paths["pred_boundary_prob_path"].relative_to(workspace)),
                "distance_pred_path": str(paths["pred_distance_pred_path"].relative_to(workspace)),
                "valid_path": str(paths["pred_valid_path"].relative_to(workspace)),
                "aoi_path": None,
            },
            "outputs": {
                "parcel_instance_path": str(paths["post_parcel_instance_path"].relative_to(workspace)),
                "parcels_gpkg_path": str(paths["post_parcels_gpkg_path"].relative_to(workspace)),
                "optional_exports": [],
            },
        },
    )

    resolved = resolve_evaluation_input_contract(
        **paths,
        predict_manifest_path=predict_manifest_path,
        postprocess_manifest_path=postprocess_manifest_path,
    )
    assert resolved.compatible is True


def test_resolve_evaluation_input_contract_accepts_unknown_vector_schema_when_features_polygonal(
    tmp_path: Path,
) -> None:
    paths = _write_happy_paths(tmp_path)
    paths["post_parcels_gpkg_path"] = _write_polygons(
        tmp_path / "parcels_unknown_schema.gpkg",
        include_confidence=True,
        schema_geometry="Unknown",
    )

    resolved = resolve_evaluation_input_contract(
        **paths,
    )
    assert resolved.compatible is True
