"""Unit tests for Stage E artifact export in module_postprocess_vectorize."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_postprocess_vectorize.export import export_postprocess_artifacts
from ai_fields.module_postprocess_vectorize.input_contract import (
    resolve_postprocess_input_contract,
)
from ai_fields.module_postprocess_vectorize.instance_core import (
    WatershedCorePolicy,
    build_parcel_instance_raster,
)
from ai_fields.module_postprocess_vectorize.marker_generation import (
    MarkerThresholdPolicy,
    build_marker_candidates,
)
from ai_fields.module_postprocess_vectorize.polygonization import (
    PolygonizationPolicy,
    build_postprocess_polygons,
)

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("scipy")
pytest.importorskip("fiona")
yaml = pytest.importorskip("yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _base_arrays(height: int = 5, width: int = 7) -> dict[str, np.ndarray]:
    extent = np.full((1, height, width), 0.9, dtype=np.float32)

    boundary_presence = np.full((height, width), 0.15, dtype=np.float32)
    boundary_presence[:, width // 2] = 0.8
    boundary = np.zeros((3, height, width), dtype=np.float32)
    boundary[1] = boundary_presence * 0.6
    boundary[2] = boundary_presence * 0.4
    boundary[0] = 1.0 - boundary_presence

    distance = np.ones((1, height, width), dtype=np.float32)
    distance[0, :, : width // 2] = 2.5

    valid = np.ones((1, height, width), dtype=np.uint8)
    return {
        "extent": extent,
        "boundary": boundary,
        "distance": distance,
        "valid": valid,
    }


def _build_chain(tmp_path: Path) -> dict[str, Any]:
    arrays = _base_arrays()
    extent_path = _write_raster(tmp_path / "extent_prob.tif", array=arrays["extent"])
    boundary_path = _write_raster(tmp_path / "boundary_prob.tif", array=arrays["boundary"])
    distance_path = _write_raster(tmp_path / "distance_pred.tif", array=arrays["distance"])
    valid_path = _write_raster(tmp_path / "valid.tif", array=arrays["valid"])

    input_contract = resolve_postprocess_input_contract(
        extent_prob_path=extent_path,
        boundary_prob_path=boundary_path,
        distance_pred_path=distance_path,
        valid_path=valid_path,
    )

    marker_policy = MarkerThresholdPolicy(
        extent_core_min_prob=0.7,
        boundary_low_max_prob=0.4,
        distance_high_min_value=1.0,
        threshold_provenance="validation_calibrated_baseline_v1",
    )
    marker_result = build_marker_candidates(
        input_contract=input_contract,
        policy=marker_policy,
    )

    watershed_policy = WatershedCorePolicy(
        extent_support_min_prob=0.5,
        threshold_provenance="validation_calibrated_baseline_v1",
        boundary_weight=2.0,
        extent_weight=1.0,
        distance_weight=0.5,
        boundary_barrier_max_prob=0.95,
    )
    instance_result = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=watershed_policy,
    )

    parcel_instance_path = _write_raster(
        tmp_path / "parcel_instance.tif",
        array=instance_result.parcel_instance.astype(np.int32)[None, :, :],
    )

    polygon_policy = PolygonizationPolicy(
        threshold_provenance="validation_calibrated_baseline_v1",
    )
    polygon_result = build_postprocess_polygons(
        input_contract=input_contract,
        instance_result=instance_result,
        output_gpkg_path=tmp_path / "parcels.gpkg",
        policy=polygon_policy,
    )

    return {
        "input_contract": input_contract,
        "marker_policy": marker_policy,
        "marker_result": marker_result,
        "watershed_policy": watershed_policy,
        "instance_result": instance_result,
        "polygon_policy": polygon_policy,
        "polygon_result": polygon_result,
        "parcel_instance_path": parcel_instance_path,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_export_postprocess_artifacts_happy_path(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)

    out_dir = tmp_path / "postprocess_run"
    artifacts = export_postprocess_artifacts(
        output_dir=out_dir,
        run_id="post_run_001",
        input_contract=chain["input_contract"],
        marker_result=chain["marker_result"],
        instance_result=chain["instance_result"],
        polygon_result=chain["polygon_result"],
        marker_policy=chain["marker_policy"],
        watershed_policy=chain["watershed_policy"],
        polygonization_policy=chain["polygon_policy"],
        parcel_instance_path=chain["parcel_instance_path"],
    )

    assert artifacts.postprocess_manifest_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.config_used_path.exists()


def test_manifest_and_summary_content_are_provenance_transparent(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    out_dir = tmp_path / "postprocess_run"

    artifacts = export_postprocess_artifacts(
        output_dir=out_dir,
        run_id="post_run_002",
        input_contract=chain["input_contract"],
        marker_result=chain["marker_result"],
        instance_result=chain["instance_result"],
        polygon_result=chain["polygon_result"],
        marker_policy=chain["marker_policy"],
        watershed_policy=chain["watershed_policy"],
        polygonization_policy=chain["polygon_policy"],
        parcel_instance_path=chain["parcel_instance_path"],
    )

    manifest = json.loads(artifacts.postprocess_manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_name"] == "postprocess_vectorize.postprocess_manifest"
    assert manifest["inputs"]["extent_prob_path"] == str(chain["input_contract"].extent_prob.path)
    assert manifest["outputs"]["parcels_gpkg_path"] == str(chain["polygon_result"].parcels_gpkg_path)
    assert manifest["outputs"]["parcel_instance_path"] == str(chain["parcel_instance_path"])
    assert manifest["stage_coverage"] == {
        "stage_a": True,
        "stage_b": True,
        "stage_c": True,
        "stage_d": True,
        "stage_e": True,
    }

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["schema_name"] == "postprocess_vectorize.summary"
    assert summary["status"] == "success"
    assert summary["parcel_count"] == int(chain["polygon_result"].polygon_count)
    assert summary["polygon_confidence_summary"]["count"] == int(chain["polygon_result"].polygon_count)
    assert 0.0 <= float(summary["polygon_confidence_summary"]["mean"]) <= 1.0


def test_config_used_contains_effective_policy_contract(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)

    artifacts = export_postprocess_artifacts(
        output_dir=tmp_path / "postprocess_run",
        run_id="post_run_003",
        input_contract=chain["input_contract"],
        marker_result=chain["marker_result"],
        instance_result=chain["instance_result"],
        polygon_result=chain["polygon_result"],
        marker_policy=chain["marker_policy"],
        watershed_policy=chain["watershed_policy"],
        polygonization_policy=chain["polygon_policy"],
        parcel_instance_path=chain["parcel_instance_path"],
    )

    payload = yaml.safe_load(artifacts.config_used_path.read_text(encoding="utf-8"))
    assert payload["module_name"] == "module_postprocess_vectorize"
    assert payload["effective_policy_contract"]["marker_policy"]["threshold_provenance"] == "validation_calibrated_baseline_v1"
    assert payload["effective_policy_contract"]["watershed_policy"]["threshold_provenance"] == "validation_calibrated_baseline_v1"
    assert payload["effective_policy_contract"]["polygonization_policy"]["confidence_policy_name"] == "rule_based_polygon_confidence_v1"


def test_missing_required_output_fails_explicitly(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    Path(chain["polygon_result"].parcels_gpkg_path).unlink()

    with pytest.raises(ContractError, match="parcels_gpkg_path does not exist"):
        export_postprocess_artifacts(
            output_dir=tmp_path / "postprocess_run",
            run_id="post_run_004",
            input_contract=chain["input_contract"],
            marker_result=chain["marker_result"],
            instance_result=chain["instance_result"],
            polygon_result=chain["polygon_result"],
            marker_policy=chain["marker_policy"],
            watershed_policy=chain["watershed_policy"],
            polygonization_policy=chain["polygon_policy"],
            parcel_instance_path=chain["parcel_instance_path"],
        )


def test_missing_required_input_valid_fails_explicitly(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    chain["input_contract"].valid.path.unlink()

    with pytest.raises(ContractError, match="Failed to read valid raster"):
        export_postprocess_artifacts(
            output_dir=tmp_path / "postprocess_run",
            run_id="post_run_005",
            input_contract=chain["input_contract"],
            marker_result=chain["marker_result"],
            instance_result=chain["instance_result"],
            polygon_result=chain["polygon_result"],
            marker_policy=chain["marker_policy"],
            watershed_policy=chain["watershed_policy"],
            polygonization_policy=chain["polygon_policy"],
            parcel_instance_path=chain["parcel_instance_path"],
        )
