"""Unit tests for run-level postprocess orchestration (Stage A->E)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import write_manifest
from ai_fields.module_postprocess_vectorize.marker_generation import (
    MarkerThresholdPolicy,
)
from ai_fields.module_postprocess_vectorize.polygonization import PolygonizationPolicy
from ai_fields.module_postprocess_vectorize.run_postprocess import (
    PostprocessRunPolicies,
    run_postprocess_for_scene,
)
from ai_fields.module_postprocess_vectorize.instance_core import WatershedCorePolicy

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


def _write_predict_like_inputs(tmp_path: Path) -> dict[str, Path]:
    arrays = _base_arrays()
    extent_path = _write_raster(tmp_path / "extent_prob.tif", array=arrays["extent"])
    boundary_path = _write_raster(tmp_path / "boundary_prob.tif", array=arrays["boundary"])
    distance_path = _write_raster(tmp_path / "distance_pred.tif", array=arrays["distance"])
    valid_path = _write_raster(tmp_path / "valid.tif", array=arrays["valid"])
    return {
        "extent_prob_path": extent_path,
        "boundary_prob_path": boundary_path,
        "distance_pred_path": distance_path,
        "valid_path": valid_path,
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
                "extent_prob": str(paths["extent_prob_path"]),
                "boundary_prob": str(paths["boundary_prob_path"]),
                "distance_pred": str(paths["distance_pred_path"]),
                "valid": str(paths["valid_path"]),
            },
        },
    )
    return manifest_path


def _baseline_policies() -> PostprocessRunPolicies:
    return PostprocessRunPolicies(
        marker_policy=MarkerThresholdPolicy(
            extent_core_min_prob=0.7,
            boundary_low_max_prob=0.4,
            distance_high_min_value=1.0,
            threshold_provenance="validation_calibrated_baseline_v1",
        ),
        watershed_policy=WatershedCorePolicy(
            extent_support_min_prob=0.5,
            threshold_provenance="validation_calibrated_baseline_v1",
            boundary_weight=2.0,
            extent_weight=1.0,
            distance_weight=0.5,
            boundary_barrier_max_prob=0.95,
        ),
        polygonization_policy=PolygonizationPolicy(
            threshold_provenance="validation_calibrated_baseline_v1",
            min_polygon_area_m2=0.0,
        ),
    )


def test_run_postprocess_for_scene_happy_path(tmp_path: Path) -> None:
    paths = _write_predict_like_inputs(tmp_path)
    predict_manifest_path = _write_predict_manifest(tmp_path, paths)

    out_root = tmp_path / "postprocess_runs"
    result = run_postprocess_for_scene(
        **paths,
        output_dir=out_root,
        policies=_baseline_policies(),
        run_id="post_scene_001",
        source_predict_manifest_path=predict_manifest_path,
    )

    assert result.success is True
    assert result.run_dir == out_root / "post_scene_001"
    assert result.parcel_instance_path == result.run_dir / "parcel_instance.tif"
    assert result.parcels_gpkg_path == result.run_dir / "parcels.gpkg"
    assert result.postprocess_manifest_path == result.run_dir / "postprocess_manifest.json"
    assert result.summary_path == result.run_dir / "summary.json"
    assert result.config_used_path == result.run_dir / "config_used.yaml"

    for path in (
        result.parcel_instance_path,
        result.parcels_gpkg_path,
        result.postprocess_manifest_path,
        result.summary_path,
        result.config_used_path,
    ):
        assert path.exists()

    manifest = json.loads(result.postprocess_manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_predict_manifest_path"] == str(predict_manifest_path)
    assert manifest["source_predict_run_id"] == "predict_run_001"
    assert manifest["outputs"]["parcel_instance_path"] == str(result.parcel_instance_path)
    assert manifest["outputs"]["parcels_gpkg_path"] == str(result.parcels_gpkg_path)


def test_run_postprocess_fail_fast_stage_a(tmp_path: Path) -> None:
    paths = _write_predict_like_inputs(tmp_path)
    out_root = tmp_path / "postprocess_runs"

    with pytest.raises(ContractError, match="Stage A"):
        run_postprocess_for_scene(
            extent_prob_path=paths["extent_prob_path"],
            boundary_prob_path=tmp_path / "missing_boundary.tif",
            distance_pred_path=paths["distance_pred_path"],
            valid_path=paths["valid_path"],
            output_dir=out_root,
            run_id="fail_stage_a",
            policies=_baseline_policies(),
        )

    assert not (out_root / "fail_stage_a" / "postprocess_manifest.json").exists()


def test_run_postprocess_fail_fast_stage_b_zero_markers(tmp_path: Path) -> None:
    paths = _write_predict_like_inputs(tmp_path)
    out_root = tmp_path / "postprocess_runs"

    strict_policies = PostprocessRunPolicies(
        marker_policy=MarkerThresholdPolicy(
            extent_core_min_prob=0.99,
            boundary_low_max_prob=0.01,
            distance_high_min_value=10.0,
            threshold_provenance="validation_calibrated_baseline_v1",
        ),
        watershed_policy=_baseline_policies().watershed_policy,
        polygonization_policy=_baseline_policies().polygonization_policy,
    )

    with pytest.raises(ContractError, match="Stage B"):
        run_postprocess_for_scene(
            **paths,
            output_dir=out_root,
            run_id="fail_stage_b",
            policies=strict_policies,
        )

    run_dir = out_root / "fail_stage_b"
    assert not (run_dir / "parcels.gpkg").exists()
    assert not (run_dir / "postprocess_manifest.json").exists()


def test_run_postprocess_fail_fast_stage_d_cleanup_drops_all(tmp_path: Path) -> None:
    paths = _write_predict_like_inputs(tmp_path)
    out_root = tmp_path / "postprocess_runs"

    bad_polygon_policy = PolygonizationPolicy(
        threshold_provenance="validation_calibrated_baseline_v1",
        min_polygon_area_m2=10_000.0,
    )
    policies = PostprocessRunPolicies(
        marker_policy=_baseline_policies().marker_policy,
        watershed_policy=_baseline_policies().watershed_policy,
        polygonization_policy=bad_polygon_policy,
    )

    with pytest.raises(ContractError, match="Stage D"):
        run_postprocess_for_scene(
            **paths,
            output_dir=out_root,
            run_id="fail_stage_d",
            policies=policies,
        )

    run_dir = out_root / "fail_stage_d"
    assert (run_dir / "parcel_instance.tif").exists()
    assert not (run_dir / "postprocess_manifest.json").exists()


def test_aoi_path_provenance_is_transparent_and_not_claimed_as_applied(tmp_path: Path) -> None:
    """AOI stored but not applied must NOT claim suppression was executed in manifest."""
    paths = _write_predict_like_inputs(tmp_path)

    # Write a trivial AOI file (just needs to exist as a file).
    aoi_path = tmp_path / "aoi.gpkg"
    aoi_path.write_bytes(b"")

    result = run_postprocess_for_scene(
        **paths,
        output_dir=tmp_path / "out",
        run_id="aoi_test",
        policies=_baseline_policies(),
        aoi_path=aoi_path,
    )

    manifest = json.loads(result.postprocess_manifest_path.read_text(encoding="utf-8"))
    aoi_policy = manifest["resolved_policy"]["aoi_policy"]

    # Must record the path (provenance).
    assert aoi_policy is not None
    assert str(aoi_path) in aoi_policy["aoi_path"]

    # Must NOT claim suppression was applied (provenance transparency rule).
    mode = aoi_policy["mode"]
    assert "not_applied" in mode, (
        f"manifest aoi_policy.mode must explicitly state suppression was not applied, got {mode!r}."
    )


def test_run_postprocess_stable_layout_across_runs(tmp_path: Path) -> None:
    paths = _write_predict_like_inputs(tmp_path)
    out_root = tmp_path / "postprocess_runs"
    policies = _baseline_policies()

    r1 = run_postprocess_for_scene(
        **paths,
        output_dir=out_root,
        run_id="scene_a",
        policies=policies,
    )
    r2 = run_postprocess_for_scene(
        **paths,
        output_dir=out_root,
        run_id="scene_b",
        policies=policies,
    )

    expected_names = {
        "parcel_instance.tif",
        "parcels.gpkg",
        "postprocess_manifest.json",
        "summary.json",
        "config_used.yaml",
    }
    assert expected_names.issubset({p.name for p in r1.run_dir.iterdir()})
    assert expected_names.issubset({p.name for p in r2.run_dir.iterdir()})

