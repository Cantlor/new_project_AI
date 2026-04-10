"""Unit tests for module_postprocess_vectorize input contract layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    SpatialContractError,
    ValidPolicyError,
)
from ai_fields.common.manifests import write_manifest
from ai_fields.module_postprocess_vectorize.input_contract import (
    resolve_postprocess_input_contract,
)

rasterio = pytest.importorskip("rasterio")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_raster(
    path: Path,
    *,
    array: np.ndarray,
    crs: str = "EPSG:32637",
    transform: Any | None = None,
    nodata: float | int | None = None,
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
        nodata=nodata,
    ) as ds:
        ds.write(array)
    return path


def _base_predict_like_inputs(
    *,
    height: int = 6,
    width: int = 8,
) -> dict[str, np.ndarray]:
    extent = np.full((1, height, width), 0.8, dtype=np.float32)

    boundary = np.zeros((3, height, width), dtype=np.float32)
    boundary[0, :, :] = 0.2
    boundary[1, :, :] = 0.5
    boundary[2, :, :] = 0.3

    distance = np.full((1, height, width), 3.5, dtype=np.float32)

    valid = np.ones((1, height, width), dtype=np.uint8)
    valid[0, 0, 0] = 0
    valid[0, 1, 1] = 0

    return {
        "extent": extent,
        "boundary": boundary,
        "distance": distance,
        "valid": valid,
    }


def _write_predict_like_artifacts(tmp_path: Path) -> dict[str, Path]:
    arrays = _base_predict_like_inputs()

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolve_postprocess_input_contract_happy_path(tmp_path: Path) -> None:
    paths = _write_predict_like_artifacts(tmp_path)

    resolved = resolve_postprocess_input_contract(**paths)

    assert resolved.compatible is True
    assert resolved.common_width == 8
    assert resolved.common_height == 6
    assert resolved.common_crs == "EPSG:32637"

    assert resolved.extent_prob.band_count == 1
    assert resolved.boundary_prob.band_count == 3
    assert resolved.distance_pred.band_count == 1
    assert resolved.valid.band_count == 1

    assert resolved.extent_value_range == pytest.approx((0.8, 0.8))
    assert resolved.distance_value_range == pytest.approx((3.5, 3.5))
    assert resolved.valid_unique_values == (0, 1)

    # Output contract skeleton must be fixed explicitly at Stage A.
    assert resolved.output_contract.parcel_instance_raster == "parcel_instance.tif"
    assert resolved.output_contract.parcels_vector == "parcels.gpkg"
    assert resolved.output_contract.required_polygon_attributes == ("polygon_confidence",)


def test_resolve_postprocess_input_contract_accepts_predict_manifest_when_consistent(
    tmp_path: Path,
) -> None:
    paths = _write_predict_like_artifacts(tmp_path)
    predict_manifest_path = _write_predict_manifest(tmp_path, paths)

    resolved = resolve_postprocess_input_contract(
        **paths,
        predict_manifest_path=predict_manifest_path,
    )

    assert resolved.predict_manifest_path == predict_manifest_path


def test_resolve_postprocess_input_contract_accepts_repo_relative_manifest_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate real predict manifests that store cwd-relative output paths
    # like "runs/module_target_predict/<run_id>/extent_prob.tif".
    monkeypatch.chdir(tmp_path)
    predict_run_dir = tmp_path / "runs" / "module_target_predict" / "predict_run_001"
    paths = _write_predict_like_artifacts(predict_run_dir)

    manifest_path = predict_run_dir / "predict_manifest.json"
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
            "created_at_utc": "2026-04-08T00:00:00Z",
            "status": "success",
            "output_paths": {
                "extent_prob": str(paths["extent_prob_path"].relative_to(tmp_path)),
                "boundary_prob": str(paths["boundary_prob_path"].relative_to(tmp_path)),
                "distance_pred": str(paths["distance_pred_path"].relative_to(tmp_path)),
                "valid": str(paths["valid_path"].relative_to(tmp_path)),
            },
        },
    )

    resolved = resolve_postprocess_input_contract(
        **paths,
        predict_manifest_path=manifest_path,
    )
    assert resolved.predict_manifest_path == manifest_path


def test_resolve_postprocess_input_contract_fails_on_spatial_mismatch(tmp_path: Path) -> None:
    arrays = _base_predict_like_inputs()

    extent_path = _write_raster(tmp_path / "extent_prob.tif", array=arrays["extent"])
    # Different width to trigger explicit spatial mismatch.
    boundary_bad = np.zeros((3, 6, 9), dtype=np.float32)
    boundary_bad[0, :, :] = 0.2
    boundary_bad[1, :, :] = 0.5
    boundary_bad[2, :, :] = 0.3
    boundary_path = _write_raster(tmp_path / "boundary_prob.tif", array=boundary_bad)
    distance_path = _write_raster(tmp_path / "distance_pred.tif", array=arrays["distance"])
    valid_path = _write_raster(tmp_path / "valid.tif", array=arrays["valid"])

    with pytest.raises(SpatialContractError, match="Spatial mismatch"):
        resolve_postprocess_input_contract(
            extent_prob_path=extent_path,
            boundary_prob_path=boundary_path,
            distance_pred_path=distance_path,
            valid_path=valid_path,
        )


def test_resolve_postprocess_input_contract_fails_on_boundary_band_count(tmp_path: Path) -> None:
    arrays = _base_predict_like_inputs()

    extent_path = _write_raster(tmp_path / "extent_prob.tif", array=arrays["extent"])
    boundary_path = _write_raster(
        tmp_path / "boundary_prob.tif",
        array=arrays["boundary"][:1, :, :],  # malformed: 1 band instead of 3
    )
    distance_path = _write_raster(tmp_path / "distance_pred.tif", array=arrays["distance"])
    valid_path = _write_raster(tmp_path / "valid.tif", array=arrays["valid"])

    with pytest.raises(ChannelCountError, match="boundary_prob must have exactly 3 band"):
        resolve_postprocess_input_contract(
            extent_prob_path=extent_path,
            boundary_prob_path=boundary_path,
            distance_pred_path=distance_path,
            valid_path=valid_path,
        )


def test_resolve_postprocess_input_contract_fails_on_non_binary_valid(tmp_path: Path) -> None:
    arrays = _base_predict_like_inputs()
    bad_valid = arrays["valid"].copy()
    bad_valid[0, 2, 2] = 255

    extent_path = _write_raster(tmp_path / "extent_prob.tif", array=arrays["extent"])
    boundary_path = _write_raster(tmp_path / "boundary_prob.tif", array=arrays["boundary"])
    distance_path = _write_raster(tmp_path / "distance_pred.tif", array=arrays["distance"])
    valid_path = _write_raster(tmp_path / "valid.tif", array=bad_valid)

    with pytest.raises(ValidPolicyError, match="valid raster must be binary"):
        resolve_postprocess_input_contract(
            extent_prob_path=extent_path,
            boundary_prob_path=boundary_path,
            distance_pred_path=distance_path,
            valid_path=valid_path,
        )


def test_resolve_postprocess_input_contract_fails_on_invalid_value_domains(tmp_path: Path) -> None:
    arrays = _base_predict_like_inputs()
    bad_extent = arrays["extent"].copy()
    bad_extent[0, 0, 0] = 1.2

    extent_path = _write_raster(tmp_path / "extent_prob.tif", array=bad_extent)
    boundary_path = _write_raster(tmp_path / "boundary_prob.tif", array=arrays["boundary"])
    distance_path = _write_raster(tmp_path / "distance_pred.tif", array=arrays["distance"])
    valid_path = _write_raster(tmp_path / "valid.tif", array=arrays["valid"])

    with pytest.raises(ContractError, match="extent_prob must be probability-like"):
        resolve_postprocess_input_contract(
            extent_prob_path=extent_path,
            boundary_prob_path=boundary_path,
            distance_pred_path=distance_path,
            valid_path=valid_path,
        )


def test_resolve_postprocess_input_contract_fails_on_malformed_boundary_probabilities(
    tmp_path: Path,
) -> None:
    arrays = _base_predict_like_inputs()
    bad_boundary = np.full((3, 6, 8), 0.6, dtype=np.float32)  # sum=1.8 on valid pixels

    extent_path = _write_raster(tmp_path / "extent_prob.tif", array=arrays["extent"])
    boundary_path = _write_raster(tmp_path / "boundary_prob.tif", array=bad_boundary)
    distance_path = _write_raster(tmp_path / "distance_pred.tif", array=arrays["distance"])
    valid_path = _write_raster(tmp_path / "valid.tif", array=arrays["valid"])

    with pytest.raises(ContractError, match="probability simplex"):
        resolve_postprocess_input_contract(
            extent_prob_path=extent_path,
            boundary_prob_path=boundary_path,
            distance_pred_path=distance_path,
            valid_path=valid_path,
        )


def test_resolve_postprocess_input_contract_fails_on_missing_required_input(tmp_path: Path) -> None:
    paths = _write_predict_like_artifacts(tmp_path)
    missing_extent = tmp_path / "missing_extent_prob.tif"

    with pytest.raises(ContractError, match="extent_prob_path does not exist"):
        resolve_postprocess_input_contract(
            extent_prob_path=missing_extent,
            boundary_prob_path=paths["boundary_prob_path"],
            distance_pred_path=paths["distance_pred_path"],
            valid_path=paths["valid_path"],
        )


def test_resolve_postprocess_input_contract_fails_on_predict_manifest_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_predict_like_artifacts(tmp_path)
    predict_manifest_path = _write_predict_manifest(tmp_path, paths)

    # Point one path to a different file than predict_manifest.output_paths.
    another_extent = _write_raster(
        tmp_path / "another_extent_prob.tif",
        array=_base_predict_like_inputs()["extent"],
    )

    with pytest.raises(ContractError, match="predict_manifest output path mismatch"):
        resolve_postprocess_input_contract(
            extent_prob_path=another_extent,
            boundary_prob_path=paths["boundary_prob_path"],
            distance_pred_path=paths["distance_pred_path"],
            valid_path=paths["valid_path"],
            predict_manifest_path=predict_manifest_path,
        )
