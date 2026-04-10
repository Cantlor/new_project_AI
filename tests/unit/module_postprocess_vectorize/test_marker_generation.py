"""Unit tests for Stage B marker generation in module_postprocess_vectorize."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.errors import ContractError, ValidPolicyError
from ai_fields.module_postprocess_vectorize.input_contract import (
    resolve_postprocess_input_contract,
)
from ai_fields.module_postprocess_vectorize.marker_generation import (
    MarkerThresholdPolicy,
    build_marker_candidates,
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


def _make_base_arrays() -> dict[str, np.ndarray]:
    extent = np.array(
        [
            [
                [0.8, 0.6, 0.9, 0.4],
                [0.75, 0.2, 0.7, 0.95],
                [0.1, 0.85, 0.65, 0.72],
            ]
        ],
        dtype=np.float32,
    )

    boundary_presence = np.array(
        [
            [0.2, 0.1, 0.5, 0.2],
            [0.3, 0.2, 0.35, 0.6],
            [0.1, 0.4, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    boundary = np.zeros((3, 3, 4), dtype=np.float32)
    boundary[1] = boundary_presence * 0.6
    boundary[2] = boundary_presence * 0.4
    boundary[0] = 1.0 - boundary_presence

    distance = np.array(
        [
            [
                [3.0, 1.0, 2.5, 4.0],
                [2.1, 0.5, 3.2, 1.0],
                [0.0, 2.2, 2.8, 1.9],
            ]
        ],
        dtype=np.float32,
    )

    valid = np.ones((1, 3, 4), dtype=np.uint8)
    valid[0, 0, 2] = 0

    return {
        "extent": extent,
        "boundary": boundary,
        "distance": distance,
        "valid": valid,
    }


def _build_input_contract(tmp_path: Path) -> Any:
    arrays = _make_base_arrays()
    extent_path = _write_raster(tmp_path / "extent_prob.tif", array=arrays["extent"])
    boundary_path = _write_raster(tmp_path / "boundary_prob.tif", array=arrays["boundary"])
    distance_path = _write_raster(tmp_path / "distance_pred.tif", array=arrays["distance"])
    valid_path = _write_raster(tmp_path / "valid.tif", array=arrays["valid"])

    return resolve_postprocess_input_contract(
        extent_prob_path=extent_path,
        boundary_prob_path=boundary_path,
        distance_pred_path=distance_path,
        valid_path=valid_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_marker_candidates_happy_path(tmp_path: Path) -> None:
    input_contract = _build_input_contract(tmp_path)
    policy = MarkerThresholdPolicy(
        extent_core_min_prob=0.7,
        boundary_low_max_prob=0.36,
        distance_high_min_value=2.0,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    result = build_marker_candidates(input_contract=input_contract, policy=policy)

    assert result.ready_for_stage_c is True
    assert result.extent_core_mask.shape == (3, 4)
    assert result.low_boundary_mask.shape == (3, 4)
    assert result.high_distance_mask.shape == (3, 4)
    assert result.marker_candidates_mask.shape == (3, 4)

    # Deterministic synthetic expectation (see fixture values).
    assert int(result.marker_candidates_mask.sum()) == 3
    assert result.policy["formula"] == "extent_core ∩ low_boundary ∩ high_distance ∩ valid"
    assert result.policy["threshold_provenance"] == "validation_calibrated_baseline_v1"


def test_build_marker_candidates_respects_valid_mask(tmp_path: Path) -> None:
    input_contract = _build_input_contract(tmp_path)
    policy = MarkerThresholdPolicy(
        extent_core_min_prob=0.7,
        boundary_low_max_prob=0.6,
        distance_high_min_value=2.0,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    result = build_marker_candidates(input_contract=input_contract, policy=policy)

    # Pixel [0,2] would satisfy extent+distance but is invalid by valid-mask.
    assert result.marker_candidates_mask[0, 2] == 0


def test_thresholds_change_marker_density_deterministically(tmp_path: Path) -> None:
    input_contract = _build_input_contract(tmp_path)

    permissive = MarkerThresholdPolicy(
        extent_core_min_prob=0.6,
        boundary_low_max_prob=0.4,
        distance_high_min_value=1.5,
        threshold_provenance="validation_calibrated_baseline_v1",
    )
    conservative = MarkerThresholdPolicy(
        extent_core_min_prob=0.75,
        boundary_low_max_prob=0.3,
        distance_high_min_value=2.0,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    res_perm = build_marker_candidates(input_contract=input_contract, policy=permissive)
    res_cons = build_marker_candidates(input_contract=input_contract, policy=conservative)

    assert int(res_perm.marker_candidates_mask.sum()) > int(res_cons.marker_candidates_mask.sum())


def test_invalid_threshold_policy_raises_explicit_error(tmp_path: Path) -> None:
    input_contract = _build_input_contract(tmp_path)
    bad_policy = MarkerThresholdPolicy(
        extent_core_min_prob=0.7,
        boundary_low_max_prob=1.2,
        distance_high_min_value=2.0,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    with pytest.raises(ContractError, match=r"boundary_low_max_prob must be in \[0,1\]"):
        build_marker_candidates(input_contract=input_contract, policy=bad_policy)


def test_zero_marker_candidates_raises_explicit_error(tmp_path: Path) -> None:
    input_contract = _build_input_contract(tmp_path)
    policy = MarkerThresholdPolicy(
        extent_core_min_prob=0.99,
        boundary_low_max_prob=0.01,
        distance_high_min_value=10.0,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    with pytest.raises(ContractError, match="zero candidates"):
        build_marker_candidates(input_contract=input_contract, policy=policy)


def test_missing_input_after_contract_resolution_raises_explicit_error(tmp_path: Path) -> None:
    input_contract = _build_input_contract(tmp_path)
    policy = MarkerThresholdPolicy(
        extent_core_min_prob=0.7,
        boundary_low_max_prob=0.35,
        distance_high_min_value=2.0,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    # Simulate runtime artifact corruption after Stage A resolution.
    input_contract.distance_pred.path.unlink()

    with pytest.raises(ContractError, match="Failed to read raster values"):
        build_marker_candidates(input_contract=input_contract, policy=policy)


def test_zero_valid_pixels_raises_valid_policy_error(tmp_path: Path) -> None:
    arrays = _make_base_arrays()
    arrays["valid"][:] = 0

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

    policy = MarkerThresholdPolicy(
        extent_core_min_prob=0.7,
        boundary_low_max_prob=0.35,
        distance_high_min_value=2.0,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    with pytest.raises(ValidPolicyError, match="zero valid pixels"):
        build_marker_candidates(input_contract=input_contract, policy=policy)
