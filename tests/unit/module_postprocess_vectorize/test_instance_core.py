"""Unit tests for Stage C instance raster core in module_postprocess_vectorize."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_postprocess_vectorize.input_contract import (
    resolve_postprocess_input_contract,
)
from ai_fields.module_postprocess_vectorize.instance_core import (
    WatershedCorePolicy,
    build_parcel_instance_raster,
)
from ai_fields.module_postprocess_vectorize.marker_generation import (
    MarkerGenerationResult,
    MarkerThresholdPolicy,
    build_marker_candidates,
)

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("scipy")


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


def _write_postprocess_inputs(tmp_path: Path, arrays: dict[str, np.ndarray]) -> Any:
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


def _base_arrays_chain() -> dict[str, np.ndarray]:
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


def _arrays_for_boundary_split() -> dict[str, np.ndarray]:
    h, w = 5, 7
    extent = np.full((1, h, w), 0.9, dtype=np.float32)

    boundary_presence = np.full((h, w), 0.05, dtype=np.float32)
    boundary_presence[:, 1] = 0.95  # strong barrier near left side

    boundary = np.zeros((3, h, w), dtype=np.float32)
    boundary[1] = boundary_presence * 0.6
    boundary[2] = boundary_presence * 0.4
    boundary[0] = 1.0 - boundary_presence

    distance = np.ones((1, h, w), dtype=np.float32)
    valid = np.ones((1, h, w), dtype=np.uint8)

    return {
        "extent": extent,
        "boundary": boundary,
        "distance": distance,
        "valid": valid,
    }


def _manual_marker_result(mask: np.ndarray) -> MarkerGenerationResult:
    return MarkerGenerationResult(
        extent_core_mask=(mask > 0).astype(np.uint8),
        low_boundary_mask=np.ones_like(mask, dtype=np.uint8),
        high_distance_mask=np.ones_like(mask, dtype=np.uint8),
        marker_candidates_mask=(mask > 0).astype(np.uint8),
        policy={"threshold_provenance": "unit_test"},
        diagnostics={},
        ready_for_stage_c=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_parcel_instance_raster_happy_path_via_stage_a_b(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _base_arrays_chain())

    marker_policy = MarkerThresholdPolicy(
        extent_core_min_prob=0.7,
        boundary_low_max_prob=0.36,
        distance_high_min_value=2.0,
        threshold_provenance="validation_calibrated_baseline_v1",
    )
    marker_result = build_marker_candidates(
        input_contract=input_contract,
        policy=marker_policy,
    )

    watershed_policy = WatershedCorePolicy(
        extent_support_min_prob=0.5,
        threshold_provenance="validation_calibrated_baseline_v1",
        boundary_weight=1.0,
        extent_weight=0.5,
        distance_weight=0.5,
        connectivity=1,
    )
    result = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=watershed_policy,
    )

    assert result.ready_for_stage_d is True
    assert result.shape == (3, 4)
    assert result.dtype == "int32"
    assert result.parcel_instance.shape == (3, 4)
    assert result.instance_count >= 1
    assert result.background_label == 0

    assert result.invalid_label == -1
    unique = np.unique(result.parcel_instance)
    # invalid pixels use -1; valid pixels are 0 (background) or 1..N (instances)
    assert unique.min() >= -1
    positive = sorted(int(v) for v in unique if int(v) > 0)
    assert positive == list(range(1, result.instance_count + 1))


def test_marker_awareness_one_marker_vs_two_markers(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _arrays_for_boundary_split())

    one = np.zeros((5, 7), dtype=np.uint8)
    one[2, 0] = 1

    two = np.zeros((5, 7), dtype=np.uint8)
    two[2, 0] = 1
    two[2, 6] = 1

    policy = WatershedCorePolicy(
        extent_support_min_prob=0.5,
        threshold_provenance="validation_calibrated_baseline_v1",
        boundary_weight=1.0,
        extent_weight=0.5,
        distance_weight=0.5,
    )

    res_one = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=_manual_marker_result(one),
        policy=policy,
    )
    res_two = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=_manual_marker_result(two),
        policy=policy,
    )

    assert res_one.instance_count == 1
    assert res_two.instance_count == 2


def test_valid_awareness_invalid_pixels_not_labeled(tmp_path: Path) -> None:
    arrays = _arrays_for_boundary_split()
    arrays["valid"][0, 0, 0] = 0
    arrays["valid"][0, 1, 0] = 0

    input_contract = _write_postprocess_inputs(tmp_path, arrays)

    markers = np.zeros((5, 7), dtype=np.uint8)
    markers[2, 0] = 1
    markers[2, 6] = 1

    policy = WatershedCorePolicy(
        extent_support_min_prob=0.5,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    result = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=_manual_marker_result(markers),
        policy=policy,
    )

    # Invalid pixels get invalid_label (-1), not background (0)
    assert result.parcel_instance[0, 0] == result.invalid_label
    assert result.parcel_instance[1, 0] == result.invalid_label


def test_boundary_cue_changes_split_behavior(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _arrays_for_boundary_split())

    markers = np.zeros((5, 7), dtype=np.uint8)
    markers[2, 0] = 1
    markers[2, 6] = 1
    marker_result = _manual_marker_result(markers)

    policy_no_boundary = WatershedCorePolicy(
        extent_support_min_prob=0.5,
        threshold_provenance="validation_calibrated_baseline_v1",
        boundary_weight=0.0,
        extent_weight=1.0,
        distance_weight=0.0,
        boundary_barrier_max_prob=1.0,
    )
    policy_with_boundary = WatershedCorePolicy(
        extent_support_min_prob=0.5,
        threshold_provenance="validation_calibrated_baseline_v1",
        boundary_weight=6.0,
        extent_weight=1.0,
        distance_weight=0.0,
        boundary_barrier_max_prob=0.5,
    )

    no_b = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=policy_no_boundary,
    )
    with_b = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=policy_with_boundary,
    )

    assert no_b.instance_count == 2
    assert with_b.instance_count == 2
    assert not np.array_equal(no_b.parcel_instance, with_b.parcel_instance)


def test_determinism_same_input_same_output(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _arrays_for_boundary_split())

    markers = np.zeros((5, 7), dtype=np.uint8)
    markers[2, 0] = 1
    markers[2, 6] = 1
    marker_result = _manual_marker_result(markers)

    policy = WatershedCorePolicy(
        extent_support_min_prob=0.5,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    res1 = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=policy,
    )
    res2 = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=policy,
    )

    assert res1.instance_count == res2.instance_count
    assert np.array_equal(res1.parcel_instance, res2.parcel_instance)


def test_zero_markers_fails_explicitly(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _arrays_for_boundary_split())

    zero = np.zeros((5, 7), dtype=np.uint8)

    policy = WatershedCorePolicy(
        extent_support_min_prob=0.5,
        threshold_provenance="validation_calibrated_baseline_v1",
    )

    with pytest.raises(ContractError, match="zero marker pixels"):
        build_parcel_instance_raster(
            input_contract=input_contract,
            marker_result=_manual_marker_result(zero),
            policy=policy,
        )
