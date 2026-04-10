"""Threshold / marker generation layer for module_postprocess_vectorize.

Stage B scope (narrow and contract-first):
- consume validated postprocess input contract;
- build extent-core / low-boundary / high-distance support masks;
- build marker candidates mask:
    marker = extent_core ∩ low_boundary ∩ high_distance ∩ valid;
- expose explicit threshold provenance and readiness signal for Stage C.

Out of scope:
- watershed,
- instance labeling,
- polygonization,
- topology cleanup,
- vector export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError, ValidPolicyError
from ai_fields.module_postprocess_vectorize.input_contract import (
    PostprocessInputContractResult,
)


@dataclass(frozen=True)
class MarkerThresholdPolicy:
    """Explicit threshold policy for Stage B marker generation."""

    extent_core_min_prob: float
    boundary_low_max_prob: float
    distance_high_min_value: float
    threshold_provenance: str
    boundary_presence_mode: str = "skeleton_plus_buffer"


@dataclass(frozen=True)
class MarkerGenerationResult:
    """Computed support masks and marker candidates for Stage C."""

    extent_core_mask: np.ndarray
    low_boundary_mask: np.ndarray
    high_distance_mask: np.ndarray
    marker_candidates_mask: np.ndarray

    policy: dict[str, Any]
    diagnostics: dict[str, Any]
    ready_for_stage_c: bool


def _require_rasterio() -> tuple[Any, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required for module_postprocess_vectorize marker generation."
        ) from exc
    return rasterio, rasterio.errors


def _validate_policy(policy: MarkerThresholdPolicy) -> MarkerThresholdPolicy:
    if not isinstance(policy, MarkerThresholdPolicy):
        raise ContractError(
            "policy must be MarkerThresholdPolicy for marker generation Stage B."
        )

    if not (0.0 <= float(policy.extent_core_min_prob) <= 1.0):
        raise ContractError(
            "policy.extent_core_min_prob must be in [0,1]."
        )
    if not (0.0 <= float(policy.boundary_low_max_prob) <= 1.0):
        raise ContractError(
            "policy.boundary_low_max_prob must be in [0,1]."
        )
    if float(policy.distance_high_min_value) < 0.0:
        raise ContractError(
            "policy.distance_high_min_value must be >= 0 for unsigned distance contract."
        )
    if (
        not isinstance(policy.threshold_provenance, str)
        or policy.threshold_provenance.strip() == ""
    ):
        raise ContractError(
            "policy.threshold_provenance must be a non-empty string."
        )
    if policy.boundary_presence_mode not in {"skeleton_plus_buffer"}:
        raise ContractError(
            "Unsupported policy.boundary_presence_mode: "
            f"{policy.boundary_presence_mode!r}."
        )
    return policy


def _read_array(path: Path) -> np.ndarray:
    rasterio, rasterio_errors = _require_rasterio()
    try:
        with rasterio.open(path) as ds:
            return ds.read()
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read raster values: {path} ({exc})") from exc


def _read_single_band(
    path: Path,
    *,
    role: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    arr = _read_array(path)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ContractError(
            f"{role} must be a single-band raster, got shape={arr.shape}."
        )
    band = arr[0]
    if band.shape != expected_shape:
        raise ContractError(
            f"{role} shape mismatch: expected {expected_shape}, got {band.shape}."
        )
    if not np.isfinite(band).all():
        raise ContractError(f"{role} contains non-finite values.")
    return band


def _read_boundary_three_band(
    path: Path,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    arr = _read_array(path)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ContractError(
            f"boundary_prob must be a 3-band raster, got shape={arr.shape}."
        )
    if arr.shape[1:] != expected_shape:
        raise ContractError(
            "boundary_prob shape mismatch: "
            f"expected (3, {expected_shape[0]}, {expected_shape[1]}), got {arr.shape}."
        )
    if not np.isfinite(arr).all():
        raise ContractError("boundary_prob contains non-finite values.")
    return arr


def _read_valid_binary(
    path: Path,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    valid_raw = _read_single_band(path, role="valid", expected_shape=expected_shape)
    unique = np.unique(valid_raw)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValidPolicyError(
            "valid raster must be binary {0,1} for marker generation, "
            f"got unique values {unique.tolist()}."
        )
    return (valid_raw > 0).astype(np.uint8)


def build_marker_candidates(
    *,
    input_contract: PostprocessInputContractResult,
    policy: MarkerThresholdPolicy,
) -> MarkerGenerationResult:
    """Compute Stage B support masks and marker candidates.

    Contract equation:
      marker_candidates = extent_core ∩ low_boundary ∩ high_distance ∩ valid
    """
    if not isinstance(input_contract, PostprocessInputContractResult):
        raise ContractError(
            "input_contract must be PostprocessInputContractResult from Stage A."
        )

    _validate_policy(policy)

    h = int(input_contract.common_height)
    w = int(input_contract.common_width)
    expected_shape = (h, w)

    extent_prob = _read_single_band(
        input_contract.extent_prob.path,
        role="extent_prob",
        expected_shape=expected_shape,
    ).astype(np.float32)
    boundary_prob = _read_boundary_three_band(
        input_contract.boundary_prob.path,
        expected_shape=expected_shape,
    ).astype(np.float32)
    distance_pred = _read_single_band(
        input_contract.distance_pred.path,
        role="distance_pred",
        expected_shape=expected_shape,
    ).astype(np.float32)
    valid01 = _read_valid_binary(
        input_contract.valid.path,
        expected_shape=expected_shape,
    )

    if not np.any(valid01 == 1):
        raise ValidPolicyError(
            "valid mask contains zero valid pixels; cannot generate marker candidates."
        )

    extent_core = (extent_prob >= float(policy.extent_core_min_prob)) & (valid01 == 1)

    if policy.boundary_presence_mode == "skeleton_plus_buffer":
        boundary_presence = boundary_prob[1] + boundary_prob[2]
    else:  # pragma: no cover
        raise ContractError(
            f"Unsupported boundary_presence_mode: {policy.boundary_presence_mode!r}."
        )

    low_boundary = (boundary_presence <= float(policy.boundary_low_max_prob)) & (valid01 == 1)
    high_distance = (distance_pred >= float(policy.distance_high_min_value)) & (valid01 == 1)

    markers = extent_core & low_boundary & high_distance

    marker_pixels = int(markers.sum())
    if marker_pixels == 0:
        raise ContractError(
            "Marker generation produced zero candidates under current threshold policy."
        )

    valid_pixels = int(valid01.sum())

    diagnostics = {
        "shape": {"height": h, "width": w},
        "pixels": {
            "valid": valid_pixels,
            "extent_core": int(extent_core.sum()),
            "low_boundary": int(low_boundary.sum()),
            "high_distance": int(high_distance.sum()),
            "marker_candidates": marker_pixels,
        },
        "ratios_over_valid": {
            "extent_core": float(extent_core.sum() / valid_pixels),
            "low_boundary": float(low_boundary.sum() / valid_pixels),
            "high_distance": float(high_distance.sum() / valid_pixels),
            "marker_candidates": float(marker_pixels / valid_pixels),
        },
    }

    policy_info = {
        "threshold_provenance": policy.threshold_provenance,
        "extent_core_min_prob": float(policy.extent_core_min_prob),
        "boundary_low_max_prob": float(policy.boundary_low_max_prob),
        "distance_high_min_value": float(policy.distance_high_min_value),
        "boundary_presence_mode": policy.boundary_presence_mode,
        "formula": "extent_core ∩ low_boundary ∩ high_distance ∩ valid",
    }

    return MarkerGenerationResult(
        extent_core_mask=extent_core.astype(np.uint8),
        low_boundary_mask=low_boundary.astype(np.uint8),
        high_distance_mask=high_distance.astype(np.uint8),
        marker_candidates_mask=markers.astype(np.uint8),
        policy=policy_info,
        diagnostics=diagnostics,
        ready_for_stage_c=True,
    )


__all__ = [
    "MarkerGenerationResult",
    "MarkerThresholdPolicy",
    "build_marker_candidates",
]
