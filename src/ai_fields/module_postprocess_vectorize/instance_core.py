"""Minimal marker-controlled watershed / instance raster core (Stage C).

This layer consumes:
- Stage A validated input contract (predict-side rasters);
- Stage B marker candidates;
and produces a baseline parcel-instance raster suitable for Stage D polygonization.

Scope is intentionally narrow:
- no polygonization,
- no topology cleanup,
- no polygon_confidence,
- no eval integration.
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
from ai_fields.module_postprocess_vectorize.marker_generation import (
    MarkerGenerationResult,
)

_EPS = 1e-9


@dataclass(frozen=True)
class WatershedCorePolicy:
    """Explicit policy for Stage C marker-controlled watershed core."""

    extent_support_min_prob: float
    threshold_provenance: str

    boundary_weight: float = 1.0
    extent_weight: float = 0.5
    distance_weight: float = 0.5
    boundary_presence_mode: str = "skeleton_plus_buffer"
    boundary_barrier_max_prob: float = 1.0

    connectivity: int = 1
    relabel_compact: bool = True
    background_label: int = 0
    invalid_label: int = -1
    min_region_pixels: int = 0
    """Minimum size (pixels) for a parcel instance to be retained after watershed.
    Instances smaller than this threshold are merged into background.  0 = no filtering."""


@dataclass(frozen=True)
class ParcelInstanceRasterResult:
    """Result of Stage C instance raster generation."""

    parcel_instance: np.ndarray
    shape: tuple[int, int]
    dtype: str

    instance_count: int
    background_label: int
    invalid_label: int

    valid_pixels: int
    invalid_pixels: int
    domain_pixels: int
    labeled_valid_pixels: int
    unlabeled_valid_pixels: int

    policy: dict[str, Any]
    diagnostics: dict[str, Any]
    ready_for_stage_d: bool


def _require_rasterio() -> tuple[Any, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required for module_postprocess_vectorize Stage C instance core."
        ) from exc
    return rasterio, rasterio.errors


def _require_scipy_ndimage() -> Any:
    try:
        import scipy.ndimage as ndi
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "scipy is required for module_postprocess_vectorize Stage C watershed core."
        ) from exc
    return ndi


def _validate_policy(policy: WatershedCorePolicy) -> WatershedCorePolicy:
    if not isinstance(policy, WatershedCorePolicy):
        raise ContractError("policy must be WatershedCorePolicy for Stage C.")

    if not (0.0 <= float(policy.extent_support_min_prob) <= 1.0):
        raise ContractError("policy.extent_support_min_prob must be in [0,1].")

    if (
        not isinstance(policy.threshold_provenance, str)
        or policy.threshold_provenance.strip() == ""
    ):
        raise ContractError("policy.threshold_provenance must be a non-empty string.")

    for name, value in (
        ("boundary_weight", policy.boundary_weight),
        ("extent_weight", policy.extent_weight),
        ("distance_weight", policy.distance_weight),
    ):
        if float(value) < 0.0:
            raise ContractError(f"policy.{name} must be >= 0.")

    if (
        float(policy.boundary_weight)
        + float(policy.extent_weight)
        + float(policy.distance_weight)
    ) <= 0.0:
        raise ContractError(
            "At least one of boundary/extent/distance weights must be > 0."
        )

    if policy.boundary_presence_mode not in {"skeleton_plus_buffer"}:
        raise ContractError(
            "Unsupported policy.boundary_presence_mode: "
            f"{policy.boundary_presence_mode!r}."
        )
    if not (0.0 <= float(policy.boundary_barrier_max_prob) <= 1.0):
        raise ContractError(
            "policy.boundary_barrier_max_prob must be in [0,1]."
        )

    if policy.connectivity not in {1, 2}:
        raise ContractError("policy.connectivity must be 1 (4-neighborhood) or 2 (8-neighborhood).")

    if policy.background_label != 0:
        raise ContractError(
            "Stage C baseline currently supports background_label=0 only."
        )
    if policy.invalid_label == policy.background_label:
        raise ContractError(
            "policy.invalid_label must differ from policy.background_label "
            "so that invalid and background pixels are explicitly distinguishable "
            "(module_postprocess_vectorize §11.3)."
        )
    if not isinstance(policy.min_region_pixels, int) or policy.min_region_pixels < 0:
        raise ContractError("policy.min_region_pixels must be a non-negative integer.")

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
    return band.astype(np.float32)


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
    return arr.astype(np.float32)


def _read_valid_binary(path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    valid_raw = _read_single_band(path, role="valid", expected_shape=expected_shape)
    unique = np.unique(valid_raw)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValidPolicyError(
            "valid raster must be binary {0,1} for Stage C, "
            f"got unique values {unique.tolist()}."
        )
    return (valid_raw > 0).astype(np.uint8)


def _validate_marker_mask(
    marker_mask: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if not isinstance(marker_mask, np.ndarray) or marker_mask.ndim != 2:
        raise ContractError(
            "marker_candidates_mask must be a 2-D numpy array."
        )
    if marker_mask.shape != expected_shape:
        raise ContractError(
            "marker_candidates_mask shape mismatch: "
            f"expected {expected_shape}, got {marker_mask.shape}."
        )

    unique = np.unique(marker_mask)
    if not np.all(np.isin(unique, [0, 1])):
        raise ContractError(
            "marker_candidates_mask must be binary {0,1}; "
            f"got unique values {unique.tolist()}."
        )

    return marker_mask.astype(np.uint8)


def _boundary_presence(boundary_prob: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "skeleton_plus_buffer":
        return np.clip(boundary_prob[1] + boundary_prob[2], 0.0, 1.0)
    raise ContractError(f"Unsupported boundary_presence_mode: {mode!r}.")


def _build_relief_surface(
    *,
    extent_prob: np.ndarray,
    boundary_prob: np.ndarray,
    distance_pred: np.ndarray,
    valid01: np.ndarray,
    domain_mask: np.ndarray,
    policy: WatershedCorePolicy,
) -> np.ndarray:
    boundary_presence = _boundary_presence(boundary_prob, mode=policy.boundary_presence_mode)

    extent_term = 1.0 - np.clip(extent_prob, 0.0, 1.0)

    # Normalize distance over valid pixels only; higher distance => lower flooding cost.
    valid_pixels = valid01 == 1
    dist_valid = distance_pred[valid_pixels]
    d_min = float(dist_valid.min())
    d_max = float(dist_valid.max())
    if d_max - d_min <= _EPS:
        dist_norm = np.zeros_like(distance_pred, dtype=np.float32)
    else:
        dist_norm = (distance_pred - d_min) / (d_max - d_min)
    distance_term = 1.0 - np.clip(dist_norm, 0.0, 1.0)

    relief = (
        float(policy.boundary_weight) * boundary_presence
        + float(policy.extent_weight) * extent_term
        + float(policy.distance_weight) * distance_term
    ).astype(np.float32)

    if not np.any(domain_mask):
        raise ContractError("Watershed domain mask is empty.")

    max_inside = float(relief[domain_mask].max())
    relief_out = relief.copy()
    relief_out[~domain_mask] = max_inside + 1.0

    r_min = float(relief_out.min())
    r_max = float(relief_out.max())
    if r_max - r_min <= _EPS:
        return np.zeros_like(relief_out, dtype=np.uint8)

    relief_scaled = ((relief_out - r_min) / (r_max - r_min) * 255.0).astype(np.uint8)
    return relief_scaled


def _filter_small_regions(
    labels: np.ndarray,
    *,
    min_pixels: int,
    background_label: int,
) -> np.ndarray:
    """Zero out (to background_label) any instance regions smaller than min_pixels."""
    if min_pixels <= 0:
        return labels
    out = labels.copy()
    for uid in np.unique(labels):
        if int(uid) <= background_label:
            continue
        if int((labels == uid).sum()) < min_pixels:
            out[labels == uid] = background_label
    return out


def _compact_relabel(labels: np.ndarray) -> tuple[np.ndarray, int]:
    unique_pos = [int(v) for v in np.unique(labels) if int(v) > 0]
    if not unique_pos:
        return np.zeros_like(labels, dtype=np.int32), 0

    out = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(sorted(unique_pos), start=1):
        out[labels == old_id] = new_id
    return out, len(unique_pos)


def build_parcel_instance_raster(
    *,
    input_contract: PostprocessInputContractResult,
    marker_result: MarkerGenerationResult,
    policy: WatershedCorePolicy,
) -> ParcelInstanceRasterResult:
    """Build minimal parcel-instance raster via marker-controlled watershed."""
    if not isinstance(input_contract, PostprocessInputContractResult):
        raise ContractError(
            "input_contract must be PostprocessInputContractResult from Stage A."
        )
    if not isinstance(marker_result, MarkerGenerationResult):
        raise ContractError("marker_result must be MarkerGenerationResult from Stage B.")

    _validate_policy(policy)

    h = int(input_contract.common_height)
    w = int(input_contract.common_width)
    shape = (h, w)

    extent_prob = _read_single_band(
        input_contract.extent_prob.path,
        role="extent_prob",
        expected_shape=shape,
    )
    boundary_prob = _read_boundary_three_band(
        input_contract.boundary_prob.path,
        expected_shape=shape,
    )
    distance_pred = _read_single_band(
        input_contract.distance_pred.path,
        role="distance_pred",
        expected_shape=shape,
    )
    valid01 = _read_valid_binary(input_contract.valid.path, expected_shape=shape)

    marker_mask = _validate_marker_mask(
        marker_result.marker_candidates_mask,
        expected_shape=shape,
    )

    if marker_result.ready_for_stage_c is False:
        raise ContractError(
            "marker_result.ready_for_stage_c is False; cannot run Stage C watershed."
        )

    boundary_presence = _boundary_presence(
        boundary_prob,
        mode=policy.boundary_presence_mode,
    )
    boundary_barrier_support = (
        boundary_presence <= float(policy.boundary_barrier_max_prob)
    )

    extent_support = extent_prob >= float(policy.extent_support_min_prob)
    domain_mask = extent_support & (valid01 == 1) & boundary_barrier_support

    valid_pixels = int(valid01.sum())
    invalid_pixels = int(valid01.size - valid_pixels)
    domain_pixels = int(domain_mask.sum())

    if valid_pixels == 0:
        raise ValidPolicyError("valid raster contains zero valid pixels for Stage C.")
    if domain_pixels == 0:
        raise ContractError(
            "Watershed domain is empty under current extent_support threshold policy."
        )

    marker_in_domain = (marker_mask == 1) & domain_mask
    marker_pixels = int(marker_in_domain.sum())
    if marker_pixels == 0:
        raise ContractError(
            "Stage C received zero marker pixels inside watershed domain."
        )

    ndi = _require_scipy_ndimage()
    structure = ndi.generate_binary_structure(2, int(policy.connectivity))
    marker_labels, marker_component_count = ndi.label(marker_in_domain.astype(np.uint8), structure=structure)
    marker_labels = marker_labels.astype(np.int32)

    if int(marker_component_count) == 0:
        raise ContractError(
            "Stage C could not derive marker components from marker_candidates."
        )

    relief = _build_relief_surface(
        extent_prob=extent_prob,
        boundary_prob=boundary_prob,
        distance_pred=distance_pred,
        valid01=valid01,
        domain_mask=domain_mask,
        policy=policy,
    )

    labels = ndi.watershed_ift(relief, marker_labels, structure=structure).astype(np.int32)

    # Enforce contract semantics: no instances outside domain / invalid.
    # invalid pixels get invalid_label (≠ background) so background and invalid
    # are explicitly distinguishable (§11.3).
    labels[~domain_mask] = int(policy.background_label)
    labels[valid01 == 0] = int(policy.invalid_label)

    # Stage C.5: region filtering — remove instances below min_region_pixels
    if policy.min_region_pixels > 0:
        labels = _filter_small_regions(
            labels,
            min_pixels=policy.min_region_pixels,
            background_label=int(policy.background_label),
        )

    if policy.relabel_compact:
        parcel_instance, instance_count = _compact_relabel(labels)
        # _compact_relabel starts from zeros; re-stamp invalid pixels (-1 lost to 0).
        parcel_instance[valid01 == 0] = int(policy.invalid_label)
    else:
        parcel_instance = labels
        instance_count = int(sum(int(v) > 0 for v in np.unique(labels)))

    labeled_valid_pixels = int(((parcel_instance > 0) & (valid01 == 1)).sum())
    unlabeled_valid_pixels = int(valid_pixels - labeled_valid_pixels)

    diagnostics = {
        "shape": {"height": h, "width": w},
        "marker_pixels": marker_pixels,
        "marker_component_count": int(marker_component_count),
        "domain_pixels": domain_pixels,
        "valid_pixels": valid_pixels,
        "labeled_valid_pixels": labeled_valid_pixels,
        "unlabeled_valid_pixels": unlabeled_valid_pixels,
    }

    policy_info = {
        "threshold_provenance": policy.threshold_provenance,
        "extent_support_min_prob": float(policy.extent_support_min_prob),
        "boundary_weight": float(policy.boundary_weight),
        "extent_weight": float(policy.extent_weight),
        "distance_weight": float(policy.distance_weight),
        "boundary_presence_mode": policy.boundary_presence_mode,
        "boundary_barrier_max_prob": float(policy.boundary_barrier_max_prob),
        "connectivity": int(policy.connectivity),
        "relabel_compact": bool(policy.relabel_compact),
        "background_label": int(policy.background_label),
        "invalid_label": int(policy.invalid_label),
        "min_region_pixels": int(policy.min_region_pixels),
        "algorithm": "marker_controlled_watershed_ift",
    }

    return ParcelInstanceRasterResult(
        parcel_instance=parcel_instance.astype(np.int32),
        shape=shape,
        dtype="int32",
        instance_count=int(instance_count),
        background_label=int(policy.background_label),
        invalid_label=int(policy.invalid_label),
        valid_pixels=valid_pixels,
        invalid_pixels=invalid_pixels,
        domain_pixels=domain_pixels,
        labeled_valid_pixels=labeled_valid_pixels,
        unlabeled_valid_pixels=unlabeled_valid_pixels,
        policy=policy_info,
        diagnostics=diagnostics,
        ready_for_stage_d=True,
    )


__all__ = [
    "ParcelInstanceRasterResult",
    "WatershedCorePolicy",
    "build_parcel_instance_raster",
]
