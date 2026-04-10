"""Postprocess input contract resolver for module_postprocess_vectorize.

This is the first, contract-first layer for postprocess/vectorize baseline:
- reads required predict raster artifacts;
- validates strict spatial compatibility;
- validates minimal semantic/value-domain assumptions;
- exposes explicit postprocess output skeleton for downstream stages.

Scope is intentionally narrow: no thresholding, no watershed, no polygonization,
and no vector export runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.constants import REQUIRED_PREDICT_OUTPUTS
from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    SpatialContractError,
    ValidPolicyError,
)
from ai_fields.common.manifests import read_manifest

_EXPECTED_PREDICT_MANIFEST_SCHEMA = "target_predict.predict_manifest"
_EPS = 1e-6
_BOUNDARY_SUM_ATOL = 5e-3


@dataclass(frozen=True)
class PostprocessRasterMetadata:
    """Minimal raster metadata needed for postprocess input contract checks."""

    role: str
    path: Path
    width: int
    height: int
    band_count: int
    crs: str
    transform_gdal: tuple[float, float, float, float, float, float]
    dtype: str
    nodata: float | int | None


@dataclass(frozen=True)
class PostprocessOutputContractSkeleton:
    """Fixed baseline output skeleton for next postprocess stages."""

    parcel_instance_raster: str
    parcels_vector: str
    required_polygon_attributes: tuple[str, ...]


@dataclass(frozen=True)
class PostprocessInputContractResult:
    """Resolved and validated postprocess input contract."""

    extent_prob: PostprocessRasterMetadata
    boundary_prob: PostprocessRasterMetadata
    distance_pred: PostprocessRasterMetadata
    valid: PostprocessRasterMetadata

    common_width: int
    common_height: int
    common_crs: str
    common_transform_gdal: tuple[float, float, float, float, float, float]

    extent_value_range: tuple[float, float]
    boundary_value_range: tuple[float, float]
    boundary_sum_range_on_valid: tuple[float, float] | None
    distance_value_range: tuple[float, float]
    valid_unique_values: tuple[int, ...]

    probability_semantics: dict[str, Any]
    valid_mask_semantics: dict[str, Any]
    output_contract: PostprocessOutputContractSkeleton

    predict_manifest_path: Path | None
    aoi_path: Path | None
    compatible: bool


def _normalize_existing_path(path: Any, *, name: str) -> Path:
    if isinstance(path, (str, PathLike)):
        normalized = Path(path)
    else:
        raise ContractError(f"{name} must be path-like, got {type(path).__name__}.")
    if str(normalized).strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    if not normalized.exists():
        raise ContractError(f"{name} does not exist: {normalized}")
    if not normalized.is_file():
        raise ContractError(f"{name} must point to a regular file: {normalized}")
    return normalized


def _normalize_optional_path(path: Any, *, name: str) -> Path | None:
    if path is None:
        return None
    return _normalize_existing_path(path, name=name)


def _require_rasterio() -> tuple[Any, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required for module_postprocess_vectorize input contract layer."
        ) from exc
    return rasterio, rasterio.errors


def _resolve_crs_string(ds: Any) -> str:
    if ds.crs is None:
        raise SpatialContractError("Input raster CRS is missing.")
    epsg = ds.crs.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else ds.crs.to_string()


def _to_nodata_value(raw: Any) -> float | int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, (float, np.floating)):
        return float(raw)
    raise ContractError(f"Unsupported nodata type: {type(raw).__name__}.")


def _read_raster_metadata(path: Path, *, role: str) -> PostprocessRasterMetadata:
    rasterio, rasterio_errors = _require_rasterio()
    try:
        with rasterio.open(path) as ds:
            if ds.width < 1 or ds.height < 1:
                raise SpatialContractError(
                    f"{role}: invalid raster dimensions ({ds.width}x{ds.height})."
                )
            if ds.count < 1:
                raise ChannelCountError(f"{role}: raster must contain at least one band.")
            crs_str = _resolve_crs_string(ds)

            dtype = str(ds.dtypes[0])
            if dtype.strip() == "":
                raise ContractError(f"{role}: failed to resolve raster dtype.")

            transform_gdal = tuple(float(v) for v in ds.transform.to_gdal())
            if len(transform_gdal) != 6:
                raise SpatialContractError(
                    f"{role}: transform must be a 6-element GDAL geotransform."
                )

            return PostprocessRasterMetadata(
                role=role,
                path=path,
                width=int(ds.width),
                height=int(ds.height),
                band_count=int(ds.count),
                crs=crs_str,
                transform_gdal=(
                    transform_gdal[0],
                    transform_gdal[1],
                    transform_gdal[2],
                    transform_gdal[3],
                    transform_gdal[4],
                    transform_gdal[5],
                ),
                dtype=dtype,
                nodata=_to_nodata_value(ds.nodata),
            )
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to open {role} raster: {path} ({exc})") from exc


def _transforms_match(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), atol=1e-9, rtol=0.0))


def _validate_spatial_alignment(
    reference: PostprocessRasterMetadata,
    candidate: PostprocessRasterMetadata,
) -> None:
    if (candidate.width, candidate.height) != (reference.width, reference.height):
        raise SpatialContractError(
            f"Spatial mismatch: {candidate.role} has {candidate.width}x{candidate.height}, "
            f"expected {reference.width}x{reference.height} from {reference.role}."
        )
    if candidate.crs != reference.crs:
        raise SpatialContractError(
            f"CRS mismatch: {candidate.role} has {candidate.crs!r}, "
            f"expected {reference.crs!r} from {reference.role}."
        )
    if not _transforms_match(candidate.transform_gdal, reference.transform_gdal):
        raise SpatialContractError(
            "Transform mismatch: "
            f"{candidate.role} transform differs from {reference.role}; "
            "hidden resampling is not allowed in postprocess input contract."
        )


def _read_array(path: Path) -> np.ndarray:
    rasterio, rasterio_errors = _require_rasterio()
    try:
        with rasterio.open(path) as ds:
            return ds.read()
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read raster values: {path} ({exc})") from exc


def _require_floating_dtype(meta: PostprocessRasterMetadata) -> None:
    if not np.issubdtype(np.dtype(meta.dtype), np.floating):
        raise ContractError(
            f"{meta.role} must have floating dtype, got {meta.dtype!r}."
        )


def _require_integer_or_bool_dtype(meta: PostprocessRasterMetadata) -> None:
    dtype = np.dtype(meta.dtype)
    if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)):
        raise ValidPolicyError(
            f"{meta.role} must have integer/bool dtype for binary valid-mask, got {meta.dtype!r}."
        )


def _validate_band_count(meta: PostprocessRasterMetadata, *, expected: int) -> None:
    if meta.band_count != expected:
        raise ChannelCountError(
            f"{meta.role} must have exactly {expected} band(s), got {meta.band_count}."
        )


def _validate_probability_range(
    array: np.ndarray,
    *,
    role: str,
) -> tuple[float, float]:
    if not np.isfinite(array).all():
        raise ContractError(f"{role} contains non-finite values.")
    min_v = float(array.min())
    max_v = float(array.max())
    if min_v < -_EPS or max_v > 1.0 + _EPS:
        raise ContractError(
            f"{role} must be probability-like in [0,1], got min={min_v}, max={max_v}."
        )
    return min_v, max_v


def _validate_distance_range(array: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(array).all():
        raise ContractError("distance_pred contains non-finite values.")
    min_v = float(array.min())
    max_v = float(array.max())
    if min_v < -_EPS:
        raise ContractError(
            "distance_pred must be non-negative for unsigned distance contract, "
            f"got min={min_v}."
        )
    return min_v, max_v


def _validate_binary_valid(array: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    if array.ndim != 2:
        raise ValidPolicyError(
            f"valid raster must be 2-D after single-band read, got shape={array.shape}."
        )

    unique = tuple(int(v) for v in np.unique(array))
    allowed = {0, 1}
    if not set(unique).issubset(allowed):
        raise ValidPolicyError(
            "valid raster must be binary {0,1}; "
            f"got unique values {list(unique)}."
        )
    valid01 = (array > 0).astype(np.uint8)
    return valid01, unique


def _validate_boundary_prob_sum(
    boundary_prob: np.ndarray,
    *,
    valid_mask01: np.ndarray,
) -> tuple[float, float] | None:
    if boundary_prob.ndim != 3 or boundary_prob.shape[0] != 3:
        raise ContractError(
            "Internal boundary_prob validation expects shape (3, H, W)."
        )

    valid_pixels = valid_mask01 == 1
    if not np.any(valid_pixels):
        return None

    sums = boundary_prob.sum(axis=0)
    sums_valid = sums[valid_pixels]
    min_sum = float(sums_valid.min())
    max_sum = float(sums_valid.max())
    if not np.all(np.abs(sums_valid - 1.0) <= _BOUNDARY_SUM_ATOL):
        raise ContractError(
            "boundary_prob on valid pixels must represent a 3-class probability simplex "
            f"(sum≈1), got range [{min_sum}, {max_sum}] with atol={_BOUNDARY_SUM_ATOL}."
        )
    return (min_sum, max_sum)


def _resolve_manifest_output_path(raw_path: str, *, manifest_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    # Support current predict manifests that may store repository-relative paths
    # (e.g. "runs/module_target_predict/.../extent_prob.tif"), while keeping
    # legacy manifest-relative behavior for plain relative filenames.
    cwd_relative = candidate
    manifest_relative = manifest_path.parent / candidate

    if cwd_relative.exists() and manifest_relative.exists():
        if cwd_relative.resolve() != manifest_relative.resolve():
            raise ContractError(
                "predict_manifest output path is ambiguous: "
                f"{raw_path!r} resolves to both {cwd_relative} and {manifest_relative}."
            )
        return cwd_relative

    if cwd_relative.exists():
        return cwd_relative
    if manifest_relative.exists():
        return manifest_relative

    # Keep deterministic fallback for paths that do not exist yet.
    return manifest_relative


def _validate_predict_manifest_consistency(
    *,
    predict_manifest_path: Path,
    extent_prob_path: Path,
    boundary_prob_path: Path,
    distance_pred_path: Path,
    valid_path: Path,
) -> None:
    manifest = read_manifest(predict_manifest_path)
    schema_name = manifest.get("schema_name")
    if schema_name != _EXPECTED_PREDICT_MANIFEST_SCHEMA:
        raise ContractError(
            "predict_manifest schema_name must be "
            f"{_EXPECTED_PREDICT_MANIFEST_SCHEMA!r}, got {schema_name!r}."
        )

    output_paths = manifest.get("output_paths")
    if not isinstance(output_paths, Mapping):
        raise ContractError("predict_manifest.output_paths must be a mapping/object.")

    for key in REQUIRED_PREDICT_OUTPUTS:
        value = output_paths.get(key)
        if not isinstance(value, str) or value.strip() == "":
            raise ContractError(
                "predict_manifest.output_paths is missing required non-empty path for "
                f"{key!r}."
            )

    expected_paths = {
        "extent_prob": extent_prob_path,
        "boundary_prob": boundary_prob_path,
        "distance_pred": distance_pred_path,
        "valid": valid_path,
    }
    for key, expected in expected_paths.items():
        actual = _resolve_manifest_output_path(
            output_paths[key], manifest_path=predict_manifest_path
        )
        if actual.resolve() != expected.resolve():
            raise ContractError(
                "predict_manifest output path mismatch for "
                f"{key!r}: manifest={actual}, provided={expected}."
            )


def resolve_postprocess_input_contract(
    *,
    extent_prob_path: str | Path,
    boundary_prob_path: str | Path,
    distance_pred_path: str | Path,
    valid_path: str | Path,
    aoi_path: str | Path | None = None,
    predict_manifest_path: str | Path | None = None,
) -> PostprocessInputContractResult:
    """Resolve and validate baseline postprocess input contract.

    Required inputs:
      - extent_prob.tif     (1 band, float, probability-like [0,1])
      - boundary_prob.tif   (3 band, float, class-prob-like [0,1], sum≈1 on valid)
      - distance_pred.tif   (1 band, float, non-negative)
      - valid.tif           (1 band, binary 0/1)

    Optional:
      - aoi_path
      - predict_manifest_path (for strict provenance/path consistency checks)
    """
    extent_prob_path = _normalize_existing_path(extent_prob_path, name="extent_prob_path")
    boundary_prob_path = _normalize_existing_path(boundary_prob_path, name="boundary_prob_path")
    distance_pred_path = _normalize_existing_path(distance_pred_path, name="distance_pred_path")
    valid_path = _normalize_existing_path(valid_path, name="valid_path")

    aoi_path = _normalize_optional_path(aoi_path, name="aoi_path")
    predict_manifest_path = _normalize_optional_path(
        predict_manifest_path, name="predict_manifest_path"
    )

    extent_meta = _read_raster_metadata(extent_prob_path, role="extent_prob")
    boundary_meta = _read_raster_metadata(boundary_prob_path, role="boundary_prob")
    distance_meta = _read_raster_metadata(distance_pred_path, role="distance_pred")
    valid_meta = _read_raster_metadata(valid_path, role="valid")

    for candidate in (boundary_meta, distance_meta, valid_meta):
        _validate_spatial_alignment(extent_meta, candidate)

    _validate_band_count(extent_meta, expected=1)
    _validate_band_count(boundary_meta, expected=3)
    _validate_band_count(distance_meta, expected=1)
    _validate_band_count(valid_meta, expected=1)

    _require_floating_dtype(extent_meta)
    _require_floating_dtype(boundary_meta)
    _require_floating_dtype(distance_meta)
    _require_integer_or_bool_dtype(valid_meta)

    extent_arr = _read_array(extent_prob_path)[0]
    boundary_arr = _read_array(boundary_prob_path)
    distance_arr = _read_array(distance_pred_path)[0]
    valid_arr = _read_array(valid_path)[0]

    valid_mask01, valid_unique = _validate_binary_valid(valid_arr)

    extent_range = _validate_probability_range(extent_arr, role="extent_prob")
    boundary_range = _validate_probability_range(boundary_arr, role="boundary_prob")
    boundary_sum_valid = _validate_boundary_prob_sum(boundary_arr, valid_mask01=valid_mask01)
    distance_range = _validate_distance_range(distance_arr)

    if predict_manifest_path is not None:
        _validate_predict_manifest_consistency(
            predict_manifest_path=predict_manifest_path,
            extent_prob_path=extent_prob_path,
            boundary_prob_path=boundary_prob_path,
            distance_pred_path=distance_pred_path,
            valid_path=valid_path,
        )

    output_contract = PostprocessOutputContractSkeleton(
        parcel_instance_raster="parcel_instance.tif",
        parcels_vector="parcels.gpkg",
        required_polygon_attributes=("polygon_confidence",),
    )

    probability_semantics: dict[str, Any] = {
        "extent_prob": {
            "format": "single_band_probability",
            "value_range": "[0,1]",
        },
        "boundary_prob": {
            "format": "three_band_class_probability",
            "class_order": {"0": "background", "1": "skeleton", "2": "buffer"},
            "value_range": "[0,1]",
            "sum_constraint_on_valid": "sum_per_pixel ~= 1.0",
        },
        "distance_pred": {
            "format": "single_band_unsigned_distance",
            "value_range": "[0,+inf)",
        },
    }

    valid_mask_semantics: dict[str, Any] = {
        "format": "single_band_binary_mask",
        "allowed_values": [0, 1],
        "source_role": "postprocess_suppression_and_qc",
    }

    return PostprocessInputContractResult(
        extent_prob=extent_meta,
        boundary_prob=boundary_meta,
        distance_pred=distance_meta,
        valid=valid_meta,
        common_width=extent_meta.width,
        common_height=extent_meta.height,
        common_crs=extent_meta.crs,
        common_transform_gdal=extent_meta.transform_gdal,
        extent_value_range=extent_range,
        boundary_value_range=boundary_range,
        boundary_sum_range_on_valid=boundary_sum_valid,
        distance_value_range=distance_range,
        valid_unique_values=valid_unique,
        probability_semantics=probability_semantics,
        valid_mask_semantics=valid_mask_semantics,
        output_contract=output_contract,
        predict_manifest_path=predict_manifest_path,
        aoi_path=aoi_path,
        compatible=True,
    )


__all__ = [
    "PostprocessInputContractResult",
    "PostprocessOutputContractSkeleton",
    "PostprocessRasterMetadata",
    "resolve_postprocess_input_contract",
]
