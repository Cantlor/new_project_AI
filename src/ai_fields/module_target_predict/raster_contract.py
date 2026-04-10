"""Minimal predict-side raster contract reader and valid-mask resolver.

This module implements the next narrow step after checkpoint-driven contract
resolution in module_target_predict:

- read input raster metadata contract;
- resolve valid-mask source with explicit priority;
- check raster/checkpoint compatibility for baseline v1 contract.

It intentionally does not implement tiled inference, feature normalization,
model forward, or output writing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.constants import CHANNEL_COUNTS, FEATURE_MODES
from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    FeatureModeError,
    SpatialContractError,
    ValidPolicyError,
)
from ai_fields.module_target_predict.checkpoint_contract import (
    CheckpointDrivenPredictContract,
)


_REQUIRED_SOURCE_RASTER_BANDS = 8


@dataclass(frozen=True)
class PredictRasterMetadata:
    """Minimal metadata contract read from an input GeoTIFF."""

    raster_path: Path
    width: int
    height: int
    band_count: int
    crs: str
    transform_gdal: tuple[float, float, float, float, float, float]
    dtype: str
    nodata: float | int | None
    mask_flags: tuple[tuple[str, ...], ...]
    has_internal_mask: bool


@dataclass(frozen=True)
class PredictValidMaskResolution:
    """Resolved valid-mask contract for predict-time preprocessing."""

    valid_mask: np.ndarray
    source: str
    valid_pixels: int
    invalid_pixels: int
    valid_ratio: float


@dataclass(frozen=True)
class PredictRasterContractResult:
    """Combined raster + valid + checkpoint-compatibility contract."""

    metadata: PredictRasterMetadata
    valid_mask_resolution: PredictValidMaskResolution
    feature_mode: str
    assembled_model_input: str
    in_channels: int
    required_source_band_count: int
    checkpoint_compatible: bool


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


def _require_rasterio() -> tuple[Any, Any, Any]:
    try:
        import rasterio
        import rasterio.errors
        from rasterio.enums import MaskFlags
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required for module_target_predict raster contract layer."
        ) from exc
    return rasterio, rasterio.errors, MaskFlags


def _resolve_crs_string(ds: Any) -> str:
    if ds.crs is None:
        raise SpatialContractError(
            "Input raster CRS is missing; predict output georeferencing would be undefined."
        )
    epsg = ds.crs.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else ds.crs.to_string()


def _resolve_mask_flags(ds: Any, MaskFlags: Any) -> tuple[tuple[str, ...], bool, bool]:
    # mask_flag_enums is per-band: list[list[MaskFlags]]
    per_band_flags: list[tuple[str, ...]] = []
    has_internal_mask = False
    has_nodata_flag = False

    for band_flags in ds.mask_flag_enums:
        names = tuple(sorted(flag.name for flag in band_flags))
        per_band_flags.append(names)

        if MaskFlags.per_dataset in band_flags or MaskFlags.alpha in band_flags:
            has_internal_mask = True
        if MaskFlags.nodata in band_flags:
            has_nodata_flag = True

    return tuple(per_band_flags), has_internal_mask, has_nodata_flag


def _build_mask_summary(mask: np.ndarray, *, source: str) -> PredictValidMaskResolution:
    if mask.ndim != 2:
        raise ContractError(f"Resolved valid mask must be 2-D (H, W), got shape={mask.shape}.")

    mask01 = (mask > 0).astype(np.uint8)
    total = int(mask01.size)
    valid_pixels = int(mask01.sum())
    invalid_pixels = total - valid_pixels
    valid_ratio = float(valid_pixels / total) if total > 0 else 0.0

    return PredictValidMaskResolution(
        valid_mask=mask01,
        source=source,
        valid_pixels=valid_pixels,
        invalid_pixels=invalid_pixels,
        valid_ratio=valid_ratio,
    )


def read_predict_raster_metadata(raster_path: str | Path) -> PredictRasterMetadata:
    """Read minimal GeoTIFF metadata required for predict-time contract checks."""

    resolved_path = _normalize_existing_path(raster_path, name="raster_path")
    rasterio, rasterio_errors, MaskFlags = _require_rasterio()

    try:
        with rasterio.open(resolved_path) as ds:
            if ds.count < 1:
                raise ChannelCountError(
                    f"Input raster must contain at least one band, got {ds.count}."
                )
            if ds.width < 1 or ds.height < 1:
                raise SpatialContractError(
                    f"Input raster has invalid dimensions: width={ds.width}, height={ds.height}."
                )

            crs_str = _resolve_crs_string(ds)

            dtype = str(ds.dtypes[0]) if ds.count > 0 else ""
            if dtype.strip() == "":
                raise ContractError("Unable to resolve raster dtype from input dataset.")

            transform_gdal = tuple(float(v) for v in ds.transform.to_gdal())
            if len(transform_gdal) != 6:
                raise SpatialContractError(
                    "Input raster transform must be a 6-element GDAL geotransform."
                )

            mask_flags, has_internal_mask, has_nodata_flag = _resolve_mask_flags(ds, MaskFlags)

            nodata_val: float | int | None
            if ds.nodata is None:
                nodata_val = None
            elif isinstance(ds.nodata, (int, np.integer)):
                nodata_val = int(ds.nodata)
            elif isinstance(ds.nodata, (float, np.floating)):
                nodata_val = float(ds.nodata)
            else:
                raise ContractError(
                    f"Unsupported nodata type in raster metadata: {type(ds.nodata).__name__}."
                )

            # nodata-only valid policy is still meaningful even if nodata is advertised
            # via mask flags but ds.nodata is null-like in this driver.
            if nodata_val is None and has_nodata_flag:
                nodata_val = None

            return PredictRasterMetadata(
                raster_path=resolved_path,
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
                nodata=nodata_val,
                mask_flags=mask_flags,
                has_internal_mask=has_internal_mask,
            )
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to open raster_path with rasterio: {resolved_path} ({exc})") from exc
    except OSError as exc:
        raise ContractError(f"Failed to read raster metadata at {resolved_path}: {exc}") from exc


def resolve_predict_valid_mask(
    *,
    raster_path: str | Path,
    explicit_valid_mask_override: np.ndarray | None = None,
) -> PredictValidMaskResolution:
    """Resolve predict-time valid mask using project priority policy.

    Priority:
      1) internal/dataset GDAL mask,
      2) nodata metadata,
      3) explicit override,
      4) explicit error.
    """

    resolved_path = _normalize_existing_path(raster_path, name="raster_path")
    metadata = read_predict_raster_metadata(resolved_path)
    rasterio, rasterio_errors, _MaskFlags = _require_rasterio()

    try:
        with rasterio.open(resolved_path) as ds:
            h, w = int(ds.height), int(ds.width)

            if metadata.has_internal_mask:
                dataset_mask = ds.dataset_mask()  # uint8: 0 invalid, 255 valid
                return _build_mask_summary(dataset_mask, source="gdal_valid_data_mask")

            if metadata.nodata is not None:
                first_band = ds.read(1)
                nodata = metadata.nodata
                if isinstance(nodata, float) and np.isnan(nodata):
                    nodata_invalid = np.isnan(first_band)
                else:
                    nodata_invalid = first_band == nodata
                nodata_mask = (~nodata_invalid).astype(np.uint8)
                return _build_mask_summary(nodata_mask, source="nodata_metadata")

            if explicit_valid_mask_override is not None:
                if not isinstance(explicit_valid_mask_override, np.ndarray):
                    raise ValidPolicyError(
                        "explicit_valid_mask_override must be a numpy ndarray when provided."
                    )
                if explicit_valid_mask_override.shape != (h, w):
                    raise ValidPolicyError(
                        "explicit_valid_mask_override shape must match raster spatial shape "
                        f"({h}, {w}), got {explicit_valid_mask_override.shape}."
                    )
                return _build_mask_summary(
                    explicit_valid_mask_override,
                    source="explicit_override",
                )

            raise ValidPolicyError(
                "Unable to resolve valid mask semantics for predict raster: "
                "no internal mask, no nodata metadata, and no explicit override."
            )
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to open raster_path with rasterio: {resolved_path} ({exc})") from exc


def _check_raster_checkpoint_compatibility(
    *,
    metadata: PredictRasterMetadata,
    checkpoint_contract: CheckpointDrivenPredictContract,
) -> None:
    if not isinstance(checkpoint_contract, CheckpointDrivenPredictContract):
        raise ContractError(
            "checkpoint_contract must be CheckpointDrivenPredictContract."
        )

    feature_mode = checkpoint_contract.feature_mode
    if feature_mode not in FEATURE_MODES:
        raise FeatureModeError(
            f"Unsupported checkpoint feature_mode {feature_mode!r}. "
            f"Expected one of {list(FEATURE_MODES)}."
        )

    expected_assembled = f"{feature_mode}_valid"
    if checkpoint_contract.assembled_model_input != expected_assembled:
        raise ChannelCountError(
            "checkpoint assembled_model_input is inconsistent with feature_mode: "
            f"expected {expected_assembled!r}, got {checkpoint_contract.assembled_model_input!r}."
        )

    expected_in_channels = CHANNEL_COUNTS[expected_assembled]
    if checkpoint_contract.in_channels != expected_in_channels:
        raise ChannelCountError(
            "checkpoint in_channels is inconsistent with assembled model input contract: "
            f"expected {expected_in_channels}, got {checkpoint_contract.in_channels}."
        )

    # Baseline v1 predict contract expects an 8-band source raster and builds
    # raw8/raw8_idx3 feature stack from this source.
    if metadata.band_count != _REQUIRED_SOURCE_RASTER_BANDS:
        raise ChannelCountError(
            "Input raster band_count is incompatible with baseline checkpoint-driven "
            "predict contract: "
            f"expected {_REQUIRED_SOURCE_RASTER_BANDS}, got {metadata.band_count}."
        )


def resolve_predict_raster_contract(
    *,
    raster_path: str | Path,
    checkpoint_contract: CheckpointDrivenPredictContract,
    explicit_valid_mask_override: np.ndarray | None = None,
) -> PredictRasterContractResult:
    """Resolve a minimal predict-time raster contract bound to checkpoint contract."""

    metadata = read_predict_raster_metadata(raster_path)
    valid_mask_resolution = resolve_predict_valid_mask(
        raster_path=raster_path,
        explicit_valid_mask_override=explicit_valid_mask_override,
    )

    _check_raster_checkpoint_compatibility(
        metadata=metadata,
        checkpoint_contract=checkpoint_contract,
    )

    return PredictRasterContractResult(
        metadata=metadata,
        valid_mask_resolution=valid_mask_resolution,
        feature_mode=checkpoint_contract.feature_mode,
        assembled_model_input=checkpoint_contract.assembled_model_input,
        in_channels=checkpoint_contract.in_channels,
        required_source_band_count=_REQUIRED_SOURCE_RASTER_BANDS,
        checkpoint_compatible=True,
    )


__all__ = [
    "PredictRasterMetadata",
    "PredictValidMaskResolution",
    "PredictRasterContractResult",
    "read_predict_raster_metadata",
    "resolve_predict_valid_mask",
    "resolve_predict_raster_contract",
]
