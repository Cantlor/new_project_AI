"""Stage 03 runtime compute: feature stack + valid mask via rasterio + numpy.

Called by prepare_features.run_prepare_features_stage when
runtime_compute_enabled=True.  Raises ContractError on any I/O failure.

Band index convention (0-indexed, 8-band sensor):
    0: coastal
    1: blue
    2: green
    3: yellow
    4: red
    5: rededge
    6: nir1
    7: nir2

Derived indices (raw8_idx3 mode):
    NDVI = (nir1 - red) / (nir1 + red + 1e-6)         channels [6, 4]
    SAVI = ((nir1 - red) / (nir1 + red + 0.5 + 1e-6)) * 1.5
    NDWI = (green - nir1) / (green + nir1 + 1e-6)     channels [2, 6]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError

FEATURES_COMPUTE_MODE = "rasterio_numpy_v1"
_UNSET = object()

# Canonical channel names per dataset-side feature mode
CHANNEL_SEMANTICS: dict[str, list[str]] = {
    "raw8": ["coastal", "blue", "green", "yellow", "red", "rededge", "nir1", "nir2"],
    "raw8_idx3": [
        "coastal", "blue", "green", "yellow", "red", "rededge", "nir1", "nir2",
        "ndvi", "savi", "ndwi",
    ],
}

_REQUIRED_BAND_COUNT = 8


def _parse_processing_bounds(processing_bounds: Any | None) -> tuple[float, float, float, float] | None:
    if processing_bounds is None:
        return None
    if isinstance(processing_bounds, (str, bytes)) or not isinstance(processing_bounds, (list, tuple)):
        raise ContractError(
            "processing_bounds must be a 4-element numeric sequence or null."
        )
    if len(processing_bounds) != 4:
        raise ContractError(
            f"processing_bounds must have length 4, got {len(processing_bounds)}."
        )
    out: list[float] = []
    for idx, value in enumerate(processing_bounds):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(
                f"processing_bounds[{idx}] must be a number, got {value!r} "
                f"({type(value).__name__})."
            )
        v = float(value)
        if not math.isfinite(v):
            raise ContractError(
                f"processing_bounds[{idx}] must be finite, got {value!r}."
            )
        out.append(v)
    if out[0] >= out[2] or out[1] >= out[3]:
        raise ContractError(
            "processing_bounds must satisfy minx < maxx and miny < maxy."
        )
    return (out[0], out[1], out[2], out[3])


def _resolve_processing_window(ds: Any, processing_bounds: Any | None) -> tuple[Any | None, list[float] | None]:
    bounds = _parse_processing_bounds(processing_bounds)
    if bounds is None:
        return None, None

    try:
        import rasterio.windows  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for processing window resolution") from exc

    ds_bounds = ds.bounds
    clamped = [
        max(bounds[0], float(ds_bounds.left)),
        max(bounds[1], float(ds_bounds.bottom)),
        min(bounds[2], float(ds_bounds.right)),
        min(bounds[3], float(ds_bounds.top)),
    ]
    if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
        raise ContractError(
            "processing_bounds do not intersect raster extent; cannot compute non-empty window."
        )

    try:
        window = rasterio.windows.from_bounds(
            clamped[0], clamped[1], clamped[2], clamped[3], transform=ds.transform
        )
        # Windowed reads/writes in this stage must be pixel-aligned and integer-sized.
        window = window.round_offsets().round_lengths()
    except Exception as exc:
        raise ContractError(
            f"Failed to derive raster window from processing_bounds: {exc}"
        ) from exc

    if window.width <= 0 or window.height <= 0:
        raise ContractError("Resolved processing window is empty after rounding.")

    full_window = rasterio.windows.Window(col_off=0, row_off=0, width=ds.width, height=ds.height)
    try:
        window = window.intersection(full_window)
    except Exception as exc:
        raise ContractError(
            f"Resolved processing window is outside raster extent: {exc}"
        ) from exc

    if window.width <= 0 or window.height <= 0:
        raise ContractError("Resolved processing window is empty after clipping to raster extent.")

    return window, clamped


def compute_valid_mask(ds: Any, *, window: Any | None = None) -> np.ndarray:
    """Compute (H, W) uint8 valid mask from a rasterio DatasetReader.

    valid=1 → pixel may be used; valid=0 → NoData / invalid.

    The mask is computed BEFORE any fill operation, as required by
    DATA_CONTRACT.md §6.2.
    """
    try:
        import rasterio.enums  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for compute_valid_mask") from exc

    if window is None:
        h, w = ds.height, ds.width
    else:
        h, w = int(window.height), int(window.width)

    # Primary: internal/dataset mask via rasterio
    # dataset_mask() returns 0 where masked (invalid), 255 where valid
    try:
        dataset_mask = ds.dataset_mask(window=window)  # (H, W) uint8: 0=masked, 255=valid
        mask_from_internal = (dataset_mask > 0).astype(np.uint8)
    except Exception:
        mask_from_internal = np.ones((h, w), dtype=np.uint8)

    # If the dataset has a nodata value, also apply pixel-value-based masking
    nodata = ds.nodata
    if nodata is not None:
        try:
            first_band = ds.read(1, window=window).astype(np.float64)
            pixel_valid = (first_band != float(nodata)).astype(np.uint8)
            # Combine: valid only if BOTH internal mask and pixel value agree
            return np.minimum(mask_from_internal, pixel_valid)
        except Exception:
            pass

    return mask_from_internal


def build_feature_stack(ds: Any, feature_mode: str, *, window: Any | None = None) -> np.ndarray:
    """Build (C, H, W) float32 feature array from a rasterio DatasetReader.

    For 'raw8': returns (8, H, W).
    For 'raw8_idx3': returns (11, H, W) = raw8 + [NDVI, SAVI, NDWI].
    """
    if ds.count < _REQUIRED_BAND_COUNT:
        raise ContractError(
            f"Raster has {ds.count} bands but feature_mode requires {_REQUIRED_BAND_COUNT}."
        )

    if feature_mode not in ("raw8", "raw8_idx3"):
        raise ContractError(
            f"Unsupported feature_mode '{feature_mode}'. "
            "Supported: 'raw8', 'raw8_idx3'."
        )

    try:
        raw = ds.read(list(range(1, _REQUIRED_BAND_COUNT + 1)), window=window).astype(np.float32)
    except Exception as exc:
        raise ContractError(f"Failed to read raster bands: {exc}") from exc

    if feature_mode == "raw8":
        return raw

    # raw8_idx3: append NDVI, SAVI, NDWI
    nir1 = raw[6]
    red = raw[4]
    green = raw[2]

    ndvi = (nir1 - red) / (nir1 + red + 1e-6)
    savi = ((nir1 - red) / (nir1 + red + 0.5 + 1e-6)) * 1.5
    ndwi = (green - nir1) / (green + nir1 + 1e-6)

    indices = np.stack([ndvi, savi, ndwi], axis=0)  # (3, H, W)
    return np.concatenate([raw, indices], axis=0)  # (11, H, W)


def _write_geotiff(path: Path, array: np.ndarray, profile: dict, *, nodata: Any = _UNSET) -> None:
    """Write a (C, H, W) or (H, W) ndarray as a GeoTIFF."""
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for _write_geotiff") from exc

    if array.ndim == 2:
        array = array[np.newaxis, ...]  # (1, H, W)

    count = array.shape[0]
    out_profile = dict(profile)
    out_profile.update(
        {
            "count": count,
            "dtype": str(array.dtype),
            "driver": "GTiff",
        }
    )
    if nodata is _UNSET:
        # Keep source profile nodata as-is (used for img.tif path).
        pass
    elif nodata is None:
        # valid.tif is semantic 0/1 mask and must not inherit source nodata tag.
        out_profile.pop("nodata", None)
    else:
        out_profile["nodata"] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(array)
    except Exception as exc:
        raise ContractError(f"Failed to write GeoTIFF to {path}: {exc}") from exc


def _write_geotiff_windowed(
    path: Path,
    window_array: np.ndarray,
    profile: dict,
    *,
    window: Any,
    nodata: Any = _UNSET,
) -> None:
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for _write_geotiff_windowed") from exc

    if window_array.ndim == 2:
        window_array = window_array[np.newaxis, ...]

    count = window_array.shape[0]
    out_profile = dict(profile)
    out_profile.update(
        {
            "count": count,
            "dtype": str(window_array.dtype),
            "driver": "GTiff",
        }
    )
    if nodata is _UNSET:
        pass
    elif nodata is None:
        out_profile.pop("nodata", None)
    else:
        out_profile["nodata"] = nodata

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(path, "w", **out_profile) as dst:
            # Intentionally write only the resolved processing window.
            # Unwritten pixels remain default-initialized (0), which is interpreted
            # as invalid in valid.tif and outside-context filler in img.tif.
            dst.write(window_array, window=window)
    except Exception as exc:
        raise ContractError(f"Failed to write windowed GeoTIFF to {path}: {exc}") from exc


def compute_and_save_features(
    raster_path: Any,
    output_dir: Any,
    feature_mode: str,
    processing_bounds: Any | None = None,
) -> dict:
    """Compute feature stack and valid mask from raster; write img.tif + valid.tif.

    Parameters
    ----------
    raster_path:
        Path to source 8-band GeoTIFF.
    output_dir:
        Directory where img.tif and valid.tif will be written.
    feature_mode:
        'raw8' or 'raw8_idx3'.
    processing_bounds:
        Optional [minx, miny, maxx, maxy] in raster CRS used to limit runtime
        reads/computation to a tighter AOI window.

    Returns
    -------
    dict with keys:
        img_path, valid_path, feature_channel_count, channel_semantics,
        features_compute_mode, processing_bounds_applied
    """
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required for compute_and_save_features but is not installed"
        ) from exc

    raster_path = Path(raster_path)
    output_dir = Path(output_dir)

    if not raster_path.exists():
        raise ContractError(f"Raster file does not exist: {raster_path}")
    if not raster_path.is_file():
        raise ContractError(f"Raster path is not a regular file: {raster_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(raster_path) as ds:
            profile = ds.profile.copy()
            window, clamped_bounds = _resolve_processing_window(ds, processing_bounds)

            # Compute valid mask BEFORE any fill
            valid_mask = compute_valid_mask(ds, window=window)  # (H, W) uint8

            # Compute feature stack
            feature_stack = build_feature_stack(ds, feature_mode, window=window)  # (C, H, W) float32
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(
            f"Failed to read raster for feature computation: {exc}"
        ) from exc

    valid_path = output_dir / "valid.tif"
    img_path = output_dir / "img.tif"

    if window is None:
        # Legacy full-raster write path (unchanged behavior).
        _write_geotiff(valid_path, valid_mask, profile, nodata=None)
        _write_geotiff(img_path, feature_stack, profile)
    else:
        _write_geotiff_windowed(valid_path, valid_mask, profile, window=window, nodata=None)
        _write_geotiff_windowed(img_path, feature_stack, profile, window=window)

    channel_semantics = CHANNEL_SEMANTICS[feature_mode]
    feature_channel_count = feature_stack.shape[0]

    return {
        "img_path": img_path,
        "valid_path": valid_path,
        "feature_channel_count": feature_channel_count,
        "channel_semantics": channel_semantics,
        "features_compute_mode": FEATURES_COMPUTE_MODE,
        "processing_bounds_applied": clamped_bounds,
    }
