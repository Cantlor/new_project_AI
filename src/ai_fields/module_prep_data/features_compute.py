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


def compute_valid_mask(ds: Any) -> np.ndarray:
    """Compute (H, W) uint8 valid mask from a rasterio DatasetReader.

    valid=1 → pixel may be used; valid=0 → NoData / invalid.

    The mask is computed BEFORE any fill operation, as required by
    DATA_CONTRACT.md §6.2.
    """
    try:
        import rasterio.enums  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for compute_valid_mask") from exc

    h, w = ds.height, ds.width

    # Primary: internal/dataset mask via rasterio
    # dataset_mask() returns 0 where masked (invalid), 255 where valid
    try:
        dataset_mask = ds.dataset_mask()  # (H, W) uint8: 0=masked, 255=valid
        mask_from_internal = (dataset_mask > 0).astype(np.uint8)
    except Exception:
        mask_from_internal = np.ones((h, w), dtype=np.uint8)

    # If the dataset has a nodata value, also apply pixel-value-based masking
    nodata = ds.nodata
    if nodata is not None:
        try:
            first_band = ds.read(1).astype(np.float64)
            pixel_valid = (first_band != float(nodata)).astype(np.uint8)
            # Combine: valid only if BOTH internal mask and pixel value agree
            return np.minimum(mask_from_internal, pixel_valid)
        except Exception:
            pass

    return mask_from_internal


def build_feature_stack(ds: Any, feature_mode: str) -> np.ndarray:
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
        raw = ds.read(list(range(1, _REQUIRED_BAND_COUNT + 1))).astype(np.float32)
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


def compute_and_save_features(
    raster_path: Any,
    output_dir: Any,
    feature_mode: str,
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

    Returns
    -------
    dict with keys:
        img_path, valid_path, feature_channel_count, channel_semantics,
        features_compute_mode
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

            # Compute valid mask BEFORE any fill
            valid_mask = compute_valid_mask(ds)  # (H, W) uint8

            # Compute feature stack
            feature_stack = build_feature_stack(ds, feature_mode)  # (C, H, W) float32
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(
            f"Failed to read raster for feature computation: {exc}"
        ) from exc

    valid_path = output_dir / "valid.tif"
    img_path = output_dir / "img.tif"

    # Write valid.tif (1 band, uint8)
    _write_geotiff(valid_path, valid_mask, profile, nodata=None)

    # Write img.tif (C bands, float32)
    _write_geotiff(img_path, feature_stack, profile)

    channel_semantics = CHANNEL_SEMANTICS[feature_mode]
    feature_channel_count = feature_stack.shape[0]

    return {
        "img_path": img_path,
        "valid_path": valid_path,
        "feature_channel_count": feature_channel_count,
        "channel_semantics": channel_semantics,
        "features_compute_mode": FEATURES_COMPUTE_MODE,
    }
