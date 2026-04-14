"""Stage 04 runtime compute: target rasters via rasterio + scipy + geopandas.

Called by prepare_targets.run_prepare_targets_stage when
runtime_compute_enabled=True.  Raises ContractError on any failure.

Outputs written to output_dir:
    extent.tif       — uint8, values: 0=background, 1=foreground, 255=ignore
    boundary.tif     — uint8, values: 0=background, 1=skeleton, 2=buffer
    boundary_raw.tif — uint8, values: 0=background, 1=raw edge pixels
    distance.tif     — float32, unsigned euclidean distance to boundary
    valid.tif        — uint8, values: 0=invalid, 1=valid (only if valid_path is None)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError

TARGETS_COMPUTE_MODE = "rasterio_scipy_v1"
_UNSET = object()

_IGNORE_LABEL = 255


def _load_rasterio_and_geopandas() -> tuple[Any, Any]:
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for targets_compute") from exc
    try:
        import geopandas as gpd  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("geopandas is required for targets_compute") from exc
    return rasterio, gpd


def load_valid_mask(valid_path_or_none: Any, ds: Any) -> np.ndarray:
    """Return (H, W) uint8 valid mask.

    If valid_path is given, read it from disk.
    Otherwise compute from ds.nodata using the same logic as features_compute.
    """
    if valid_path_or_none is not None:
        try:
            import rasterio  # noqa: PLC0415

            p = Path(valid_path_or_none)
            if not p.exists():
                raise ContractError(f"valid_path does not exist: {p}")
            with rasterio.open(p) as vds:
                mask = vds.read(1)
            return mask.astype(np.uint8)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"Failed to read valid mask from {valid_path_or_none}: {exc}") from exc

    # Compute from dataset
    h, w = ds.height, ds.width
    try:
        dataset_mask = ds.dataset_mask()
        mask_from_internal = (dataset_mask > 0).astype(np.uint8)
    except Exception:
        mask_from_internal = np.ones((h, w), dtype=np.uint8)

    nodata = ds.nodata
    if nodata is not None:
        try:
            first_band = ds.read(1).astype(np.float64)
            pixel_valid = (first_band != float(nodata)).astype(np.uint8)
            return np.minimum(mask_from_internal, pixel_valid)
        except Exception:
            pass
    return mask_from_internal


def rasterize_extent(ds: Any, vector_gdf: Any) -> np.ndarray:
    """Rasterize field extents to (H, W) uint8 aligned to raster grid.

    Values: 0=background, 1=foreground field, 255=ignore (near NoData border).
    """
    h, w = ds.height, ds.width

    # Reproject vector to raster CRS
    raster_crs = ds.crs
    if raster_crs is not None and vector_gdf.crs is not None:
        try:
            if str(vector_gdf.crs) != str(raster_crs):
                vector_gdf = vector_gdf.to_crs(raster_crs)
        except Exception as exc:
            raise ContractError(
                f"Failed to reproject vector to raster CRS: {exc}"
            ) from exc

    geoms = [geom for geom in vector_gdf.geometry if geom is not None and not geom.is_empty]
    return rasterize_extent_for_window(
        transform=ds.transform,
        height=h,
        width=w,
        vector_geometries=geoms,
    )


def build_boundary_target(
    ds: Any, vector_gdf: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Build boundary target (encoding: 0=background, 1=skeleton, 2=buffer).

    Returns (boundary, boundary_raw) both (H, W) uint8.

    boundary_raw: raw rasterized polygon edge pixels (0/1).
    boundary:     boundary_raw + dilation buffer zone encoded as 2.
    """
    h, w = ds.height, ds.width
    raster_crs = ds.crs

    if raster_crs is not None and vector_gdf.crs is not None:
        try:
            if str(vector_gdf.crs) != str(raster_crs):
                vector_gdf = vector_gdf.to_crs(raster_crs)
        except Exception as exc:
            raise ContractError(
                f"Failed to reproject vector for boundary: {exc}"
            ) from exc

    geoms = [geom for geom in vector_gdf.geometry if geom is not None and not geom.is_empty]
    return build_boundary_target_for_window(
        transform=ds.transform,
        height=h,
        width=w,
        vector_geometries=geoms,
        buffer_iterations=3,
    )


def compute_distance_target(
    boundary_raw: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Compute (H, W) float32 unsigned euclidean distance to nearest boundary pixel.

    Invalid pixels are zeroed out.
    """
    try:
        from scipy.ndimage import distance_transform_edt  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("scipy is required for compute_distance_target") from exc

    try:
        not_boundary = ~boundary_raw.astype(bool)
        dist = distance_transform_edt(not_boundary).astype(np.float32)
    except Exception as exc:
        raise ContractError(f"Failed to compute distance transform: {exc}") from exc

    # Zero out invalid pixels
    dist[valid_mask == 0] = 0.0
    return dist


def rasterize_extent_for_window(
    *,
    transform: Any,
    height: int,
    width: int,
    vector_geometries: list[Any],
) -> np.ndarray:
    """Rasterize extent for a window/grid definition.

    Returns uint8 (H, W), values:
      - 0 background
      - 1 foreground
    """
    try:
        import rasterio.features  # noqa: PLC0415
        from shapely.geometry import mapping  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio.features and shapely are required") from exc

    shapes = [(mapping(geom), 1) for geom in vector_geometries if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    try:
        extent = rasterio.features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
    except Exception as exc:
        raise ContractError(f"Failed to rasterize extent for window: {exc}") from exc
    return extent.astype(np.uint8)


def build_boundary_target_for_window(
    *,
    transform: Any,
    height: int,
    width: int,
    vector_geometries: list[Any],
    buffer_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build boundary + boundary_raw for a specific window/grid."""
    try:
        import rasterio.features  # noqa: PLC0415
        from scipy.ndimage import binary_dilation  # noqa: PLC0415
        from shapely.geometry import mapping  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio.features, scipy, and shapely are required") from exc

    boundary_geoms = []
    for geom in vector_geometries:
        if geom is None or geom.is_empty:
            continue
        try:
            bnd = geom.boundary
            if bnd is not None and not bnd.is_empty:
                boundary_geoms.append((mapping(bnd), 1))
        except Exception:
            continue

    if not boundary_geoms:
        empty = np.zeros((height, width), dtype=np.uint8)
        return empty, empty

    try:
        boundary_raw = rasterio.features.rasterize(
            boundary_geoms,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
    except Exception as exc:
        raise ContractError(f"Failed to rasterize boundary for window: {exc}") from exc

    try:
        dilated = binary_dilation(boundary_raw.astype(bool), iterations=buffer_iterations)
        buffer_zone = (dilated & ~boundary_raw.astype(bool)).astype(np.uint8)
    except Exception as exc:
        raise ContractError(f"Failed to compute boundary buffer for window: {exc}") from exc

    boundary = np.zeros((height, width), dtype=np.uint8)
    boundary[buffer_zone == 1] = 2
    boundary[boundary_raw == 1] = 1
    return boundary, boundary_raw.astype(np.uint8)


def compute_distance_target_clipped(
    boundary_raw: np.ndarray,
    valid_mask: np.ndarray,
    *,
    distance_clip_px: int,
) -> np.ndarray:
    """Compute unsigned distance target and clip to the configured max distance."""
    if isinstance(distance_clip_px, bool) or not isinstance(distance_clip_px, int):
        raise ContractError(
            "distance_clip_px must be an integer, "
            f"got {type(distance_clip_px).__name__}."
        )
    if distance_clip_px <= 0:
        raise ContractError(f"distance_clip_px must be > 0, got {distance_clip_px}.")

    dist = compute_distance_target(boundary_raw=boundary_raw, valid_mask=valid_mask)
    np.minimum(dist, float(distance_clip_px), out=dist)
    return dist


def _write_geotiff(path: Path, array: np.ndarray, profile: dict, *, nodata: Any = _UNSET) -> None:
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for _write_geotiff") from exc

    if array.ndim == 2:
        array = array[np.newaxis, ...]

    count = array.shape[0]
    out_profile = dict(profile)
    out_profile.update({"count": count, "dtype": str(array.dtype), "driver": "GTiff"})
    if nodata is _UNSET:
        # Keep source profile nodata as-is (used for distance.tif path).
        pass
    elif nodata is None:
        # Semantic uint8 targets (extent/boundary/boundary_raw/valid) must not
        # inherit source nodata metadata.
        out_profile.pop("nodata", None)
    else:
        out_profile["nodata"] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(array)
    except Exception as exc:
        raise ContractError(f"Failed to write GeoTIFF to {path}: {exc}") from exc


def compute_and_save_targets(
    raster_path: Any,
    vector_path: Any,
    output_dir: Any,
    valid_path: Any | None = None,
    distance_clip_px: int | None = None,
) -> dict:
    """Compute and write target rasters aligned to the source raster grid.

    Parameters
    ----------
    raster_path:
        Source 8-band GeoTIFF (grid reference).
    vector_path:
        GT vector file with field polygons.
    output_dir:
        Directory where target GeoTIFFs will be written.
    valid_path:
        Optional pre-computed valid.tif.  If None, valid mask is computed here.

    Returns
    -------
    dict with keys:
        extent_path, boundary_path, boundary_raw_path, distance_path, valid_path
    """
    rasterio, gpd = _load_rasterio_and_geopandas()

    raster_path = Path(raster_path)
    vector_path = Path(vector_path)
    output_dir = Path(output_dir)

    if not raster_path.exists():
        raise ContractError(f"Raster file does not exist: {raster_path}")
    if not raster_path.is_file():
        raise ContractError(f"Raster path is not a regular file: {raster_path}")
    if not vector_path.exists():
        raise ContractError(f"Vector file does not exist: {vector_path}")
    if not vector_path.is_file():
        raise ContractError(f"Vector path is not a regular file: {vector_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        vec_gdf = gpd.read_file(vector_path)
    except Exception as exc:
        raise ContractError(f"Failed to read vector file: {exc}") from exc

    try:
        with rasterio.open(raster_path) as ds:
            profile = ds.profile.copy()

            # Compute valid mask (from provided path or from raster)
            valid_mask = load_valid_mask(valid_path, ds)

            # Compute extent
            extent = rasterize_extent(ds, vec_gdf)
            extent[valid_mask == 0] = _IGNORE_LABEL

            # Compute boundary targets
            boundary, boundary_raw = build_boundary_target(ds, vec_gdf)

            # Compute distance target
            if distance_clip_px is None:
                distance = compute_distance_target(boundary_raw, valid_mask)
            else:
                distance = compute_distance_target_clipped(
                    boundary_raw=boundary_raw,
                    valid_mask=valid_mask,
                    distance_clip_px=int(distance_clip_px),
                )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(
            f"Failed to compute targets from raster/vector: {exc}"
        ) from exc

    # Write targets
    extent_path = output_dir / "extent.tif"
    boundary_path = output_dir / "boundary.tif"
    boundary_raw_path = output_dir / "boundary_raw.tif"
    distance_path = output_dir / "distance.tif"

    _write_geotiff(extent_path, extent, profile, nodata=None)
    _write_geotiff(boundary_path, boundary, profile, nodata=None)
    _write_geotiff(boundary_raw_path, boundary_raw, profile, nodata=None)
    _write_geotiff(distance_path, distance, profile)

    # Write valid.tif if not provided
    out_valid_path: Path
    if valid_path is None:
        out_valid_path = output_dir / "valid.tif"
        _write_geotiff(out_valid_path, valid_mask, profile, nodata=None)
    else:
        out_valid_path = Path(valid_path)

    return {
        "extent_path": extent_path,
        "boundary_path": boundary_path,
        "boundary_raw_path": boundary_raw_path,
        "distance_path": distance_path,
        "valid_path": out_valid_path,
        "targets_compute_mode": TARGETS_COMPUTE_MODE,
    }
