"""Stage 02 runtime compute: spatial context resolution via rasterio + geopandas.

Called by prepare_spatial_context.run_prepare_spatial_context_stage when
runtime_compute_enabled=True.  Raises ContractError on any I/O or CRS failure
so the stage runner can return status="failed".
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError

SPATIAL_COMPUTE_MODE = "rasterio_geopandas_v1"

_MB = 1024 * 1024


def _bounds_are_finite(bounds: list[float]) -> bool:
    return len(bounds) == 4 and all(math.isfinite(v) for v in bounds)


def _clamp_bounds_to_raster(
    *,
    raw_bounds: list[float],
    raster_bounds: list[float],
) -> list[float]:
    minx = max(float(raw_bounds[0]), float(raster_bounds[0]))
    miny = max(float(raw_bounds[1]), float(raster_bounds[1]))
    maxx = min(float(raw_bounds[2]), float(raster_bounds[2]))
    maxy = min(float(raw_bounds[3]), float(raster_bounds[3]))
    if minx >= maxx or miny >= maxy:
        raise ContractError(
            "Derived AOI does not intersect raster extent after clamping; "
            "cannot resolve non-empty effective extent."
        )
    return [minx, miny, maxx, maxy]


def _detect_total_ram_mb() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return int((pages * page_size) / _MB)


def _resolve_memory_budget_mb(total_ram_mb: int | None, requested_mb: int | None) -> int:
    if requested_mb is not None:
        if isinstance(requested_mb, bool) or not isinstance(requested_mb, int):
            raise ContractError(
                "memory_budget_mb override must be an integer, "
                f"got {type(requested_mb).__name__}."
            )
        if requested_mb <= 0:
            raise ContractError(f"memory_budget_mb override must be > 0, got {requested_mb}.")
        return requested_mb

    if total_ram_mb is None:
        return 3072
    if total_ram_mb <= 16 * 1024:
        return 3072
    if total_ram_mb <= 32 * 1024:
        return 6144
    if total_ram_mb <= 64 * 1024:
        return 10240

    # Conservative cap for larger machines; avoid "use all RAM" behaviour.
    return min(16384, max(12288, int(total_ram_mb * 0.2)))


def _dtype_itemsize_bytes(dtype_name: str) -> int:
    try:
        import numpy as np  # noqa: PLC0415

        return int(np.dtype(dtype_name).itemsize)
    except Exception:
        return 4


def _extract_block_metadata(ds: Any) -> tuple[bool, list[list[int]]]:
    is_tiled = bool(getattr(ds, "is_tiled", False))
    raw_block_shapes = getattr(ds, "block_shapes", None)
    block_shapes: list[list[int]] = []
    if raw_block_shapes is not None:
        seen: set[tuple[int, int]] = set()
        for shape in raw_block_shapes:
            try:
                rows = int(shape[0])
                cols = int(shape[1])
            except Exception:
                continue
            if rows > 0 and cols > 0:
                seen.add((rows, cols))
        block_shapes = [[r, c] for r, c in sorted(seen)]
    return is_tiled, block_shapes


def _resolve_large_scene_mode(
    *,
    approx_source_bytes: int,
    is_tiled: bool,
    memory_budget_mb: int,
) -> tuple[bool, str]:
    approx_source_mb = approx_source_bytes / _MB
    if approx_source_mb > memory_budget_mb:
        return True, "approx_source_bytes_exceeds_memory_budget"
    if not is_tiled and approx_source_mb > memory_budget_mb * 0.5:
        return True, "source_not_tiled_and_large"
    if approx_source_mb > memory_budget_mb * 0.75:
        return True, "approx_source_bytes_near_memory_budget"
    return False, "within_memory_budget"


def _copy_to_block_cache(*, rasterio: Any, source_path: Path, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(source_path) as src:
            profile = src.profile.copy()
            profile.update(
                {
                    "driver": "GTiff",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "compress": "deflate",
                }
            )
            with rasterio.open(cache_path, "w", **profile) as dst:
                for _, window in src.block_windows(1):
                    data = src.read(window=window)
                    dst.write(data, window=window)
    except Exception as exc:
        raise ContractError(
            f"Failed to materialize block_cache raster '{cache_path}': {exc}"
        ) from exc


def _prepare_canonical_runtime_source(
    *,
    rasterio: Any,
    raster_path: Path,
    output_root: Path | None,
    is_tiled: bool,
    aoi_present: bool,
) -> tuple[str, str]:
    """Resolve source_kind and source_path for downstream windowed runtime."""
    if is_tiled and not aoi_present:
        return "direct_raster", str(raster_path)

    if is_tiled and aoi_present:
        if output_root is None:
            return "direct_raster", str(raster_path)
        vrt_path = output_root / "runtime_source.vrt"
        try:
            from rasterio.shutil import copy as rio_copy  # noqa: PLC0415

            rio_copy(str(raster_path), str(vrt_path), driver="VRT")
        except Exception as exc:
            raise ContractError(
                f"Failed to build VRT runtime source '{vrt_path}': {exc}"
            ) from exc
        return "vrt", str(vrt_path)

    if output_root is None:
        # Standalone helper callers may omit output_dir; keep function usable
        # by falling back to direct source access. Stage-02 runtime path always
        # provides output_dir and can materialize block_cache when needed.
        return "direct_raster", str(raster_path)

    cache_path = output_root / "runtime_source_block_cache.tif"
    _copy_to_block_cache(
        rasterio=rasterio,
        source_path=raster_path,
        cache_path=cache_path,
    )
    return "block_cache", str(cache_path)


def compute_spatial_context(
    raster_path: Any,
    vector_path: Any,
    aoi_path: Any | None,
    buffer_m: float,
    output_dir: Any | None = None,
    derive_aoi_from_labels: bool = False,
    memory_budget_mb: int | None = None,
) -> dict:
    """Compute spatial context from real files.

    Parameters
    ----------
    raster_path:
        Path to source GeoTIFF (raster grid reference).
    vector_path:
        Path to vector GT file.
    aoi_path:
        Optional AOI vector file. When provided the effective extent is derived
        from buffered + dissolved AOI bounds reprojected to raster CRS.
    buffer_m:
        Buffer in metres to apply around the AOI geometry.
    derive_aoi_from_labels:
        When True and aoi_path is None, auto-derive AOI from the buffered bbox
        of reprojected label polygons.  Writes aoi_derived_from_labels.gpkg.

    Returns
    -------
    dict with keys:
        raster_crs, vector_crs, vector_crs_source, vector_reprojected,
        vector_runtime_path, aoi_crs, aoi_crs_source, aoi_reprojected,
        aoi_reprojected_path, aoi_source_type, aoi_derivation_method,
        effective_extent_bounds, raster_bounds, raster_width, raster_height,
        source_kind, source_path, is_tiled, block_shapes, approx_source_bytes,
        memory_budget_mb, large_scene_mode, auto_mode_reason,
        full_scene_materialization_allowed, spatial_compute_mode
    """
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required for spatial_context_compute but is not installed"
        ) from exc

    try:
        import geopandas as gpd  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "geopandas is required for spatial_context_compute but is not installed"
        ) from exc

    raster_path = Path(raster_path)
    vector_path = Path(vector_path)

    # --- Open raster ---
    if not raster_path.exists():
        raise ContractError(f"Raster file does not exist: {raster_path}")
    if not raster_path.is_file():
        raise ContractError(f"Raster path is not a regular file: {raster_path}")

    try:
        with rasterio.open(raster_path) as ds:
            raster_crs_obj = ds.crs
            if raster_crs_obj is None:
                raise ContractError(
                    f"Raster has no CRS, cannot resolve spatial context: {raster_path}"
                )
            raster_crs = raster_crs_obj.to_string()
            raster_width = ds.width
            raster_height = ds.height
            bounds = ds.bounds
            raster_bounds = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            is_tiled, block_shapes = _extract_block_metadata(ds)
            dtype_name = str(ds.dtypes[0]) if ds.count > 0 else "float32"
            approx_source_bytes = int(
                ds.width * ds.height * ds.count * _dtype_itemsize_bytes(dtype_name)
            )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"Failed to open raster for spatial context: {exc}") from exc

    total_ram_mb = _detect_total_ram_mb()
    resolved_memory_budget_mb = _resolve_memory_budget_mb(total_ram_mb, memory_budget_mb)
    large_scene_mode, auto_mode_reason = _resolve_large_scene_mode(
        approx_source_bytes=approx_source_bytes,
        is_tiled=is_tiled,
        memory_budget_mb=resolved_memory_budget_mb,
    )

    output_root: Path | None = None
    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

    source_kind, source_path = _prepare_canonical_runtime_source(
        rasterio=rasterio,
        raster_path=raster_path,
        output_root=output_root,
        is_tiled=is_tiled,
        aoi_present=bool(aoi_path is not None or derive_aoi_from_labels),
    )
    try:
        with rasterio.open(source_path) as source_ds:
            is_tiled, block_shapes = _extract_block_metadata(source_ds)
    except Exception as exc:
        raise ContractError(
            f"Failed to open canonical runtime source '{source_path}': {exc}"
        ) from exc

    # --- Open vector ---
    if not vector_path.exists():
        raise ContractError(f"Vector file does not exist: {vector_path}")
    if not vector_path.is_file():
        raise ContractError(f"Vector path is not a regular file: {vector_path}")

    try:
        vec_gdf = gpd.read_file(vector_path)
    except Exception as exc:
        raise ContractError(f"Failed to read vector file for spatial context: {exc}") from exc

    if vec_gdf.crs is None:
        raise ContractError(
            f"Vector file has no CRS, cannot resolve spatial context: {vector_path}"
        )
    vector_crs_source = str(vec_gdf.crs)
    try:
        vec_reproj = vec_gdf.to_crs(raster_crs)
    except Exception as exc:
        raise ContractError(
            f"Failed to reproject vector labels to raster CRS {raster_crs}: {exc}"
        ) from exc
    vector_crs = str(vec_reproj.crs)
    vector_reprojected = vector_crs_source != vector_crs
    vector_runtime_path: str | None = None
    if output_root is not None:
        vector_out = output_root / "vector_in_raster_crs.gpkg"
        try:
            vec_reproj.to_file(vector_out, driver="GPKG")
        except Exception as exc:
            raise ContractError(
                f"Failed to write reprojected vector artifact: {vector_out} ({exc})"
            ) from exc
        vector_runtime_path = str(vector_out)
    else:
        vector_runtime_path = str(vector_path)

    # --- Optional AOI ---
    aoi_crs: str | None = None
    aoi_crs_source: str | None = None
    aoi_reprojected: bool | None = None
    aoi_reprojected_path: str | None = None
    aoi_source_type: str | None = None
    aoi_derivation_method: str | None = None
    effective_extent_bounds: list[float]

    if aoi_path is not None:
        aoi_source_type = "user_provided"
        aoi_path = Path(aoi_path)
        if not aoi_path.exists():
            raise ContractError(f"AOI file does not exist: {aoi_path}")
        if not aoi_path.is_file():
            raise ContractError(f"AOI path is not a regular file: {aoi_path}")

        try:
            aoi_gdf = gpd.read_file(aoi_path)
        except Exception as exc:
            raise ContractError(
                f"Failed to read AOI file for spatial context: {exc}"
            ) from exc

        if aoi_gdf.crs is None:
            raise ContractError(f"AOI file has no CRS: {aoi_path}")

        aoi_crs_source = str(aoi_gdf.crs)

        # Reproject AOI to raster CRS
        try:
            aoi_reproj = aoi_gdf.to_crs(raster_crs)
        except Exception as exc:
            raise ContractError(
                f"Failed to reproject AOI to raster CRS {raster_crs}: {exc}"
            ) from exc

        aoi_crs = str(aoi_reproj.crs)
        aoi_reprojected = aoi_crs_source != aoi_crs
        if output_root is not None:
            aoi_out = output_root / "aoi_in_raster_crs.gpkg"
            try:
                aoi_reproj.to_file(aoi_out, driver="GPKG")
            except Exception as exc:
                raise ContractError(
                    f"Failed to write reprojected AOI artifact: {aoi_out} ({exc})"
                ) from exc
            aoi_reprojected_path = str(aoi_out)

        # Validate CRS is projected (metric) for buffer to make sense
        try:
            from pyproj import CRS as ProjCRS  # noqa: PLC0415

            _proj_crs = ProjCRS.from_user_input(aoi_crs)
            if _proj_crs.is_geographic:
                raise ContractError(
                    f"Raster CRS {raster_crs} is geographic (not projected/metric); "
                    "cannot apply metre-based AOI buffer. "
                    "Reproject the raster to a projected CRS first."
                )
        except ContractError:
            raise
        except Exception:
            pass  # pyproj check is best-effort; proceed

        # Buffer and dissolve
        try:
            aoi_buffered = aoi_reproj.buffer(buffer_m)
            from shapely.ops import unary_union  # noqa: PLC0415

            dissolved = unary_union(aoi_buffered)
            bounds_dissolved = dissolved.bounds  # (minx, miny, maxx, maxy)
            effective_extent_bounds = list(bounds_dissolved)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(
                f"Failed to compute effective extent from AOI buffer: {exc}"
            ) from exc
    elif derive_aoi_from_labels:
        # Derive AOI from buffered bbox of reprojected label polygons.
        from shapely.geometry import box as shapely_box  # noqa: PLC0415

        total_bounds = [float(v) for v in vec_reproj.total_bounds]  # [minx, miny, maxx, maxy]
        if not _bounds_are_finite(total_bounds):
            raise ContractError(
                "Cannot derive AOI from labels: vector bounds are empty or non-finite."
            )

        # Keep the derived AOI strictly rectangular: numeric bbox expansion.
        expanded_bounds = [
            total_bounds[0] - float(buffer_m),
            total_bounds[1] - float(buffer_m),
            total_bounds[2] + float(buffer_m),
            total_bounds[3] + float(buffer_m),
        ]
        clamped_bounds = _clamp_bounds_to_raster(
            raw_bounds=expanded_bounds,
            raster_bounds=raster_bounds,
        )
        derived_rect = shapely_box(*clamped_bounds)

        effective_extent_bounds = clamped_bounds
        aoi_crs = raster_crs
        aoi_source_type = "derived_from_labels"
        aoi_derivation_method = "labels_bbox_buffered"

        if output_root is not None:
            derived_gdf = gpd.GeoDataFrame(geometry=[derived_rect], crs=raster_crs)
            aoi_derived_out = output_root / "aoi_derived_from_labels.gpkg"
            try:
                derived_gdf.to_file(aoi_derived_out, driver="GPKG")
            except Exception as exc:
                raise ContractError(
                    f"Failed to write derived AOI artifact: {aoi_derived_out} ({exc})"
                ) from exc
            aoi_reprojected_path = str(aoi_derived_out)
    else:
        effective_extent_bounds = list(raster_bounds)

    return {
        "raster_crs": raster_crs,
        "vector_crs": vector_crs,
        "vector_crs_source": vector_crs_source,
        "vector_reprojected": vector_reprojected,
        "vector_runtime_path": vector_runtime_path,
        "aoi_crs": aoi_crs,
        "aoi_crs_source": aoi_crs_source,
        "aoi_reprojected": aoi_reprojected,
        "aoi_reprojected_path": aoi_reprojected_path,
        "aoi_source_type": aoi_source_type,
        "aoi_derivation_method": aoi_derivation_method,
        "effective_extent_bounds": effective_extent_bounds,
        "raster_bounds": raster_bounds,
        "raster_width": raster_width,
        "raster_height": raster_height,
        "spatial_compute_mode": SPATIAL_COMPUTE_MODE,
        "source_kind": source_kind,
        "source_path": source_path,
        "is_tiled": is_tiled,
        "block_shapes": block_shapes,
        "approx_source_bytes": approx_source_bytes,
        "memory_budget_mb": resolved_memory_budget_mb,
        "total_system_ram_mb": total_ram_mb,
        "large_scene_mode": large_scene_mode,
        "auto_mode_reason": auto_mode_reason,
        "full_scene_materialization_allowed": (not large_scene_mode),
        "canonical_runtime_source": {
            "source_kind": source_kind,
            "source_path": source_path,
        },
    }
