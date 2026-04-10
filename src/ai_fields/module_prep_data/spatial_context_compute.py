"""Stage 02 runtime compute: spatial context resolution via rasterio + geopandas.

Called by prepare_spatial_context.run_prepare_spatial_context_stage when
runtime_compute_enabled=True.  Raises ContractError on any I/O or CRS failure
so the stage runner can return status="failed".
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError

SPATIAL_COMPUTE_MODE = "rasterio_geopandas_v1"


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


def compute_spatial_context(
    raster_path: Any,
    vector_path: Any,
    aoi_path: Any | None,
    buffer_m: float,
    output_dir: Any | None = None,
    derive_aoi_from_labels: bool = False,
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
        vector_reprojected_path, aoi_crs, aoi_crs_source, aoi_reprojected,
        aoi_reprojected_path, aoi_source_type, aoi_derivation_method,
        effective_extent_bounds, raster_bounds, raster_width, raster_height,
        spatial_compute_mode
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
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"Failed to open raster for spatial context: {exc}") from exc

    output_root: Path | None = None
    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

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
    vector_reprojected_path: str | None = None
    if output_root is not None:
        vector_out = output_root / "vector_in_raster_crs.gpkg"
        try:
            vec_reproj.to_file(vector_out, driver="GPKG")
        except Exception as exc:
            raise ContractError(
                f"Failed to write reprojected vector artifact: {vector_out} ({exc})"
            ) from exc
        vector_reprojected_path = str(vector_out)

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
        "vector_reprojected_path": vector_reprojected_path,
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
    }
