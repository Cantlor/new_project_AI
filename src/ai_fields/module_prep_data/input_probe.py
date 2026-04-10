"""Runtime file probes for stage 01_check_inputs.

Provides minimal metadata extraction from real GeoTIFF and vector files
using rasterio and fiona. Called only when runtime_probe_enabled=True and
no explicit metadata dict is provided to run_check_inputs_stage.

These functions are intentionally narrow:
  - check file existence and readability;
  - extract the minimal metadata fields required by module_prep_data validators.

They do NOT perform reprojection, resampling, nodata fill, or feature contract
assembly. Those belong to later stages.

Source references:
  - module_prep_data.md §6 (01_check_inputs runtime slice)
  - DATA_CONTRACT.md §3.1, §5, §6, §7
  - DECISIONS.md DEC-002 (valid dual role)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError

# Identifies the probe implementation version recorded in manifests.
PROBE_MODE = "rasterio_fiona_probe_v1"


def probe_raster(path: Path) -> dict[str, Any]:
    """Open a GeoTIFF with rasterio and return minimal metadata for validators.

    Returned dict keys (all required by module_prep_data validators):
      crs          -- EPSG string like "EPSG:32637", or None if CRS absent
      band_count   -- number of bands (int)
      width        -- raster width in pixels (int)
      height       -- raster height in pixels (int)
      dtype        -- numpy dtype string of first band, e.g. "uint16"
      nodata       -- nodata scalar or None
      has_valid_mask -- True if any band uses a non-trivial mask (not all-valid)
      readable     -- always True (set so validators accept this metadata)

    The `has_valid_mask` flag covers the case where nodata is set at the
    dataset level (MaskFlags.nodata) or where an alpha/per-dataset mask exists.
    This satisfies the DATA_CONTRACT.md §6.2 valid resolution requirement.

    Raises:
        ContractError: if rasterio is not installed, file is missing, is not
            a regular file, or cannot be read by rasterio.
    """
    try:
        import rasterio
        from rasterio.enums import MaskFlags
    except ImportError as exc:
        raise ContractError(
            "rasterio is required for runtime probing. "
            "Install it with: pip install rasterio"
        ) from exc

    if not path.exists():
        raise ContractError(f"raster_path file does not exist: {path}")
    if not path.is_file():
        raise ContractError(f"raster_path must point to a regular file: {path}")

    try:
        with rasterio.open(path) as ds:
            crs_str: str | None = None
            if ds.crs is not None:
                epsg = ds.crs.to_epsg()
                crs_str = f"EPSG:{epsg}" if epsg is not None else ds.crs.to_string()

            # has_valid_mask: True when any band has a mask beyond "all pixels valid".
            # Covers: MaskFlags.nodata (nodata value set), MaskFlags.alpha,
            # MaskFlags.per_dataset (shared mask band).
            has_valid_mask = any(
                MaskFlags.all_valid not in ds.mask_flag_enums[i]
                for i in range(ds.count)
            )

            return {
                "crs": crs_str,
                "band_count": ds.count,
                "width": ds.width,
                "height": ds.height,
                "dtype": str(ds.dtypes[0]) if ds.count > 0 else None,
                "nodata": ds.nodata,
                "has_valid_mask": has_valid_mask,
                "readable": True,
            }
    except rasterio.errors.RasterioIOError as exc:
        raise ContractError(
            f"raster_path is not readable by rasterio: {path} ({exc})"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"Failed to probe raster at {path}: {exc}") from exc


def probe_vector(path: Path) -> dict[str, Any]:
    """Open a vector file with fiona and return minimal metadata for validators.

    Returned dict keys (all required by module_prep_data validators):
      crs            -- EPSG string like "EPSG:32637", or None if CRS absent
      feature_count  -- number of features (int)
      geometry_types -- sorted list of geometry type strings from layer schema
      readable       -- always True (set so validators accept this metadata)

    Geometry types are read from the layer schema. If the schema reports
    "Unknown" or no geometry type, a scan of up to 100 features is performed
    as a fallback.

    Raises:
        ContractError: if fiona or rasterio is not installed, file is missing,
            is not a regular file, or cannot be read by fiona.
    """
    try:
        import fiona
        import fiona.errors
        from rasterio.crs import CRS as RasterioCRS
    except ImportError as exc:
        raise ContractError(
            "fiona and rasterio are required for runtime probing. "
            "Install them with: pip install fiona rasterio"
        ) from exc

    if not path.exists():
        raise ContractError(f"vector_path file does not exist: {path}")
    if not path.is_file():
        raise ContractError(f"vector_path must point to a regular file: {path}")

    try:
        with fiona.open(path) as collection:
            # CRS: prefer authoritative WKT, fall back to CRS dict/object.
            crs_str: str | None = None
            raw_crs = getattr(collection, "crs_wkt", None) or None
            if raw_crs:
                rc = RasterioCRS.from_wkt(raw_crs)
                epsg = rc.to_epsg()
                crs_str = f"EPSG:{epsg}" if epsg is not None else rc.to_string()
            elif collection.crs:
                rc = RasterioCRS.from_user_input(collection.crs)
                epsg = rc.to_epsg()
                crs_str = f"EPSG:{epsg}" if epsg is not None else rc.to_string()

            # Geometry types: prefer schema, scan features as fallback.
            geometry_types: set[str] = set()
            schema_geom = (collection.schema or {}).get("geometry", "")
            if schema_geom and schema_geom.upper() not in ("", "UNKNOWN", "GEOMETRY", "NONE"):
                geometry_types.add(schema_geom)

            if not geometry_types:
                for i, feat in enumerate(collection):
                    if i >= 100:
                        break
                    if feat and feat.get("geometry"):
                        geometry_types.add(feat["geometry"]["type"])

            return {
                "crs": crs_str,
                "feature_count": len(collection),
                "geometry_types": sorted(geometry_types),
                "readable": True,
            }
    except fiona.errors.DriverError as exc:
        raise ContractError(
            f"vector_path is not readable by fiona: {path} ({exc})"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"Failed to probe vector at {path}: {exc}") from exc
