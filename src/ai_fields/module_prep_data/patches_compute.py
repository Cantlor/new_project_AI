"""Stage 05 runtime compute: patch extraction via rasterio + numpy.

Called by make_patches.run_make_patches_stage when
runtime_compute_enabled=True.  Raises ContractError on any failure.

Patch classification (sampling_class):
    near_invalid       — valid_ratio < 0.5
    boundary_positive  — edge_ratio >= 0.1 and valid_ratio >= 0.5
    hard_negative      — valid_ratio >= 0.8 and edge_ratio == 0.0
    center_positive    — valid_ratio >= 0.8 and 0 < edge_ratio < 0.1
    (rejection: valid_ratio < 0.05)

Patch files written per patch_id inside output_dir/patches/:
    <patch_id>_img.tif
    <patch_id>_extent.tif
    <patch_id>_boundary.tif
    <patch_id>_distance.tif
    <patch_id>_valid.tif
    <patch_id>_meta.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError
from ai_fields.common.progress import progress_bar
from ai_fields.module_prep_data.features_compute import (
    build_feature_stack_from_array,
)
from ai_fields.module_prep_data.targets_compute import (
    build_boundary_target_for_window,
    compute_distance_target_clipped,
    rasterize_extent_for_window,
)

PATCHES_COMPUTE_MODE = "rasterio_numpy_v1"
_UNSET = object()

_REJECTION_VALID_RATIO_THRESHOLD = 0.05
_IGNORE_LABEL = 255


def classify_patch(valid_ratio: float, edge_ratio: float) -> str | None:
    """Return sampling class for a patch or None if the patch should be rejected."""
    if valid_ratio < _REJECTION_VALID_RATIO_THRESHOLD:
        return None  # reject
    if valid_ratio < 0.5:
        return "near_invalid"
    if edge_ratio >= 0.1:
        return "boundary_positive"
    if valid_ratio >= 0.8 and edge_ratio == 0.0:
        return "hard_negative"
    return "center_positive"


def generate_patch_windows(
    height: int, width: int, patch_size: int, stride: int | None = None
) -> list[dict]:
    """Generate non-overlapping (or strided) patch windows.

    Returns list of full-size windows where:
      - every window has width == patch_size and height == patch_size;
      - every window is fully inside raster bounds;
      - right/bottom borders are covered by adding a final aligned window when
        stride does not land exactly on the last possible offset.
    """
    if stride is None:
        stride = patch_size // 2
    if patch_size <= 0:
        raise ContractError(f"patch_size must be > 0, got {patch_size}.")
    if stride <= 0:
        raise ContractError(f"stride must be > 0, got {stride}.")
    if height < patch_size or width < patch_size:
        return []

    def _axis_offsets(length: int) -> list[int]:
        last = length - patch_size
        offsets = list(range(0, last + 1, stride))
        if not offsets:
            return [last]
        if offsets[-1] != last:
            offsets.append(last)
        return offsets

    windows = []
    for yoff in _axis_offsets(height):
        for xoff in _axis_offsets(width):
            windows.append(
                {"xoff": xoff, "yoff": yoff, "width": patch_size, "height": patch_size}
            )
    return windows


def _write_patch_tif(
    path: Path, array: np.ndarray, profile: dict, window_transform: Any, *, nodata: Any = _UNSET
) -> None:
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for patch writing") from exc

    if array.ndim == 2:
        array = array[np.newaxis, ...]
    count = array.shape[0]

    out_profile = dict(profile)
    out_profile.update(
        {
            "count": count,
            "dtype": str(array.dtype),
            "driver": "GTiff",
            "width": array.shape[2],
            "height": array.shape[1],
            "transform": window_transform,
        }
    )
    if nodata is _UNSET:
        # Keep source profile nodata as-is (img/distance patch paths).
        pass
    elif nodata is None:
        # Semantic uint8 patch layers must not inherit source nodata metadata.
        out_profile.pop("nodata", None)
    else:
        out_profile["nodata"] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(array)
    except Exception as exc:
        raise ContractError(f"Failed to write patch GeoTIFF to {path}: {exc}") from exc


def extract_and_save_patch(
    datasets_dict: dict,
    window: dict,
    patch_id: str,
    output_dir: Path,
    meta_fields: dict,
) -> dict | None:
    """Extract a windowed patch from all datasets and save to disk.

    Returns dict with patch metadata, or None if patch is rejected.
    """
    try:
        import rasterio.windows  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for patch extraction") from exc

    xoff = window["xoff"]
    yoff = window["yoff"]
    w = window.get("width", meta_fields.get("patch_size", 512))
    h = window.get("height", meta_fields.get("patch_size", 512))
    expected_patch_size = int(meta_fields.get("patch_size", 512))

    if (w, h) != (expected_patch_size, expected_patch_size):
        raise ContractError(
            "Patch window size contract violation: "
            f"window=({h},{w}) but patch_size={expected_patch_size} "
            f"(patch_id={patch_id}, xoff={xoff}, yoff={yoff})."
        )

    rio_window = rasterio.windows.Window(
        col_off=xoff, row_off=yoff, width=w, height=h
    )

    # Read valid band first for classification
    valid_ds = datasets_dict["valid"]
    try:
        valid_arr = valid_ds.read(1, window=rio_window)
    except Exception as exc:
        raise ContractError(f"Failed to read valid band at window {window}: {exc}") from exc

    total_pixels = valid_arr.size
    valid_pixels = int(np.sum(valid_arr > 0))
    valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0

    # Compute edge_ratio from boundary
    boundary_ds = datasets_dict["boundary"]
    try:
        boundary_arr = boundary_ds.read(1, window=rio_window)
    except Exception as exc:
        raise ContractError(
            f"Failed to read boundary band at window {window}: {exc}"
        ) from exc

    edge_pixels = int(np.sum(boundary_arr == 1))  # skeleton pixels only
    edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0.0

    sampling_class = classify_patch(valid_ratio, edge_ratio)
    if sampling_class is None:
        return None  # rejected

    # Compute window transform from reference dataset
    ref_ds = datasets_dict.get("img", datasets_dict["valid"])
    try:
        window_transform = rasterio.windows.transform(rio_window, ref_ds.transform)
    except Exception as exc:
        raise ContractError(f"Failed to compute window transform: {exc}") from exc

    base_profile = ref_ds.profile.copy()

    patch_dir = output_dir
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Write each layer
    layers = {
        "img": (datasets_dict["img"], _UNSET),
        "extent": (datasets_dict["extent"], None),
        "boundary": (datasets_dict["boundary"], None),
        "distance": (datasets_dict["distance"], _UNSET),
        "valid": (datasets_dict["valid"], None),
    }

    for layer_name, layer_desc in layers.items():
        ds, nodata_override = layer_desc
        try:
            arr = ds.read(window=rio_window)
        except Exception as exc:
            raise ContractError(
                f"Failed to read {layer_name} at window {window}: {exc}"
            ) from exc
        out_path = patch_dir / f"{patch_id}_{layer_name}.tif"
        _write_patch_tif(
            out_path,
            arr,
            base_profile,
            window_transform,
            nodata=nodata_override,
        )

    # Write meta.json
    patch_meta = {
        "patch_id": patch_id,
        "feature_mode": meta_fields.get("feature_mode", "raw8"),
        "feature_channel_count": meta_fields.get("feature_channel_count", 8),
        "channel_names": meta_fields.get("channel_names", []),
        "valid_saved_separately": True,
        "xoff": xoff,
        "yoff": yoff,
        "patch_width": w,
        "patch_height": h,
        "valid_ratio": round(valid_ratio, 4),
        "edge_ratio": round(edge_ratio, 4),
        "sampling_class": sampling_class,
        "source_crs": str(ref_ds.crs) if ref_ds.crs else None,
    }
    meta_path = patch_dir / f"{patch_id}_meta.json"
    try:
        meta_path.write_text(json.dumps(patch_meta, indent=2), encoding="utf-8")
    except Exception as exc:
        raise ContractError(f"Failed to write patch meta.json: {exc}") from exc

    return patch_meta


def _normalize_effective_bounds(raw_bounds: Any | None) -> tuple[float, float, float, float] | None:
    if raw_bounds is None:
        return None
    if isinstance(raw_bounds, (str, bytes)) or not isinstance(raw_bounds, (list, tuple)):
        raise ContractError("effective_extent_bounds must be a 4-element numeric sequence or null.")
    if len(raw_bounds) != 4:
        raise ContractError(
            f"effective_extent_bounds must have length 4, got {len(raw_bounds)}."
        )
    out: list[float] = []
    for idx, value in enumerate(raw_bounds):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(
                f"effective_extent_bounds[{idx}] must be numeric, got {value!r}."
            )
        out.append(float(value))
    if out[0] >= out[2] or out[1] >= out[3]:
        raise ContractError(
            "effective_extent_bounds must satisfy minx < maxx and miny < maxy."
        )
    return (out[0], out[1], out[2], out[3])


def _align_window_to_blocks(window: Any, *, ds: Any) -> tuple[Any, list[int] | None]:
    try:
        import rasterio.windows  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for block-aware window alignment") from exc

    block_shapes = getattr(ds, "block_shapes", None)
    if not block_shapes:
        return window, None

    try:
        rows = int(block_shapes[0][0])
        cols = int(block_shapes[0][1])
    except Exception:
        return window, None
    if rows <= 0 or cols <= 0:
        return window, None

    col_off = int(window.col_off)
    row_off = int(window.row_off)
    width = int(window.width)
    height = int(window.height)
    col_end = col_off + width
    row_end = row_off + height

    aligned_col_off = (col_off // cols) * cols
    aligned_row_off = (row_off // rows) * rows
    aligned_col_end = int(math.ceil(col_end / cols) * cols)
    aligned_row_end = int(math.ceil(row_end / rows) * rows)

    aligned_col_end = min(aligned_col_end, int(ds.width))
    aligned_row_end = min(aligned_row_end, int(ds.height))

    aligned = rasterio.windows.Window(
        col_off=aligned_col_off,
        row_off=aligned_row_off,
        width=max(0, aligned_col_end - aligned_col_off),
        height=max(0, aligned_row_end - aligned_row_off),
    )
    if aligned.width <= 0 or aligned.height <= 0:
        return window, None
    return aligned, [rows, cols]


def _read_raw_and_valid_window(
    ds: Any,
    *,
    window: Any,
    align_to_blocks: bool,
) -> tuple[np.ndarray, np.ndarray, list[int] | None]:
    """Read raw8 and valid mask for a window with optional block alignment."""
    read_window = window
    used_block_shape: list[int] | None = None
    if align_to_blocks:
        read_window, used_block_shape = _align_window_to_blocks(window, ds=ds)

    if ds.count < 8:
        raise ContractError(
            f"Canonical runtime source has {ds.count} bands, but raw8 requires 8 bands."
        )

    try:
        raw_read = ds.read(list(range(1, 9)), window=read_window).astype(np.float32)
        dataset_mask = ds.dataset_mask(window=read_window)
    except Exception as exc:
        raise ContractError(f"Failed to read source raster window: {exc}") from exc

    valid_read = (dataset_mask > 0).astype(np.uint8)
    nodata = ds.nodata
    if nodata is not None:
        try:
            nodata_valid = (raw_read[0].astype(np.float64) != float(nodata)).astype(np.uint8)
            valid_read = np.minimum(valid_read, nodata_valid)
        except Exception:
            pass

    if read_window != window:
        r0 = int(window.row_off - read_window.row_off)
        c0 = int(window.col_off - read_window.col_off)
        r1 = r0 + int(window.height)
        c1 = c0 + int(window.width)
        raw = raw_read[:, r0:r1, c0:c1]
        valid = valid_read[r0:r1, c0:c1]
    else:
        raw = raw_read
        valid = valid_read

    return raw, valid, used_block_shape


def _resolve_generation_window(
    ds: Any,
    *,
    spatial_context_mode: str,
    effective_extent_bounds: tuple[float, float, float, float] | None,
) -> Any:
    try:
        import rasterio.windows  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for generation window resolution") from exc

    full_window = rasterio.windows.Window(
        col_off=0,
        row_off=0,
        width=int(ds.width),
        height=int(ds.height),
    )
    if spatial_context_mode != "aoi_limited" or effective_extent_bounds is None:
        return full_window

    bounds = effective_extent_bounds
    ds_bounds = ds.bounds
    clamped = [
        max(float(bounds[0]), float(ds_bounds.left)),
        max(float(bounds[1]), float(ds_bounds.bottom)),
        min(float(bounds[2]), float(ds_bounds.right)),
        min(float(bounds[3]), float(ds_bounds.top)),
    ]
    if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
        raise ContractError(
            "AOI-limited effective_extent_bounds do not intersect source raster bounds."
        )
    try:
        window = rasterio.windows.from_bounds(
            clamped[0],
            clamped[1],
            clamped[2],
            clamped[3],
            transform=ds.transform,
        )
        window = window.round_offsets().round_lengths()
        window = window.intersection(full_window)
    except Exception as exc:
        raise ContractError(
            f"Failed to resolve AOI-limited generation window from bounds: {exc}"
        ) from exc
    if window.width <= 0 or window.height <= 0:
        raise ContractError("Resolved AOI-limited generation window is empty.")
    return window


def _build_vector_index(vector_runtime_path: Any, raster_crs: Any) -> tuple[list[Any], Any, dict[str, Any]]:
    try:
        import geopandas as gpd  # noqa: PLC0415
        from shapely.strtree import STRtree  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("geopandas + shapely are required for patch-first target compute") from exc

    vector_path = Path(vector_runtime_path)
    if not vector_path.exists():
        raise ContractError(f"vector_runtime_path does not exist: {vector_path}")
    if not vector_path.is_file():
        raise ContractError(f"vector_runtime_path is not a regular file: {vector_path}")

    try:
        gdf = gpd.read_file(vector_path)
    except Exception as exc:
        raise ContractError(f"Failed to read vector_runtime_path '{vector_path}': {exc}") from exc

    if gdf.crs is None:
        raise ContractError(f"vector_runtime_path has no CRS: {vector_path}")
    if raster_crs is not None and str(gdf.crs) != str(raster_crs):
        try:
            gdf = gdf.to_crs(raster_crs)
        except Exception as exc:
            raise ContractError(
                f"Failed to reproject vector_runtime_path to raster CRS: {exc}"
            ) from exc

    geometries = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        raise ContractError(
            f"vector_runtime_path '{vector_path}' does not contain non-empty geometries."
        )

    tree = None
    try:
        tree = STRtree(geometries)
    except Exception:
        tree = None

    policy = {
        "mode": "in_memory_full_gdf_with_strtree_v1",
        "accepted_residual_risk": "vector_memory_scales_with_polygon_count",
        "vector_feature_count_loaded": len(geometries),
        "strtree_built": tree is not None,
    }
    return geometries, tree, policy


def _query_geometries_for_bounds(
    *,
    all_geometries: list[Any],
    tree: Any,
    bounds: tuple[float, float, float, float],
) -> list[Any]:
    if tree is None:
        return all_geometries
    try:
        from shapely.geometry import box as shapely_box  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("shapely is required for STRtree queries") from exc

    query_geom = shapely_box(*bounds)
    try:
        hits = tree.query(query_geom)
    except Exception:
        return all_geometries

    if hits is None:
        return []
    if isinstance(hits, np.ndarray):
        hit_list = hits.tolist()
    else:
        hit_list = list(hits)
    if not hit_list:
        return []

    first = hit_list[0]
    if isinstance(first, (int, np.integer)):
        out: list[Any] = []
        for idx in hit_list:
            try:
                out.append(all_geometries[int(idx)])
            except Exception:
                continue
        return out

    return [geom for geom in hit_list if geom is not None and not geom.is_empty]


def _build_patch_meta(
    *,
    patch_id: str,
    feature_mode: str,
    feature_channel_count: int,
    channel_names: list[str],
    xoff: int,
    yoff: int,
    patch_size: int,
    valid_ratio: float,
    edge_ratio: float,
    sampling_class: str,
    source_crs: Any,
    halo_px: int,
    distance_clip_px: int,
) -> dict[str, Any]:
    return {
        "patch_id": patch_id,
        "feature_mode": feature_mode,
        "feature_channel_count": feature_channel_count,
        "channel_names": channel_names,
        "valid_saved_separately": True,
        "xoff": xoff,
        "yoff": yoff,
        "patch_width": patch_size,
        "patch_height": patch_size,
        "valid_ratio": round(valid_ratio, 4),
        "edge_ratio": round(edge_ratio, 4),
        "sampling_class": sampling_class,
        "source_crs": str(source_crs) if source_crs is not None else None,
        "halo_px": halo_px,
        "distance_clip_px": distance_clip_px,
        "distance_mode": "patch_local_clipped",
        "double_read_avoided": True,
    }


def compute_and_save_patches_from_source(
    raster_source_path: Any,
    vector_runtime_path: Any,
    output_dir: Any,
    config: Any,
    feature_mode: str,
    *,
    source_kind: str = "direct_raster",
    spatial_context_mode: str = "full_raster",
    effective_extent_bounds: Any | None = None,
    progress_enabled: bool | None = None,
) -> dict:
    """Patch-first runtime materializer from canonical source + runtime vector.

    This is the baseline production path. It does not depend on full-scene
    intermediates from stage 03/04.
    """
    try:
        import rasterio  # noqa: PLC0415
        import rasterio.windows  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for patch-first stage 05 runtime") from exc

    raster_source_path = Path(raster_source_path)
    if not raster_source_path.exists():
        raise ContractError(f"canonical source_path does not exist: {raster_source_path}")
    if not raster_source_path.is_file():
        raise ContractError(f"canonical source_path is not a regular file: {raster_source_path}")

    patch_size = int(config.patches.patch_size)
    halo_px = int(config.patches.halo_px)
    distance_clip_px = int(config.distance.distance_clip_px)
    if distance_clip_px > halo_px:
        raise ContractError(
            "distance_clip_px must be <= halo_px for patch-local clipped distance, got "
            f"distance_clip_px={distance_clip_px}, halo_px={halo_px}."
        )
    stride = getattr(config.patches, "stride", None)
    if stride is None:
        stride = patch_size // 2

    from ai_fields.module_prep_data.features_compute import CHANNEL_SEMANTICS  # noqa: PLC0415

    channel_names = list(CHANNEL_SEMANTICS.get(feature_mode, []))
    patches_subdir = Path(output_dir) / "patches"
    patches_subdir.mkdir(parents=True, exist_ok=True)

    written_total = 0
    written_center = 0
    written_boundary = 0
    written_negative = 0
    written_near_invalid = 0
    rejected_count = 0
    block_aligned_reads = 0
    block_shape_used: list[int] | None = None

    effective_bounds = _normalize_effective_bounds(effective_extent_bounds)

    with rasterio.open(raster_source_path) as ds:
        generation_window = _resolve_generation_window(
            ds,
            spatial_context_mode=spatial_context_mode,
            effective_extent_bounds=effective_bounds,
        )
        windows_local = generate_patch_windows(
            int(generation_window.height),
            int(generation_window.width),
            patch_size,
            stride=stride,
        )
        if not windows_local:
            raise ContractError(
                "No full-size patch windows can be generated from spatial context envelope: "
                f"generation_window=({int(generation_window.height)},{int(generation_window.width)}), "
                f"patch_size={patch_size}."
            )

        candidate_windows = [
            {
                "xoff": int(generation_window.col_off) + int(win["xoff"]),
                "yoff": int(generation_window.row_off) + int(win["yoff"]),
                "width": patch_size,
                "height": patch_size,
            }
            for win in windows_local
        ]

        all_geometries, tree, vector_policy = _build_vector_index(
            vector_runtime_path=vector_runtime_path,
            raster_crs=ds.crs,
        )

        base_profile = ds.profile.copy()
        with progress_bar(
            total=len(candidate_windows),
            desc="prep_data: patch-first materialize",
            unit="patch",
            progress_enabled=progress_enabled,
            leave=False,
        ) as bar:
            for idx, win in enumerate(candidate_windows):
                xoff = int(win["xoff"])
                yoff = int(win["yoff"])
                central_window = rasterio.windows.Window(
                    col_off=xoff,
                    row_off=yoff,
                    width=patch_size,
                    height=patch_size,
                )
                if (
                    xoff < 0
                    or yoff < 0
                    or xoff + patch_size > ds.width
                    or yoff + patch_size > ds.height
                ):
                    rejected_count += 1
                    bar.update(1)
                    continue

                halo_col_off = max(0, xoff - halo_px)
                halo_row_off = max(0, yoff - halo_px)
                halo_col_end = min(int(ds.width), xoff + patch_size + halo_px)
                halo_row_end = min(int(ds.height), yoff + patch_size + halo_px)
                halo_window = rasterio.windows.Window(
                    col_off=halo_col_off,
                    row_off=halo_row_off,
                    width=max(0, halo_col_end - halo_col_off),
                    height=max(0, halo_row_end - halo_row_off),
                )
                if halo_window.width <= 0 or halo_window.height <= 0:
                    rejected_count += 1
                    bar.update(1)
                    continue

                raw_halo, valid_halo, used_block_shape = _read_raw_and_valid_window(
                    ds,
                    window=halo_window,
                    align_to_blocks=True,
                )
                if used_block_shape is not None:
                    block_aligned_reads += 1
                    block_shape_used = used_block_shape

                feature_halo = build_feature_stack_from_array(raw_halo, feature_mode)

                halo_transform = rasterio.windows.transform(halo_window, ds.transform)
                halo_bounds = rasterio.windows.bounds(halo_window, ds.transform)
                local_geometries = _query_geometries_for_bounds(
                    all_geometries=all_geometries,
                    tree=tree,
                    bounds=(
                        float(halo_bounds[0]),
                        float(halo_bounds[1]),
                        float(halo_bounds[2]),
                        float(halo_bounds[3]),
                    ),
                )

                extent_halo = rasterize_extent_for_window(
                    transform=halo_transform,
                    height=int(halo_window.height),
                    width=int(halo_window.width),
                    vector_geometries=local_geometries,
                )
                boundary_halo, boundary_raw_halo = build_boundary_target_for_window(
                    transform=halo_transform,
                    height=int(halo_window.height),
                    width=int(halo_window.width),
                    vector_geometries=local_geometries,
                    buffer_iterations=3,
                )
                distance_halo = compute_distance_target_clipped(
                    boundary_raw=boundary_raw_halo,
                    valid_mask=valid_halo,
                    distance_clip_px=distance_clip_px,
                )

                r0 = int(yoff - halo_row_off)
                c0 = int(xoff - halo_col_off)
                r1 = r0 + patch_size
                c1 = c0 + patch_size

                img_patch = feature_halo[:, r0:r1, c0:c1]
                extent_patch = extent_halo[r0:r1, c0:c1].astype(np.uint8, copy=False)
                boundary_patch = boundary_halo[r0:r1, c0:c1].astype(np.uint8, copy=False)
                distance_patch = distance_halo[r0:r1, c0:c1].astype(np.float32, copy=False)
                valid_patch = valid_halo[r0:r1, c0:c1].astype(np.uint8, copy=False)

                if (
                    img_patch.shape[-2:] != (patch_size, patch_size)
                    or extent_patch.shape != (patch_size, patch_size)
                    or boundary_patch.shape != (patch_size, patch_size)
                    or distance_patch.shape != (patch_size, patch_size)
                    or valid_patch.shape != (patch_size, patch_size)
                ):
                    rejected_count += 1
                    bar.update(1)
                    continue

                # Explicit ignore semantics for extent target.
                extent_patch = extent_patch.copy()
                extent_patch[valid_patch == 0] = _IGNORE_LABEL

                total_pixels = float(patch_size * patch_size)
                valid_ratio = float(np.sum(valid_patch > 0) / total_pixels)
                edge_ratio = float(np.sum(boundary_patch == 1) / total_pixels)
                sampling_class = classify_patch(valid_ratio=valid_ratio, edge_ratio=edge_ratio)
                if sampling_class is None:
                    rejected_count += 1
                    bar.update(1)
                    continue

                patch_id = f"patch_{idx:06d}"
                central_transform = rasterio.windows.transform(central_window, ds.transform)

                _write_patch_tif(
                    patches_subdir / f"{patch_id}_img.tif",
                    img_patch,
                    base_profile,
                    central_transform,
                    nodata=_UNSET,
                )
                _write_patch_tif(
                    patches_subdir / f"{patch_id}_extent.tif",
                    extent_patch,
                    base_profile,
                    central_transform,
                    nodata=None,
                )
                _write_patch_tif(
                    patches_subdir / f"{patch_id}_boundary.tif",
                    boundary_patch,
                    base_profile,
                    central_transform,
                    nodata=None,
                )
                _write_patch_tif(
                    patches_subdir / f"{patch_id}_distance.tif",
                    distance_patch,
                    base_profile,
                    central_transform,
                    nodata=_UNSET,
                )
                _write_patch_tif(
                    patches_subdir / f"{patch_id}_valid.tif",
                    valid_patch,
                    base_profile,
                    central_transform,
                    nodata=None,
                )

                patch_meta = _build_patch_meta(
                    patch_id=patch_id,
                    feature_mode=feature_mode,
                    feature_channel_count=int(img_patch.shape[0]),
                    channel_names=channel_names,
                    xoff=xoff,
                    yoff=yoff,
                    patch_size=patch_size,
                    valid_ratio=valid_ratio,
                    edge_ratio=edge_ratio,
                    sampling_class=sampling_class,
                    source_crs=ds.crs,
                    halo_px=halo_px,
                    distance_clip_px=distance_clip_px,
                )
                (patches_subdir / f"{patch_id}_meta.json").write_text(
                    json.dumps(patch_meta, indent=2),
                    encoding="utf-8",
                )

                written_total += 1
                if sampling_class == "center_positive":
                    written_center += 1
                elif sampling_class == "boundary_positive":
                    written_boundary += 1
                elif sampling_class == "hard_negative":
                    written_negative += 1
                elif sampling_class == "near_invalid":
                    written_near_invalid += 1
                bar.update(1)
                bar.set_postfix(written=written_total, rejected=rejected_count)

    return {
        "written_total": written_total,
        "written_center": written_center,
        "written_boundary": written_boundary,
        "written_negative": written_negative,
        "written_near_invalid": written_near_invalid,
        "rejection_stats": {
            "invalid_ratio_rejects": rejected_count,
            "mask_ratio_rejects": None,
            "boundary_quality_rejects": None,
            "duplicate_or_overlap_rejects": None,
        },
        "patches_subdir": patches_subdir,
        "patches_compute_mode": PATCHES_COMPUTE_MODE,
        "runtime_path": "patch_first_from_source",
        "source_kind": source_kind,
        "spatial_context_mode": spatial_context_mode,
        "effective_extent_bounds_used": (
            list(effective_bounds) if effective_bounds is not None else None
        ),
        "halo_px": halo_px,
        "distance_clip_px": distance_clip_px,
        "double_read_avoided": True,
        "candidate_windows_total": len(candidate_windows),
        "vector_load_policy": vector_policy,
        "block_aware_reading": {
            "enabled": True,
            "block_aligned_read_count": block_aligned_reads,
            "block_shape_used": block_shape_used,
        },
    }


def compute_and_save_patches(
    img_path: Any,
    extent_path: Any,
    boundary_path: Any,
    distance_path: Any,
    valid_path: Any,
    output_dir: Any,
    config: Any,
    feature_mode: str,
    progress_enabled: bool | None = None,
) -> dict:
    """Extract patches from all layer GeoTIFFs and write to output_dir/patches/.

    Parameters
    ----------
    img_path, extent_path, boundary_path, distance_path, valid_path:
        Paths to the corresponding GeoTIFF layers.
    output_dir:
        Parent directory; patches go in output_dir/patches/.
    config:
        PrepDataConfig or compatible object with .patches.patch_size, .patches.stride_m.
    feature_mode:
        'raw8' or 'raw8_idx3'.

    Returns
    -------
    dict with keys:
        written_total, written_center, written_boundary, written_negative,
        written_near_invalid, rejection_stats, patches_subdir
    """
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for compute_and_save_patches") from exc

    for name, p in [
        ("img_path", img_path),
        ("extent_path", extent_path),
        ("boundary_path", boundary_path),
        ("distance_path", distance_path),
        ("valid_path", valid_path),
    ]:
        p = Path(p)
        if not p.exists():
            raise ContractError(f"{name} does not exist: {p}")

    output_dir = Path(output_dir)
    patches_subdir = output_dir / "patches"
    patches_subdir.mkdir(parents=True, exist_ok=True)

    patch_size = config.patches.patch_size
    stride = getattr(config.patches, "stride", None)

    try:
        from ai_fields.module_prep_data.features_compute import CHANNEL_SEMANTICS  # noqa: PLC0415

        channel_names = CHANNEL_SEMANTICS.get(feature_mode, [])
    except Exception:
        channel_names = []

    meta_fields = {
        "feature_mode": feature_mode,
        "feature_channel_count": len(channel_names),
        "channel_names": channel_names,
        "patch_size": patch_size,
    }

    # Open all datasets
    layer_paths = {
        "img": Path(img_path),
        "extent": Path(extent_path),
        "boundary": Path(boundary_path),
        "distance": Path(distance_path),
        "valid": Path(valid_path),
    }

    written_total = 0
    written_center = 0
    written_boundary = 0
    written_negative = 0
    written_near_invalid = 0
    rejected_count = 0

    try:
        datasets = {name: rasterio.open(path) for name, path in layer_paths.items()}
        try:
            ref_ds = datasets["img"]
            height, width = ref_ds.height, ref_ds.width

            windows = generate_patch_windows(
                height, width, patch_size, stride=stride
            )
            if not windows:
                raise ContractError(
                    "No full-size patch windows can be generated: "
                    f"raster_size=({height},{width}), patch_size={patch_size}. "
                    "Provide inputs/config that allow at least one full patch."
                )

            with progress_bar(
                total=len(windows),
                desc="prep_data: extract patches",
                unit="patch",
                progress_enabled=progress_enabled,
                leave=False,
            ) as bar:
                for idx, window in enumerate(windows):
                    patch_id = f"patch_{idx:06d}"
                    try:
                        patch_meta = extract_and_save_patch(
                            datasets_dict=datasets,
                            window=window,
                            patch_id=patch_id,
                            output_dir=patches_subdir,
                            meta_fields=meta_fields,
                        )
                    except ContractError:
                        raise

                    if patch_meta is None:
                        rejected_count += 1
                        bar.update(1)
                        bar.set_postfix(
                            written=written_total,
                            rejected=rejected_count,
                        )
                        continue

                    written_total += 1
                    cls = patch_meta["sampling_class"]
                    if cls == "center_positive":
                        written_center += 1
                    elif cls == "boundary_positive":
                        written_boundary += 1
                    elif cls == "hard_negative":
                        written_negative += 1
                    elif cls == "near_invalid":
                        written_near_invalid += 1
                    bar.update(1)
                    bar.set_postfix(
                        written=written_total,
                        rejected=rejected_count,
                    )
        finally:
            for ds in datasets.values():
                try:
                    ds.close()
                except Exception:
                    pass
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"Failed during patch extraction: {exc}") from exc

    return {
        "written_total": written_total,
        "written_center": written_center,
        "written_boundary": written_boundary,
        "written_negative": written_negative,
        "written_near_invalid": written_near_invalid,
        "rejection_stats": {
            "invalid_ratio_rejects": rejected_count,
            "mask_ratio_rejects": None,
            "boundary_quality_rejects": None,
            "duplicate_or_overlap_rejects": None,
        },
        "patches_subdir": patches_subdir,
    }
