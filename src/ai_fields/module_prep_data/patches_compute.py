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
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError
from ai_fields.common.progress import progress_bar

PATCHES_COMPUTE_MODE = "rasterio_numpy_v1"
_UNSET = object()

_REJECTION_VALID_RATIO_THRESHOLD = 0.05


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
