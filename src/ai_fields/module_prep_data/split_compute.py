"""Stage 06 runtime compute: dataset split and normalization stats.

Called by split_dataset.run_split_dataset_stage when runtime_compute_enabled=True.
Raises ContractError on any failure so the stage runner can return status="failed".
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError
from ai_fields.common.progress import iter_progress

SPLIT_COMPUTE_MODE = "numpy_spatial_v1"

_DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)
_REQUIRED_PATCH_FILES = ("_img.tif", "_extent.tif", "_boundary.tif", "_distance.tif", "_valid.tif")
_NORM_STATS_FILENAME = "norm_stats.json"
_NORM_METHOD_EXACT = "exact_full_accumulation"
_NORM_METHOD_RESERVOIR = "reservoir_sampling"
_DEFAULT_NORM_EXACT_THRESHOLD_PIXELS = 1_000_000
_DEFAULT_RESERVOIR_CAPACITY_PER_BAND = 200_000


def load_patch_meta_list(
    patches_dir: Any,
    *,
    progress_enabled: bool | None = None,
) -> list[dict]:
    """Read all *_meta.json files in patches_dir and return list of meta dicts."""
    patches_dir = Path(patches_dir)
    if not patches_dir.exists():
        raise ContractError(f"patches_dir does not exist: {patches_dir}")

    meta_files = sorted(patches_dir.glob("*_meta.json"))
    if not meta_files:
        raise ContractError(f"No *_meta.json files found in {patches_dir}")

    metas = []
    for mf in iter_progress(
        meta_files,
        total=len(meta_files),
        desc="prep_data: read patch meta",
        unit="file",
        progress_enabled=progress_enabled,
        leave=False,
    ):
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ContractError(f"Failed to read patch meta {mf}: {exc}") from exc
        metas.append(data)
    return metas


def assign_splits(
    meta_list: list[dict],
    policy: str,
    random_seed: int | None,
    ratios: tuple[float, float, float] = _DEFAULT_SPLIT_RATIOS,
) -> dict[str, list[str]]:
    """Assign patch_ids to train/val/test splits.

    Parameters
    ----------
    meta_list:
        List of patch meta dicts, each with a 'patch_id' key.
    policy:
        'random' or 'spatial_stratified'.
    random_seed:
        Seed for reproducibility.
    ratios:
        (train_ratio, val_ratio, test_ratio); must sum to 1.0.

    Returns
    -------
    dict with 'train', 'val', 'test' keys → lists of patch_id strings.
    """
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("numpy is required for split assignment") from exc

    if not meta_list:
        raise ContractError("Cannot split an empty patch list.")

    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ContractError(
            f"Split ratios must sum to 1.0, got {train_r + val_r + test_r:.4f}"
        )

    if policy == "spatial_stratified":
        # Sort by (yoff, xoff) for spatial separation
        ordered = sorted(
            meta_list,
            key=lambda m: (m.get("yoff", 0), m.get("xoff", 0)),
        )
    else:
        # random
        rng = np.random.default_rng(seed=random_seed)
        idx = rng.permutation(len(meta_list))
        ordered = [meta_list[i] for i in idx]

    n = len(ordered)
    n_train = max(1, round(n * train_r))
    n_val = max(0, round(n * val_r))
    # test gets the remainder
    n_test = max(0, n - n_train - n_val)

    train_ids = [m["patch_id"] for m in ordered[:n_train]]
    val_ids = [m["patch_id"] for m in ordered[n_train : n_train + n_val]]
    test_ids = [m["patch_id"] for m in ordered[n_train + n_val :]]

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def create_export_layout(
    patches_dir: Path,
    dataset_dir: Path,
    split_assignment: dict[str, list[str]],
    expected_patch_size: int,
    *,
    progress_enabled: bool | None = None,
) -> dict[str, int]:
    """Create the canonical dataset directory layout and link/copy patch files.

    Creates dataset_dir/{train,val,test}/{img,extent,boundary,distance,valid,meta}/
    and places each patch's files in the correct split subdirectory.

    Returns dict with 'train_count', 'val_count', 'test_count'.
    """
    layer_dirs = ["img", "extent", "boundary", "distance", "valid", "meta"]
    counts: dict[str, int] = {}

    total_patch_count = sum(len(ids) for ids in split_assignment.values())
    written_patch_count = 0

    for split_name, patch_ids in split_assignment.items():
        split_dir = dataset_dir / split_name
        for layer in layer_dirs:
            (split_dir / layer).mkdir(parents=True, exist_ok=True)

        for pid in iter_progress(
            patch_ids,
            total=len(patch_ids),
            desc=f"prep_data: export {split_name}",
            unit="patch",
            progress_enabled=progress_enabled,
            leave=False,
        ):
            _validate_patch_sample_shape_contract(
                patches_dir=patches_dir,
                patch_id=pid,
                expected_patch_size=expected_patch_size,
            )
            # Copy or link GeoTIFF layers
            for suffix in ("_img.tif", "_extent.tif", "_boundary.tif", "_distance.tif", "_valid.tif"):
                layer_name = suffix.lstrip("_").removesuffix(".tif")
                src = patches_dir / f"{pid}{suffix}"
                if not src.exists():
                    raise ContractError(f"Patch file missing: {src}")
                dst = split_dir / layer_name / src.name
                try:
                    dst.hardlink_to(src)
                except (AttributeError, OSError):
                    shutil.copy2(src, dst)

            # Copy meta.json
            meta_src = patches_dir / f"{pid}_meta.json"
            if not meta_src.exists():
                raise ContractError(f"Patch meta missing: {meta_src}")
            meta_dst = split_dir / "meta" / meta_src.name
            try:
                meta_dst.hardlink_to(meta_src)
            except (AttributeError, OSError):
                shutil.copy2(meta_src, meta_dst)
            written_patch_count += 1

        counts[f"{split_name}_count"] = len(patch_ids)

    # Keep explicit count for debugging/progress diagnostics in callers.
    counts["written_patch_count"] = written_patch_count
    counts["requested_patch_count"] = total_patch_count

    return counts


def _validate_patch_sample_shape_contract(
    *,
    patches_dir: Path,
    patch_id: str,
    expected_patch_size: int,
) -> None:
    """Fail-fast guardrail against malformed patch rasters entering split exports."""
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required for patch shape validation in split export."
        ) from exc

    expected_shape = (expected_patch_size, expected_patch_size)
    reference_shape: tuple[int, int] | None = None
    reference_layer: str | None = None

    for suffix in _REQUIRED_PATCH_FILES:
        layer_name = suffix.removeprefix("_").removesuffix(".tif")
        path = patches_dir / f"{patch_id}{suffix}"
        if not path.exists():
            raise ContractError(f"Patch file missing: {path}")

        try:
            with rasterio.open(path) as ds:
                shape = (ds.height, ds.width)
        except Exception as exc:
            raise ContractError(
                f"Failed to read patch raster for shape check: {path}: {exc}"
            ) from exc

        if reference_shape is None:
            reference_shape = shape
            reference_layer = layer_name
        elif shape != reference_shape:
            raise ContractError(
                "Patch layer shape mismatch before split export: "
                f"patch_id={patch_id}, layer={layer_name}, shape={shape}, "
                f"reference_layer={reference_layer}, reference_shape={reference_shape}."
            )

        if shape != expected_shape:
            raise ContractError(
                "Patch size contract violation before split export: "
                f"patch_id={patch_id}, layer={layer_name}, shape={shape}, "
                f"expected_shape={expected_shape}."
            )


def _reservoir_update(
    *,
    reservoir: np.ndarray,
    current_size: int,
    seen_count: int,
    values: np.ndarray,
    rng: Any,
) -> tuple[int, int]:
    size = int(current_size)
    seen = int(seen_count)
    cap = int(reservoir.shape[0])
    for value in values:
        seen += 1
        if size < cap:
            reservoir[size] = float(value)
            size += 1
            continue
        j = int(rng.integers(0, seen))
        if j < cap:
            reservoir[j] = float(value)
    return size, seen


def compute_normalization_stats(
    dataset_dir: Path,
    clip_percentiles: tuple[float, float] = (0.5, 99.5),
    *,
    progress_enabled: bool | None = None,
    random_seed: int = 42,
    exact_threshold_pixels: int = _DEFAULT_NORM_EXACT_THRESHOLD_PIXELS,
    reservoir_capacity_per_band: int = _DEFAULT_RESERVOIR_CAPACITY_PER_BAND,
) -> dict:
    """Compute per-band normalization stats in bounded memory.

    Baseline policy:
      - exact accumulation for small train-valid sets (<= exact_threshold_pixels);
      - automatic fallback to per-band reservoir sampling above the threshold.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio and numpy are required for normalization stats") from exc

    if exact_threshold_pixels <= 0:
        raise ContractError(
            f"exact_threshold_pixels must be > 0, got {exact_threshold_pixels}."
        )
    if reservoir_capacity_per_band <= 0:
        raise ContractError(
            "reservoir_capacity_per_band must be > 0, got "
            f"{reservoir_capacity_per_band}."
        )

    train_img_dir = dataset_dir / "train" / "img"
    train_valid_dir = dataset_dir / "train" / "valid"
    img_files = sorted(train_img_dir.glob("*_img.tif"))
    if not img_files:
        raise ContractError(f"No img files found in {train_img_dir}")

    p_lo, p_hi = clip_percentiles
    rng = np.random.default_rng(seed=random_seed)

    exact_mode = True
    total_valid_pixels = 0
    exact_values: dict[int, list[np.ndarray]] = {}
    reservoir: dict[int, np.ndarray] = {}
    reservoir_sizes: dict[int, int] = {}
    reservoir_seen: dict[int, int] = {}
    band_count: int | None = None

    for img_path in iter_progress(
        img_files,
        total=len(img_files),
        desc="prep_data: norm stats",
        unit="patch",
        progress_enabled=progress_enabled,
        leave=False,
    ):
        patch_stem = img_path.stem.replace("_img", "")
        valid_path = train_valid_dir / f"{patch_stem}_valid.tif"
        if not valid_path.exists():
            continue
        try:
            with rasterio.open(img_path) as img_ds:
                img_data = img_ds.read().astype(np.float32)
            with rasterio.open(valid_path) as val_ds:
                valid_mask = val_ds.read(1).astype(bool)
        except Exception as exc:
            raise ContractError(f"Failed to read {img_path}: {exc}") from exc

        if band_count is None:
            band_count = int(img_data.shape[0])
        elif band_count != int(img_data.shape[0]):
            raise ContractError(
                "Inconsistent band count across train/img patches during normalization."
            )

        n_patch_valid = int(valid_mask.sum())
        if n_patch_valid <= 0:
            continue
        total_valid_pixels += n_patch_valid

        if exact_mode and total_valid_pixels <= exact_threshold_pixels:
            for band_idx in range(img_data.shape[0]):
                vals = img_data[band_idx][valid_mask]
                if vals.size == 0:
                    continue
                exact_values.setdefault(band_idx, []).append(vals)
            continue

        # Switch to reservoir mode if threshold exceeded.
        if exact_mode:
            exact_mode = False
            for band_idx in range(img_data.shape[0]):
                reservoir[band_idx] = np.empty(
                    reservoir_capacity_per_band, dtype=np.float32
                )
                reservoir_sizes[band_idx] = 0
                reservoir_seen[band_idx] = 0
                for chunk in exact_values.get(band_idx, []):
                    size, seen = _reservoir_update(
                        reservoir=reservoir[band_idx],
                        current_size=reservoir_sizes[band_idx],
                        seen_count=reservoir_seen[band_idx],
                        values=chunk.astype(np.float32, copy=False),
                        rng=rng,
                    )
                    reservoir_sizes[band_idx] = size
                    reservoir_seen[band_idx] = seen
            exact_values = {}

        for band_idx in range(img_data.shape[0]):
            vals = img_data[band_idx][valid_mask]
            if vals.size == 0:
                continue
            size, seen = _reservoir_update(
                reservoir=reservoir[band_idx],
                current_size=reservoir_sizes[band_idx],
                seen_count=reservoir_seen[band_idx],
                values=vals.astype(np.float32, copy=False),
                rng=rng,
            )
            reservoir_sizes[band_idx] = size
            reservoir_seen[band_idx] = seen

    if band_count is None:
        raise ContractError("No readable train/img patches for normalization stats.")

    band_stats: list[dict[str, float | int]] = []
    if exact_mode:
        method = _NORM_METHOD_EXACT
        approximation = False
        for band_idx in range(band_count):
            chunks = exact_values.get(band_idx, [])
            if not chunks:
                raise ContractError(
                    f"No valid train pixels for band_idx={band_idx} in exact normalization mode."
                )
            all_vals = np.concatenate(chunks)
            band_stats.append(
                {
                    "band_idx": int(band_idx),
                    "p_lo": float(np.percentile(all_vals, p_lo)),
                    "p_hi": float(np.percentile(all_vals, p_hi)),
                }
            )
    else:
        method = _NORM_METHOD_RESERVOIR
        approximation = True
        for band_idx in range(band_count):
            if reservoir_sizes.get(band_idx, 0) <= 0:
                raise ContractError(
                    f"Reservoir sampling produced no samples for band_idx={band_idx}."
                )
            sample = reservoir[band_idx][: reservoir_sizes[band_idx]]
            band_stats.append(
                {
                    "band_idx": int(band_idx),
                    "p_lo": float(np.percentile(sample, p_lo)),
                    "p_hi": float(np.percentile(sample, p_hi)),
                }
            )

    norm_stats = {
        "band_stats": band_stats,
        "n_valid_pixels": int(total_valid_pixels),
        "computed_on": "valid_train_pixels",
        "clip_percentiles": [float(p_lo), float(p_hi)],
        "method": method,
        "approximation": approximation,
        "rng_seed": int(random_seed),
        "exact_threshold_pixels": int(exact_threshold_pixels),
        "reservoir_capacity_per_band": int(reservoir_capacity_per_band),
    }

    norm_stats_path = dataset_dir / _NORM_STATS_FILENAME
    try:
        norm_stats_path.write_text(json.dumps(norm_stats, indent=2), encoding="utf-8")
    except Exception as exc:
        raise ContractError(f"Failed to write norm_stats.json: {exc}") from exc

    norm_stats["norm_stats_path"] = str(norm_stats_path)
    return norm_stats


def compute_and_save_split(
    patches_dir: Any,
    output_dir: Any,
    config: Any,
    *,
    progress_enabled: bool | None = None,
) -> dict:
    """Run the full split: load metas, assign splits, create layout, compute norm stats.

    Parameters
    ----------
    patches_dir:
        Directory containing *_img.tif, *_meta.json, etc. patch files.
    output_dir:
        Parent output directory; dataset layout is created inside.
    config:
        PrepDataConfig with .split.policy, .split.random_seed,
        .normalization.clip_percentiles.

    Returns
    -------
    dict with train_count, val_count, test_count, norm_stats_path,
    split_assignment_executed, export_layout_materialized.
    """
    patches_dir = Path(patches_dir)
    output_dir = Path(output_dir)
    dataset_dir = output_dir / "dataset"

    meta_list = load_patch_meta_list(
        patches_dir,
        progress_enabled=progress_enabled,
    )

    policy = getattr(getattr(config, "split", None), "policy", "random")
    random_seed_val = getattr(getattr(config, "split", None), "random_seed", 42)

    split_assignment = assign_splits(
        meta_list,
        policy=policy,
        random_seed=random_seed_val,
    )

    counts = create_export_layout(
        patches_dir,
        dataset_dir,
        split_assignment,
        expected_patch_size=int(config.patches.patch_size),
        progress_enabled=progress_enabled,
    )

    clip_percentiles_cfg = getattr(
        getattr(config, "normalization", None), "clip_percentiles", (0.5, 99.5)
    )
    norm_stats = compute_normalization_stats(
        dataset_dir,
        clip_percentiles=tuple(float(v) for v in clip_percentiles_cfg),
        progress_enabled=progress_enabled,
        random_seed=int(random_seed_val) if random_seed_val is not None else 42,
    )

    return {
        "train_count": counts.get("train_count", 0),
        "val_count": counts.get("val_count", 0),
        "test_count": counts.get("test_count", 0),
        "norm_stats_path": norm_stats.get("norm_stats_path"),
        "normalization_method": norm_stats.get("method"),
        "normalization_approximation": norm_stats.get("approximation"),
        "normalization_rng_seed": norm_stats.get("rng_seed"),
        "normalization_exact_threshold_pixels": norm_stats.get("exact_threshold_pixels"),
        "normalization_reservoir_capacity_per_band": norm_stats.get(
            "reservoir_capacity_per_band"
        ),
        "split_assignment_executed": True,
        "export_layout_materialized": True,
        "split_compute_mode": SPLIT_COMPUTE_MODE,
    }
