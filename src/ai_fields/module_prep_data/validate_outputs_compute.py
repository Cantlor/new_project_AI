"""Stage 07 runtime compute: dataset integrity validation via rasterio.

Called by validate_outputs.run_validate_outputs_stage when runtime_compute_enabled=True.
Raises ContractError on any failure so the stage runner can return status="failed".
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError

VALIDATE_COMPUTE_MODE = "rasterio_scan_v1"

_SPLITS = ("train", "val", "test")
_REQUIRED_LAYER_SUBDIRS = ("img", "extent", "boundary", "distance", "valid", "meta")

# Expected value domains per target layer
_EXTENT_VALID_VALUES = frozenset({0, 1, 255})
_BOUNDARY_VALID_VALUES = frozenset({0, 1, 2})
_VALID_VALID_VALUES = frozenset({0, 1})


def scan_split_dir(
    split_dir: Path, required_subdirs: tuple[str, ...] = _REQUIRED_LAYER_SUBDIRS
) -> dict[str, list[Path]]:
    """Scan split directory and return per-layer file lists.

    Returns dict mapping layer_name → sorted list of Path objects.
    Raises ContractError if split_dir doesn't exist or a required subdir is missing.
    """
    if not split_dir.exists():
        raise ContractError(f"Split directory does not exist: {split_dir}")

    result: dict[str, list[Path]] = {}
    for subdir_name in required_subdirs:
        subdir = split_dir / subdir_name
        if not subdir.exists():
            raise ContractError(
                f"Required subdirectory missing: {subdir} in {split_dir}"
            )
        if subdir_name == "meta":
            files = sorted(subdir.glob("*_meta.json"))
        else:
            files = sorted(subdir.glob(f"*_{subdir_name}.tif"))
        result[subdir_name] = files
    return result


def check_shapes_consistency(dataset_dir: Path, expected_patch_size: int | None = None) -> dict:
    """Check that all layers within each patch have matching (H, W).

    Samples all patches across train/val/test splits.
    Returns {consistent: bool, mismatches: list, checked_count: int}.
    When expected_patch_size is provided, also verifies every patch layer has
    exact shape (expected_patch_size, expected_patch_size).
    """
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for shapes consistency check") from exc

    mismatches = []
    checked_count = 0

    for split_name in _SPLITS:
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            continue

        img_files = sorted((split_dir / "img").glob("*_img.tif"))
        for img_path in img_files:
            patch_stem = img_path.stem.replace("_img", "")
            try:
                with rasterio.open(img_path) as ds:
                    ref_h, ref_w = ds.height, ds.width

                if expected_patch_size is not None and (
                    ref_h != expected_patch_size or ref_w != expected_patch_size
                ):
                    mismatches.append(
                        f"{split_name}/{patch_stem}: img shape ({ref_h},{ref_w}) "
                        f"!= expected ({expected_patch_size},{expected_patch_size})"
                    )

                for layer_name in ("extent", "boundary", "distance", "valid"):
                    layer_path = split_dir / layer_name / f"{patch_stem}_{layer_name}.tif"
                    if not layer_path.exists():
                        mismatches.append(
                            f"{split_name}/{patch_stem}: {layer_name} file missing"
                        )
                        continue
                    with rasterio.open(layer_path) as ds:
                        if ds.height != ref_h or ds.width != ref_w:
                            mismatches.append(
                                f"{split_name}/{patch_stem}: {layer_name} shape "
                                f"({ds.height},{ds.width}) != img ({ref_h},{ref_w})"
                            )
                        elif expected_patch_size is not None and (
                            ds.height != expected_patch_size or ds.width != expected_patch_size
                        ):
                            mismatches.append(
                                f"{split_name}/{patch_stem}: {layer_name} shape "
                                f"({ds.height},{ds.width}) != expected "
                                f"({expected_patch_size},{expected_patch_size})"
                            )
            except Exception as exc:
                raise ContractError(
                    f"Failed to read {img_path} during shape check: {exc}"
                ) from exc

            checked_count += 1

    return {
        "consistent": len(mismatches) == 0,
        "mismatches": mismatches,
        "checked_count": checked_count,
    }


def check_target_value_domains(dataset_dir: Path, n_samples: int = 50) -> dict:
    """Sample patches and verify target value domains.

    extent ⊆ {0, 1, 255}, boundary ⊆ {0, 1, 2}, valid ⊆ {0, 1},
    distance ≥ 0.

    Returns {all_ok: bool, issues: list}.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio and numpy are required for domain checks") from exc

    issues = []
    train_img_dir = dataset_dir / "train" / "img"
    if not train_img_dir.exists():
        return {"all_ok": True, "issues": ["train/img not found; skipped"]}

    all_img_files = sorted(train_img_dir.glob("*_img.tif"))
    sample_size = min(n_samples, len(all_img_files))
    sampled = random.sample(all_img_files, k=sample_size) if sample_size > 0 else []

    for img_path in sampled:
        patch_stem = img_path.stem.replace("_img", "")
        split_dir = dataset_dir / "train"

        for layer_name, valid_set in [
            ("extent", _EXTENT_VALID_VALUES),
            ("boundary", _BOUNDARY_VALID_VALUES),
            ("valid", _VALID_VALID_VALUES),
        ]:
            layer_path = split_dir / layer_name / f"{patch_stem}_{layer_name}.tif"
            if not layer_path.exists():
                issues.append(f"{patch_stem}: {layer_name} missing")
                continue
            try:
                with rasterio.open(layer_path) as ds:
                    data = ds.read(1)
                unique_vals = set(int(v) for v in np.unique(data))
                unexpected = unique_vals - valid_set
                if unexpected:
                    issues.append(
                        f"{patch_stem}/{layer_name}: unexpected values {unexpected}"
                    )
            except Exception as exc:
                issues.append(f"{patch_stem}/{layer_name}: read error — {exc}")

        # distance ≥ 0
        dist_path = split_dir / "distance" / f"{patch_stem}_distance.tif"
        if dist_path.exists():
            try:
                import numpy as np  # noqa: PLC0415

                with rasterio.open(dist_path) as ds:
                    data = ds.read(1).astype(np.float32)
                if float(data.min()) < 0.0:
                    issues.append(f"{patch_stem}/distance: contains negative values")
            except Exception as exc:
                issues.append(f"{patch_stem}/distance: read error — {exc}")

    return {"all_ok": len(issues) == 0, "issues": issues}


def check_metadata_contract(dataset_dir: Path, feature_mode: str, n_samples: int = 20) -> dict:
    """Sample meta.json files and check required keys and channel_count consistency.

    Returns {ok: bool, issues: list}.
    """
    required_keys = {
        "patch_id",
        "feature_mode",
        "feature_channel_count",
        "valid_ratio",
        "sampling_class",
    }
    issues = []

    train_meta_dir = dataset_dir / "train" / "meta"
    if not train_meta_dir.exists():
        return {"ok": True, "issues": ["train/meta not found; skipped"]}

    all_meta = sorted(train_meta_dir.glob("*_meta.json"))
    sample_size = min(n_samples, len(all_meta))
    sampled = random.sample(all_meta, k=sample_size) if sample_size > 0 else []

    for meta_path in sampled:
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"{meta_path.name}: read error — {exc}")
            continue

        missing = required_keys - set(data.keys())
        if missing:
            issues.append(f"{meta_path.name}: missing keys {missing}")
            continue

        if data.get("feature_mode") != feature_mode:
            issues.append(
                f"{meta_path.name}: feature_mode mismatch "
                f"(expected {feature_mode}, got {data.get('feature_mode')})"
            )

    return {"ok": len(issues) == 0, "issues": issues}


def validate_dataset(dataset_dir: Any, config: Any) -> dict:
    """Run all three validation checks and aggregate results.

    Returns:
        shapes_ok, domains_ok, metadata_ok, issues, checked_count,
        validation_compute_mode.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise ContractError(f"dataset_dir does not exist: {dataset_dir}")

    feature_mode = getattr(config, "feature_mode", "raw8")

    expected_patch_size = int(getattr(getattr(config, "patches", None), "patch_size", 0))
    shapes_result = check_shapes_consistency(
        dataset_dir,
        expected_patch_size=expected_patch_size if expected_patch_size > 0 else None,
    )
    domains_result = check_target_value_domains(dataset_dir)
    meta_result = check_metadata_contract(dataset_dir, feature_mode=feature_mode)

    all_issues = (
        shapes_result.get("mismatches", [])
        + domains_result.get("issues", [])
        + meta_result.get("issues", [])
    )

    return {
        "shapes_ok": shapes_result["consistent"],
        "domains_ok": domains_result["all_ok"],
        "metadata_ok": meta_result["ok"],
        "issues": all_issues,
        "checked_count": shapes_result.get("checked_count", 0),
        "validation_compute_mode": VALIDATE_COMPUTE_MODE,
    }
