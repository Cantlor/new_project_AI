"""Minimal run-level train orchestration for module_net_train.

This layer intentionally provides only a narrow baseline flow:
  config -> dataset/dataloader -> model -> loss -> trainer -> export.

It is not a full experiment manager or CLI framework.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.constants import CHANNEL_COUNTS, FEATURE_MODES
from ai_fields.common.errors import ContractError
from ai_fields.common.progress import progress_bar
from ai_fields.module_net_train.dataset import (
    FieldsDataset,
    assemble_model_input,
    fields_collate_fn,
    list_sample_ids,
    read_sample,
)
from ai_fields.module_net_train.export import (
    NetTrainExportArtifacts,
    build_checkpoint_payload,
    export_training_artifacts,
    save_checkpoint,
)
from ai_fields.module_net_train.losses import MultitaskLoss
from ai_fields.module_net_train.model import build_model
from ai_fields.module_net_train.schemas import (
    CoverageAwareSamplingConfig,
    DEPRECATED_MONITORED_METRIC_NAMES,
    MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1,
    MONITORED_METRIC_EXPECTED_MODE,
)
from ai_fields.module_net_train.trainer import (
    build_scheduler,
    build_optimizer,
    evaluate_one_epoch,
    resolve_runtime_execution,
    train_one_epoch,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import rasterio

    _RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    rasterio = None  # type: ignore[assignment]
    _RASTERIO_AVAILABLE = False

_BASELINE_COMPOSITE_METRIC_NAME = MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1
_REQUIRED_SPLIT_LAYER_DIRS = ("img", "extent", "boundary", "distance", "valid", "meta")
_REQUIRED_RASTER_LAYERS = ("img", "extent", "boundary", "distance", "valid")
_HISTORY_COLUMNS = [
    "epoch",
    "train_extent",
    "train_boundary",
    "train_distance",
    "train_total",
    "val_extent",
    "val_boundary",
    "val_distance",
    "val_total",
    "monitored_metric",
    "is_best",
    "train_n_valid",
    "val_n_valid",
    "n_aux",
    "train_extent_f1",
    "train_extent_precision",
    "train_extent_recall",
    "train_extent_iou",
    "train_boundary_f1",
    "train_boundary_precision",
    "train_boundary_recall",
    "train_boundary_skeleton_f1",
    "train_boundary_buffer_f1",
    "val_extent_f1",
    "val_extent_precision",
    "val_extent_recall",
    "val_extent_iou",
    "val_boundary_f1",
    "val_boundary_precision",
    "val_boundary_recall",
    "val_boundary_skeleton_f1",
    "val_boundary_buffer_f1",
]


@dataclass(frozen=True)
class NetTrainRunResult:
    """Result contract for a minimal baseline training run."""

    run_dir: Path
    run_id: str
    train_summary: dict[str, Any]
    val_summary: dict[str, Any]
    checkpoint_path: Path
    last_checkpoint_path: Path
    checkpoint_metadata_path: Path
    train_manifest_path: Path
    summary_path: Path
    config_used_path: Path
    history_path: Path
    eval_val_path: Path
    bucket_metrics_val_path: Path
    epochs_completed: int
    best_metric_name: str
    best_metric_value: float
    best_epoch: int
    monitored_metric_mode: str
    monitored_metric_policy_note: str | None
    device_requested: str | None
    device_resolved: str
    amp_requested: bool
    amp_used: bool
    oom_fallbacks_applied: tuple[str, ...]
    dataset_root: Path | None
    dataset_patch_size: int


@dataclass(frozen=True)
class NetTrainDatasetContractResult:
    """Resolved contract summary for one selected fixed-size train dataset."""

    train_split_dir: Path
    val_split_dir: Path
    patch_size: int
    feature_mode: str
    expected_channels: int
    train_sample_count: int
    val_sample_count: int


@dataclass(frozen=True)
class _RuntimeInputNormalizer:
    """Per-batch normalizer built from prep_data norm_stats + scaling contract."""

    feature_channels: int
    scale_min: float
    scale_max: float
    p_lo: "torch.Tensor"
    p_hi: "torch.Tensor"

    def __call__(self, image_batch: "torch.Tensor") -> "torch.Tensor":
        if image_batch.ndim != 4:
            raise ContractError(
                f"image batch for normalization must have shape (B,C,H,W), got {tuple(image_batch.shape)}."
            )
        if image_batch.shape[1] < self.feature_channels:
            raise ContractError(
                "image batch channel count is smaller than normalization feature channels: "
                f"{image_batch.shape[1]} < {self.feature_channels}."
            )

        lo = self.p_lo.to(device=image_batch.device, dtype=image_batch.dtype).view(
            1, self.feature_channels, 1, 1
        )
        hi = self.p_hi.to(device=image_batch.device, dtype=image_batch.dtype).view(
            1, self.feature_channels, 1, 1
        )
        denom = hi - lo
        if torch.any(denom <= 0):
            raise ContractError(
                "Normalization denominator has non-positive values after loading stats."
            )

        out = image_batch.clone()
        feat = out[:, : self.feature_channels]
        feat = torch.clamp(feat, min=lo, max=hi)
        feat = (feat - lo) / denom
        feat = feat * (self.scale_max - self.scale_min) + self.scale_min
        out[:, : self.feature_channels] = feat

        if not torch.isfinite(out).all():
            n_nan = int(torch.isnan(out).sum().item())
            n_inf = int(torch.isinf(out).sum().item())
            raise ContractError(
                "Normalized model input contains non-finite values: "
                f"nan={n_nan}, inf={n_inf}."
            )
        return out


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ContractError(
            "torch is required for module_net_train run orchestration. Install torch to use this layer."
        )


def _normalize_path(path: Any, *, name: str) -> Path:
    if isinstance(path, (str, PathLike)):
        p = Path(path)
    else:
        raise ContractError(f"{name} must be path-like, got {type(path).__name__}.")
    if str(p).strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    return p


def _validate_scaling_range(value: Any) -> tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or isinstance(value, (str, bytes))
    ):
        raise ContractError("normalization.scaling_range must be a 2-item numeric sequence.")
    lo, hi = value
    if isinstance(lo, bool) or not isinstance(lo, (int, float)):
        raise ContractError("normalization.scaling_range[0] must be numeric.")
    if isinstance(hi, bool) or not isinstance(hi, (int, float)):
        raise ContractError("normalization.scaling_range[1] must be numeric.")
    lo_f = float(lo)
    hi_f = float(hi)
    if hi_f <= lo_f:
        raise ContractError(
            "normalization.scaling_range must satisfy max > min "
            f"(got {lo_f}..{hi_f})."
        )
    return lo_f, hi_f


def _build_runtime_input_normalizer(
    *,
    normalization: Any,
    feature_mode: str,
) -> _RuntimeInputNormalizer:
    _require_torch()
    if feature_mode not in FEATURE_MODES:
        raise ContractError(
            f"feature_mode must be one of {sorted(FEATURE_MODES)}, got {feature_mode!r}."
        )
    if not isinstance(normalization, Mapping):
        raise ContractError(
            f"normalization must be a mapping/object, got {type(normalization).__name__}."
        )
    stats_source = normalization.get("stats_source")
    if not isinstance(stats_source, str) or stats_source.strip() == "":
        raise ContractError("normalization.stats_source must be a non-empty path string.")
    stats_path = _normalize_path(stats_source, name="normalization.stats_source")
    if not stats_path.exists():
        raise ContractError(f"normalization.stats_source does not exist: {stats_path}")

    scale_min, scale_max = _validate_scaling_range(normalization.get("scaling_range"))

    try:
        stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"Failed to read normalization stats at {stats_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid normalization stats JSON at {stats_path}: {exc}") from exc
    if not isinstance(stats_payload, dict):
        raise ContractError("Normalization stats JSON must contain a top-level object.")
    raw_band_stats = stats_payload.get("band_stats")
    if not isinstance(raw_band_stats, list) or len(raw_band_stats) == 0:
        raise ContractError(
            "Normalization stats JSON must include non-empty 'band_stats' list."
        )

    expected_channels = int(CHANNEL_COUNTS[feature_mode])
    lo_by_idx: dict[int, float] = {}
    hi_by_idx: dict[int, float] = {}
    for entry in raw_band_stats:
        if not isinstance(entry, Mapping):
            raise ContractError("Each entry in normalization band_stats must be an object.")
        band_idx = entry.get("band_idx")
        p_lo = entry.get("p_lo")
        p_hi = entry.get("p_hi")
        if isinstance(band_idx, bool) or not isinstance(band_idx, int):
            raise ContractError("normalization band_stats[*].band_idx must be an integer.")
        if band_idx < 0:
            raise ContractError("normalization band_stats[*].band_idx must be >= 0.")
        if isinstance(p_lo, bool) or not isinstance(p_lo, (int, float)):
            raise ContractError("normalization band_stats[*].p_lo must be numeric.")
        if isinstance(p_hi, bool) or not isinstance(p_hi, (int, float)):
            raise ContractError("normalization band_stats[*].p_hi must be numeric.")
        lo_f = float(p_lo)
        hi_f = float(p_hi)
        if not (lo_f < hi_f):
            raise ContractError(
                "normalization band stats must satisfy p_lo < p_hi for every band "
                f"(band_idx={band_idx}, p_lo={lo_f}, p_hi={hi_f})."
            )
        lo_by_idx[band_idx] = lo_f
        hi_by_idx[band_idx] = hi_f

    missing = [idx for idx in range(expected_channels) if idx not in lo_by_idx]
    if missing:
        raise ContractError(
            "Normalization stats are incomplete for selected feature_mode "
            f"{feature_mode!r}; missing band indices: {missing}."
        )

    p_lo = torch.tensor([lo_by_idx[idx] for idx in range(expected_channels)], dtype=torch.float32)
    p_hi = torch.tensor([hi_by_idx[idx] for idx in range(expected_channels)], dtype=torch.float32)
    return _RuntimeInputNormalizer(
        feature_channels=expected_channels,
        scale_min=scale_min,
        scale_max=scale_max,
        p_lo=p_lo,
        p_hi=p_hi,
    )


def _require_rasterio() -> None:
    if not _RASTERIO_AVAILABLE:
        raise ContractError(
            "rasterio is required to validate train-ready dataset spatial contract."
        )


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"{name} must be an integer >= 1.")
    return value


def _validate_split_layer_dirs(split_dir: Path, *, split_name: str) -> None:
    for layer in _REQUIRED_SPLIT_LAYER_DIRS:
        layer_dir = split_dir / layer
        if not layer_dir.exists() or not layer_dir.is_dir():
            raise ContractError(
                f"{split_name} split is missing required layer directory: {layer_dir}"
            )


def _sample_path(split_dir: Path, layer: str, sample_id: str) -> Path:
    suffix = "json" if layer == "meta" else "tif"
    return split_dir / layer / f"{sample_id}_{layer}.{suffix}"


def _validate_one_sample_contract(
    *,
    split_dir: Path,
    split_name: str,
    sample_id: str,
    expected_channels: int,
    expected_patch_size: int | None,
    resolved_patch_size: int | None,
) -> int:
    layer_shapes: dict[str, tuple[int, int]] = {}
    img_count: int | None = None

    for layer in _REQUIRED_RASTER_LAYERS:
        path = _sample_path(split_dir, layer, sample_id)
        if not path.exists():
            raise ContractError(
                f"{split_name} split sample {sample_id!r} is missing required layer file: {path}"
            )
        with rasterio.open(path) as ds:  # type: ignore[union-attr]
            layer_shapes[layer] = (int(ds.height), int(ds.width))
            if layer == "img":
                img_count = int(ds.count)

    meta_path = _sample_path(split_dir, "meta", sample_id)
    if not meta_path.exists():
        raise ContractError(
            f"{split_name} split sample {sample_id!r} is missing required meta file: {meta_path}"
        )
    try:
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"Failed to read meta JSON at {meta_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid meta JSON at {meta_path}: {exc}") from exc
    if not isinstance(meta_payload, dict):
        raise ContractError(f"Meta payload at {meta_path} must be a JSON object.")

    if img_count is None:
        raise ContractError(f"Could not resolve image channel count for sample {sample_id!r}.")
    if img_count != expected_channels:
        raise ContractError(
            f"{split_name} split sample {sample_id!r} has img channels={img_count}, "
            f"expected {expected_channels} for selected feature_mode."
        )

    unique_shapes = set(layer_shapes.values())
    if len(unique_shapes) != 1:
        raise ContractError(
            f"{split_name} split sample {sample_id!r} has inconsistent layer shapes: {layer_shapes}."
        )
    height, width = layer_shapes["img"]
    if height != width:
        raise ContractError(
            f"{split_name} split sample {sample_id!r} is not square: {height}x{width}."
        )

    if expected_patch_size is not None and width != expected_patch_size:
        raise ContractError(
            f"{split_name} split sample {sample_id!r} has patch size {width}, "
            f"expected {expected_patch_size}."
        )
    if resolved_patch_size is not None and width != resolved_patch_size:
        raise ContractError(
            f"{split_name} split sample {sample_id!r} has patch size {width}, "
            f"but dataset already resolved as {resolved_patch_size}."
        )
    return width


def validate_train_ready_dataset_contract(
    *,
    train_split_dir: str | Path,
    val_split_dir: str | Path,
    feature_mode: str,
    expected_patch_size: int | None = None,
    progress_enabled: bool | None = None,
) -> NetTrainDatasetContractResult:
    """Validate one selected train-ready dataset for fixed-size training contract.

    The function is intentionally strict:
      - checks canonical split layout (`img/extent/boundary/distance/valid/meta`);
      - rejects mixed patch sizes across samples and splits;
      - validates image channel count for selected dataset-side feature_mode.
    """
    _require_rasterio()
    train_path = _normalize_path(train_split_dir, name="train_split_dir")
    val_path = _normalize_path(val_split_dir, name="val_split_dir")
    if not train_path.exists():
        raise ContractError(f"train_split_dir does not exist: {train_path}")
    if not val_path.exists():
        raise ContractError(f"val_split_dir does not exist: {val_path}")

    if feature_mode not in FEATURE_MODES:
        raise ContractError(
            f"feature_mode must be one of {sorted(FEATURE_MODES)}, got {feature_mode!r}."
        )
    expected_channels = CHANNEL_COUNTS[feature_mode]
    if expected_patch_size is not None:
        expected_patch_size = _require_positive_int(
            expected_patch_size, name="expected_patch_size"
        )

    _validate_split_layer_dirs(train_path, split_name="train")
    _validate_split_layer_dirs(val_path, split_name="val")

    train_ids = list_sample_ids(train_path)
    val_ids = list_sample_ids(val_path)
    resolved_patch_size: int | None = None

    for split_name, split_path, sample_ids in (
        ("train", train_path, train_ids),
        ("val", val_path, val_ids),
    ):
        with progress_bar(
            total=len(sample_ids),
            desc=f"net_train: validate {split_name} split",
            unit="sample",
            progress_enabled=progress_enabled,
            leave=False,
        ) as bar:
            for sample_id in sample_ids:
                resolved_patch_size = _validate_one_sample_contract(
                    split_dir=split_path,
                    split_name=split_name,
                    sample_id=sample_id,
                    expected_channels=expected_channels,
                    expected_patch_size=expected_patch_size,
                    resolved_patch_size=resolved_patch_size,
                )
                bar.update(1)

    if resolved_patch_size is None:
        raise ContractError("Could not resolve patch_size from selected train-ready dataset.")

    return NetTrainDatasetContractResult(
        train_split_dir=train_path,
        val_split_dir=val_path,
        patch_size=resolved_patch_size,
        feature_mode=feature_mode,
        expected_channels=expected_channels,
        train_sample_count=len(train_ids),
        val_sample_count=len(val_ids),
    )


def _set_global_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError("seed must be an integer.")
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _build_dataloader(
    *,
    split_dir: Path,
    feature_mode: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: bool = False,
    sampler: Any | None = None,
) -> Any:
    dataset = FieldsDataset(split_dir=split_dir, feature_mode=feature_mode, augment=augment)
    if len(dataset) == 0:
        raise ContractError(f"Split dataset is empty: {split_dir}")
    if sampler is not None and shuffle:
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=fields_collate_fn,
        pin_memory=False,
        drop_last=False,
    )


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _coverage_bucket_for(
    coverage: float,
    *,
    q_low: float,
    q_high: float,
) -> str:
    if coverage <= q_low:
        return "low"
    if coverage <= q_high:
        return "medium"
    return "high"


def _resolve_coverage_aware_sampling_config(training_cfg: Any) -> CoverageAwareSamplingConfig:
    raw_cfg = getattr(training_cfg, "coverage_aware_sampling", None)
    if raw_cfg is None:
        cfg = CoverageAwareSamplingConfig()
        cfg.validate()
        return cfg
    if isinstance(raw_cfg, CoverageAwareSamplingConfig):
        raw_cfg.validate()
        return raw_cfg
    if isinstance(raw_cfg, Mapping):
        cfg = CoverageAwareSamplingConfig(**dict(raw_cfg))
        cfg.validate()
        return cfg
    raise ContractError(
        "training.coverage_aware_sampling must be a mapping/object or "
        "CoverageAwareSamplingConfig."
    )


def _sample_extent_coverage(
    *,
    split_dir: Path,
    sample_id: str,
) -> float:
    _require_rasterio()
    extent_path = split_dir / "extent" / f"{sample_id}_extent.tif"
    valid_path = split_dir / "valid" / f"{sample_id}_valid.tif"
    if not extent_path.exists() or not valid_path.exists():
        raise ContractError(
            f"Missing extent/valid raster for coverage sampling: sample_id={sample_id!r}."
        )
    with rasterio.open(extent_path) as ds_extent, rasterio.open(valid_path) as ds_valid:  # type: ignore[union-attr]
        extent = ds_extent.read(1) > 0
        valid = ds_valid.read(1) > 0
    valid_count = int(valid.sum())
    if valid_count <= 0:
        return 0.0
    return _safe_div(float((extent & valid).sum()), float(valid_count))


def _compute_split_coverages(
    *,
    split_dir: Path,
    sample_ids: list[str],
    progress_enabled: bool | None,
    progress_desc: str,
) -> dict[str, float]:
    if np is None:
        raise ContractError("numpy is required for coverage-aware sampling diagnostics.")
    coverages: dict[str, float] = {}
    with progress_bar(
        total=len(sample_ids),
        desc=progress_desc,
        unit="sample",
        progress_enabled=progress_enabled,
        leave=False,
    ) as bar:
        for sample_id in sample_ids:
            coverages[sample_id] = _sample_extent_coverage(split_dir=split_dir, sample_id=sample_id)
            bar.update(1)
    return coverages


def _build_coverage_aware_train_sampler(
    *,
    split_dir: Path,
    seed: int,
    coverage_cfg: CoverageAwareSamplingConfig,
    progress_enabled: bool | None,
) -> tuple[Any | None, dict[str, Any]]:
    coverage_cfg.validate()
    if not coverage_cfg.enabled:
        return None, {
            "enabled": False,
            "reason": "disabled_by_config",
            "bucket_weights": dict(coverage_cfg.bucket_weights),
            "replacement": bool(coverage_cfg.replacement),
            "coverage_quantile_low": float(coverage_cfg.coverage_quantile_low),
            "coverage_quantile_high": float(coverage_cfg.coverage_quantile_high),
        }

    if np is None:
        raise ContractError("numpy is required for coverage-aware sampling.")
    _require_torch()

    sample_ids = list_sample_ids(split_dir)
    if len(sample_ids) == 0:
        raise ContractError(f"Cannot build coverage-aware sampler for empty split: {split_dir}")

    coverages = _compute_split_coverages(
        split_dir=split_dir,
        sample_ids=sample_ids,
        progress_enabled=progress_enabled,
        progress_desc="net_train: train coverage buckets",
    )
    coverage_values = np.asarray([coverages[sid] for sid in sample_ids], dtype=np.float64)
    q_low = float(np.quantile(coverage_values, float(coverage_cfg.coverage_quantile_low)))
    q_high = float(np.quantile(coverage_values, float(coverage_cfg.coverage_quantile_high)))
    if q_high < q_low:
        raise ContractError(
            "Coverage quantiles resolved to invalid order for coverage-aware sampling: "
            f"q_low={q_low}, q_high={q_high}."
        )

    bucket_weights = {k: float(v) for k, v in coverage_cfg.bucket_weights.items()}
    bucket_counts = {"low": 0, "medium": 0, "high": 0}
    sample_weights: list[float] = []
    for sample_id in sample_ids:
        coverage = float(coverages[sample_id])
        bucket = _coverage_bucket_for(coverage, q_low=q_low, q_high=q_high)
        bucket_counts[bucket] += 1
        sample_weights.append(bucket_weights[bucket])

    torch_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch_weights,
        num_samples=len(sample_ids),
        replacement=bool(coverage_cfg.replacement),
        generator=generator,
    )

    diagnostics = {
        "enabled": True,
        "strategy": "coverage_bucket_weighted_random_sampler",
        "coverage_quantile_low": float(coverage_cfg.coverage_quantile_low),
        "coverage_quantile_high": float(coverage_cfg.coverage_quantile_high),
        "resolved_coverage_quantiles": {"q_low": q_low, "q_high": q_high},
        "bucket_definition": "low: coverage<=q_low; medium: q_low<coverage<=q_high; high: coverage>q_high",
        "bucket_weights": bucket_weights,
        "bucket_counts": bucket_counts,
        "replacement": bool(coverage_cfg.replacement),
        "num_samples": len(sample_ids),
        "seed": int(seed),
    }
    return sampler, diagnostics


def _extent_bucket_metrics_from_counts(*, tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    denom = (2 * tp) + fp + fn
    f1 = _safe_div(2 * tp, denom)
    return {
        "extent_precision": precision,
        "extent_recall": recall,
        "extent_f1": f1,
        "extent_precision_recall_gap": precision - recall,
    }


def _compute_bucket_metrics_for_split(
    *,
    model: Any,
    split_dir: Path,
    feature_mode: str,
    run_id: str,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    monitored_metric_name: str,
    monitored_metric_mode: str,
    input_normalizer: Any | None,
    device: str,
    thresholds: tuple[float, ...] = (0.50, 0.55),
    progress_enabled: bool | None = None,
) -> dict[str, Any]:
    if np is None:
        raise ContractError("numpy is required for bucket metrics diagnostics.")
    _require_torch()
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()

    sample_ids = list_sample_ids(split_dir)
    if len(sample_ids) == 0:
        raise ContractError(f"Cannot compute bucket metrics: split is empty ({split_dir}).")

    coverages = _compute_split_coverages(
        split_dir=split_dir,
        sample_ids=sample_ids,
        progress_enabled=progress_enabled,
        progress_desc="net_train: val coverage buckets",
    )
    coverage_values = np.asarray([coverages[sid] for sid in sample_ids], dtype=np.float64)
    q33 = float(np.quantile(coverage_values, 1.0 / 3.0))
    q66 = float(np.quantile(coverage_values, 2.0 / 3.0))

    bucket_counts = {"low": 0, "medium": 0, "high": 0}
    counts: dict[str, dict[str, dict[str, int]]] = {
        f"{t:.2f}": {
            "low": {"tp": 0, "fp": 0, "fn": 0},
            "medium": {"tp": 0, "fp": 0, "fn": 0},
            "high": {"tp": 0, "fp": 0, "fn": 0},
        }
        for t in thresholds
    }

    with torch.no_grad(), progress_bar(
        total=len(sample_ids),
        desc="net_train: eval val buckets",
        unit="sample",
        progress_enabled=progress_enabled,
        leave=False,
    ) as bar:
        for sample_id in sample_ids:
            sample = read_sample(split_dir, sample_id, feature_mode)
            coverage = float(coverages[sample_id])
            bucket = _coverage_bucket_for(coverage, q_low=q33, q_high=q66)
            bucket_counts[bucket] += 1

            valid_mask = sample["valid"] > 0
            extent_gt = sample["extent"] > 0
            assembled = assemble_model_input(sample["img"], sample["valid"]).astype(np.float32)
            batch = torch.from_numpy(assembled).unsqueeze(0).to(device=resolved_device, dtype=torch.float32)
            if input_normalizer is not None:
                batch = input_normalizer(batch)
            model_out = model(batch)
            extent_logits = model_out.get("extent")
            if not isinstance(extent_logits, torch.Tensor):
                raise ContractError("Model output missing 'extent' tensor for bucket metrics.")
            extent_prob = torch.sigmoid(extent_logits)[0, 0].detach().cpu().numpy()

            for thr in thresholds:
                key = f"{thr:.2f}"
                pred = np.logical_and(extent_prob >= float(thr), valid_mask)
                gt_pos = np.logical_and(extent_gt, valid_mask)
                bg_valid = np.logical_and(np.logical_not(extent_gt), valid_mask)
                tp = int(np.logical_and(pred, gt_pos).sum())
                fp = int(np.logical_and(pred, bg_valid).sum())
                fn = int(np.logical_and(np.logical_not(pred), gt_pos).sum())
                counts[key][bucket]["tp"] += tp
                counts[key][bucket]["fp"] += fp
                counts[key][bucket]["fn"] += fn
            bar.update(1)

    bucket_metrics: dict[str, Any] = {}
    for thr_key, by_bucket in counts.items():
        bucket_metrics[thr_key] = {}
        for bucket in ("low", "medium", "high"):
            tp = int(by_bucket[bucket]["tp"])
            fp = int(by_bucket[bucket]["fp"])
            fn = int(by_bucket[bucket]["fn"])
            metrics = _extent_bucket_metrics_from_counts(tp=tp, fp=fp, fn=fn)
            bucket_metrics[thr_key][bucket] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                **metrics,
            }

    return {
        "schema_name": "net_train.bucket_metrics_split",
        "schema_version": "v1",
        "run_id": run_id,
        "split": "val",
        "feature_mode": feature_mode,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "monitored_metric_name": monitored_metric_name,
        "monitored_metric_mode": monitored_metric_mode,
        "thresholds": [float(t) for t in thresholds],
        "bucket_definition": {
            "source_split": "val",
            "source_metric": "gt_extent_coverage_with_valid_mask",
            "rule": "low: coverage<=q33; medium: q33<coverage<=q66; high: coverage>q66",
            "q33": q33,
            "q66": q66,
        },
        "bucket_counts": bucket_counts,
        "metrics": bucket_metrics,
    }


def _extract_float(summary: dict[str, Any], key: str, *, context: str) -> float:
    if key not in summary:
        raise ContractError(f"{context} summary is missing required key: {key!r}.")
    value = summary[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(
            f"{context} summary key {key!r} must be numeric, got {type(value).__name__}."
        )
    return float(value)


def _resolve_metric_mode(metric_name: str, mode_override: str | None) -> tuple[str, str | None]:
    if metric_name in DEPRECATED_MONITORED_METRIC_NAMES:
        raise ContractError(
            f"monitored_metric_name={metric_name!r} is deprecated and no longer supported. "
            f"Use {_BASELINE_COMPOSITE_METRIC_NAME!r}."
        )
    expected_mode = MONITORED_METRIC_EXPECTED_MODE.get(metric_name)
    if expected_mode is None:
        raise ContractError(
            f"Unsupported monitored_metric_name={metric_name!r}. "
            "Supported: "
            f"{_BASELINE_COMPOSITE_METRIC_NAME!r}."
        )
    policy_note: str | None = None

    if mode_override is None:
        return expected_mode, policy_note
    if mode_override not in {"min", "max"}:
        raise ContractError("monitored_metric_mode must be 'min' or 'max'.")
    if mode_override != expected_mode:
        raise ContractError(
            f"monitored_metric_mode={mode_override!r} is inconsistent with "
            f"monitored_metric_name={metric_name!r}; expected {expected_mode!r}."
        )
    return mode_override, policy_note


def _compute_monitored_metric(val_summary: dict[str, Any], metric_name: str) -> float:
    if metric_name == _BASELINE_COMPOSITE_METRIC_NAME:
        boundary_f1 = _extract_float(val_summary, "boundary_f1", context="val")
        extent_f1 = _extract_float(val_summary, "extent_f1", context="val")
        return 0.6 * boundary_f1 + 0.4 * extent_f1
    raise ContractError(f"Unsupported monitored_metric_name={metric_name!r}.")


def _is_improved(*, candidate: float, best_value: float | None, mode: str) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return candidate < best_value
    return candidate > best_value


def _history_row(
    *,
    epoch: int,
    train_summary: dict[str, Any],
    val_summary: dict[str, Any],
    monitored_metric: float,
    is_best: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_extent": _extract_float(train_summary, "extent", context="train"),
        "train_boundary": _extract_float(train_summary, "boundary", context="train"),
        "train_distance": _extract_float(train_summary, "distance", context="train"),
        "train_total": _extract_float(train_summary, "total", context="train"),
        "val_extent": _extract_float(val_summary, "extent", context="val"),
        "val_boundary": _extract_float(val_summary, "boundary", context="val"),
        "val_distance": _extract_float(val_summary, "distance", context="val"),
        "val_total": _extract_float(val_summary, "total", context="val"),
        "train_extent_f1": _extract_float(train_summary, "extent_f1", context="train"),
        "train_extent_precision": _extract_float(train_summary, "extent_precision", context="train"),
        "train_extent_recall": _extract_float(train_summary, "extent_recall", context="train"),
        "train_extent_iou": _extract_float(train_summary, "extent_iou", context="train"),
        "train_boundary_f1": _extract_float(train_summary, "boundary_f1", context="train"),
        "train_boundary_precision": _extract_float(train_summary, "boundary_precision", context="train"),
        "train_boundary_recall": _extract_float(train_summary, "boundary_recall", context="train"),
        "train_boundary_skeleton_f1": _extract_float(
            train_summary, "boundary_skeleton_f1", context="train"
        ),
        "train_boundary_buffer_f1": _extract_float(
            train_summary, "boundary_buffer_f1", context="train"
        ),
        "val_extent_f1": _extract_float(val_summary, "extent_f1", context="val"),
        "val_extent_precision": _extract_float(val_summary, "extent_precision", context="val"),
        "val_extent_recall": _extract_float(val_summary, "extent_recall", context="val"),
        "val_extent_iou": _extract_float(val_summary, "extent_iou", context="val"),
        "val_boundary_f1": _extract_float(val_summary, "boundary_f1", context="val"),
        "val_boundary_precision": _extract_float(val_summary, "boundary_precision", context="val"),
        "val_boundary_recall": _extract_float(val_summary, "boundary_recall", context="val"),
        "val_boundary_skeleton_f1": _extract_float(
            val_summary, "boundary_skeleton_f1", context="val"
        ),
        "val_boundary_buffer_f1": _extract_float(
            val_summary, "boundary_buffer_f1", context="val"
        ),
        "monitored_metric": monitored_metric,
        "is_best": bool(is_best),
        "train_n_valid": int(train_summary.get("n_valid", 0)),
        "val_n_valid": int(val_summary.get("n_valid", 0)),
        "n_aux": int(val_summary.get("n_aux", train_summary.get("n_aux", 0))),
    }


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ContractError("Cannot write history.csv: no epoch rows were recorded.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_HISTORY_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise ContractError(f"Failed to write history CSV at {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Failed to serialize JSON at {path}: {exc}") from exc
    except OSError as exc:
        raise ContractError(f"Failed to write JSON at {path}: {exc}") from exc


def _is_out_of_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda out of memory" in text


def _load_checkpoint_state_dict(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except OSError as exc:
        raise ContractError(f"Failed to read checkpoint from {path}: {exc}") from exc
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ContractError(f"Checkpoint payload at {path} is missing 'model_state_dict'.")
    state = payload["model_state_dict"]
    if not isinstance(state, dict):
        raise ContractError(f"model_state_dict at {path} must be a mapping/object.")
    return state


def run_train_baseline(
    *,
    config: Any,
    train_split_dir: str | Path,
    val_split_dir: str | Path,
    run_dir: str | Path,
    run_id: str,
    dataset_source_run_id: str,
    dataset_source_manifest_path: str | Path,
    normalization: dict[str, Any],
    dataset_root: str | Path | None = None,
    dataset_patch_size: int | None = None,
    dataset_feature_mode: str | None = None,
    epochs_override: int | None = None,
    device_override: str | None = None,
    monitored_metric_name: str | None = None,
    monitored_metric_mode: str | None = None,
    progress_enabled: bool | None = None,
) -> NetTrainRunResult:
    """Execute a minimal baseline train+val run and export Stage E artifacts.

    This function orchestrates already implemented layers and intentionally
    does not implement a large training framework.
    """
    _require_torch()
    if not hasattr(config, "validate") or not callable(config.validate):
        raise ContractError("config must provide validate() and match NetTrainConfig contract.")
    config.validate()

    train_path = _normalize_path(train_split_dir, name="train_split_dir")
    val_path = _normalize_path(val_split_dir, name="val_split_dir")
    run_path = _normalize_path(run_dir, name="run_dir")
    source_manifest_path = _normalize_path(
        dataset_source_manifest_path, name="dataset_source_manifest_path"
    )
    resolved_dataset_root = (
        _normalize_path(dataset_root, name="dataset_root")
        if dataset_root is not None
        else None
    )
    if not train_path.exists():
        raise ContractError(f"train_split_dir does not exist: {train_path}")
    if not val_path.exists():
        raise ContractError(f"val_split_dir does not exist: {val_path}")
    if not source_manifest_path.exists():
        raise ContractError(
            f"dataset_source_manifest_path does not exist: {source_manifest_path}"
        )
    if not isinstance(run_id, str) or run_id.strip() == "":
        raise ContractError("run_id must be a non-empty string.")
    if not isinstance(dataset_source_run_id, str) or dataset_source_run_id.strip() == "":
        raise ContractError("dataset_source_run_id must be a non-empty string.")
    if dataset_patch_size is not None:
        dataset_patch_size = _require_positive_int(dataset_patch_size, name="dataset_patch_size")
    if dataset_feature_mode is not None:
        if dataset_feature_mode not in FEATURE_MODES:
            raise ContractError(
                f"dataset_feature_mode must be one of {sorted(FEATURE_MODES)}, "
                f"got {dataset_feature_mode!r}."
            )
        if dataset_feature_mode != config.feature_mode:
            raise ContractError(
                "dataset_feature_mode is inconsistent with config.feature_mode: "
                f"{dataset_feature_mode!r} != {config.feature_mode!r}."
            )
    if resolved_dataset_root is None:
        inferred_root = train_path.parent if train_path.parent == val_path.parent else None
        if inferred_root is not None:
            resolved_dataset_root = inferred_root

    input_normalizer = _build_runtime_input_normalizer(
        normalization=normalization,
        feature_mode=config.feature_mode,
    )

    dataset_contract = validate_train_ready_dataset_contract(
        train_split_dir=train_path,
        val_split_dir=val_path,
        feature_mode=config.feature_mode,
        expected_patch_size=dataset_patch_size,
        progress_enabled=progress_enabled,
    )
    resolved_dataset_patch_size = dataset_contract.patch_size

    epochs = config.training.num_epochs if epochs_override is None else epochs_override
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ContractError("epochs must be an integer >= 1.")

    cfg_monitoring = getattr(config, "monitoring", None)
    if cfg_monitoring is None:
        raise ContractError(
            "config.monitoring is required in NetTrainConfig for monitored metric policy."
        )
    resolved_metric_name = (
        monitored_metric_name
        if monitored_metric_name is not None
        else getattr(cfg_monitoring, "monitored_metric_name", None)
    )
    if not isinstance(resolved_metric_name, str) or resolved_metric_name.strip() == "":
        raise ContractError("Resolved monitored metric name must be a non-empty string.")
    if monitored_metric_mode is not None:
        resolved_mode_input = monitored_metric_mode
    elif monitored_metric_name is not None:
        # Explicit name override should not inherit a potentially inconsistent mode
        # from config; let mode resolve from the metric name contract.
        resolved_mode_input = None
    else:
        resolved_mode_input = getattr(cfg_monitoring, "monitored_metric_mode", None)
    resolved_metric_mode, policy_note = _resolve_metric_mode(
        resolved_metric_name, resolved_mode_input
    )

    run_path.mkdir(parents=True, exist_ok=True)
    history_path = run_path / "history.csv"
    best_checkpoint_path = run_path / "best.ckpt"
    last_checkpoint_path = run_path / "last.ckpt"

    _set_global_seed(config.training.seed)

    coverage_sampling_cfg = _resolve_coverage_aware_sampling_config(config.training)
    train_sampler, coverage_sampling_runtime = _build_coverage_aware_train_sampler(
        split_dir=train_path,
        seed=config.training.seed,
        coverage_cfg=coverage_sampling_cfg,
        progress_enabled=progress_enabled,
    )
    train_loader = _build_dataloader(
        split_dir=train_path,
        feature_mode=config.feature_mode,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=train_sampler is None,
        augment=config.training.augment,
        sampler=train_sampler,
    )
    val_loader = _build_dataloader(
        split_dir=val_path,
        feature_mode=config.feature_mode,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        sampler=None,
    )

    model = build_model(config)
    loss_fn = MultitaskLoss.from_config(config.loss)
    optimizer = build_optimizer(model, config)
    scheduler, scheduler_step_policy = build_scheduler(
        optimizer,
        config,
        total_epochs=epochs,
    )
    scheduler_last_lr = float(optimizer.param_groups[0]["lr"])

    runtime_policy = resolve_runtime_execution(
        requested_device=device_override if device_override is not None else config.training.device,
        amp_requested=bool(config.training.amp),
    )
    resolved_device = str(runtime_policy["device_resolved"])
    amp_requested = bool(runtime_policy["amp_requested"])
    amp_used = bool(runtime_policy["amp_used"])
    oom_fallbacks_applied: list[str] = []

    train_summary: dict[str, Any] = {}
    val_summary: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []
    best_metric_value: float | None = None
    best_epoch: int | None = None

    with progress_bar(
        total=epochs,
        desc="net_train: epochs",
        unit="epoch",
        progress_enabled=progress_enabled,
        leave=True,
    ) as epoch_bar:
        for epoch in range(1, epochs + 1):
            while True:
                try:
                    train_summary = train_one_epoch(
                        model,
                        train_loader,
                        loss_fn,
                        optimizer,
                        device=resolved_device,
                        amp_enabled=amp_used,
                        aux_weight=config.loss.aux_weight,
                        extent_aux_weight=getattr(config.loss, "extent_aux_weight", None),
                        boundary_aux_weight=getattr(config.loss, "boundary_aux_weight", None),
                        distance_aux_weight=getattr(config.loss, "distance_aux_weight", None),
                        gradient_clip_norm=config.training.gradient_clip,
                        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                        progress_enabled=progress_enabled,
                        input_normalizer=input_normalizer,
                    )
                    val_summary = evaluate_one_epoch(
                        model,
                        val_loader,
                        loss_fn,
                        device=resolved_device,
                        amp_enabled=amp_used,
                        aux_weight=config.loss.aux_weight,
                        extent_aux_weight=getattr(config.loss, "extent_aux_weight", None),
                        boundary_aux_weight=getattr(config.loss, "boundary_aux_weight", None),
                        distance_aux_weight=getattr(config.loss, "distance_aux_weight", None),
                        progress_enabled=progress_enabled,
                        input_normalizer=input_normalizer,
                    )
                    break
                except RuntimeError as exc:
                    if not _is_out_of_memory_error(exc):
                        raise
                    if amp_used:
                        amp_used = False
                        fallback_note = f"epoch_{epoch}:disable_amp_after_oom"
                        oom_fallbacks_applied.append(fallback_note)
                        if _TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise ContractError(
                        "Out-of-memory during module_net_train epoch execution and no further "
                        "runtime fallback is available. Baseline fallback policy can disable AMP once."
                    ) from exc

            monitored_metric = _compute_monitored_metric(val_summary, resolved_metric_name)
            is_best = _is_improved(
                candidate=monitored_metric,
                best_value=best_metric_value,
                mode=resolved_metric_mode,
            )
            if is_best:
                best_metric_value = monitored_metric
                best_epoch = epoch

            history_rows.append(
                _history_row(
                    epoch=epoch,
                    train_summary=train_summary,
                    val_summary=val_summary,
                    monitored_metric=monitored_metric,
                    is_best=is_best,
                )
            )

            checkpoint_payload = build_checkpoint_payload(
                config=config,
                model=model,
                epochs_completed=epoch,
            )
            save_checkpoint(last_checkpoint_path, checkpoint_payload)
            if is_best:
                save_checkpoint(best_checkpoint_path, checkpoint_payload)

            if scheduler_step_policy == "epoch_end":
                scheduler.step()
            elif scheduler_step_policy == "epoch_end_val_total":
                scheduler.step(_extract_float(val_summary, "total", context="val"))
            else:
                raise ContractError(
                    f"Unsupported scheduler_step_policy {scheduler_step_policy!r}."
                )
            scheduler_last_lr = float(optimizer.param_groups[0]["lr"])

            epoch_bar.update(1)
            epoch_bar.set_postfix(
                train=f"{train_summary.get('total', 0.0):.4f}",
                val=f"{val_summary.get('total', 0.0):.4f}",
                best=f"{best_metric_value:.4f}" if best_metric_value is not None else "—",
                lr=f"{scheduler_last_lr:.2e}",
            )

    if best_metric_value is None or best_epoch is None:
        raise ContractError("Failed to resolve best checkpoint: monitored metric was never recorded.")
    if not best_checkpoint_path.exists():
        raise ContractError(f"Best checkpoint was not written: {best_checkpoint_path}")
    if not last_checkpoint_path.exists():
        raise ContractError(f"Last checkpoint was not written: {last_checkpoint_path}")

    _write_history_csv(history_path, history_rows)

    best_state_dict = _load_checkpoint_state_dict(best_checkpoint_path)
    model.load_state_dict(best_state_dict)
    try:
        eval_val_summary = evaluate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device=resolved_device,
            amp_enabled=amp_used,
            aux_weight=config.loss.aux_weight,
            extent_aux_weight=getattr(config.loss, "extent_aux_weight", None),
            boundary_aux_weight=getattr(config.loss, "boundary_aux_weight", None),
            distance_aux_weight=getattr(config.loss, "distance_aux_weight", None),
            progress_enabled=progress_enabled,
            input_normalizer=input_normalizer,
        )
    except RuntimeError as exc:
        if not _is_out_of_memory_error(exc):
            raise
        if not amp_used:
            raise ContractError(
                "Out-of-memory during eval_val generation and no further runtime fallback is available."
            ) from exc
        amp_used = False
        oom_fallbacks_applied.append("eval_val:disable_amp_after_oom")
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_val_summary = evaluate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device=resolved_device,
            amp_enabled=amp_used,
            aux_weight=config.loss.aux_weight,
            extent_aux_weight=getattr(config.loss, "extent_aux_weight", None),
            boundary_aux_weight=getattr(config.loss, "boundary_aux_weight", None),
            distance_aux_weight=getattr(config.loss, "distance_aux_weight", None),
            progress_enabled=progress_enabled,
            input_normalizer=input_normalizer,
        )
    eval_val_path = run_path / "eval_val.json"
    _write_json(
        eval_val_path,
        {
            "schema_name": "net_train.eval_split",
            "schema_version": "v1",
            "run_id": run_id,
            "split": "val",
            "checkpoint_path": str(best_checkpoint_path),
            "checkpoint_epoch": int(best_epoch),
            "monitored_metric_name": resolved_metric_name,
            "monitored_metric_mode": resolved_metric_mode,
            "monitored_metric_value": _compute_monitored_metric(eval_val_summary, resolved_metric_name),
            "metrics": dict(eval_val_summary),
        },
    )
    bucket_metrics_val = _compute_bucket_metrics_for_split(
        model=model,
        split_dir=val_path,
        feature_mode=config.feature_mode,
        run_id=run_id,
        checkpoint_path=best_checkpoint_path,
        checkpoint_epoch=best_epoch,
        monitored_metric_name=resolved_metric_name,
        monitored_metric_mode=resolved_metric_mode,
        input_normalizer=input_normalizer,
        device=resolved_device,
        thresholds=(0.50, 0.55),
        progress_enabled=progress_enabled,
    )
    bucket_metrics_val_path = run_path / "bucket_metrics_val.json"
    _write_json(bucket_metrics_val_path, bucket_metrics_val)

    export_artifacts: NetTrainExportArtifacts = export_training_artifacts(
        run_dir=run_path,
        run_id=run_id,
        config=config,
        model=model,
        dataset_source_run_id=dataset_source_run_id,
        dataset_source_manifest_path=source_manifest_path,
        dataset_root=resolved_dataset_root,
        dataset_patch_size=resolved_dataset_patch_size,
        dataset_feature_mode=dataset_feature_mode or config.feature_mode,
        normalization=normalization,
        epochs_completed=epochs,
        best_metric_name=resolved_metric_name,
        best_metric_value=best_metric_value,
        checkpoint_filename=best_checkpoint_path.name,
        monitored_metric_mode=resolved_metric_mode,
        best_epoch=best_epoch,
        last_checkpoint_path=last_checkpoint_path,
        history_path=history_path,
        train_summary=train_summary,
        val_summary=val_summary,
        eval_val_summary=eval_val_summary,
        eval_val_path=eval_val_path,
        bucket_metrics_val_summary=bucket_metrics_val,
        bucket_metrics_val_path=bucket_metrics_val_path,
        coverage_aware_sampling=coverage_sampling_runtime,
        monitored_metric_policy_note=policy_note,
        scheduler_state_dict=scheduler.state_dict(),
        scheduler_step_policy=scheduler_step_policy,
        scheduler_last_lr=scheduler_last_lr,
        device_requested=runtime_policy["device_requested"],
        device_resolved=str(runtime_policy["device_resolved"]),
        amp_requested=amp_requested,
        amp_used=amp_used,
        oom_fallbacks_applied=tuple(oom_fallbacks_applied),
    )

    return NetTrainRunResult(
        run_dir=export_artifacts.run_dir,
        run_id=run_id,
        train_summary=train_summary,
        val_summary=val_summary,
        checkpoint_path=export_artifacts.checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        checkpoint_metadata_path=export_artifacts.checkpoint_metadata_path,
        train_manifest_path=export_artifacts.train_manifest_path,
        summary_path=export_artifacts.summary_path,
        config_used_path=export_artifacts.config_used_path,
        history_path=history_path,
        eval_val_path=eval_val_path,
        bucket_metrics_val_path=bucket_metrics_val_path,
        epochs_completed=epochs,
        best_metric_name=resolved_metric_name,
        best_metric_value=best_metric_value,
        best_epoch=best_epoch,
        monitored_metric_mode=resolved_metric_mode,
        monitored_metric_policy_note=policy_note,
        device_requested=runtime_policy["device_requested"],
        device_resolved=str(runtime_policy["device_resolved"]),
        amp_requested=amp_requested,
        amp_used=amp_used,
        oom_fallbacks_applied=tuple(oom_fallbacks_applied),
        dataset_root=resolved_dataset_root,
        dataset_patch_size=resolved_dataset_patch_size,
    )
