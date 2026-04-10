"""Stage E: checkpoint and export metadata layer for module_net_train.

This module provides a minimal, contract-first export utility that writes:
  - checkpoint file
  - checkpoint_metadata.json
  - train_manifest.json
  - summary.json
  - config_used.yaml

It intentionally avoids building a full experiment manager framework.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.constants import CHANNEL_COUNTS, DATA_CONTRACT_VERSION, FEATURE_MODES
from ai_fields.common.errors import ContractError, FeatureModeError
from ai_fields.common.manifests import write_manifest, write_summary

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


_DATASET_CHANNEL_SEMANTICS: dict[str, list[str]] = {
    "raw8": ["coastal", "blue", "green", "yellow", "red", "rededge", "nir1", "nir2"],
    "raw8_idx3": [
        "coastal",
        "blue",
        "green",
        "yellow",
        "red",
        "rededge",
        "nir1",
        "nir2",
        "NDVI",
        "SAVI",
        "NDWI",
    ],
}
_EXTENT_IGNORE_LABEL = 255


@dataclass(frozen=True)
class NetTrainExportArtifacts:
    """Paths to exported Stage E artifacts."""

    run_dir: Path
    checkpoint_path: Path
    checkpoint_metadata_path: Path
    train_manifest_path: Path
    summary_path: Path
    config_used_path: Path


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ContractError(
            "torch is required for checkpoint export in module_net_train. Install torch to use this module."
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_path(path: Any, *, name: str) -> Path:
    if isinstance(path, (str, PathLike)):
        normalized = Path(path)
    else:
        raise ContractError(
            f"{name} must be path-like (str or Path), got {type(path).__name__}."
        )
    if str(normalized).strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    return normalized


def _require_non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{name} must be a non-empty string.")
    return value


def _require_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number, got {type(value).__name__}.")
    return float(value)


def _require_pair_of_numbers(value: Any, *, name: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ContractError(f"{name} must be a 2-item numeric sequence.")
    return [_require_number(value[0], name=f"{name}[0]"), _require_number(value[1], name=f"{name}[1]")]


def _normalize_normalization_contract(normalization: Any) -> dict[str, Any]:
    if not isinstance(normalization, Mapping):
        raise ContractError(
            f"normalization must be a mapping/object, got {type(normalization).__name__}."
        )
    norm_name = _require_non_empty_string(
        normalization.get("normalization_name"), name="normalization.normalization_name"
    )
    stats_source = _require_non_empty_string(
        normalization.get("stats_source"), name="normalization.stats_source"
    )
    clip_percentiles = _require_pair_of_numbers(
        normalization.get("clip_percentiles"), name="normalization.clip_percentiles"
    )
    scaling_range = _require_pair_of_numbers(
        normalization.get("scaling_range"), name="normalization.scaling_range"
    )
    return {
        "normalization_name": norm_name,
        "stats_source": stats_source,
        "clip_percentiles": clip_percentiles,
        "scaling_range": scaling_range,
    }


def _assembled_contract(feature_mode: str) -> tuple[str, int, int, list[str]]:
    if feature_mode not in FEATURE_MODES:
        raise FeatureModeError(
            f"Unsupported feature_mode {feature_mode!r}. Expected one of {list(FEATURE_MODES)}."
        )
    assembled_model_input = f"{feature_mode}_valid"
    if assembled_model_input not in CHANNEL_COUNTS:
        raise ContractError(
            f"No channel count registered for assembled model input: {assembled_model_input!r}."
        )
    feature_channels = CHANNEL_COUNTS[feature_mode]
    final_channels = CHANNEL_COUNTS[assembled_model_input]
    channel_semantics = [*_DATASET_CHANNEL_SEMANTICS[feature_mode], "valid"]
    if len(channel_semantics) != final_channels:
        raise ContractError(
            "Assembled channel semantics length does not match final input channels: "
            f"{len(channel_semantics)} != {final_channels}."
        )
    return assembled_model_input, feature_channels, final_channels, channel_semantics


def _config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise ContractError(
        "config must be a dataclass instance or mapping/object to write config_used.yaml."
    )


def _write_config_used(path: Path, config: Any) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ContractError("PyYAML is required to write config_used.yaml.") from exc

    payload = _config_to_dict(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            yaml.safe_dump(payload, sort_keys=True, allow_unicode=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ContractError(f"Failed to write config_used.yaml at {path}: {exc}") from exc


def _manifest_base(
    *,
    schema_name: str,
    run_id: str,
    stage_name: str,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_name": schema_name,
        "schema_version": "v1",
        "module_name": "module_net_train",
        "module_version": None,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "run_id": run_id,
        "stage_name": stage_name,
        "created_at_utc": _utc_now_iso(),
        "status": status,
    }


def _resolve_export_contract(config: Any, model: Any) -> dict[str, Any]:
    if not isinstance(getattr(config, "valid_as_input_channel", None), bool):
        raise ContractError("config.valid_as_input_channel must be a boolean.")
    if config.valid_as_input_channel is not True:
        raise ContractError(
            "config.valid_as_input_channel must be True for baseline v1 export contract."
        )

    feature_mode = getattr(config, "feature_mode", None)
    if not isinstance(feature_mode, str):
        raise ContractError("config.feature_mode must be a string.")

    assembled_model_input, feature_channels, final_channels, channel_semantics = _assembled_contract(
        feature_mode
    )

    model_in_channels = getattr(model, "in_channels", None)
    if (
        isinstance(model_in_channels, bool)
        or not isinstance(model_in_channels, int)
        or model_in_channels < 1
    ):
        raise ContractError(
            "model.in_channels must exist and be a positive integer to export checkpoint metadata."
        )
    if model_in_channels != final_channels:
        raise ContractError(
            f"model.in_channels={model_in_channels} does not match assembled contract "
            f"{assembled_model_input} ({final_channels})."
        )

    return {
        "feature_mode": feature_mode,
        "assembled_model_input": assembled_model_input,
        "feature_channel_count": feature_channels,
        "final_input_channel_count": final_channels,
        "channel_semantics": channel_semantics,
        "in_channels": model_in_channels,
    }


def _resolve_model_architecture_contract(config: Any, model: Any) -> dict[str, Any]:
    model_cfg = getattr(config, "model", None)
    architecture = _require_non_empty_string(
        getattr(model_cfg, "architecture", None),
        name="config.model.architecture",
    )

    encoder_depth = getattr(model_cfg, "encoder_depth", None)
    if isinstance(encoder_depth, bool) or not isinstance(encoder_depth, int) or encoder_depth < 2:
        raise ContractError(
            "config.model.encoder_depth must be an integer >= 2 for checkpoint metadata export."
        )

    base_channels = getattr(model_cfg, "base_channels", None)
    if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels < 1:
        raise ContractError(
            "config.model.base_channels must be an integer >= 1 for checkpoint metadata export."
        )

    model_encoder_depth = getattr(model, "encoder_depth", None)
    if (
        isinstance(model_encoder_depth, bool)
        or not isinstance(model_encoder_depth, int)
        or model_encoder_depth < 2
    ):
        raise ContractError(
            "model.encoder_depth must exist and be an integer >= 2 to export checkpoint metadata."
        )
    if model_encoder_depth != encoder_depth:
        raise ContractError(
            "model.encoder_depth is inconsistent with config.model.encoder_depth: "
            f"{model_encoder_depth} != {encoder_depth}."
        )

    model_base_channels = getattr(model, "base_channels", None)
    if (
        isinstance(model_base_channels, bool)
        or not isinstance(model_base_channels, int)
        or model_base_channels < 1
    ):
        raise ContractError(
            "model.base_channels must exist and be an integer >= 1 to export checkpoint metadata."
        )
    if model_base_channels != base_channels:
        raise ContractError(
            "model.base_channels is inconsistent with config.model.base_channels: "
            f"{model_base_channels} != {base_channels}."
        )

    return {
        "model_architecture": architecture,
        "encoder_depth": encoder_depth,
        "base_channels": base_channels,
    }


def build_checkpoint_payload(
    *,
    config: Any,
    model: Any,
    epochs_completed: int,
    optimizer_state_dict: Mapping[str, Any] | None = None,
    scheduler_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a strict checkpoint payload for net_train artifacts."""
    _require_torch()
    if (
        isinstance(epochs_completed, bool)
        or not isinstance(epochs_completed, int)
        or epochs_completed < 0
    ):
        raise ContractError("epochs_completed must be an integer >= 0.")

    contract = _resolve_export_contract(config, model)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "feature_mode": contract["feature_mode"],
        "assembled_model_input": contract["assembled_model_input"],
        "in_channels": contract["in_channels"],
        "channel_semantics": contract["channel_semantics"],
        "valid_as_input_channel": True,
        "epochs_completed": epochs_completed,
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = dict(optimizer_state_dict)
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = dict(scheduler_state_dict)
    return payload


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Save checkpoint payload to disk and return resolved path."""
    _require_torch()
    checkpoint_path = _normalize_path(path, name="checkpoint path")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(dict(payload), checkpoint_path)
    except OSError as exc:
        raise ContractError(f"Failed to write checkpoint to {checkpoint_path}: {exc}") from exc
    return checkpoint_path


def export_training_artifacts(
    *,
    run_dir: str | Path,
    run_id: str,
    config: Any,
    model: Any,
    dataset_source_run_id: str,
    dataset_source_manifest_path: str | Path,
    normalization: Mapping[str, Any],
    epochs_completed: int,
    best_metric_name: str,
    best_metric_value: float | None,
    dataset_root: str | Path | None = None,
    dataset_patch_size: int | None = None,
    dataset_feature_mode: str | None = None,
    checkpoint_filename: str = "best.ckpt",
    summary_warnings: Sequence[str] | None = None,
    model_version: str | None = None,
    checkpoint_status: str = "success",
    optimizer_state_dict: Mapping[str, Any] | None = None,
    scheduler_state_dict: Mapping[str, Any] | None = None,
    monitored_metric_mode: str | None = None,
    best_epoch: int | None = None,
    last_checkpoint_path: str | Path | None = None,
    history_path: str | Path | None = None,
    train_summary: Mapping[str, Any] | None = None,
    val_summary: Mapping[str, Any] | None = None,
    monitored_metric_policy_note: str | None = None,
    scheduler_step_policy: str | None = None,
    scheduler_last_lr: float | None = None,
) -> NetTrainExportArtifacts:
    """Export minimal Stage E artifacts for module_net_train.

    This function is intentionally strict and explicit:
      - no hidden fallback metadata defaults for critical fields;
      - explicit ContractError on missing downstream-required metadata.
    """
    _require_torch()
    resolved_run_dir = _normalize_path(run_dir, name="run_dir")
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    run_id = _require_non_empty_string(run_id, name="run_id")
    dataset_source_run_id = _require_non_empty_string(
        dataset_source_run_id, name="dataset_source_run_id"
    )
    source_manifest_path = _normalize_path(
        dataset_source_manifest_path, name="dataset_source_manifest_path"
    )
    resolved_dataset_root = (
        _normalize_path(dataset_root, name="dataset_root")
        if dataset_root is not None
        else None
    )
    if not source_manifest_path.exists():
        raise ContractError(
            f"dataset_source_manifest_path does not exist: {source_manifest_path}"
        )
    if dataset_patch_size is not None:
        if isinstance(dataset_patch_size, bool) or not isinstance(dataset_patch_size, int):
            raise ContractError("dataset_patch_size must be an integer when provided.")
        if dataset_patch_size < 1:
            raise ContractError("dataset_patch_size must be >= 1 when provided.")
    if dataset_feature_mode is not None:
        dataset_feature_mode = _require_non_empty_string(
            dataset_feature_mode, name="dataset_feature_mode"
        )
    best_metric_name = _require_non_empty_string(best_metric_name, name="best_metric_name")
    if (
        isinstance(epochs_completed, bool)
        or not isinstance(epochs_completed, int)
        or epochs_completed < 0
    ):
        raise ContractError("epochs_completed must be an integer >= 0.")
    contract = _resolve_export_contract(config, model)
    model_arch_contract = _resolve_model_architecture_contract(config, model)
    feature_mode = contract["feature_mode"]
    if dataset_feature_mode is not None and dataset_feature_mode != feature_mode:
        raise ContractError(
            "dataset_feature_mode is inconsistent with config.feature_mode: "
            f"{dataset_feature_mode!r} != {feature_mode!r}."
        )
    resolved_dataset_feature_mode = dataset_feature_mode or feature_mode
    assembled_model_input = contract["assembled_model_input"]
    feature_channels = contract["feature_channel_count"]
    final_channels = contract["final_input_channel_count"]
    channel_semantics = contract["channel_semantics"]
    model_in_channels = contract["in_channels"]

    normalized_normalization = _normalize_normalization_contract(normalization)
    if monitored_metric_mode is not None:
        monitored_metric_mode = _require_non_empty_string(
            monitored_metric_mode, name="monitored_metric_mode"
        )
        if monitored_metric_mode not in {"min", "max"}:
            raise ContractError("monitored_metric_mode must be 'min' or 'max'.")
    if scheduler_step_policy is not None:
        scheduler_step_policy = _require_non_empty_string(
            scheduler_step_policy, name="scheduler_step_policy"
        )
    if scheduler_last_lr is not None:
        scheduler_last_lr = _require_number(scheduler_last_lr, name="scheduler_last_lr")
    if best_epoch is not None:
        if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 1:
            raise ContractError("best_epoch must be an integer >= 1 when provided.")
    resolved_last_checkpoint_path = (
        _normalize_path(last_checkpoint_path, name="last_checkpoint_path")
        if last_checkpoint_path is not None
        else None
    )
    resolved_history_path = (
        _normalize_path(history_path, name="history_path")
        if history_path is not None
        else None
    )

    checkpoint_path = resolved_run_dir / checkpoint_filename
    checkpoint_metadata_path = resolved_run_dir / "checkpoint_metadata.json"
    train_manifest_path = resolved_run_dir / "train_manifest.json"
    summary_path = resolved_run_dir / "summary.json"
    config_used_path = resolved_run_dir / "config_used.yaml"

    checkpoint_payload = build_checkpoint_payload(
        config=config,
        model=model,
        epochs_completed=epochs_completed,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
    )
    save_checkpoint(checkpoint_path, checkpoint_payload)

    _write_config_used(config_used_path, config)

    checkpoint_metadata = _manifest_base(
        schema_name="net_train.checkpoint_metadata",
        run_id=run_id,
        stage_name="checkpoint_export",
        status=checkpoint_status,
    )
    checkpoint_metadata.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "feature_mode": feature_mode,
            "assembled_model_input": assembled_model_input,
            "in_channels": model_in_channels,
            "channel_semantics": channel_semantics,
            "valid_as_input_channel": True,
            "normalization": normalized_normalization,
            "target_heads": {
                "extent": {
                    "type": "binary_segmentation",
                    "ignore_label": _EXTENT_IGNORE_LABEL,
                },
                "boundary": {
                    "type": "multiclass_segmentation",
                    "classes": {"0": "background", "1": "skeleton", "2": "buffer"},
                },
                "distance": {
                    "type": "regression",
                    "definition": "unsigned_distance_to_nearest_boundary",
                    "loss": "smooth_l1",
                },
            },
            "model_version": model_version,
            "model_architecture": model_arch_contract["model_architecture"],
            "encoder_depth": model_arch_contract["encoder_depth"],
            "base_channels": model_arch_contract["base_channels"],
            "model": {
                "architecture_name": model_arch_contract["model_architecture"],
                "encoder_depth": model_arch_contract["encoder_depth"],
                "base_channels": model_arch_contract["base_channels"],
            },
            "provenance": {
                "source_run_ids": [dataset_source_run_id],
                "source_manifest_paths": [str(source_manifest_path)],
                "source_config_paths": [],
                "code_version": None,
                "git_commit": None,
            },
            "dataset_root": str(resolved_dataset_root) if resolved_dataset_root is not None else None,
            "patch_size": dataset_patch_size,
            "dataset_feature_mode": resolved_dataset_feature_mode,
        }
    )
    write_manifest(checkpoint_metadata_path, checkpoint_metadata)

    train_manifest = _manifest_base(
        schema_name="net_train.train_manifest",
        run_id=run_id,
        stage_name="train",
        status=checkpoint_status,
    )
    train_manifest.update(
        {
            "dataset_source_run_id": dataset_source_run_id,
            "dataset_source_manifest_path": str(source_manifest_path),
            "dataset_root": str(resolved_dataset_root) if resolved_dataset_root is not None else None,
            "patch_size": dataset_patch_size,
            "dataset_feature_mode": resolved_dataset_feature_mode,
            "feature_mode": feature_mode,
            "assembled_model_input": assembled_model_input,
            "feature_channel_count": feature_channels,
            "final_input_channel_count": final_channels,
            "channel_semantics": channel_semantics,
            "valid_as_input_channel": True,
            "model": {
                "architecture_name": model_arch_contract["model_architecture"],
                "encoder_depth": model_arch_contract["encoder_depth"],
                "base_channels": model_arch_contract["base_channels"],
                "encoder_name": None,
                "heads": ["extent", "boundary", "distance"],
            },
            "loss": {
                "extent_loss_name": "focal_bce_plus_soft_dice",
                "boundary_loss_name": "weighted_focal_ce_plus_skeleton_soft_dice",
                "distance_loss_name": "smooth_l1",
                "loss_weights": {
                    "extent": getattr(getattr(config, "loss", None), "extent_weight", None),
                    "boundary": getattr(getattr(config, "loss", None), "boundary_weight", None),
                    "distance": getattr(getattr(config, "loss", None), "distance_weight", None),
                },
            },
            "optimizer": {
                "name": getattr(getattr(config, "optimizer", None), "name", None),
                "lr": getattr(getattr(config, "optimizer", None), "lr", None),
            },
            "scheduler": {
                "name": getattr(getattr(config, "scheduler", None), "name", None),
                "step_policy": scheduler_step_policy,
                "last_lr": scheduler_last_lr,
            },
            "training": {
                "batch_size": getattr(getattr(config, "training", None), "batch_size", None),
                "epochs_completed": epochs_completed,
                "amp_used": bool(getattr(getattr(config, "training", None), "amp", False)),
                "augment": bool(getattr(getattr(config, "training", None), "augment", True)),
                "best_checkpoint_metric": best_metric_name,
                "best_metric_value": best_metric_value,
                "best_epoch": best_epoch,
                "monitored_metric_mode": monitored_metric_mode,
            },
            "config": {
                "config_used_path": str(config_used_path),
                "config_hash": None,
                "config_overrides": None,
            },
            "provenance": {
                "source_run_ids": [dataset_source_run_id],
                "source_manifest_paths": [str(source_manifest_path)],
                "source_config_paths": [],
                "code_version": None,
                "git_commit": None,
            },
            "inputs": {
                "artifacts": [
                    {
                        "path": str(source_manifest_path),
                        "role": "dataset_source_manifest",
                        "format": "json",
                        "is_required": True,
                        "exists": source_manifest_path.exists(),
                    }
                ]
            },
            "outputs": {
                "artifacts": [
                    {
                        "path": str(checkpoint_path),
                        "role": "checkpoint",
                        "format": "pytorch",
                        "is_required": True,
                        "exists": checkpoint_path.exists(),
                    },
                    {
                        "path": str(resolved_last_checkpoint_path)
                        if resolved_last_checkpoint_path is not None
                        else None,
                        "role": "last_checkpoint",
                        "format": "pytorch",
                        "is_required": False,
                        "exists": bool(
                            resolved_last_checkpoint_path is not None
                            and resolved_last_checkpoint_path.exists()
                        ),
                    },
                    {
                        "path": str(checkpoint_metadata_path),
                        "role": "checkpoint_metadata",
                        "format": "json",
                        "is_required": True,
                        "exists": True,
                    },
                ]
            },
            "resolved_contract": {
                "features": {
                    "dataset_feature_mode": feature_mode,
                    "assembled_model_input": assembled_model_input,
                    "feature_channel_count": feature_channels,
                    "final_input_channel_count": final_channels,
                    "channel_semantics": channel_semantics,
                    "valid_as_input_channel": True,
                },
                "normalization": normalized_normalization,
                "valid_policy": {
                    "valid_source": "prep_data_export",
                    "valid_representation": "separate_binary_layer_plus_input_channel",
                    "invalid_handling": "exclude_from_loss_metrics_and_runtime_checks",
                    "nodata_policy": "no_silent_fallback",
                },
                "spatial": {},
                "dataset": {
                    "dataset_root": str(resolved_dataset_root)
                    if resolved_dataset_root is not None
                    else None,
                    "patch_size": dataset_patch_size,
                    "feature_mode": resolved_dataset_feature_mode,
                },
                "aoi_policy": None,
            },
            "runtime": {
                "device_requested": getattr(getattr(config, "training", None), "device", None),
                "device_resolved": None,
                "amp_requested": bool(getattr(getattr(config, "training", None), "amp", False)),
                "amp_used": bool(getattr(getattr(config, "training", None), "amp", False)),
                "oom_fallbacks_applied": [],
                "notes": [],
            },
            "diagnostics": {"warnings": [], "errors": []},
            "history_path": str(resolved_history_path) if resolved_history_path is not None else None,
            "last_checkpoint_path": str(resolved_last_checkpoint_path)
            if resolved_last_checkpoint_path is not None
            else None,
            "best_checkpoint_path": str(checkpoint_path),
            "monitored_metric_name": best_metric_name,
            "monitored_metric_mode": monitored_metric_mode,
            "monitored_metric_policy_note": monitored_metric_policy_note,
            "train_summary": dict(train_summary) if train_summary is not None else None,
            "val_summary": dict(val_summary) if val_summary is not None else None,
        }
    )
    write_manifest(train_manifest_path, train_manifest)

    warnings_list = list(summary_warnings) if summary_warnings is not None else []
    summary_payload = {
        "schema_name": "net_train.summary",
        "status": checkpoint_status,
        "feature_mode": feature_mode,
        "dataset_feature_mode": resolved_dataset_feature_mode,
        "dataset_root": str(resolved_dataset_root) if resolved_dataset_root is not None else None,
        "patch_size": dataset_patch_size,
        "assembled_model_input": assembled_model_input,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "monitored_metric_mode": monitored_metric_mode,
        "best_checkpoint_path": str(checkpoint_path),
        "last_checkpoint_path": str(resolved_last_checkpoint_path)
        if resolved_last_checkpoint_path is not None
        else None,
        "scheduler_name": getattr(getattr(config, "scheduler", None), "name", None),
        "scheduler_step_policy": scheduler_step_policy,
        "scheduler_last_lr": scheduler_last_lr,
        "history_path": str(resolved_history_path) if resolved_history_path is not None else None,
        "epochs_completed": epochs_completed,
        "warnings": warnings_list,
        "run_id": run_id,
        "created_at_utc": _utc_now_iso(),
        "checkpoint_metadata_path": str(checkpoint_metadata_path),
        "train_manifest_path": str(train_manifest_path),
        "monitored_metric_policy_note": monitored_metric_policy_note,
        "train": dict(train_summary) if train_summary is not None else None,
        "val": dict(val_summary) if val_summary is not None else None,
    }
    write_summary(summary_path, summary_payload)

    return NetTrainExportArtifacts(
        run_dir=resolved_run_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        train_manifest_path=train_manifest_path,
        summary_path=summary_path,
        config_used_path=config_used_path,
    )
