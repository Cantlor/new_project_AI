#!/usr/bin/env bash
set -euo pipefail

# Canonical runner for module_net_train.
# Quick-edit block: set defaults here for local operator workflows.
PYTHON_CMD="python3"
CONFIG_PATH=""
PREP_RUN_DIR=""
DATASET_ROOT=""
TRAIN_SPLIT_DIR=""
VAL_SPLIT_DIR=""
DATASET_SOURCE_MANIFEST=""
DATASET_SOURCE_RUN_ID=""
EXPECTED_PATCH_SIZE=""
RUNS_ROOT="runs/module_net_train"
RUN_ID="net-train-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR=""
NORMALIZATION_NAME=""
NORMALIZATION_CLIP_PERCENTILES=""
NORMALIZATION_SCALING_RANGE=""
NORMALIZATION_STATS_SOURCE=""
EPOCHS_OVERRIDE=""
DEVICE_OVERRIDE=""
MONITORED_METRIC_NAME=""
MONITORED_METRIC_MODE=""
FROM_STEP="train"
TO_STEP="train"
DRY_RUN=false

readonly STEPS=("train")

print_help() {
  cat <<'EOF'
Usage:
  bash tools/run_module_net_train.sh --config <path> [options]

Required:
  --config <path>                  Training config YAML (primary config selection mode)

Dataset input (provide one of the practical modes):
  --prep-run-dir <path>            module_prep_data run dir; auto-resolves:
                                   <run>/06_split_dataset/dataset
                                   <run>/06_split_dataset/split_manifest.json
  --dataset-root <path>            One selected train-ready dataset root with train/val splits
  --train-split-dir <path>         Explicit train split dir (default: <dataset-root>/train)
  --val-split-dir <path>           Explicit val split dir (default: <dataset-root>/val)
  --dataset-source-manifest <path> split_manifest.json from module_prep_data
  --dataset-source-run-id <id>     Source prep run id (defaults from manifest run_id)
  --patch-size <int>               Optional strict consistency check (256|384|512)

Run/output:
  --runs-root <path>               Root directory for net_train runs
  --run-id <id>                    Run id (default: net-train-<UTC timestamp>)
  --run-dir <path>                 Full run dir override (has priority over runs-root/run-id)

Optional runtime overrides (passed to run_train_baseline):
  --epochs-override <int>
  --device-override <cuda|mps|cpu>
  --monitored-metric-name <name>
  --monitored-metric-mode <min|max>

Optional normalization contract overrides:
  --normalization-name <name>
  --normalization-clip-percentiles <lo,hi>
  --normalization-scaling-range <min,max>
  --normalization-stats-source <path-or-id>

Step control:
  --from-step <name>               Current module exposes one operator step: train
  --to-step <name>                 Current module exposes one operator step: train

Other:
  --python-cmd <cmd>               Python executable (default: python3)
  --dry-run                        Print resolved plan and exit
  -h, --help                       Show this help
EOF
}

is_valid_step() {
  local candidate="$1"
  local step
  for step in "${STEPS[@]}"; do
    if [[ "${step}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"; shift 2 ;;
    --prep-run-dir)
      PREP_RUN_DIR="${2:-}"; shift 2 ;;
    --dataset-root)
      DATASET_ROOT="${2:-}"; shift 2 ;;
    --train-split-dir)
      TRAIN_SPLIT_DIR="${2:-}"; shift 2 ;;
    --val-split-dir)
      VAL_SPLIT_DIR="${2:-}"; shift 2 ;;
    --dataset-source-manifest)
      DATASET_SOURCE_MANIFEST="${2:-}"; shift 2 ;;
    --dataset-source-run-id)
      DATASET_SOURCE_RUN_ID="${2:-}"; shift 2 ;;
    --patch-size)
      EXPECTED_PATCH_SIZE="${2:-}"; shift 2 ;;
    --runs-root)
      RUNS_ROOT="${2:-}"; shift 2 ;;
    --run-id)
      RUN_ID="${2:-}"; shift 2 ;;
    --run-dir)
      RUN_DIR="${2:-}"; shift 2 ;;
    --epochs-override)
      EPOCHS_OVERRIDE="${2:-}"; shift 2 ;;
    --device-override)
      DEVICE_OVERRIDE="${2:-}"; shift 2 ;;
    --monitored-metric-name)
      MONITORED_METRIC_NAME="${2:-}"; shift 2 ;;
    --monitored-metric-mode)
      MONITORED_METRIC_MODE="${2:-}"; shift 2 ;;
    --normalization-name)
      NORMALIZATION_NAME="${2:-}"; shift 2 ;;
    --normalization-clip-percentiles)
      NORMALIZATION_CLIP_PERCENTILES="${2:-}"; shift 2 ;;
    --normalization-scaling-range)
      NORMALIZATION_SCALING_RANGE="${2:-}"; shift 2 ;;
    --normalization-stats-source)
      NORMALIZATION_STATS_SOURCE="${2:-}"; shift 2 ;;
    --from-step)
      FROM_STEP="${2:-}"; shift 2 ;;
    --to-step)
      TO_STEP="${2:-}"; shift 2 ;;
    --python-cmd)
      PYTHON_CMD="${2:-}"; shift 2 ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    -h|--help)
      print_help; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      print_help
      exit 2 ;;
  esac
done

# Config selection policy:
# 1) primary: --config
# 2) fallback: CONFIG_PATH in quick-edit block
# 3) otherwise: fail
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Missing config path. Provide --config or set CONFIG_PATH at the top of this script." >&2
  exit 2
fi

if [[ -n "${PREP_RUN_DIR}" && -n "${DATASET_ROOT}" ]]; then
  echo "[ERROR] Ambiguous dataset selection: provide either --prep-run-dir or --dataset-root, not both." >&2
  exit 2
fi
if [[ -n "${PREP_RUN_DIR}" && -n "${DATASET_SOURCE_MANIFEST}" ]]; then
  echo "[ERROR] Ambiguous dataset selection: --prep-run-dir already defines split manifest; do not pass --dataset-source-manifest." >&2
  exit 2
fi
if [[ -n "${PREP_RUN_DIR}" && -n "${DATASET_SOURCE_RUN_ID}" ]]; then
  echo "[ERROR] Ambiguous dataset selection: --prep-run-dir already defines source run id; do not pass --dataset-source-run-id." >&2
  exit 2
fi
if [[ -n "${PREP_RUN_DIR}" && -n "${TRAIN_SPLIT_DIR}" ]]; then
  echo "[ERROR] Ambiguous dataset selection: do not pass --train-split-dir together with --prep-run-dir." >&2
  exit 2
fi
if [[ -n "${PREP_RUN_DIR}" && -n "${VAL_SPLIT_DIR}" ]]; then
  echo "[ERROR] Ambiguous dataset selection: do not pass --val-split-dir together with --prep-run-dir." >&2
  exit 2
fi

if [[ -n "${PREP_RUN_DIR}" ]]; then
  DATASET_ROOT="${PREP_RUN_DIR%/}/06_split_dataset/dataset"
  DATASET_SOURCE_MANIFEST="${PREP_RUN_DIR%/}/06_split_dataset/split_manifest.json"
  if [[ -z "${DATASET_SOURCE_RUN_ID}" ]]; then
    DATASET_SOURCE_RUN_ID="$(basename "${PREP_RUN_DIR%/}")"
  fi
fi

if [[ -z "${PREP_RUN_DIR}" && -z "${DATASET_ROOT}" ]]; then
  echo "[ERROR] Missing dataset selection. Provide one explicit source: --prep-run-dir or --dataset-root." >&2
  exit 2
fi
if [[ -z "${TRAIN_SPLIT_DIR}" ]]; then
  TRAIN_SPLIT_DIR="${DATASET_ROOT%/}/train"
fi
if [[ -z "${VAL_SPLIT_DIR}" ]]; then
  VAL_SPLIT_DIR="${DATASET_ROOT%/}/val"
fi
if [[ -z "${DATASET_SOURCE_MANIFEST}" ]]; then
  echo "[ERROR] Missing dataset source manifest. Provide --dataset-source-manifest (or use --prep-run-dir mode)." >&2
  exit 2
fi
if [[ -n "${EXPECTED_PATCH_SIZE}" ]]; then
  if ! [[ "${EXPECTED_PATCH_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --patch-size must be an integer (256, 384, 512)." >&2
    exit 2
  fi
  if [[ "${EXPECTED_PATCH_SIZE}" != "256" && "${EXPECTED_PATCH_SIZE}" != "384" && "${EXPECTED_PATCH_SIZE}" != "512" ]]; then
    echo "[ERROR] --patch-size supports baseline values 256, 384, 512." >&2
    exit 2
  fi
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${RUNS_ROOT%/}/${RUN_ID}"
fi

if ! is_valid_step "${FROM_STEP}"; then
  echo "[ERROR] Invalid --from-step: ${FROM_STEP}. Supported: ${STEPS[*]}." >&2
  exit 2
fi
if ! is_valid_step "${TO_STEP}"; then
  echo "[ERROR] Invalid --to-step: ${TO_STEP}. Supported: ${STEPS[*]}." >&2
  exit 2
fi
if [[ "${FROM_STEP}" != "train" || "${TO_STEP}" != "train" ]]; then
  echo "[ERROR] module_net_train currently exposes one operator step only: train." >&2
  exit 2
fi

echo "[INFO] module_net_train canonical operator step: train"
echo "[INFO] Requested step range: ${FROM_STEP} -> ${TO_STEP}"
echo "[INFO] Config: ${CONFIG_PATH}"
if [[ -n "${PREP_RUN_DIR}" ]]; then
  echo "[INFO] Dataset selection mode: prep-run-dir"
  echo "[INFO] Prep run dir: ${PREP_RUN_DIR}"
else
  echo "[INFO] Dataset selection mode: dataset-root"
fi
echo "[INFO] Dataset root: ${DATASET_ROOT}"
echo "[INFO] Train split dir: ${TRAIN_SPLIT_DIR}"
echo "[INFO] Val split dir: ${VAL_SPLIT_DIR}"
echo "[INFO] Dataset source manifest: ${DATASET_SOURCE_MANIFEST}"
if [[ -n "${DATASET_SOURCE_RUN_ID}" ]]; then
  echo "[INFO] Dataset source run id: ${DATASET_SOURCE_RUN_ID}"
else
  echo "[INFO] Dataset source run id: <auto-from-manifest>"
fi
if [[ -n "${EXPECTED_PATCH_SIZE}" ]]; then
  echo "[INFO] Patch size consistency check: ${EXPECTED_PATCH_SIZE}"
else
  echo "[INFO] Patch size consistency check: <auto-from-manifest-or-data>"
fi
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Runner uses entrypoint: ai_fields.module_net_train.run_train.run_train_baseline"

if [[ "${DRY_RUN}" == true ]]; then
  echo "[DRY-RUN] Training will not start."
  echo "[DRY-RUN] Python command: ${PYTHON_CMD}"
  exit 0
fi

if [[ ! -d "${TRAIN_SPLIT_DIR}" ]]; then
  echo "[ERROR] train split dir does not exist: ${TRAIN_SPLIT_DIR}" >&2
  exit 2
fi
if [[ ! -d "${VAL_SPLIT_DIR}" ]]; then
  echo "[ERROR] val split dir does not exist: ${VAL_SPLIT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${DATASET_SOURCE_MANIFEST}" ]]; then
  echo "[ERROR] dataset source manifest does not exist: ${DATASET_SOURCE_MANIFEST}" >&2
  exit 2
fi

echo "[STEP 1/1] train: building config + resolving normalization + running baseline training"
PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" \
NT_CONFIG_PATH="${CONFIG_PATH}" \
NT_DATASET_ROOT="${DATASET_ROOT}" \
NT_TRAIN_SPLIT_DIR="${TRAIN_SPLIT_DIR}" \
NT_VAL_SPLIT_DIR="${VAL_SPLIT_DIR}" \
NT_DATASET_SOURCE_MANIFEST="${DATASET_SOURCE_MANIFEST}" \
NT_DATASET_SOURCE_RUN_ID="${DATASET_SOURCE_RUN_ID}" \
NT_EXPECTED_PATCH_SIZE="${EXPECTED_PATCH_SIZE}" \
NT_RUN_DIR="${RUN_DIR}" \
NT_RUN_ID="${RUN_ID}" \
NT_NORMALIZATION_NAME="${NORMALIZATION_NAME}" \
NT_NORMALIZATION_CLIP_PERCENTILES="${NORMALIZATION_CLIP_PERCENTILES}" \
NT_NORMALIZATION_SCALING_RANGE="${NORMALIZATION_SCALING_RANGE}" \
NT_NORMALIZATION_STATS_SOURCE="${NORMALIZATION_STATS_SOURCE}" \
NT_EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE}" \
NT_DEVICE_OVERRIDE="${DEVICE_OVERRIDE}" \
NT_MONITORED_METRIC_NAME="${MONITORED_METRIC_NAME}" \
NT_MONITORED_METRIC_MODE="${MONITORED_METRIC_MODE}" \
"${PYTHON_CMD}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from ai_fields.common.errors import ContractError
from ai_fields.module_net_train.run_train import (
    run_train_baseline,
    validate_train_ready_dataset_contract,
)
from ai_fields.module_net_train.schemas import (
    LossConfig,
    ModelConfig,
    MonitoringConfig,
    NetTrainConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _parse_number_pair(raw: Any, *, name: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ContractError(f"{name} must be a 2-item sequence.")
    out: list[float] = []
    for idx, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"{name}[{idx}] must be numeric.")
        out.append(float(value))
    return out


def _parse_pair_override(raw: str, *, name: str) -> list[float] | None:
    if raw == "":
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ContractError(f"{name} must have format '<a>,<b>'.")
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError as exc:
        raise ContractError(f"{name} must contain numeric values.") from exc


def _optional_int(raw: str, *, name: str) -> int | None:
    if raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ContractError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ContractError(f"{name} must be >= 1.")
    return value


def _optional_str(raw: str) -> str | None:
    return raw if raw != "" else None


def _optional_patch_size(raw: str) -> int | None:
    if raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ContractError("--patch-size must be an integer.") from exc
    if value not in {256, 384, 512}:
        raise ContractError("--patch-size supports baseline values 256, 384, 512.")
    return value


def _load_train_config(path: Path) -> NetTrainConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"Failed to read config YAML at {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ContractError(f"Config YAML is invalid at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("Training config YAML must contain a top-level mapping/object.")

    required_sections = ("model", "loss", "optimizer", "scheduler", "training", "monitoring")
    for section in required_sections:
        if not isinstance(payload.get(section), dict):
            raise ContractError(f"Config section '{section}' is required and must be a mapping.")

    cfg = NetTrainConfig(
        feature_mode=payload.get("feature_mode"),
        valid_as_input_channel=payload.get("valid_as_input_channel"),
        model=ModelConfig(**payload["model"]),
        loss=LossConfig(**payload["loss"]),
        optimizer=OptimizerConfig(**payload["optimizer"]),
        scheduler=SchedulerConfig(**payload["scheduler"]),
        training=TrainingConfig(**payload["training"]),
        monitoring=MonitoringConfig(**payload["monitoring"]),
    )
    cfg.validate()
    return cfg


def _resolve_dataset_source_run_id(manifest_path: Path, manifest_payload: dict[str, Any]) -> str:
    explicit = _env("NT_DATASET_SOURCE_RUN_ID")
    if explicit:
        return explicit
    run_id = manifest_payload.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id.strip()
    # common prep path: <prep-run-id>/06_split_dataset/split_manifest.json
    if manifest_path.parent.name == "06_split_dataset":
        candidate = manifest_path.parent.parent.name
        if candidate.strip():
            return candidate
    raise ContractError(
        "Could not resolve dataset_source_run_id. "
        "Provide --dataset-source-run-id explicitly."
    )


def _resolve_normalization(
    *,
    manifest_payload: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    override_name = _optional_str(_env("NT_NORMALIZATION_NAME"))
    override_clip = _parse_pair_override(
        _env("NT_NORMALIZATION_CLIP_PERCENTILES"),
        name="--normalization-clip-percentiles",
    )
    override_scale = _parse_pair_override(
        _env("NT_NORMALIZATION_SCALING_RANGE"),
        name="--normalization-scaling-range",
    )
    override_stats = _optional_str(_env("NT_NORMALIZATION_STATS_SOURCE"))

    norm_block: dict[str, Any] = {}
    resolved_contract = manifest_payload.get("resolved_contract")
    if isinstance(resolved_contract, dict):
        candidate = resolved_contract.get("normalization")
        if isinstance(candidate, dict):
            norm_block = candidate
    if not norm_block:
        candidate = manifest_payload.get("normalization")
        if isinstance(candidate, dict):
            norm_block = candidate

    norm_name = override_name or norm_block.get("normalization_name")
    if not isinstance(norm_name, str) or not norm_name.strip():
        raise ContractError(
            "Could not resolve normalization.normalization_name from split manifest. "
            "Provide --normalization-name."
        )

    clip = override_clip or _parse_number_pair(
        norm_block.get("clip_percentiles"),
        name="normalization.clip_percentiles",
    )

    scale_raw = norm_block.get("scaling_range")
    if scale_raw is None:
        scale_raw = norm_block.get("scale_range")
    scale = override_scale or _parse_number_pair(
        scale_raw,
        name="normalization.scaling_range",
    )

    stats_source = override_stats
    if not stats_source:
        maybe_stats = norm_block.get("stats_source")
        if isinstance(maybe_stats, str) and maybe_stats.strip():
            stats_source = maybe_stats.strip()
    if not stats_source:
        maybe_norm_stats_path = manifest_payload.get("norm_stats_path")
        if isinstance(maybe_norm_stats_path, str) and maybe_norm_stats_path.strip():
            stats_source = maybe_norm_stats_path.strip()
    if not stats_source:
        candidate = dataset_root / "norm_stats.json"
        if candidate.exists():
            stats_source = str(candidate)
    if not stats_source:
        raise ContractError(
            "Could not resolve normalization.stats_source. "
            "Provide --normalization-stats-source (recommended: <dataset-root>/norm_stats.json)."
        )

    return {
        "normalization_name": norm_name.strip(),
        "stats_source": stats_source,
        "clip_percentiles": clip,
        "scaling_range": scale,
    }


def _resolve_dataset_feature_mode(manifest_payload: dict[str, Any]) -> str | None:
    feature_mode = manifest_payload.get("feature_mode")
    if isinstance(feature_mode, str) and feature_mode.strip():
        return feature_mode.strip()
    config_block = manifest_payload.get("config")
    if isinstance(config_block, dict):
        candidate = config_block.get("feature_mode")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    resolved_contract = manifest_payload.get("resolved_contract")
    if isinstance(resolved_contract, dict):
        features = resolved_contract.get("features")
        if isinstance(features, dict):
            candidate = features.get("feature_mode")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _resolve_manifest_patch_size(manifest_payload: dict[str, Any]) -> int | None:
    patch_size = manifest_payload.get("patch_size")
    if isinstance(patch_size, bool):
        return None
    if isinstance(patch_size, int) and patch_size > 0:
        return patch_size
    return None


config_path = Path(_env("NT_CONFIG_PATH"))
dataset_root = Path(_env("NT_DATASET_ROOT"))
train_split_dir = Path(_env("NT_TRAIN_SPLIT_DIR"))
val_split_dir = Path(_env("NT_VAL_SPLIT_DIR"))
dataset_source_manifest_path = Path(_env("NT_DATASET_SOURCE_MANIFEST"))
run_dir = Path(_env("NT_RUN_DIR"))
run_id = _env("NT_RUN_ID")
expected_patch_size = _optional_patch_size(_env("NT_EXPECTED_PATCH_SIZE"))

if run_id == "":
    run_id = run_dir.name
if run_id == "":
    raise ContractError("run_id must be non-empty (set --run-id or --run-dir).")

cfg = _load_train_config(config_path)

try:
    manifest_payload = json.loads(dataset_source_manifest_path.read_text(encoding="utf-8"))
except OSError as exc:
    raise ContractError(
        f"Failed to read dataset source manifest at {dataset_source_manifest_path}: {exc}"
    ) from exc
except json.JSONDecodeError as exc:
    raise ContractError(
        f"Dataset source manifest is invalid JSON at {dataset_source_manifest_path}: {exc}"
    ) from exc
if not isinstance(manifest_payload, dict):
    raise ContractError("dataset source manifest must contain a top-level JSON object.")

dataset_source_run_id = _resolve_dataset_source_run_id(dataset_source_manifest_path, manifest_payload)
dataset_feature_mode = _resolve_dataset_feature_mode(manifest_payload)
if dataset_feature_mode is not None and dataset_feature_mode != cfg.feature_mode:
    raise ContractError(
        "Selected dataset feature_mode is inconsistent with config.feature_mode: "
        f"{dataset_feature_mode!r} != {cfg.feature_mode!r}."
    )
manifest_patch_size = _resolve_manifest_patch_size(manifest_payload)
if (
    expected_patch_size is not None
    and manifest_patch_size is not None
    and expected_patch_size != manifest_patch_size
):
    raise ContractError(
        "Requested --patch-size is inconsistent with split manifest patch_size: "
        f"{expected_patch_size} != {manifest_patch_size}."
    )

dataset_contract = validate_train_ready_dataset_contract(
    train_split_dir=train_split_dir,
    val_split_dir=val_split_dir,
    feature_mode=cfg.feature_mode,
    expected_patch_size=expected_patch_size or manifest_patch_size,
)
resolved_patch_size = int(dataset_contract.patch_size)
if manifest_patch_size is not None and resolved_patch_size != manifest_patch_size:
    raise ContractError(
        "Resolved dataset patch_size from files is inconsistent with split manifest patch_size: "
        f"{resolved_patch_size} != {manifest_patch_size}."
    )

normalization = _resolve_normalization(manifest_payload=manifest_payload, dataset_root=dataset_root)

print("[INFO] Resolved dataset feature_mode=", dataset_feature_mode or cfg.feature_mode, sep="")
print("[INFO] Resolved dataset patch_size=", resolved_patch_size, sep="")
print("[INFO] Resolved train samples=", dataset_contract.train_sample_count, sep="")
print("[INFO] Resolved val samples=", dataset_contract.val_sample_count, sep="")

result = run_train_baseline(
    config=cfg,
    train_split_dir=train_split_dir,
    val_split_dir=val_split_dir,
    run_dir=run_dir,
    run_id=run_id,
    dataset_source_run_id=dataset_source_run_id,
    dataset_source_manifest_path=dataset_source_manifest_path,
    dataset_root=dataset_root,
    dataset_patch_size=resolved_patch_size,
    dataset_feature_mode=dataset_feature_mode or cfg.feature_mode,
    normalization=normalization,
    epochs_override=_optional_int(_env("NT_EPOCHS_OVERRIDE"), name="--epochs-override"),
    device_override=_optional_str(_env("NT_DEVICE_OVERRIDE")),
    monitored_metric_name=_optional_str(_env("NT_MONITORED_METRIC_NAME")),
    monitored_metric_mode=_optional_str(_env("NT_MONITORED_METRIC_MODE")),
)

print("[RESULT] run_id=", result.run_id, sep="")
print("[RESULT] run_dir=", result.run_dir, sep="")
print("[RESULT] checkpoint=", result.checkpoint_path, sep="")
print("[RESULT] checkpoint_metadata=", result.checkpoint_metadata_path, sep="")
print("[RESULT] train_manifest=", result.train_manifest_path, sep="")
print("[RESULT] summary=", result.summary_path, sep="")
print("[RESULT] config_used=", result.config_used_path, sep="")
print("[RESULT] history=", result.history_path, sep="")
print("[RESULT] dataset_root=", result.dataset_root, sep="")
print("[RESULT] dataset_patch_size=", result.dataset_patch_size, sep="")
print("[RESULT] best_metric_name=", result.best_metric_name, sep="")
print("[RESULT] best_metric_value=", result.best_metric_value, sep="")
print("[RESULT] best_epoch=", result.best_epoch, sep="")
PY

echo "[DONE] module_net_train run completed successfully."
