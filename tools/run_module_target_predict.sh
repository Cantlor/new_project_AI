#!/usr/bin/env bash
set -euo pipefail

# Canonical runner for module_target_predict (single-scene, checkpoint-driven).
# Quick-edit block: set defaults here for local operator workflows.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi
CHECKPOINT_PATH=""
CHECKPOINT_METADATA_PATH=""
TRAIN_MANIFEST_PATH=""
CONFIG_USED_PATH=""
NORMALIZATION_STATS_PATH=""
INPUT_RASTER_PATH=""
AOI_PATH=""
OUTPUT_DIR="runs/module_target_predict"
RUN_ID="target-predict-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR=""
DEVICE_OVERRIDE=""
TILE_SIZE_OVERRIDE=""
OVERLAP_OVERRIDE=""
PROGRESS_OVERRIDE=""
DRY_RUN=false

print_help() {
  cat <<'EOF'
Usage:
  bash tools/run_module_target_predict.sh --checkpoint <path> --input-raster <path> [options]

Required:
  --checkpoint <path>               Trained checkpoint (.ckpt)
  --input-raster <path>             Input 8-band GeoTIFF for predict

Optional artifact overrides:
  --checkpoint-metadata <path>      checkpoint_metadata.json (default: sibling of checkpoint)
  --train-manifest <path>           train_manifest.json (default: sibling auto-detect)
  --config-used <path>              config_used.yaml (default: sibling auto-detect)
  --normalization-stats <path>      Explicit normalization stats JSON override

Runtime:
  --output-dir <path>               Output root (default: runs/module_target_predict)
  --run-id <id>                     Run id (default: target-predict-<UTC timestamp>)
  --run-dir <path>                  Full run dir override (priority over output-dir/run-id)
  --device-override <cpu|cuda|mps>  Device override
  --tile-size <int>                 Tile size override
  --overlap <float>                 Overlap fraction in [0,1)
  --progress-enabled                Force progress bars on
  --no-progress                     Force progress bars off

Other:
  --aoi <path>                      Currently unsupported in executable baseline path
  --python-cmd <cmd>                Python executable (default: .venv/bin/python if present, else python3)
  --dry-run                         Print resolved inputs and exit
  -h, --help                        Show this help

Notes:
  - Runner uses Python entrypoint: ai_fields.module_target_predict.predict_run.run_predict_for_scene
  - Predict remains raster-only: no postprocess/vectorization/eval in this runner.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT_PATH="${2:-}"; shift 2 ;;
    --checkpoint-metadata)
      CHECKPOINT_METADATA_PATH="${2:-}"; shift 2 ;;
    --train-manifest)
      TRAIN_MANIFEST_PATH="${2:-}"; shift 2 ;;
    --config-used)
      CONFIG_USED_PATH="${2:-}"; shift 2 ;;
    --normalization-stats)
      NORMALIZATION_STATS_PATH="${2:-}"; shift 2 ;;
    --input-raster)
      INPUT_RASTER_PATH="${2:-}"; shift 2 ;;
    --aoi)
      AOI_PATH="${2:-}"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"; shift 2 ;;
    --run-id)
      RUN_ID="${2:-}"; shift 2 ;;
    --run-dir)
      RUN_DIR="${2:-}"; shift 2 ;;
    --device-override)
      DEVICE_OVERRIDE="${2:-}"; shift 2 ;;
    --tile-size)
      TILE_SIZE_OVERRIDE="${2:-}"; shift 2 ;;
    --overlap)
      OVERLAP_OVERRIDE="${2:-}"; shift 2 ;;
    --progress-enabled)
      PROGRESS_OVERRIDE="true"; shift ;;
    --no-progress)
      PROGRESS_OVERRIDE="false"; shift ;;
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

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] Missing checkpoint path. Provide --checkpoint or set CHECKPOINT_PATH at the top of this script." >&2
  exit 2
fi
if [[ -z "${INPUT_RASTER_PATH}" ]]; then
  echo "[ERROR] Missing input raster path. Provide --input-raster or set INPUT_RASTER_PATH at the top of this script." >&2
  exit 2
fi
if [[ -n "${AOI_PATH}" ]]; then
  echo "[ERROR] AOI inference mode is not implemented in the current executable module_target_predict path. Provide full raster input without --aoi." >&2
  exit 2
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${OUTPUT_DIR%/}/${RUN_ID}"
fi

if [[ -z "${CHECKPOINT_METADATA_PATH}" ]]; then
  checkpoint_dir="$(dirname "${CHECKPOINT_PATH}")"
  checkpoint_metadata_candidate="${checkpoint_dir}/checkpoint_metadata.json"
  if [[ -f "${checkpoint_metadata_candidate}" ]]; then
    CHECKPOINT_METADATA_PATH="${checkpoint_metadata_candidate}"
  else
    echo "[ERROR] Could not resolve checkpoint metadata automatically. Expected sibling file: ${checkpoint_metadata_candidate}. Use --checkpoint-metadata explicitly." >&2
    exit 2
  fi
fi
if [[ -z "${TRAIN_MANIFEST_PATH}" ]]; then
  train_manifest_candidate="$(dirname "${CHECKPOINT_PATH}")/train_manifest.json"
  if [[ -f "${train_manifest_candidate}" ]]; then
    TRAIN_MANIFEST_PATH="${train_manifest_candidate}"
  fi
fi
if [[ -z "${CONFIG_USED_PATH}" ]]; then
  config_used_candidate="$(dirname "${CHECKPOINT_PATH}")/config_used.yaml"
  if [[ -f "${config_used_candidate}" ]]; then
    CONFIG_USED_PATH="${config_used_candidate}"
  fi
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT_METADATA_PATH}" ]]; then
  echo "[ERROR] Checkpoint metadata does not exist: ${CHECKPOINT_METADATA_PATH}" >&2
  exit 2
fi
if [[ ! -f "${INPUT_RASTER_PATH}" ]]; then
  echo "[ERROR] Input raster does not exist: ${INPUT_RASTER_PATH}" >&2
  exit 2
fi
if [[ -n "${TRAIN_MANIFEST_PATH}" && ! -f "${TRAIN_MANIFEST_PATH}" ]]; then
  echo "[ERROR] train_manifest path does not exist: ${TRAIN_MANIFEST_PATH}" >&2
  exit 2
fi
if [[ -n "${CONFIG_USED_PATH}" && ! -f "${CONFIG_USED_PATH}" ]]; then
  echo "[ERROR] config_used path does not exist: ${CONFIG_USED_PATH}" >&2
  exit 2
fi
if [[ -n "${NORMALIZATION_STATS_PATH}" && ! -f "${NORMALIZATION_STATS_PATH}" ]]; then
  echo "[ERROR] normalization stats path does not exist: ${NORMALIZATION_STATS_PATH}" >&2
  exit 2
fi

echo "[INFO] module_target_predict single-scene runner"
echo "[INFO] Entry point: ai_fields.module_target_predict.predict_run.run_predict_for_scene"
echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"
echo "[INFO] Checkpoint metadata: ${CHECKPOINT_METADATA_PATH}"
if [[ -n "${TRAIN_MANIFEST_PATH}" ]]; then
  echo "[INFO] Train manifest: ${TRAIN_MANIFEST_PATH}"
else
  echo "[INFO] Train manifest: <not provided>"
fi
if [[ -n "${CONFIG_USED_PATH}" ]]; then
  echo "[INFO] Config used: ${CONFIG_USED_PATH}"
else
  echo "[INFO] Config used: <not provided>"
fi
if [[ -n "${NORMALIZATION_STATS_PATH}" ]]; then
  echo "[INFO] Normalization stats override: ${NORMALIZATION_STATS_PATH}"
else
  echo "[INFO] Normalization stats override: <checkpoint-driven resolution>"
fi
echo "[INFO] Input raster: ${INPUT_RASTER_PATH}"
echo "[INFO] Output root: ${OUTPUT_DIR}"
echo "[INFO] Run id: ${RUN_ID}"
echo "[INFO] Run dir: ${RUN_DIR}"
if [[ -n "${DEVICE_OVERRIDE}" ]]; then
  echo "[INFO] Device override: ${DEVICE_OVERRIDE}"
else
  echo "[INFO] Device override: <auto>"
fi
if [[ -n "${TILE_SIZE_OVERRIDE}" ]]; then
  echo "[INFO] Tile size override: ${TILE_SIZE_OVERRIDE}"
else
  echo "[INFO] Tile size override: <default from predict entrypoint: 512>"
fi
if [[ -n "${OVERLAP_OVERRIDE}" ]]; then
  echo "[INFO] Overlap override: ${OVERLAP_OVERRIDE}"
else
  echo "[INFO] Overlap override: <default from predict entrypoint: 0.25>"
fi
if [[ -n "${PROGRESS_OVERRIDE}" ]]; then
  echo "[INFO] Progress override: ${PROGRESS_OVERRIDE}"
else
  echo "[INFO] Progress override: <auto (TTY/env policy)>"
fi

if [[ "${DRY_RUN}" == true ]]; then
  echo "[DRY-RUN] Predict run will not start."
  echo "[DRY-RUN] Python command: ${PYTHON_CMD}"
  exit 0
fi

echo "[RUN] predict_scene"
PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" \
TP_CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
TP_CHECKPOINT_METADATA_PATH="${CHECKPOINT_METADATA_PATH}" \
TP_TRAIN_MANIFEST_PATH="${TRAIN_MANIFEST_PATH}" \
TP_CONFIG_USED_PATH="${CONFIG_USED_PATH}" \
TP_NORMALIZATION_STATS_PATH="${NORMALIZATION_STATS_PATH}" \
TP_INPUT_RASTER_PATH="${INPUT_RASTER_PATH}" \
TP_RUN_DIR="${RUN_DIR}" \
TP_RUN_ID="${RUN_ID}" \
TP_DEVICE_OVERRIDE="${DEVICE_OVERRIDE}" \
TP_TILE_SIZE_OVERRIDE="${TILE_SIZE_OVERRIDE}" \
TP_OVERLAP_OVERRIDE="${OVERLAP_OVERRIDE}" \
TP_PROGRESS_OVERRIDE="${PROGRESS_OVERRIDE}" \
"${PYTHON_CMD}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

from ai_fields.common.errors import ContractError
from ai_fields.module_target_predict.predict_run import run_predict_for_scene


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


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


def _optional_float(raw: str, *, name: str) -> float | None:
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ContractError(f"{name} must be a float.") from exc
    return value


def _optional_bool(raw: str, *, name: str) -> bool | None:
    if raw == "":
        return None
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ContractError(f"{name} must be one of true/false/1/0/yes/no/on/off.")


def _optional_path(raw: str) -> str | None:
    return raw if raw != "" else None


tile_size = _optional_int(_env("TP_TILE_SIZE_OVERRIDE"), name="--tile-size")
overlap = _optional_float(_env("TP_OVERLAP_OVERRIDE"), name="--overlap")
if overlap is not None and not (0.0 <= overlap < 1.0):
    raise ContractError("--overlap must be in [0, 1).")
progress_enabled = _optional_bool(_env("TP_PROGRESS_OVERRIDE"), name="progress override")

kwargs = {
    "raster_path": Path(_env("TP_INPUT_RASTER_PATH")),
    "checkpoint_path": Path(_env("TP_CHECKPOINT_PATH")),
    "checkpoint_metadata_path": Path(_env("TP_CHECKPOINT_METADATA_PATH")),
    "output_dir": Path(_env("TP_RUN_DIR")),
    "device": _optional_path(_env("TP_DEVICE_OVERRIDE")),
    "tile_size": tile_size if tile_size is not None else 512,
    "overlap": overlap if overlap is not None else 0.25,
    "normalization_stats_path": _optional_path(_env("TP_NORMALIZATION_STATS_PATH")),
    "run_id": _env("TP_RUN_ID"),
    "train_manifest_path": _optional_path(_env("TP_TRAIN_MANIFEST_PATH")),
    "config_used_path": _optional_path(_env("TP_CONFIG_USED_PATH")),
    "progress_enabled": progress_enabled,
}

run_info = run_predict_for_scene(**kwargs)
print("[DONE] predict_scene")
print("[DONE] output_dir:", run_info["output_dir"])
print("[DONE] manifest_path:", run_info["manifest_path"])
print("[DONE] summary_path:", run_info["summary_path"])
print("[DONE] config_path:", run_info["config_path"])
print("[DONE] output_paths:", json.dumps(run_info["output_paths"], ensure_ascii=False))
PY
