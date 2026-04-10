#!/usr/bin/env bash
set -euo pipefail

# Canonical runner for module_postprocess_vectorize (single-scene path).
# Quick-edit block: set defaults here for local operator workflows.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi

PREDICT_RUN_DIR=""
EXTENT_PROB_PATH=""
BOUNDARY_PROB_PATH=""
DISTANCE_PRED_PATH=""
VALID_PATH=""
SOURCE_PREDICT_MANIFEST_PATH=""
SOURCE_PREDICT_RUN_ID=""
AOI_PATH=""
OUTPUT_DIR="runs/module_postprocess_vectorize"
RUN_ID="postprocess-vectorize-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR_OVERRIDE=""
LAYER_NAME="parcels"

# Baseline policy defaults (aligned with existing Stage A->E unit-chain fixtures).
MARKER_EXTENT_CORE_MIN_PROB="0.7"
MARKER_BOUNDARY_LOW_MAX_PROB="0.4"
MARKER_DISTANCE_HIGH_MIN_VALUE="1.0"
THRESHOLD_PROVENANCE="validation_calibrated_baseline_v1"

WATERSHED_EXTENT_SUPPORT_MIN_PROB="0.5"
WATERSHED_BOUNDARY_WEIGHT="2.0"
WATERSHED_EXTENT_WEIGHT="1.0"
WATERSHED_DISTANCE_WEIGHT="0.5"
WATERSHED_BOUNDARY_BARRIER_MAX_PROB="0.95"
WATERSHED_MIN_REGION_PIXELS="0"

POLYGON_MIN_AREA_M2="0.0"
POLYGON_NUM_WORKERS="1"

AOI_SUPPRESSION_ENABLED="false"
BOUNDARY_REPAIR_ENABLED="false"
BOUNDARY_REPAIR_CLOSING_RADIUS="2"
PROGRESS_OVERRIDE=""
DRY_RUN=false

print_help() {
  cat <<'EOF_HELP'
Usage:
  bash tools/run_module_postprocess_vectorize.sh [input mode] [options]

Input mode A (recommended):
  --predict-run-dir <path>         Resolve from one module_target_predict run dir:
                                   extent_prob.tif
                                   boundary_prob.tif
                                   distance_pred.tif
                                   valid.tif
                                   predict_manifest.json (optional auto-link)

Input mode B (explicit raster paths):
  --extent-prob <path>
  --boundary-prob <path>
  --distance-pred <path>
  --valid <path>

Optional provenance:
  --source-predict-manifest <path> Explicit predict_manifest.json
  --source-predict-run-id <id>     Explicit predict run id for manifest provenance

Run/output:
  --output-dir <path>              Output root (default: runs/module_postprocess_vectorize)
  --run-id <id>                    Run id (default: postprocess-vectorize-<UTC timestamp>)
  --run-dir <path>                 Full run dir override (has priority over output-dir/run-id)
  --layer-name <name>              Output GPKG layer name (default: parcels)

Optional runtime controls:
  --aoi <path>                     AOI vector path
  --aoi-suppression-enabled        Enable AOI suppression into effective_valid.tif
  --no-aoi-suppression             Disable AOI suppression (default)
  --boundary-repair-enabled        Enable Stage B.5 morphological boundary repair
  --no-boundary-repair             Disable Stage B.5 boundary repair (default)
  --boundary-repair-closing-radius <int>
  --progress-enabled               Force progress bars on
  --no-progress                    Force progress bars off

Optional baseline policy overrides (keep defaults unless calibrated):
  --marker-extent-core-min-prob <float>
  --marker-boundary-low-max-prob <float>
  --marker-distance-high-min-value <float>
  --watershed-extent-support-min-prob <float>
  --watershed-boundary-weight <float>
  --watershed-extent-weight <float>
  --watershed-distance-weight <float>
  --watershed-boundary-barrier-max-prob <float>
  --watershed-min-region-pixels <int>
  --polygon-min-area-m2 <float>
  --workers <int>
  --num-workers <int>
  --threshold-provenance <string>

Other:
  --python-cmd <cmd>               Python executable (default: .venv/bin/python if present, else python3)
  --dry-run                        Print resolved plan and exit
  -h, --help                       Show this help

Notes:
  - Runner uses Python entrypoint:
      ai_fields.module_postprocess_vectorize.run_postprocess.run_postprocess_for_scene
  - This runner performs postprocess/vectorization only.
    It does not run eval or training logic.
EOF_HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predict-run-dir)
      PREDICT_RUN_DIR="${2:-}"; shift 2 ;;
    --extent-prob)
      EXTENT_PROB_PATH="${2:-}"; shift 2 ;;
    --boundary-prob)
      BOUNDARY_PROB_PATH="${2:-}"; shift 2 ;;
    --distance-pred)
      DISTANCE_PRED_PATH="${2:-}"; shift 2 ;;
    --valid)
      VALID_PATH="${2:-}"; shift 2 ;;
    --source-predict-manifest)
      SOURCE_PREDICT_MANIFEST_PATH="${2:-}"; shift 2 ;;
    --source-predict-run-id)
      SOURCE_PREDICT_RUN_ID="${2:-}"; shift 2 ;;
    --aoi)
      AOI_PATH="${2:-}"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"; shift 2 ;;
    --run-id)
      RUN_ID="${2:-}"; shift 2 ;;
    --run-dir)
      RUN_DIR_OVERRIDE="${2:-}"; shift 2 ;;
    --layer-name)
      LAYER_NAME="${2:-}"; shift 2 ;;

    --aoi-suppression-enabled)
      AOI_SUPPRESSION_ENABLED="true"; shift ;;
    --no-aoi-suppression)
      AOI_SUPPRESSION_ENABLED="false"; shift ;;
    --boundary-repair-enabled)
      BOUNDARY_REPAIR_ENABLED="true"; shift ;;
    --no-boundary-repair)
      BOUNDARY_REPAIR_ENABLED="false"; shift ;;
    --boundary-repair-closing-radius)
      BOUNDARY_REPAIR_CLOSING_RADIUS="${2:-}"; shift 2 ;;
    --progress-enabled)
      PROGRESS_OVERRIDE="true"; shift ;;
    --no-progress)
      PROGRESS_OVERRIDE="false"; shift ;;

    --marker-extent-core-min-prob)
      MARKER_EXTENT_CORE_MIN_PROB="${2:-}"; shift 2 ;;
    --marker-boundary-low-max-prob)
      MARKER_BOUNDARY_LOW_MAX_PROB="${2:-}"; shift 2 ;;
    --marker-distance-high-min-value)
      MARKER_DISTANCE_HIGH_MIN_VALUE="${2:-}"; shift 2 ;;
    --watershed-extent-support-min-prob)
      WATERSHED_EXTENT_SUPPORT_MIN_PROB="${2:-}"; shift 2 ;;
    --watershed-boundary-weight)
      WATERSHED_BOUNDARY_WEIGHT="${2:-}"; shift 2 ;;
    --watershed-extent-weight)
      WATERSHED_EXTENT_WEIGHT="${2:-}"; shift 2 ;;
    --watershed-distance-weight)
      WATERSHED_DISTANCE_WEIGHT="${2:-}"; shift 2 ;;
    --watershed-boundary-barrier-max-prob)
      WATERSHED_BOUNDARY_BARRIER_MAX_PROB="${2:-}"; shift 2 ;;
    --watershed-min-region-pixels)
      WATERSHED_MIN_REGION_PIXELS="${2:-}"; shift 2 ;;
    --polygon-min-area-m2)
      POLYGON_MIN_AREA_M2="${2:-}"; shift 2 ;;
    --workers|--num-workers)
      POLYGON_NUM_WORKERS="${2:-}"; shift 2 ;;
    --threshold-provenance)
      THRESHOLD_PROVENANCE="${2:-}"; shift 2 ;;

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

if [[ -n "${RUN_DIR_OVERRIDE}" ]]; then
  OUTPUT_DIR="$(dirname "${RUN_DIR_OVERRIDE}")"
  RUN_ID="$(basename "${RUN_DIR_OVERRIDE}")"
fi

# Input mode consistency checks.
if [[ -n "${PREDICT_RUN_DIR}" ]]; then
  if [[ -n "${EXTENT_PROB_PATH}" || -n "${BOUNDARY_PROB_PATH}" || -n "${DISTANCE_PRED_PATH}" || -n "${VALID_PATH}" ]]; then
    echo "[ERROR] Ambiguous input mode: use either --predict-run-dir or explicit raster paths, not both." >&2
    exit 2
  fi

  if [[ ! -d "${PREDICT_RUN_DIR}" ]]; then
    echo "[ERROR] predict run dir does not exist: ${PREDICT_RUN_DIR}" >&2
    exit 2
  fi

  EXTENT_PROB_PATH="${PREDICT_RUN_DIR%/}/extent_prob.tif"
  BOUNDARY_PROB_PATH="${PREDICT_RUN_DIR%/}/boundary_prob.tif"
  DISTANCE_PRED_PATH="${PREDICT_RUN_DIR%/}/distance_pred.tif"
  VALID_PATH="${PREDICT_RUN_DIR%/}/valid.tif"

  if [[ -z "${SOURCE_PREDICT_MANIFEST_PATH}" ]]; then
    candidate_manifest="${PREDICT_RUN_DIR%/}/predict_manifest.json"
    if [[ -f "${candidate_manifest}" ]]; then
      SOURCE_PREDICT_MANIFEST_PATH="${candidate_manifest}"
    fi
  fi
  if [[ -z "${SOURCE_PREDICT_RUN_ID}" && -z "${SOURCE_PREDICT_MANIFEST_PATH}" ]]; then
    SOURCE_PREDICT_RUN_ID="$(basename "${PREDICT_RUN_DIR%/}")"
  fi
else
  if [[ -z "${EXTENT_PROB_PATH}" || -z "${BOUNDARY_PROB_PATH}" || -z "${DISTANCE_PRED_PATH}" || -z "${VALID_PATH}" ]]; then
    echo "[ERROR] Missing explicit raster inputs. Provide all of: --extent-prob, --boundary-prob, --distance-pred, --valid (or use --predict-run-dir)." >&2
    exit 2
  fi
fi

if [[ ! -f "${EXTENT_PROB_PATH}" ]]; then
  echo "[ERROR] Missing required raster input: ${EXTENT_PROB_PATH}" >&2
  exit 2
fi
if [[ ! -f "${BOUNDARY_PROB_PATH}" ]]; then
  echo "[ERROR] Missing required raster input: ${BOUNDARY_PROB_PATH}" >&2
  exit 2
fi
if [[ ! -f "${DISTANCE_PRED_PATH}" ]]; then
  echo "[ERROR] Missing required raster input: ${DISTANCE_PRED_PATH}" >&2
  exit 2
fi
if [[ ! -f "${VALID_PATH}" ]]; then
  echo "[ERROR] Missing required raster input: ${VALID_PATH}" >&2
  exit 2
fi

if [[ -n "${SOURCE_PREDICT_MANIFEST_PATH}" && ! -f "${SOURCE_PREDICT_MANIFEST_PATH}" ]]; then
  echo "[ERROR] source predict manifest does not exist: ${SOURCE_PREDICT_MANIFEST_PATH}" >&2
  exit 2
fi
if [[ -n "${AOI_PATH}" && ! -f "${AOI_PATH}" ]]; then
  echo "[ERROR] AOI path does not exist: ${AOI_PATH}" >&2
  exit 2
fi
if [[ -z "${RUN_ID}" ]]; then
  echo "[ERROR] run id must be non-empty." >&2
  exit 2
fi
if [[ -z "${LAYER_NAME}" ]]; then
  echo "[ERROR] layer name must be non-empty." >&2
  exit 2
fi
if [[ -z "${THRESHOLD_PROVENANCE}" ]]; then
  echo "[ERROR] threshold provenance must be non-empty." >&2
  exit 2
fi

if [[ -n "${PREDICT_RUN_DIR}" ]]; then
  echo "[INFO] Input mode: predict-run-dir"
  echo "[INFO] Predict run dir: ${PREDICT_RUN_DIR}"
else
  echo "[INFO] Input mode: explicit raster paths"
fi

echo "[INFO] module_postprocess_vectorize single-scene runner"
echo "[INFO] Entry point: ai_fields.module_postprocess_vectorize.run_postprocess.run_postprocess_for_scene"
echo "[INFO] extent_prob: ${EXTENT_PROB_PATH}"
echo "[INFO] boundary_prob: ${BOUNDARY_PROB_PATH}"
echo "[INFO] distance_pred: ${DISTANCE_PRED_PATH}"
echo "[INFO] valid: ${VALID_PATH}"
if [[ -n "${AOI_PATH}" ]]; then
  echo "[INFO] AOI path: ${AOI_PATH}"
else
  echo "[INFO] AOI path: <not provided>"
fi
if [[ -n "${SOURCE_PREDICT_MANIFEST_PATH}" ]]; then
  echo "[INFO] Source predict manifest: ${SOURCE_PREDICT_MANIFEST_PATH}"
else
  echo "[INFO] Source predict manifest: <not provided>"
fi
if [[ -n "${SOURCE_PREDICT_RUN_ID}" ]]; then
  echo "[INFO] Source predict run id: ${SOURCE_PREDICT_RUN_ID}"
else
  echo "[INFO] Source predict run id: <auto-from-manifest-if-provided>"
fi
echo "[INFO] Output root: ${OUTPUT_DIR}"
echo "[INFO] Run id: ${RUN_ID}"
echo "[INFO] Run dir: ${OUTPUT_DIR%/}/${RUN_ID}"
echo "[INFO] GPKG layer name: ${LAYER_NAME}"
echo "[INFO] AOI suppression enabled: ${AOI_SUPPRESSION_ENABLED}"
echo "[INFO] Boundary repair enabled: ${BOUNDARY_REPAIR_ENABLED}"
echo "[INFO] Boundary repair closing radius: ${BOUNDARY_REPAIR_CLOSING_RADIUS}"
if [[ -n "${PROGRESS_OVERRIDE}" ]]; then
  echo "[INFO] Progress override: ${PROGRESS_OVERRIDE}"
else
  echo "[INFO] Progress override: <auto (TTY/env policy)>"
fi

echo "[INFO] Policy (marker): extent_core_min_prob=${MARKER_EXTENT_CORE_MIN_PROB}, boundary_low_max_prob=${MARKER_BOUNDARY_LOW_MAX_PROB}, distance_high_min_value=${MARKER_DISTANCE_HIGH_MIN_VALUE}"
echo "[INFO] Policy (watershed): extent_support_min_prob=${WATERSHED_EXTENT_SUPPORT_MIN_PROB}, boundary_weight=${WATERSHED_BOUNDARY_WEIGHT}, extent_weight=${WATERSHED_EXTENT_WEIGHT}, distance_weight=${WATERSHED_DISTANCE_WEIGHT}, boundary_barrier_max_prob=${WATERSHED_BOUNDARY_BARRIER_MAX_PROB}, min_region_pixels=${WATERSHED_MIN_REGION_PIXELS}"
echo "[INFO] Policy (polygonization): min_polygon_area_m2=${POLYGON_MIN_AREA_M2}, num_workers=${POLYGON_NUM_WORKERS}"
echo "[INFO] Threshold provenance: ${THRESHOLD_PROVENANCE}"

if [[ "${DRY_RUN}" == true ]]; then
  echo "[DRY-RUN] Postprocess run will not start."
  echo "[DRY-RUN] Python command: ${PYTHON_CMD}"
  exit 0
fi

echo "[RUN] postprocess_scene"
PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" \
PP_EXTENT_PROB_PATH="${EXTENT_PROB_PATH}" \
PP_BOUNDARY_PROB_PATH="${BOUNDARY_PROB_PATH}" \
PP_DISTANCE_PRED_PATH="${DISTANCE_PRED_PATH}" \
PP_VALID_PATH="${VALID_PATH}" \
PP_AOI_PATH="${AOI_PATH}" \
PP_SOURCE_PREDICT_MANIFEST_PATH="${SOURCE_PREDICT_MANIFEST_PATH}" \
PP_SOURCE_PREDICT_RUN_ID="${SOURCE_PREDICT_RUN_ID}" \
PP_OUTPUT_DIR="${OUTPUT_DIR}" \
PP_RUN_ID="${RUN_ID}" \
PP_LAYER_NAME="${LAYER_NAME}" \
PP_MARKER_EXTENT_CORE_MIN_PROB="${MARKER_EXTENT_CORE_MIN_PROB}" \
PP_MARKER_BOUNDARY_LOW_MAX_PROB="${MARKER_BOUNDARY_LOW_MAX_PROB}" \
PP_MARKER_DISTANCE_HIGH_MIN_VALUE="${MARKER_DISTANCE_HIGH_MIN_VALUE}" \
PP_THRESHOLD_PROVENANCE="${THRESHOLD_PROVENANCE}" \
PP_WATERSHED_EXTENT_SUPPORT_MIN_PROB="${WATERSHED_EXTENT_SUPPORT_MIN_PROB}" \
PP_WATERSHED_BOUNDARY_WEIGHT="${WATERSHED_BOUNDARY_WEIGHT}" \
PP_WATERSHED_EXTENT_WEIGHT="${WATERSHED_EXTENT_WEIGHT}" \
PP_WATERSHED_DISTANCE_WEIGHT="${WATERSHED_DISTANCE_WEIGHT}" \
PP_WATERSHED_BOUNDARY_BARRIER_MAX_PROB="${WATERSHED_BOUNDARY_BARRIER_MAX_PROB}" \
PP_WATERSHED_MIN_REGION_PIXELS="${WATERSHED_MIN_REGION_PIXELS}" \
PP_POLYGON_MIN_AREA_M2="${POLYGON_MIN_AREA_M2}" \
PP_POLYGON_NUM_WORKERS="${POLYGON_NUM_WORKERS}" \
PP_AOI_SUPPRESSION_ENABLED="${AOI_SUPPRESSION_ENABLED}" \
PP_BOUNDARY_REPAIR_ENABLED="${BOUNDARY_REPAIR_ENABLED}" \
PP_BOUNDARY_REPAIR_CLOSING_RADIUS="${BOUNDARY_REPAIR_CLOSING_RADIUS}" \
PP_PROGRESS_OVERRIDE="${PROGRESS_OVERRIDE}" \
"${PYTHON_CMD}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

from ai_fields.common.errors import ContractError
from ai_fields.module_postprocess_vectorize.boundary_repair import BoundaryRepairPolicy
from ai_fields.module_postprocess_vectorize.instance_core import WatershedCorePolicy
from ai_fields.module_postprocess_vectorize.marker_generation import MarkerThresholdPolicy
from ai_fields.module_postprocess_vectorize.polygonization import PolygonizationPolicy
from ai_fields.module_postprocess_vectorize.run_postprocess import (
    PostprocessRunPolicies,
    run_postprocess_for_scene,
)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _optional_str(raw: str) -> str | None:
    return raw if raw != "" else None


def _parse_float(raw: str, *, name: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ContractError(f"{name} must be a float.") from exc
    return value


def _parse_int(raw: str, *, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ContractError(f"{name} must be an integer.") from exc
    return value


def _parse_bool(raw: str, *, name: str) -> bool:
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ContractError(f"{name} must be one of true/false/1/0/yes/no/on/off.")


def _optional_bool(raw: str, *, name: str) -> bool | None:
    if raw == "":
        return None
    return _parse_bool(raw, name=name)


marker_policy = MarkerThresholdPolicy(
    extent_core_min_prob=_parse_float(
        _env("PP_MARKER_EXTENT_CORE_MIN_PROB"),
        name="marker extent_core_min_prob",
    ),
    boundary_low_max_prob=_parse_float(
        _env("PP_MARKER_BOUNDARY_LOW_MAX_PROB"),
        name="marker boundary_low_max_prob",
    ),
    distance_high_min_value=_parse_float(
        _env("PP_MARKER_DISTANCE_HIGH_MIN_VALUE"),
        name="marker distance_high_min_value",
    ),
    threshold_provenance=_env("PP_THRESHOLD_PROVENANCE"),
)

watershed_policy = WatershedCorePolicy(
    extent_support_min_prob=_parse_float(
        _env("PP_WATERSHED_EXTENT_SUPPORT_MIN_PROB"),
        name="watershed extent_support_min_prob",
    ),
    threshold_provenance=_env("PP_THRESHOLD_PROVENANCE"),
    boundary_weight=_parse_float(
        _env("PP_WATERSHED_BOUNDARY_WEIGHT"),
        name="watershed boundary_weight",
    ),
    extent_weight=_parse_float(
        _env("PP_WATERSHED_EXTENT_WEIGHT"),
        name="watershed extent_weight",
    ),
    distance_weight=_parse_float(
        _env("PP_WATERSHED_DISTANCE_WEIGHT"),
        name="watershed distance_weight",
    ),
    boundary_barrier_max_prob=_parse_float(
        _env("PP_WATERSHED_BOUNDARY_BARRIER_MAX_PROB"),
        name="watershed boundary_barrier_max_prob",
    ),
    min_region_pixels=_parse_int(
        _env("PP_WATERSHED_MIN_REGION_PIXELS"),
        name="watershed min_region_pixels",
    ),
)

polygonization_policy = PolygonizationPolicy(
    threshold_provenance=_env("PP_THRESHOLD_PROVENANCE"),
    min_polygon_area_m2=_parse_float(
        _env("PP_POLYGON_MIN_AREA_M2"),
        name="polygon min_polygon_area_m2",
    ),
    num_workers=_parse_int(
        _env("PP_POLYGON_NUM_WORKERS"),
        name="polygon num_workers",
    ),
)

aoi_suppression_enabled = _parse_bool(
    _env("PP_AOI_SUPPRESSION_ENABLED"),
    name="aoi suppression enabled",
)
boundary_repair_enabled = _parse_bool(
    _env("PP_BOUNDARY_REPAIR_ENABLED"),
    name="boundary repair enabled",
)

boundary_repair_policy = None
if boundary_repair_enabled:
    boundary_repair_policy = BoundaryRepairPolicy(
        enabled=True,
        closing_radius=_parse_int(
            _env("PP_BOUNDARY_REPAIR_CLOSING_RADIUS"),
            name="boundary repair closing_radius",
        ),
    )

policies = PostprocessRunPolicies(
    marker_policy=marker_policy,
    watershed_policy=watershed_policy,
    polygonization_policy=polygonization_policy,
    aoi_suppression_enabled=aoi_suppression_enabled,
    boundary_repair_policy=boundary_repair_policy,
)

result = run_postprocess_for_scene(
    extent_prob_path=Path(_env("PP_EXTENT_PROB_PATH")),
    boundary_prob_path=Path(_env("PP_BOUNDARY_PROB_PATH")),
    distance_pred_path=Path(_env("PP_DISTANCE_PRED_PATH")),
    valid_path=Path(_env("PP_VALID_PATH")),
    output_dir=Path(_env("PP_OUTPUT_DIR")),
    run_id=_env("PP_RUN_ID"),
    policies=policies,
    aoi_path=_optional_str(_env("PP_AOI_PATH")),
    source_predict_manifest_path=_optional_str(_env("PP_SOURCE_PREDICT_MANIFEST_PATH")),
    source_predict_run_id=_optional_str(_env("PP_SOURCE_PREDICT_RUN_ID")),
    layer_name=_env("PP_LAYER_NAME"),
    progress_enabled=_optional_bool(_env("PP_PROGRESS_OVERRIDE"), name="progress override"),
)

print("[DONE] postprocess_scene")
print(
    "[DONE] result:",
    json.dumps(
        {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "parcel_instance_path": str(result.parcel_instance_path),
            "parcels_gpkg_path": str(result.parcels_gpkg_path),
            "postprocess_manifest_path": str(result.postprocess_manifest_path),
            "summary_path": str(result.summary_path),
            "config_used_path": str(result.config_used_path),
            "instance_count": int(result.instance_count),
            "polygon_count": int(result.polygon_count),
            "success": bool(result.success),
        },
        ensure_ascii=False,
    ),
)
PY
