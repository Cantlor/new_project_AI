#!/usr/bin/env bash
set -euo pipefail

# Canonical runner for module_eval (single-scene eval path).
# Quick-edit block: set defaults here for local operator workflows.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi

PREP_RUN_DIR=""
PREDICT_RUN_DIR=""
POSTPROCESS_RUN_DIR=""

GT_EXTENT_PATH=""
GT_BOUNDARY_PATH=""
GT_VALID_PATH=""
GT_DISTANCE_PATH=""
GT_PARCELS_PATH=""

PRED_EXTENT_PROB_PATH=""
PRED_BOUNDARY_PROB_PATH=""
PRED_DISTANCE_PRED_PATH=""
PRED_VALID_PATH=""
PREDICT_MANIFEST_PATH=""

POST_PARCEL_INSTANCE_PATH=""
POST_PARCELS_GPKG_PATH=""
POSTPROCESS_MANIFEST_PATH=""

OUTPUT_DIR="runs/module_eval"
RUN_ID="eval-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR_OVERRIDE=""
EVAL_MODE="end_to_end_single_scene"

PIXEL_EXTENT_THRESHOLD="0.5"
PIXEL_PREDICTION_RULE="gte"
PIXEL_POSITIVE_GT_LABEL="1"
PIXEL_IGNORE_GT_LABEL="255"
PIXEL_THRESHOLD_PROVENANCE="eval_pixel_threshold_v1"

BOUNDARY_PREDICTION_INTERPRETATION="argmax_non_background"
BOUNDARY_GT_INTERPRETATION="non_background"
BOUNDARY_THRESHOLD_PROVENANCE="eval_boundary_policy_v1"
BOUNDARY_NON_BACKGROUND_THRESHOLD="0.5"
BOUNDARY_BDE_ENABLED="true"
BOUNDARY_EMPTY_HANDLING="explicit_error"

OBJECT_THRESHOLD_PROVENANCE="eval_object_matching_v1"
OBJECT_MIN_IOU_THRESHOLD="0.1"
OBJECT_MIN_OVERLAP_GT_THRESHOLD="0.1"
OBJECT_MIN_OVERLAP_PRED_THRESHOLD="0.1"
OBJECT_MATCH_RULE="iou_or_overlap"
OBJECT_EMPTY_HANDLING="explicit_error"

DISTANCE_THRESHOLD_PROVENANCE="eval_distance_policy_v1"
DISTANCE_ABSENT_GT_POLICY="skip"

BUCKET_ENABLED="false"
BUCKET_THRESHOLD_PROVENANCE="eval_bucket_policy_v1"
BUCKET_USE_PROJECTED_AREA="true"
BUCKET_SMALL_MAX_PIXELS="200"
BUCKET_MEDIUM_MAX_PIXELS="2000"
BUCKET_SMALL_MAX_M2="5000"
BUCKET_MEDIUM_MAX_M2="50000"

PROGRESS_OVERRIDE=""
DRY_RUN=false

print_help() {
  cat <<'EOF_HELP'
Usage:
  bash tools/run_module_eval.sh [input mode] [options]

Input mode A (recommended):
  --prep-run-dir <path>            Resolve GT-side artifacts from module_prep_data run dir
  --predict-run-dir <path>         Resolve prediction rasters from module_target_predict run dir
  --postprocess-run-dir <path>     Resolve vectorized outputs from module_postprocess_vectorize run dir

Input mode B (explicit paths):
  --gt-extent <path>
  --gt-boundary <path>
  --gt-valid <path>
  --gt-distance <path>             Optional
  --gt-parcels <path>
  --pred-extent-prob <path>
  --pred-boundary-prob <path>
  --pred-distance-pred <path>
  --pred-valid <path>
  --post-parcel-instance <path>    Optional
  --post-parcels-gpkg <path>
  --predict-manifest <path>        Optional
  --postprocess-manifest <path>    Optional

Run/output:
  --output-dir <path>              Output root (default: runs/module_eval)
  --run-id <id>                    Run id (default: eval-<UTC timestamp>)
  --run-dir <path>                 Full run dir override (priority over output-dir/run-id)
  --eval-mode <name>               Eval mode label stored in artifacts

Policy overrides (optional):
  --extent-threshold <float>
  --pixel-threshold-provenance <string>
  --boundary-threshold-provenance <string>
  --object-threshold-provenance <string>
  --distance-threshold-provenance <string>
  --bde-enabled | --no-bde
  --bucket-enabled | --no-bucket

Progress:
  --progress-enabled               Force progress bars on
  --no-progress                    Force progress bars off

Other:
  --python-cmd <cmd>               Python executable (default: .venv/bin/python if present, else python3)
  --dry-run                        Print resolved plan and exit
  -h, --help                       Show this help

Notes:
  - Runner uses Python entrypoint: ai_fields.module_eval.run_eval.run_eval
  - Current executable path is single-scene run-level orchestration (Stage A->E).
  - Comparison/leaderboard modes are intentionally out of scope for this runner.
EOF_HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep-run-dir)
      PREP_RUN_DIR="${2:-}"; shift 2 ;;
    --predict-run-dir)
      PREDICT_RUN_DIR="${2:-}"; shift 2 ;;
    --postprocess-run-dir)
      POSTPROCESS_RUN_DIR="${2:-}"; shift 2 ;;

    --gt-extent)
      GT_EXTENT_PATH="${2:-}"; shift 2 ;;
    --gt-boundary)
      GT_BOUNDARY_PATH="${2:-}"; shift 2 ;;
    --gt-valid)
      GT_VALID_PATH="${2:-}"; shift 2 ;;
    --gt-distance)
      GT_DISTANCE_PATH="${2:-}"; shift 2 ;;
    --gt-parcels)
      GT_PARCELS_PATH="${2:-}"; shift 2 ;;

    --pred-extent-prob)
      PRED_EXTENT_PROB_PATH="${2:-}"; shift 2 ;;
    --pred-boundary-prob)
      PRED_BOUNDARY_PROB_PATH="${2:-}"; shift 2 ;;
    --pred-distance-pred)
      PRED_DISTANCE_PRED_PATH="${2:-}"; shift 2 ;;
    --pred-valid)
      PRED_VALID_PATH="${2:-}"; shift 2 ;;
    --predict-manifest)
      PREDICT_MANIFEST_PATH="${2:-}"; shift 2 ;;

    --post-parcel-instance)
      POST_PARCEL_INSTANCE_PATH="${2:-}"; shift 2 ;;
    --post-parcels-gpkg)
      POST_PARCELS_GPKG_PATH="${2:-}"; shift 2 ;;
    --postprocess-manifest)
      POSTPROCESS_MANIFEST_PATH="${2:-}"; shift 2 ;;

    --output-dir)
      OUTPUT_DIR="${2:-}"; shift 2 ;;
    --run-id)
      RUN_ID="${2:-}"; shift 2 ;;
    --run-dir)
      RUN_DIR_OVERRIDE="${2:-}"; shift 2 ;;
    --eval-mode)
      EVAL_MODE="${2:-}"; shift 2 ;;

    --extent-threshold)
      PIXEL_EXTENT_THRESHOLD="${2:-}"; shift 2 ;;
    --pixel-threshold-provenance)
      PIXEL_THRESHOLD_PROVENANCE="${2:-}"; shift 2 ;;
    --boundary-threshold-provenance)
      BOUNDARY_THRESHOLD_PROVENANCE="${2:-}"; shift 2 ;;
    --object-threshold-provenance)
      OBJECT_THRESHOLD_PROVENANCE="${2:-}"; shift 2 ;;
    --distance-threshold-provenance)
      DISTANCE_THRESHOLD_PROVENANCE="${2:-}"; shift 2 ;;
    --bde-enabled)
      BOUNDARY_BDE_ENABLED="true"; shift ;;
    --no-bde)
      BOUNDARY_BDE_ENABLED="false"; shift ;;
    --bucket-enabled)
      BUCKET_ENABLED="true"; shift ;;
    --no-bucket)
      BUCKET_ENABLED="false"; shift ;;

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

if [[ -n "${RUN_DIR_OVERRIDE}" ]]; then
  OUTPUT_DIR="$(dirname "${RUN_DIR_OVERRIDE}")"
  RUN_ID="$(basename "${RUN_DIR_OVERRIDE}")"
  RUN_DIR="${RUN_DIR_OVERRIDE}"
else
  RUN_DIR="${OUTPUT_DIR%/}/${RUN_ID}"
fi

if [[ -n "${PREP_RUN_DIR}" ]]; then
  if [[ -n "${GT_EXTENT_PATH}" || -n "${GT_BOUNDARY_PATH}" || -n "${GT_VALID_PATH}" || -n "${GT_DISTANCE_PATH}" || -n "${GT_PARCELS_PATH}" ]]; then
    echo "[ERROR] Ambiguous GT input mode: use either --prep-run-dir or explicit --gt-* paths." >&2
    exit 2
  fi
  if [[ ! -d "${PREP_RUN_DIR}" ]]; then
    echo "[ERROR] prep run dir does not exist: ${PREP_RUN_DIR}" >&2
    exit 2
  fi
  GT_EXTENT_PATH="${PREP_RUN_DIR%/}/04_prepare_targets/extent.tif"
  GT_BOUNDARY_PATH="${PREP_RUN_DIR%/}/04_prepare_targets/boundary.tif"
  GT_VALID_PATH="${PREP_RUN_DIR%/}/03_prepare_features/valid.tif"

  candidate_gt_distance="${PREP_RUN_DIR%/}/04_prepare_targets/distance.tif"
  if [[ -f "${candidate_gt_distance}" ]]; then
    GT_DISTANCE_PATH="${candidate_gt_distance}"
  fi

  candidate_gt_parcels="${PREP_RUN_DIR%/}/02_prepare_spatial_context/vector_in_raster_crs.gpkg"
  if [[ -f "${candidate_gt_parcels}" ]]; then
    GT_PARCELS_PATH="${candidate_gt_parcels}"
  fi

  PREP_TARGETS_MANIFEST_PATH="${PREP_RUN_DIR%/}/04_prepare_targets/targets_manifest.json"
  PREP_SPLIT_MANIFEST_PATH="${PREP_RUN_DIR%/}/06_split_dataset/split_manifest.json"
  PREP_VALIDATE_MANIFEST_PATH="${PREP_RUN_DIR%/}/07_validate_outputs/validate_outputs_manifest.json"
else
  PREP_TARGETS_MANIFEST_PATH=""
  PREP_SPLIT_MANIFEST_PATH=""
  PREP_VALIDATE_MANIFEST_PATH=""
fi

if [[ -n "${PREDICT_RUN_DIR}" ]]; then
  if [[ -n "${PRED_EXTENT_PROB_PATH}" || -n "${PRED_BOUNDARY_PROB_PATH}" || -n "${PRED_DISTANCE_PRED_PATH}" || -n "${PRED_VALID_PATH}" ]]; then
    echo "[ERROR] Ambiguous predict input mode: use either --predict-run-dir or explicit --pred-* paths." >&2
    exit 2
  fi
  if [[ ! -d "${PREDICT_RUN_DIR}" ]]; then
    echo "[ERROR] predict run dir does not exist: ${PREDICT_RUN_DIR}" >&2
    exit 2
  fi
  PRED_EXTENT_PROB_PATH="${PREDICT_RUN_DIR%/}/extent_prob.tif"
  PRED_BOUNDARY_PROB_PATH="${PREDICT_RUN_DIR%/}/boundary_prob.tif"
  PRED_DISTANCE_PRED_PATH="${PREDICT_RUN_DIR%/}/distance_pred.tif"
  PRED_VALID_PATH="${PREDICT_RUN_DIR%/}/valid.tif"

  if [[ -z "${PREDICT_MANIFEST_PATH}" ]]; then
    candidate_predict_manifest="${PREDICT_RUN_DIR%/}/predict_manifest.json"
    if [[ -f "${candidate_predict_manifest}" ]]; then
      PREDICT_MANIFEST_PATH="${candidate_predict_manifest}"
    fi
  fi
fi

if [[ -n "${POSTPROCESS_RUN_DIR}" ]]; then
  if [[ -n "${POST_PARCEL_INSTANCE_PATH}" || -n "${POST_PARCELS_GPKG_PATH}" ]]; then
    echo "[ERROR] Ambiguous postprocess input mode: use either --postprocess-run-dir or explicit --post-* paths." >&2
    exit 2
  fi
  if [[ ! -d "${POSTPROCESS_RUN_DIR}" ]]; then
    echo "[ERROR] postprocess run dir does not exist: ${POSTPROCESS_RUN_DIR}" >&2
    exit 2
  fi

  candidate_post_instance="${POSTPROCESS_RUN_DIR%/}/parcel_instance.tif"
  if [[ -f "${candidate_post_instance}" ]]; then
    POST_PARCEL_INSTANCE_PATH="${candidate_post_instance}"
  fi
  POST_PARCELS_GPKG_PATH="${POSTPROCESS_RUN_DIR%/}/parcels.gpkg"

  if [[ -z "${POSTPROCESS_MANIFEST_PATH}" ]]; then
    candidate_post_manifest="${POSTPROCESS_RUN_DIR%/}/postprocess_manifest.json"
    if [[ -f "${candidate_post_manifest}" ]]; then
      POSTPROCESS_MANIFEST_PATH="${candidate_post_manifest}"
    fi
  fi
fi

# Required contract inputs for current executable run_eval path.
if [[ -z "${GT_EXTENT_PATH}" || -z "${GT_BOUNDARY_PATH}" || -z "${GT_VALID_PATH}" ]]; then
  echo "[ERROR] Missing GT rasters. Provide --prep-run-dir or explicit --gt-extent/--gt-boundary/--gt-valid." >&2
  exit 2
fi
if [[ -z "${GT_PARCELS_PATH}" ]]; then
  echo "[ERROR] Missing GT parcels vector for object metrics. Provide --gt-parcels or use --prep-run-dir with stage-02 vector artifact." >&2
  exit 2
fi
if [[ -z "${PRED_EXTENT_PROB_PATH}" || -z "${PRED_BOUNDARY_PROB_PATH}" || -z "${PRED_DISTANCE_PRED_PATH}" || -z "${PRED_VALID_PATH}" ]]; then
  echo "[ERROR] Missing prediction rasters. Provide --predict-run-dir or explicit --pred-* paths." >&2
  exit 2
fi
if [[ -z "${POST_PARCELS_GPKG_PATH}" ]]; then
  echo "[ERROR] Missing postprocess polygons vector for object metrics. Provide --postprocess-run-dir or --post-parcels-gpkg." >&2
  exit 2
fi

for required_path in \
  "${GT_EXTENT_PATH}" \
  "${GT_BOUNDARY_PATH}" \
  "${GT_VALID_PATH}" \
  "${GT_PARCELS_PATH}" \
  "${PRED_EXTENT_PROB_PATH}" \
  "${PRED_BOUNDARY_PROB_PATH}" \
  "${PRED_DISTANCE_PRED_PATH}" \
  "${PRED_VALID_PATH}" \
  "${POST_PARCELS_GPKG_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "[ERROR] Missing required input artifact: ${required_path}" >&2
    exit 2
  fi
done

if [[ -n "${GT_DISTANCE_PATH}" && ! -f "${GT_DISTANCE_PATH}" ]]; then
  echo "[ERROR] gt_distance path does not exist: ${GT_DISTANCE_PATH}" >&2
  exit 2
fi
if [[ -n "${POST_PARCEL_INSTANCE_PATH}" && ! -f "${POST_PARCEL_INSTANCE_PATH}" ]]; then
  echo "[ERROR] post parcel_instance path does not exist: ${POST_PARCEL_INSTANCE_PATH}" >&2
  exit 2
fi
if [[ -n "${PREDICT_MANIFEST_PATH}" && ! -f "${PREDICT_MANIFEST_PATH}" ]]; then
  echo "[ERROR] predict manifest path does not exist: ${PREDICT_MANIFEST_PATH}" >&2
  exit 2
fi
if [[ -n "${POSTPROCESS_MANIFEST_PATH}" && ! -f "${POSTPROCESS_MANIFEST_PATH}" ]]; then
  echo "[ERROR] postprocess manifest path does not exist: ${POSTPROCESS_MANIFEST_PATH}" >&2
  exit 2
fi

if [[ -z "${RUN_ID}" ]]; then
  echo "[ERROR] run id must be non-empty." >&2
  exit 2
fi
if [[ -z "${EVAL_MODE}" ]]; then
  echo "[ERROR] eval mode must be non-empty." >&2
  exit 2
fi

echo "[INFO] module_eval single-scene runner"
echo "[INFO] Entry point: ai_fields.module_eval.run_eval.run_eval"
echo "[INFO] Eval mode: ${EVAL_MODE}"
if [[ -n "${PREP_RUN_DIR}" ]]; then
  echo "[INFO] Prep run dir: ${PREP_RUN_DIR}"
else
  echo "[INFO] Prep run dir: <not provided>"
fi
if [[ -n "${PREDICT_RUN_DIR}" ]]; then
  echo "[INFO] Predict run dir: ${PREDICT_RUN_DIR}"
else
  echo "[INFO] Predict run dir: <not provided>"
fi
if [[ -n "${POSTPROCESS_RUN_DIR}" ]]; then
  echo "[INFO] Postprocess run dir: ${POSTPROCESS_RUN_DIR}"
else
  echo "[INFO] Postprocess run dir: <not provided>"
fi

echo "[INFO] GT extent: ${GT_EXTENT_PATH}"
echo "[INFO] GT boundary: ${GT_BOUNDARY_PATH}"
echo "[INFO] GT valid: ${GT_VALID_PATH}"
if [[ -n "${GT_DISTANCE_PATH}" ]]; then
  echo "[INFO] GT distance: ${GT_DISTANCE_PATH}"
else
  echo "[INFO] GT distance: <not provided; Stage B.5 may skip>"
fi
echo "[INFO] GT parcels: ${GT_PARCELS_PATH}"

echo "[INFO] Pred extent_prob: ${PRED_EXTENT_PROB_PATH}"
echo "[INFO] Pred boundary_prob: ${PRED_BOUNDARY_PROB_PATH}"
echo "[INFO] Pred distance_pred: ${PRED_DISTANCE_PRED_PATH}"
echo "[INFO] Pred valid: ${PRED_VALID_PATH}"
if [[ -n "${PREDICT_MANIFEST_PATH}" ]]; then
  echo "[INFO] Predict manifest: ${PREDICT_MANIFEST_PATH}"
else
  echo "[INFO] Predict manifest: <not provided>"
fi

if [[ -n "${POST_PARCEL_INSTANCE_PATH}" ]]; then
  echo "[INFO] Post parcel_instance: ${POST_PARCEL_INSTANCE_PATH}"
else
  echo "[INFO] Post parcel_instance: <not provided>"
fi
echo "[INFO] Post parcels.gpkg: ${POST_PARCELS_GPKG_PATH}"
if [[ -n "${POSTPROCESS_MANIFEST_PATH}" ]]; then
  echo "[INFO] Postprocess manifest: ${POSTPROCESS_MANIFEST_PATH}"
else
  echo "[INFO] Postprocess manifest: <not provided>"
fi

echo "[INFO] Output root: ${OUTPUT_DIR}"
echo "[INFO] Run id: ${RUN_ID}"
echo "[INFO] Run dir: ${RUN_DIR}"

echo "[INFO] Pixel policy: threshold=${PIXEL_EXTENT_THRESHOLD}, rule=${PIXEL_PREDICTION_RULE}, provenance=${PIXEL_THRESHOLD_PROVENANCE}"
echo "[INFO] Boundary policy: prediction_interpretation=${BOUNDARY_PREDICTION_INTERPRETATION}, gt_interpretation=${BOUNDARY_GT_INTERPRETATION}, bde_enabled=${BOUNDARY_BDE_ENABLED}, provenance=${BOUNDARY_THRESHOLD_PROVENANCE}"
echo "[INFO] Object policy: match_rule=${OBJECT_MATCH_RULE}, min_iou=${OBJECT_MIN_IOU_THRESHOLD}, min_overlap_gt=${OBJECT_MIN_OVERLAP_GT_THRESHOLD}, min_overlap_pred=${OBJECT_MIN_OVERLAP_PRED_THRESHOLD}, provenance=${OBJECT_THRESHOLD_PROVENANCE}"
echo "[INFO] Distance policy: absent_gt_policy=${DISTANCE_ABSENT_GT_POLICY}, provenance=${DISTANCE_THRESHOLD_PROVENANCE}"
echo "[INFO] Bucketed eval enabled: ${BUCKET_ENABLED}"
if [[ -n "${PROGRESS_OVERRIDE}" ]]; then
  echo "[INFO] Progress override: ${PROGRESS_OVERRIDE}"
else
  echo "[INFO] Progress override: <auto (TTY/env policy)>"
fi

if [[ "${DRY_RUN}" == true ]]; then
  echo "[DRY-RUN] Eval run will not start."
  echo "[DRY-RUN] Python command: ${PYTHON_CMD}"
  exit 0
fi

echo "[RUN] eval_scene"
PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" \
EV_RUN_ID="${RUN_ID}" \
EV_OUTPUT_DIR="${RUN_DIR}" \
EV_EVAL_MODE="${EVAL_MODE}" \
EV_GT_EXTENT_PATH="${GT_EXTENT_PATH}" \
EV_GT_BOUNDARY_PATH="${GT_BOUNDARY_PATH}" \
EV_GT_VALID_PATH="${GT_VALID_PATH}" \
EV_GT_DISTANCE_PATH="${GT_DISTANCE_PATH}" \
EV_GT_PARCELS_PATH="${GT_PARCELS_PATH}" \
EV_PRED_EXTENT_PROB_PATH="${PRED_EXTENT_PROB_PATH}" \
EV_PRED_BOUNDARY_PROB_PATH="${PRED_BOUNDARY_PROB_PATH}" \
EV_PRED_DISTANCE_PRED_PATH="${PRED_DISTANCE_PRED_PATH}" \
EV_PRED_VALID_PATH="${PRED_VALID_PATH}" \
EV_POST_PARCEL_INSTANCE_PATH="${POST_PARCEL_INSTANCE_PATH}" \
EV_POST_PARCELS_GPKG_PATH="${POST_PARCELS_GPKG_PATH}" \
EV_PREDICT_MANIFEST_PATH="${PREDICT_MANIFEST_PATH}" \
EV_POSTPROCESS_MANIFEST_PATH="${POSTPROCESS_MANIFEST_PATH}" \
EV_PREP_TARGETS_MANIFEST_PATH="${PREP_TARGETS_MANIFEST_PATH}" \
EV_PREP_SPLIT_MANIFEST_PATH="${PREP_SPLIT_MANIFEST_PATH}" \
EV_PREP_VALIDATE_MANIFEST_PATH="${PREP_VALIDATE_MANIFEST_PATH}" \
EV_PREP_RUN_DIR="${PREP_RUN_DIR}" \
EV_PREDICT_RUN_DIR="${PREDICT_RUN_DIR}" \
EV_POSTPROCESS_RUN_DIR="${POSTPROCESS_RUN_DIR}" \
EV_PIXEL_EXTENT_THRESHOLD="${PIXEL_EXTENT_THRESHOLD}" \
EV_PIXEL_PREDICTION_RULE="${PIXEL_PREDICTION_RULE}" \
EV_PIXEL_POSITIVE_GT_LABEL="${PIXEL_POSITIVE_GT_LABEL}" \
EV_PIXEL_IGNORE_GT_LABEL="${PIXEL_IGNORE_GT_LABEL}" \
EV_PIXEL_THRESHOLD_PROVENANCE="${PIXEL_THRESHOLD_PROVENANCE}" \
EV_BOUNDARY_PREDICTION_INTERPRETATION="${BOUNDARY_PREDICTION_INTERPRETATION}" \
EV_BOUNDARY_GT_INTERPRETATION="${BOUNDARY_GT_INTERPRETATION}" \
EV_BOUNDARY_THRESHOLD_PROVENANCE="${BOUNDARY_THRESHOLD_PROVENANCE}" \
EV_BOUNDARY_NON_BACKGROUND_THRESHOLD="${BOUNDARY_NON_BACKGROUND_THRESHOLD}" \
EV_BOUNDARY_BDE_ENABLED="${BOUNDARY_BDE_ENABLED}" \
EV_BOUNDARY_EMPTY_HANDLING="${BOUNDARY_EMPTY_HANDLING}" \
EV_OBJECT_THRESHOLD_PROVENANCE="${OBJECT_THRESHOLD_PROVENANCE}" \
EV_OBJECT_MIN_IOU_THRESHOLD="${OBJECT_MIN_IOU_THRESHOLD}" \
EV_OBJECT_MIN_OVERLAP_GT_THRESHOLD="${OBJECT_MIN_OVERLAP_GT_THRESHOLD}" \
EV_OBJECT_MIN_OVERLAP_PRED_THRESHOLD="${OBJECT_MIN_OVERLAP_PRED_THRESHOLD}" \
EV_OBJECT_MATCH_RULE="${OBJECT_MATCH_RULE}" \
EV_OBJECT_EMPTY_HANDLING="${OBJECT_EMPTY_HANDLING}" \
EV_DISTANCE_THRESHOLD_PROVENANCE="${DISTANCE_THRESHOLD_PROVENANCE}" \
EV_DISTANCE_ABSENT_GT_POLICY="${DISTANCE_ABSENT_GT_POLICY}" \
EV_BUCKET_ENABLED="${BUCKET_ENABLED}" \
EV_BUCKET_THRESHOLD_PROVENANCE="${BUCKET_THRESHOLD_PROVENANCE}" \
EV_BUCKET_USE_PROJECTED_AREA="${BUCKET_USE_PROJECTED_AREA}" \
EV_BUCKET_SMALL_MAX_PIXELS="${BUCKET_SMALL_MAX_PIXELS}" \
EV_BUCKET_MEDIUM_MAX_PIXELS="${BUCKET_MEDIUM_MAX_PIXELS}" \
EV_BUCKET_SMALL_MAX_M2="${BUCKET_SMALL_MAX_M2}" \
EV_BUCKET_MEDIUM_MAX_M2="${BUCKET_MEDIUM_MAX_M2}" \
EV_PROGRESS_OVERRIDE="${PROGRESS_OVERRIDE}" \
"${PYTHON_CMD}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError
from ai_fields.module_eval.boundary_metrics import BoundaryEvaluationPolicy
from ai_fields.module_eval.bucketed_eval import BucketSizePolicy
from ai_fields.module_eval.distance_metrics import DistanceEvaluationPolicy
from ai_fields.module_eval.object_metrics import ObjectMatchingPolicy
from ai_fields.module_eval.pixel_metrics import PixelBinarizationPolicy
from ai_fields.module_eval.run_eval import EvalRunInputs, EvalRunPolicies, run_eval


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _optional_path(name: str) -> Path | None:
    raw = _env(name)
    if raw == "":
        return None
    return Path(raw)


def _bool(value: str, *, name: str) -> bool:
    norm = value.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    raise ContractError(f"{name} must be one of true/false/1/0/yes/no.")


def _optional_bool(raw: str, *, name: str) -> bool | None:
    if raw == "":
        return None
    return _bool(raw, name=name)


def _int(raw: str, *, name: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ContractError(f"{name} must be an integer.") from exc


def _float(raw: str, *, name: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ContractError(f"{name} must be a float.") from exc


run_id = _env("EV_RUN_ID")
output_dir = Path(_env("EV_OUTPUT_DIR"))
eval_mode = _env("EV_EVAL_MODE")

inputs = EvalRunInputs(
    gt_extent_path=Path(_env("EV_GT_EXTENT_PATH")),
    gt_boundary_path=Path(_env("EV_GT_BOUNDARY_PATH")),
    gt_valid_path=Path(_env("EV_GT_VALID_PATH")),
    pred_extent_prob_path=Path(_env("EV_PRED_EXTENT_PROB_PATH")),
    pred_boundary_prob_path=Path(_env("EV_PRED_BOUNDARY_PROB_PATH")),
    pred_distance_pred_path=Path(_env("EV_PRED_DISTANCE_PRED_PATH")),
    pred_valid_path=Path(_env("EV_PRED_VALID_PATH")),
    gt_distance_path=_optional_path("EV_GT_DISTANCE_PATH"),
    gt_parcels_path=_optional_path("EV_GT_PARCELS_PATH"),
    post_parcel_instance_path=_optional_path("EV_POST_PARCEL_INSTANCE_PATH"),
    post_parcels_gpkg_path=_optional_path("EV_POST_PARCELS_GPKG_PATH"),
    predict_manifest_path=_optional_path("EV_PREDICT_MANIFEST_PATH"),
    postprocess_manifest_path=_optional_path("EV_POSTPROCESS_MANIFEST_PATH"),
)

pixel_policy = PixelBinarizationPolicy(
    extent_prob_threshold=_float(_env("EV_PIXEL_EXTENT_THRESHOLD"), name="--extent-threshold"),
    threshold_provenance=_env("EV_PIXEL_THRESHOLD_PROVENANCE"),
    positive_gt_label=_int(_env("EV_PIXEL_POSITIVE_GT_LABEL"), name="pixel positive_gt_label"),
    ignore_gt_label=_int(_env("EV_PIXEL_IGNORE_GT_LABEL"), name="pixel ignore_gt_label"),
    prediction_rule=_env("EV_PIXEL_PREDICTION_RULE"),
)

boundary_policy = BoundaryEvaluationPolicy(
    prediction_interpretation=_env("EV_BOUNDARY_PREDICTION_INTERPRETATION"),
    gt_interpretation=_env("EV_BOUNDARY_GT_INTERPRETATION"),
    threshold_provenance=_env("EV_BOUNDARY_THRESHOLD_PROVENANCE"),
    non_background_prob_threshold=_float(
        _env("EV_BOUNDARY_NON_BACKGROUND_THRESHOLD"),
        name="boundary non_background_prob_threshold",
    ),
    bde_enabled=_bool(_env("EV_BOUNDARY_BDE_ENABLED"), name="--bde-enabled/--no-bde"),
    empty_boundary_handling=_env("EV_BOUNDARY_EMPTY_HANDLING"),
)

object_policy = ObjectMatchingPolicy(
    threshold_provenance=_env("EV_OBJECT_THRESHOLD_PROVENANCE"),
    min_iou_threshold=_float(_env("EV_OBJECT_MIN_IOU_THRESHOLD"), name="object min_iou_threshold"),
    min_overlap_gt_threshold=_float(
        _env("EV_OBJECT_MIN_OVERLAP_GT_THRESHOLD"),
        name="object min_overlap_gt_threshold",
    ),
    min_overlap_pred_threshold=_float(
        _env("EV_OBJECT_MIN_OVERLAP_PRED_THRESHOLD"),
        name="object min_overlap_pred_threshold",
    ),
    match_rule=_env("EV_OBJECT_MATCH_RULE"),
    empty_object_handling=_env("EV_OBJECT_EMPTY_HANDLING"),
)

distance_policy = DistanceEvaluationPolicy(
    threshold_provenance=_env("EV_DISTANCE_THRESHOLD_PROVENANCE"),
    absent_gt_policy=_env("EV_DISTANCE_ABSENT_GT_POLICY"),
)

bucket_policy = None
if _bool(_env("EV_BUCKET_ENABLED"), name="--bucket-enabled/--no-bucket"):
    bucket_policy = BucketSizePolicy(
        threshold_provenance=_env("EV_BUCKET_THRESHOLD_PROVENANCE"),
        use_projected_area=_bool(_env("EV_BUCKET_USE_PROJECTED_AREA"), name="bucket use_projected_area"),
        small_max_pixels=_int(_env("EV_BUCKET_SMALL_MAX_PIXELS"), name="bucket small_max_pixels"),
        medium_max_pixels=_int(_env("EV_BUCKET_MEDIUM_MAX_PIXELS"), name="bucket medium_max_pixels"),
        small_max_m2=_float(_env("EV_BUCKET_SMALL_MAX_M2"), name="bucket small_max_m2"),
        medium_max_m2=_float(_env("EV_BUCKET_MEDIUM_MAX_M2"), name="bucket medium_max_m2"),
    )

policies = EvalRunPolicies(
    eval_mode=eval_mode,
    pixel_policy=pixel_policy,
    boundary_policy=boundary_policy,
    object_policy=object_policy,
    distance_policy=distance_policy,
    bucket_policy=bucket_policy,
)

extra_source_manifest_paths: list[Path] = []
for key in (
    "EV_PREP_TARGETS_MANIFEST_PATH",
    "EV_PREP_SPLIT_MANIFEST_PATH",
    "EV_PREP_VALIDATE_MANIFEST_PATH",
):
    p = _optional_path(key)
    if p is not None and p.exists():
        extra_source_manifest_paths.append(p)

extra_effective_config: dict[str, Any] = {
    "runner": {
        "runner_name": "tools/run_module_eval.sh",
        "input_mode": {
            "prep_run_dir": _env("EV_PREP_RUN_DIR") or None,
            "predict_run_dir": _env("EV_PREDICT_RUN_DIR") or None,
            "postprocess_run_dir": _env("EV_POSTPROCESS_RUN_DIR") or None,
        },
    }
}

result = run_eval(
    run_id=run_id,
    inputs=inputs,
    policies=policies,
    output_dir=output_dir,
    extra_source_manifest_paths=extra_source_manifest_paths,
    extra_effective_config=extra_effective_config,
    progress_enabled=_optional_bool(_env("EV_PROGRESS_OVERRIDE"), name="progress override"),
)

payload = {
    "run_id": result.run_id,
    "eval_mode": result.eval_mode,
    "run_dir": str(result.run_dir),
    "eval_manifest_path": str(result.eval_manifest_path),
    "summary_path": str(result.summary_path),
    "config_used_path": str(result.config_used_path),
    "error_taxonomy_path": str(result.error_taxonomy_path),
    "ready_for_next_stage": bool(result.ready_for_next_stage),
}
print("[DONE] eval_scene completed")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

echo "[DONE] module_eval run finished"
