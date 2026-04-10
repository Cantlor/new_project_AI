#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENTS=(A1 A2 A3 A4)

# Source-of-truth baseline references from module_eval_baseline_and_experiment_plan.md
BASELINE_PREP_RUN_DIR="runs/module_prep_data/prep-raw8-256-ps256"
BASELINE_PREDICT_RUN_DIR="runs/module_target_predict/target-predict-20260408T133539Z"
BASELINE_POSTPROCESS_RUN_ID="postprocess-vectorize-20260409T092257Z"
BASELINE_EVAL_RUN_ID="eval-20260410T070540Z"

POSTPROCESS_OUTPUT_ROOT="runs/module_postprocess_vectorize"
EVAL_OUTPUT_ROOT="runs/module_eval"
LOG_ROOT="runs/experiment_logs/sprint_a"
CONFIG_ROOT="configs/module_postprocess_vectorize/experiments/sprint_a"

RUN_ONE=""
RUN_ALL=false
BATCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DRY_RUN=false

usage() {
  cat <<'EOF_USAGE'
Usage:
  bash tools/run_postprocess_sprint_a.sh --experiment <A1|A2|A3|A4> [--batch-id <id>] [--dry-run]
  bash tools/run_postprocess_sprint_a.sh --all [--batch-id <id>] [--dry-run]

Behavior:
  - Uses canonical runners only:
      tools/run_module_postprocess_vectorize.sh
      tools/run_module_eval.sh
  - Uses fixed baseline predict/prep inputs for all experiments.
  - Writes per-experiment logs under:
      runs/experiment_logs/sprint_a/<batch-id>/
  - Uses predictable run IDs:
      postprocess-sprint-a-<batch-id>-<exp-lower>
      eval-sprint-a-<batch-id>-<exp-lower>

Notes:
  - A0 baseline is reused (not rerun):
      postprocess-vectorize-20260409T092257Z
      eval-20260410T070540Z
  - A5 is intentionally deferred.
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      RUN_ONE="${2:-}"
      shift 2
      ;;
    --all)
      RUN_ALL=true
      shift
      ;;
    --batch-id)
      BATCH_ID="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${RUN_ALL}" == true && -n "${RUN_ONE}" ]]; then
  echo "[ERROR] Use either --all or --experiment, not both." >&2
  exit 2
fi
if [[ "${RUN_ALL}" == false && -z "${RUN_ONE}" ]]; then
  echo "[ERROR] Specify --all or --experiment." >&2
  exit 2
fi
if [[ -z "${BATCH_ID}" ]]; then
  echo "[ERROR] batch-id must be non-empty." >&2
  exit 2
fi

if [[ -n "${RUN_ONE}" ]]; then
  valid=false
  for e in "${EXPERIMENTS[@]}"; do
    if [[ "${RUN_ONE}" == "${e}" ]]; then
      valid=true
      break
    fi
  done
  if [[ "${valid}" != true ]]; then
    echo "[ERROR] Unsupported experiment '${RUN_ONE}'. Allowed: ${EXPERIMENTS[*]}" >&2
    exit 2
  fi
fi

mkdir -p "${LOG_ROOT}/${BATCH_ID}"
INDEX_TSV="${LOG_ROOT}/${BATCH_ID}/run_index.tsv"
if [[ ! -f "${INDEX_TSV}" ]]; then
  {
    echo -e "experiment\tpostprocess_run_id\teval_run_id\tpostprocess_log\teval_log"
  } > "${INDEX_TSV}"
fi

reset_baseline_knobs() {
  MARKER_EXTENT_CORE_MIN_PROB="0.7"
  MARKER_BOUNDARY_LOW_MAX_PROB="0.4"
  MARKER_DISTANCE_HIGH_MIN_VALUE="1.0"

  WATERSHED_EXTENT_SUPPORT_MIN_PROB="0.5"
  WATERSHED_BOUNDARY_WEIGHT="2.0"
  WATERSHED_EXTENT_WEIGHT="1.0"
  WATERSHED_DISTANCE_WEIGHT="0.5"
  WATERSHED_BOUNDARY_BARRIER_MAX_PROB="0.95"
  WATERSHED_MIN_REGION_PIXELS="0"

  POLYGON_MIN_AREA_M2="0.0"
  NUM_WORKERS="4"

  BOUNDARY_REPAIR_ENABLED="false"
  BOUNDARY_REPAIR_CLOSING_RADIUS="2"

  THRESHOLD_PROVENANCE="validation_calibrated_baseline_v1"
  EXPERIMENT_DESCRIPTION=""
}

run_experiment() {
  local exp_id="$1"
  local exp_lower="${exp_id,,}"
  local cfg="${CONFIG_ROOT}/${exp_id}.env"

  if [[ ! -f "${cfg}" ]]; then
    echo "[ERROR] Missing experiment config: ${cfg}" >&2
    exit 2
  fi

  reset_baseline_knobs
  # shellcheck disable=SC1090
  source "${cfg}"

  local pp_run_id="postprocess-sprint-a-${BATCH_ID}-${exp_lower}"
  local eval_run_id="eval-sprint-a-${BATCH_ID}-${exp_lower}"

  local pp_log="${LOG_ROOT}/${BATCH_ID}/${exp_id}_postprocess.log"
  local eval_log="${LOG_ROOT}/${BATCH_ID}/${exp_id}_eval.log"

  echo "[INFO] ${exp_id}: ${EXPERIMENT_DESCRIPTION}"
  echo "[INFO] ${exp_id}: postprocess_run_id=${pp_run_id}"
  echo "[INFO] ${exp_id}: eval_run_id=${eval_run_id}"
  echo "[INFO] ${exp_id}: postprocess_log=${pp_log}"
  echo "[INFO] ${exp_id}: eval_log=${eval_log}"

  {
    echo "[INFO] experiment=${exp_id}"
    echo "[INFO] batch_id=${BATCH_ID}"
    echo "[INFO] postprocess_run_id=${pp_run_id}"
    echo "[INFO] eval_run_id=${eval_run_id}"
    echo "[INFO] baseline_predict_run_dir=${BASELINE_PREDICT_RUN_DIR}"
    echo "[INFO] baseline_prep_run_dir=${BASELINE_PREP_RUN_DIR}"
    echo "[INFO] threshold_provenance=${THRESHOLD_PROVENANCE}"
    echo "[INFO] marker_extent_core_min_prob=${MARKER_EXTENT_CORE_MIN_PROB}"
    echo "[INFO] marker_boundary_low_max_prob=${MARKER_BOUNDARY_LOW_MAX_PROB}"
    echo "[INFO] marker_distance_high_min_value=${MARKER_DISTANCE_HIGH_MIN_VALUE}"
    echo "[INFO] watershed_extent_support_min_prob=${WATERSHED_EXTENT_SUPPORT_MIN_PROB}"
    echo "[INFO] watershed_boundary_weight=${WATERSHED_BOUNDARY_WEIGHT}"
    echo "[INFO] watershed_extent_weight=${WATERSHED_EXTENT_WEIGHT}"
    echo "[INFO] watershed_distance_weight=${WATERSHED_DISTANCE_WEIGHT}"
    echo "[INFO] watershed_boundary_barrier_max_prob=${WATERSHED_BOUNDARY_BARRIER_MAX_PROB}"
    echo "[INFO] watershed_min_region_pixels=${WATERSHED_MIN_REGION_PIXELS}"
    echo "[INFO] polygon_min_area_m2=${POLYGON_MIN_AREA_M2}"
    echo "[INFO] num_workers=${NUM_WORKERS}"
    echo "[INFO] boundary_repair_enabled=${BOUNDARY_REPAIR_ENABLED}"
    echo "[INFO] boundary_repair_closing_radius=${BOUNDARY_REPAIR_CLOSING_RADIUS}"
  } > "${pp_log}"

  local -a pp_cmd=(
    bash tools/run_module_postprocess_vectorize.sh
    --predict-run-dir "${BASELINE_PREDICT_RUN_DIR}"
    --output-dir "${POSTPROCESS_OUTPUT_ROOT}"
    --run-id "${pp_run_id}"
    --marker-extent-core-min-prob "${MARKER_EXTENT_CORE_MIN_PROB}"
    --marker-boundary-low-max-prob "${MARKER_BOUNDARY_LOW_MAX_PROB}"
    --marker-distance-high-min-value "${MARKER_DISTANCE_HIGH_MIN_VALUE}"
    --watershed-extent-support-min-prob "${WATERSHED_EXTENT_SUPPORT_MIN_PROB}"
    --watershed-boundary-weight "${WATERSHED_BOUNDARY_WEIGHT}"
    --watershed-extent-weight "${WATERSHED_EXTENT_WEIGHT}"
    --watershed-distance-weight "${WATERSHED_DISTANCE_WEIGHT}"
    --watershed-boundary-barrier-max-prob "${WATERSHED_BOUNDARY_BARRIER_MAX_PROB}"
    --watershed-min-region-pixels "${WATERSHED_MIN_REGION_PIXELS}"
    --polygon-min-area-m2 "${POLYGON_MIN_AREA_M2}"
    --num-workers "${NUM_WORKERS}"
    --threshold-provenance "${THRESHOLD_PROVENANCE}"
    --no-progress
  )
  if [[ "${BOUNDARY_REPAIR_ENABLED}" == "true" ]]; then
    pp_cmd+=(--boundary-repair-enabled --boundary-repair-closing-radius "${BOUNDARY_REPAIR_CLOSING_RADIUS}")
  else
    pp_cmd+=(--no-boundary-repair)
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    pp_cmd+=(--dry-run)
  fi

  printf '[CMD] %q ' "${pp_cmd[@]}" >> "${pp_log}"
  echo >> "${pp_log}"
  "${pp_cmd[@]}" >> "${pp_log}" 2>&1

  {
    echo "[INFO] experiment=${exp_id}"
    echo "[INFO] batch_id=${BATCH_ID}"
    echo "[INFO] postprocess_run_id=${pp_run_id}"
    echo "[INFO] eval_run_id=${eval_run_id}"
  } > "${eval_log}"

  local eval_postprocess_dir="${POSTPROCESS_OUTPUT_ROOT}/${pp_run_id}"
  if [[ "${DRY_RUN}" == true ]]; then
    # In dry-run mode postprocess output is not written, so validate eval runner
    # argument plumbing against existing baseline postprocess artifacts.
    eval_postprocess_dir="${POSTPROCESS_OUTPUT_ROOT}/${BASELINE_POSTPROCESS_RUN_ID}"
  fi
  echo "[INFO] postprocess_run_dir_for_eval=${eval_postprocess_dir}" >> "${eval_log}"

  local -a eval_cmd=(
    bash tools/run_module_eval.sh
    --prep-run-dir "${BASELINE_PREP_RUN_DIR}"
    --predict-run-dir "${BASELINE_PREDICT_RUN_DIR}"
    --postprocess-run-dir "${eval_postprocess_dir}"
    --output-dir "${EVAL_OUTPUT_ROOT}"
    --run-id "${eval_run_id}"
    --no-progress
  )
  if [[ "${DRY_RUN}" == true ]]; then
    eval_cmd+=(--dry-run)
  fi

  printf '[CMD] %q ' "${eval_cmd[@]}" >> "${eval_log}"
  echo >> "${eval_log}"
  "${eval_cmd[@]}" >> "${eval_log}" 2>&1

  echo -e "${exp_id}\t${pp_run_id}\t${eval_run_id}\t${pp_log}\t${eval_log}" >> "${INDEX_TSV}"

  echo "[DONE] ${exp_id} completed"
}

echo "[INFO] Baseline A0 reused (not rerun):"
echo "[INFO]   predict_run_dir=${BASELINE_PREDICT_RUN_DIR}"
echo "[INFO]   postprocess_run_id=${BASELINE_POSTPROCESS_RUN_ID}"
echo "[INFO]   eval_run_id=${BASELINE_EVAL_RUN_ID}"
echo "[INFO] Batch ID: ${BATCH_ID}"
echo "[INFO] Dry-run mode: ${DRY_RUN}"
echo "[INFO] Log root: ${LOG_ROOT}/${BATCH_ID}"

declare -a target_experiments=()
if [[ "${RUN_ALL}" == true ]]; then
  target_experiments=("${EXPERIMENTS[@]}")
else
  target_experiments=("${RUN_ONE}")
fi

for exp in "${target_experiments[@]}"; do
  run_experiment "${exp}"
done

echo "[DONE] Sprint A runner finished for: ${target_experiments[*]}"
echo "[DONE] run index: ${INDEX_TSV}"
