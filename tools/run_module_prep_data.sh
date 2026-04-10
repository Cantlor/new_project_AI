#!/usr/bin/env bash
set -euo pipefail

# Canonical runner for module_prep_data.
# Quick-edit block: change defaults here if you prefer editing the script
# instead of passing CLI arguments each time.
PYTHON_CMD="python3"
CONFIG_PATH=""
RASTER_PATH=""
VECTOR_PATH=""
AOI_PATH=""
OUTPUT_DIR="runs/module_prep_data"
RUN_ID="prep-data-$(date -u +%Y%m%dT%H%M%SZ)"
FROM_STAGE="01_check_inputs"
TO_STAGE="07_validate_outputs"
VALID_PATH=""
PATCHES_DIR=""
DATASET_DIR=""
PATCH_SIZES=""
MULTI_SIZE_EXPORT_ROOT=""

RUNTIME_COMPUTE_ENABLED=true
RUNTIME_PROBE_ENABLED=true
METADATA_SIDECAR_FALLBACK_ENABLED=true
OVERWRITE=false
DRY_RUN=false

readonly STAGES=(
  "01_check_inputs"
  "02_prepare_spatial_context"
  "03_prepare_features"
  "04_prepare_targets"
  "05_make_patches"
  "06_split_dataset"
  "07_validate_outputs"
)

print_help() {
  cat <<'EOF'
Usage:
  bash tools/run_module_prep_data.sh --config <path> --raster <path> --vector <path> [options]

Required:
  --config <path>      Prep config YAML (primary config selection mode)
  --raster <path>      Input 8-band GeoTIFF
  --vector <path>      Input labels vector (polygons/multipolygons)

Optional:
  --aoi <path>         Optional AOI vector
  --output-dir <path>  Output root (default: runs/module_prep_data)
  --run-id <id>        Run id (default: prep-data-<UTC timestamp>)
  --from-stage <name>  Start stage (default: 01_check_inputs)
  --to-stage <name>    Stop stage (default: 07_validate_outputs)
  --valid-path <path>  Optional precomputed valid raster
  --patches-dir <path> Optional stage-06 patches dir
  --dataset-dir <path> Optional stage-07 dataset dir
  --python-cmd <cmd>   Python executable (default: python3)
  --patch-sizes <csv>  Multi-size mode, e.g. 256,384,512
  --multi-size-export-root <path>
                       Multi-size dataset export root (default handled by Python runner)
  --overwrite          Pass --overwrite to run_pipeline (only valid from stage 01)
  --runtime-compute-enabled | --no-runtime-compute-enabled
  --runtime-probe-enabled | --no-runtime-probe-enabled
  --metadata-sidecar-fallback-enabled | --no-metadata-sidecar-fallback-enabled
  --dry-run            Print command and exit without running
  -h, --help           Show this help

Canonical stage names:
  01_check_inputs
  02_prepare_spatial_context
  03_prepare_features
  04_prepare_targets
  05_make_patches
  06_split_dataset
  07_validate_outputs
EOF
}

is_valid_stage() {
  local candidate="$1"
  local stage
  for stage in "${STAGES[@]}"; do
    if [[ "$stage" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"; shift 2 ;;
    --raster)
      RASTER_PATH="${2:-}"; shift 2 ;;
    --vector)
      VECTOR_PATH="${2:-}"; shift 2 ;;
    --aoi)
      AOI_PATH="${2:-}"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"; shift 2 ;;
    --run-id)
      RUN_ID="${2:-}"; shift 2 ;;
    --from-stage)
      FROM_STAGE="${2:-}"; shift 2 ;;
    --to-stage)
      TO_STAGE="${2:-}"; shift 2 ;;
    --valid-path)
      VALID_PATH="${2:-}"; shift 2 ;;
    --patches-dir)
      PATCHES_DIR="${2:-}"; shift 2 ;;
    --dataset-dir)
      DATASET_DIR="${2:-}"; shift 2 ;;
    --python-cmd)
      PYTHON_CMD="${2:-}"; shift 2 ;;
    --patch-sizes)
      PATCH_SIZES="${2:-}"; shift 2 ;;
    --multi-size-export-root)
      MULTI_SIZE_EXPORT_ROOT="${2:-}"; shift 2 ;;
    --overwrite)
      OVERWRITE=true; shift ;;
    --runtime-compute-enabled)
      RUNTIME_COMPUTE_ENABLED=true; shift ;;
    --no-runtime-compute-enabled)
      RUNTIME_COMPUTE_ENABLED=false; shift ;;
    --runtime-probe-enabled)
      RUNTIME_PROBE_ENABLED=true; shift ;;
    --no-runtime-probe-enabled)
      RUNTIME_PROBE_ENABLED=false; shift ;;
    --metadata-sidecar-fallback-enabled)
      METADATA_SIDECAR_FALLBACK_ENABLED=true; shift ;;
    --no-metadata-sidecar-fallback-enabled)
      METADATA_SIDECAR_FALLBACK_ENABLED=false; shift ;;
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
# 1) primary: --config argument
# 2) fallback: CONFIG_PATH variable in quick-edit block
# 3) otherwise: fail with clear error
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Missing config path. Provide --config or set CONFIG_PATH at the top of this script." >&2
  exit 2
fi
if [[ -z "${RASTER_PATH}" ]]; then
  echo "[ERROR] Missing raster path. Provide --raster or set RASTER_PATH at the top of this script." >&2
  exit 2
fi
if [[ -z "${VECTOR_PATH}" ]]; then
  echo "[ERROR] Missing vector path. Provide --vector or set VECTOR_PATH at the top of this script." >&2
  exit 2
fi
if ! is_valid_stage "${FROM_STAGE}"; then
  echo "[ERROR] Invalid --from-stage: ${FROM_STAGE}" >&2
  exit 2
fi
if ! is_valid_stage "${TO_STAGE}"; then
  echo "[ERROR] Invalid --to-stage: ${TO_STAGE}" >&2
  exit 2
fi

if [[ -z "${PATCHES_DIR}" ]]; then
  PATCHES_DIR="${OUTPUT_DIR}/${RUN_ID}/05_make_patches/patches"
fi
if [[ -z "${DATASET_DIR}" ]]; then
  DATASET_DIR="${OUTPUT_DIR}/${RUN_ID}/06_split_dataset/dataset"
fi

cmd=(
  "${PYTHON_CMD}" -m ai_fields.module_prep_data.run_pipeline
  --config "${CONFIG_PATH}"
  --raster "${RASTER_PATH}"
  --vector "${VECTOR_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --start-from-stage "${FROM_STAGE}"
  --stop-after-stage "${TO_STAGE}"
)

if [[ -n "${AOI_PATH}" ]]; then
  cmd+=(--aoi "${AOI_PATH}")
fi
if [[ -n "${VALID_PATH}" ]]; then
  cmd+=(--valid-path "${VALID_PATH}")
fi
if [[ -n "${PATCH_SIZES}" ]]; then
  cmd+=(--patch-sizes "${PATCH_SIZES}")
fi
if [[ -n "${MULTI_SIZE_EXPORT_ROOT}" ]]; then
  cmd+=(--multi-size-export-root "${MULTI_SIZE_EXPORT_ROOT}")
fi
if [[ "${OVERWRITE}" == true ]]; then
  cmd+=(--overwrite)
fi
if [[ "${RUNTIME_COMPUTE_ENABLED}" == true ]]; then
  cmd+=(--runtime-compute-enabled)
else
  cmd+=(--no-runtime-compute-enabled)
fi
if [[ "${RUNTIME_PROBE_ENABLED}" == true ]]; then
  cmd+=(--runtime-probe-enabled)
else
  cmd+=(--no-runtime-probe-enabled)
fi
if [[ "${METADATA_SIDECAR_FALLBACK_ENABLED}" == true ]]; then
  cmd+=(--metadata-sidecar-fallback-enabled)
else
  cmd+=(--no-metadata-sidecar-fallback-enabled)
fi

echo "[INFO] module_prep_data canonical stages:"
printf '  - %s\n' "${STAGES[@]}"
echo "[INFO] Requested range: ${FROM_STAGE} -> ${TO_STAGE}"
echo "[INFO] Config: ${CONFIG_PATH}"
echo "[INFO] Raster: ${RASTER_PATH}"
echo "[INFO] Vector: ${VECTOR_PATH}"
if [[ -n "${AOI_PATH}" ]]; then
  echo "[INFO] AOI: ${AOI_PATH}"
else
  echo "[INFO] AOI: <none>"
fi
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Run id: ${RUN_ID}"
if [[ -z "${PATCH_SIZES}" ]]; then
  cmd+=(--patches-dir "${PATCHES_DIR}")
  cmd+=(--dataset-dir "${DATASET_DIR}")
  echo "[INFO] Patches dir (for stage 06): ${PATCHES_DIR}"
  echo "[INFO] Dataset dir (for stage 07): ${DATASET_DIR}"
else
  echo "[INFO] Multi-size mode patch sizes: ${PATCH_SIZES}"
  if [[ -n "${MULTI_SIZE_EXPORT_ROOT}" ]]; then
    echo "[INFO] Multi-size export root override: ${MULTI_SIZE_EXPORT_ROOT}"
  else
    echo "[INFO] Multi-size export root: <default in run_pipeline>"
  fi
fi
echo "[INFO] runtime_compute_enabled=${RUNTIME_COMPUTE_ENABLED}"
echo "[INFO] runtime_probe_enabled=${RUNTIME_PROBE_ENABLED}"
echo "[INFO] metadata_sidecar_fallback_enabled=${METADATA_SIDECAR_FALLBACK_ENABLED}"
echo "[INFO] Running entrypoint: ${PYTHON_CMD} -m ai_fields.module_prep_data.run_pipeline"
echo "[INFO] Stage-by-stage progress will be printed by run_pipeline ([RUN]/[DONE])."

if [[ "${DRY_RUN}" == true ]]; then
  echo "[DRY-RUN] Command:"
  printf ' %q' "${cmd[@]}"
  echo
  exit 0
fi

"${cmd[@]}"
