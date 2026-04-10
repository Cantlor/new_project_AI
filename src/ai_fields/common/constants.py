# Constants for the AI Fields project.
# All values are sourced from the accepted project documents.
# Do not change these without updating DATA_CONTRACT.md and DECISIONS.md.

# DATA_CONTRACT.md §18.1
DATA_CONTRACT_VERSION = "v1"

# DATA_CONTRACT.md §7.1 — only these two dataset-side feature modes are supported in v1
FEATURE_MODES = ("raw8", "raw8_idx3")

# DATA_CONTRACT.md §7.4 — assembled model input contracts (feature stack + valid channel)
ASSEMBLED_MODEL_INPUTS = ("raw8_valid", "raw8_idx3_valid")

# DATA_CONTRACT.md §7.3 — fixed derived indices for raw8_idx3 mode
DERIVED_INDICES = ("NDVI", "SAVI", "NDWI")

# DATA_CONTRACT.md §7.5 — canonical channel counts
# raw8 = 8 spectral bands
# raw8_idx3 = 8 spectral + 3 derived indices = 11 (dataset-side)
# raw8_valid = raw8 + valid channel = 9 (assembled model input)
# raw8_idx3_valid = raw8_idx3 + valid channel = 12 (assembled model input)
CHANNEL_COUNTS: dict[str, int] = {
    "raw8": 8,
    "raw8_idx3": 11,
    "raw8_valid": 9,
    "raw8_idx3_valid": 12,
}

# DATA_CONTRACT.md §16.1 — canonical layer names used across all modules
CANONICAL_LAYER_NAMES = (
    "img",
    "extent",
    "boundary",
    "distance",
    "valid",
    "meta",
    "extent_prob",
    "boundary_prob",
    "distance_pred",
    "parcel_instance",
)

# DATA_CONTRACT.md §2 — pipeline order is an invariant; do not reorder
PIPELINE_MODULES = (
    "module_prep_data",
    "module_net_train",
    "module_target_predict",
    "module_postprocess_vectorize",
    "module_eval",
)

# DATA_CONTRACT.md §8.1 — mandatory layers in every train-ready sample
REQUIRED_SAMPLE_LAYERS = ("img", "extent", "boundary", "distance", "valid", "meta")

# DATA_CONTRACT.md §12.3 — mandatory raster outputs from predict
REQUIRED_PREDICT_OUTPUTS = ("extent_prob", "boundary_prob", "distance_pred", "valid")
