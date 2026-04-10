"""Unit tests for ai_fields.common.constants.

Verifies that all canonical contract values match the accepted project documents.
These tests must never be edited to accommodate code changes — if a constant
changes, the source document must change first. (REPO_CONVENTIONS.md §18.2)
"""

import pytest

from ai_fields.common.constants import (
    ASSEMBLED_MODEL_INPUTS,
    CANONICAL_LAYER_NAMES,
    CHANNEL_COUNTS,
    DATA_CONTRACT_VERSION,
    DERIVED_INDICES,
    FEATURE_MODES,
    PIPELINE_MODULES,
    REQUIRED_PREDICT_OUTPUTS,
    REQUIRED_SAMPLE_LAYERS,
)


# ---------------------------------------------------------------------------
# DATA_CONTRACT_VERSION
# ---------------------------------------------------------------------------


def test_data_contract_version():
    # DATA_CONTRACT.md §18.1
    assert DATA_CONTRACT_VERSION == "v1"


# ---------------------------------------------------------------------------
# CHANNEL_COUNTS — DATA_CONTRACT.md §7.5
# ---------------------------------------------------------------------------


def test_channel_count_raw8_valid():
    # raw8 (8 spectral) + valid = 9 assembled model input channels
    assert CHANNEL_COUNTS["raw8_valid"] == 9


def test_channel_count_raw8_idx3_valid():
    # raw8_idx3 (8 spectral + 3 derived indices) + valid = 12 assembled model input channels
    assert CHANNEL_COUNTS["raw8_idx3_valid"] == 12


def test_channel_count_raw8_idx3_dataset_side():
    # dataset-side raw8_idx3 = 8 + NDVI + SAVI + NDWI = 11 (without valid)
    assert CHANNEL_COUNTS["raw8_idx3"] == 11


def test_channel_count_raw8_dataset_side():
    # dataset-side raw8 = 8 spectral bands only (without valid)
    assert CHANNEL_COUNTS["raw8"] == 8


def test_channel_counts_assembled_greater_than_dataset_side():
    # assembled inputs always include valid as an extra channel
    assert CHANNEL_COUNTS["raw8_valid"] == CHANNEL_COUNTS["raw8"] + 1
    assert CHANNEL_COUNTS["raw8_idx3_valid"] == CHANNEL_COUNTS["raw8_idx3"] + 1


# ---------------------------------------------------------------------------
# FEATURE_MODES — DATA_CONTRACT.md §7.1
# ---------------------------------------------------------------------------


def test_feature_modes_exact():
    # Only these two dataset-side modes are supported in v1; no others
    assert set(FEATURE_MODES) == {"raw8", "raw8_idx3"}


def test_feature_modes_no_extras():
    assert len(FEATURE_MODES) == 2


# ---------------------------------------------------------------------------
# ASSEMBLED_MODEL_INPUTS — DATA_CONTRACT.md §7.4
# ---------------------------------------------------------------------------


def test_assembled_model_inputs_exact():
    assert set(ASSEMBLED_MODEL_INPUTS) == {"raw8_valid", "raw8_idx3_valid"}


def test_assembled_model_inputs_no_extras():
    assert len(ASSEMBLED_MODEL_INPUTS) == 2


# ---------------------------------------------------------------------------
# DERIVED_INDICES — DATA_CONTRACT.md §7.3
# ---------------------------------------------------------------------------


def test_derived_indices_exact():
    assert set(DERIVED_INDICES) == {"NDVI", "SAVI", "NDWI"}


def test_derived_indices_count():
    # Exactly three fixed derived indices in raw8_idx3 mode
    assert len(DERIVED_INDICES) == 3


# ---------------------------------------------------------------------------
# CANONICAL_LAYER_NAMES — DATA_CONTRACT.md §16.1
# ---------------------------------------------------------------------------


def test_valid_in_canonical_layer_names():
    assert "valid" in CANONICAL_LAYER_NAMES


def test_canonical_layer_names_complete():
    required = {
        "img", "extent", "boundary", "distance", "valid", "meta",
        "extent_prob", "boundary_prob", "distance_pred", "parcel_instance",
    }
    assert required <= set(CANONICAL_LAYER_NAMES)


# ---------------------------------------------------------------------------
# REQUIRED_SAMPLE_LAYERS — DATA_CONTRACT.md §8.1
# ---------------------------------------------------------------------------


def test_img_in_required_sample_layers():
    assert "img" in REQUIRED_SAMPLE_LAYERS


def test_required_sample_layers_complete():
    # All six mandatory layers per DATA_CONTRACT.md §8.1
    expected = {"img", "extent", "boundary", "distance", "valid", "meta"}
    assert set(REQUIRED_SAMPLE_LAYERS) == expected


def test_required_sample_layers_count():
    assert len(REQUIRED_SAMPLE_LAYERS) == 6


# ---------------------------------------------------------------------------
# REQUIRED_PREDICT_OUTPUTS — DATA_CONTRACT.md §12.3
# ---------------------------------------------------------------------------


def test_required_predict_outputs_complete():
    expected = {"extent_prob", "boundary_prob", "distance_pred", "valid"}
    assert set(REQUIRED_PREDICT_OUTPUTS) == expected


def test_valid_in_required_predict_outputs():
    # valid is a mandatory predict output (DATA_CONTRACT.md §12.3, DEC-002)
    assert "valid" in REQUIRED_PREDICT_OUTPUTS


# ---------------------------------------------------------------------------
# PIPELINE_MODULES — DATA_CONTRACT.md §2
# ---------------------------------------------------------------------------


def test_pipeline_modules_order():
    # Pipeline order is a project invariant and must not be reordered
    assert list(PIPELINE_MODULES) == [
        "module_prep_data",
        "module_net_train",
        "module_target_predict",
        "module_postprocess_vectorize",
        "module_eval",
    ]


def test_pipeline_modules_count():
    assert len(PIPELINE_MODULES) == 5
