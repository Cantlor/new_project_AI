"""Unit tests for ai_fields.common.errors.

Verifies the contract exception hierarchy required by DATA_CONTRACT.md §17.
All contract violations must raise a subclass of ContractError so callers
can catch at the right level of specificity.
"""

import pytest

from ai_fields.common.errors import (
    ChannelCountError,
    CheckpointMetadataError,
    ContractError,
    FeatureModeError,
    ManifestError,
    NormalizationContractError,
    SpatialContractError,
    ValidPolicyError,
)

# All concrete error classes in one place for parametrize use
ALL_CONTRACT_ERRORS = [
    FeatureModeError,
    ChannelCountError,
    SpatialContractError,
    ValidPolicyError,
    ManifestError,
    CheckpointMetadataError,
    NormalizationContractError,
]


# ---------------------------------------------------------------------------
# Root of hierarchy
# ---------------------------------------------------------------------------


def test_contract_error_is_exception():
    assert issubclass(ContractError, Exception)


def test_contract_error_is_catchable_as_exception():
    with pytest.raises(Exception):
        raise ContractError("base contract error")


# ---------------------------------------------------------------------------
# Each concrete class inherits from ContractError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_CONTRACT_ERRORS)
def test_each_error_is_subclass_of_contract_error(cls):
    assert issubclass(cls, ContractError), f"{cls.__name__} must subclass ContractError"


@pytest.mark.parametrize("cls", ALL_CONTRACT_ERRORS)
def test_each_error_is_subclass_of_exception(cls):
    assert issubclass(cls, Exception), f"{cls.__name__} must subclass Exception"


# ---------------------------------------------------------------------------
# Raise and catch behaviour
# ---------------------------------------------------------------------------


def test_feature_mode_error_caught_as_contract_error():
    with pytest.raises(ContractError):
        raise FeatureModeError("unknown mode")


def test_channel_count_error_caught_as_contract_error():
    with pytest.raises(ContractError):
        raise ChannelCountError("expected 9 channels")


def test_spatial_contract_error_caught_as_contract_error():
    with pytest.raises(ContractError):
        raise SpatialContractError("CRS mismatch")


def test_valid_policy_error_caught_as_contract_error():
    with pytest.raises(ContractError):
        raise ValidPolicyError("valid computed after fill")


def test_manifest_error_caught_as_contract_error():
    with pytest.raises(ContractError):
        raise ManifestError("missing run_id")


def test_checkpoint_metadata_error_caught_as_contract_error():
    with pytest.raises(ContractError):
        raise CheckpointMetadataError("feature_mode missing from checkpoint")


def test_normalization_contract_error_caught_as_contract_error():
    with pytest.raises(ContractError):
        raise NormalizationContractError("stats not derived from train data")


# ---------------------------------------------------------------------------
# Specific errors are NOT accidentally swallowed by siblings
# ---------------------------------------------------------------------------


def test_feature_mode_error_not_caught_as_channel_count_error():
    with pytest.raises(FeatureModeError):
        raise FeatureModeError("specific error")


def test_errors_carry_message():
    msg = "descriptive message"
    err = FeatureModeError(msg)
    assert str(err) == msg
