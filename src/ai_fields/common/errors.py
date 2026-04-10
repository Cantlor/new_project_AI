# Contract exception hierarchy for the AI Fields project.
# DATA_CONTRACT.md §17 requires explicit errors on all contract violations.
# Validators must raise these; callers must never silently continue on ContractError.


class ContractError(Exception):
    """Base class for all data and feature contract violations."""


class FeatureModeError(ContractError):
    """Raised when an unsupported or inconsistent feature mode is encountered.

    Covers: unknown feature_mode names, mismatch between feature_mode and
    in_channels, hidden channel reordering. (DATA_CONTRACT.md §7.1, §7.6)
    """


class ChannelCountError(ContractError):
    """Raised when in_channels does not match the assembled model input contract.

    in_channels must equal the assembled input channel count (9 or 12),
    not the dataset-side feature count. (DATA_CONTRACT.md §7.5)
    """


class SpatialContractError(ContractError):
    """Raised when spatial alignment or CRS contract is violated.

    Covers: CRS mismatch, half-pixel drift, undocumented reprojection,
    raster/vector misalignment. (DATA_CONTRACT.md §5)
    """


class ValidPolicyError(ContractError):
    """Raised when the valid/NoData contract is violated.

    Covers: valid computed after NoData fill, valid semantics lost,
    invalid pixels silently treated as background. (DATA_CONTRACT.md §6)
    """


class ManifestError(ContractError):
    """Raised when a manifest is missing required fields or provenance is lost.

    (DATA_CONTRACT.md §15, MANIFEST_SCHEMAS.md §3, §14)
    """


class CheckpointMetadataError(ContractError):
    """Raised when checkpoint metadata is insufficient for predict-time contract restoration.

    Predict must fail explicitly rather than guess feature_mode or in_channels.
    (DATA_CONTRACT.md §12.4, DECISIONS.md DEC-007)
    """


class NormalizationContractError(ContractError):
    """Raised when normalization rules are inconsistent between train and predict,
    or when train-derived stats are unavailable at predict time.

    (DATA_CONTRACT.md §10, DECISIONS.md DEC-008)
    """
