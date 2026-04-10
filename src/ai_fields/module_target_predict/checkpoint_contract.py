"""Checkpoint-driven predict contract resolver for module_target_predict.

This module implements the first, minimal predict-side layer from module_target_predict
specification: restore runtime-critical inference contract from train export artifacts
without silent fallbacks.

Scope of this layer is intentionally narrow:
- read and validate net_train checkpoint metadata;
- restore assembled model input contract (`raw8_valid` / `raw8_idx3_valid`);
- restore normalization contract and target-head semantics;
- expose required predict output skeleton.

It does not implement tiled inference, raster reading, model forward, or output writing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.constants import (
    ASSEMBLED_MODEL_INPUTS,
    CHANNEL_COUNTS,
    FEATURE_MODES,
    REQUIRED_PREDICT_OUTPUTS,
)
from ai_fields.common.errors import (
    ChannelCountError,
    CheckpointMetadataError,
    ContractError,
    FeatureModeError,
    NormalizationContractError,
    ValidPolicyError,
)
from ai_fields.common.manifests import read_manifest


_EXPECTED_CHECKPOINT_METADATA_SCHEMA = "net_train.checkpoint_metadata"
_REQUIRED_TARGET_HEADS = ("extent", "boundary", "distance")
_OUTPUT_RASTER_FILENAMES: dict[str, str] = {
    "extent_prob": "extent_prob.tif",
    "boundary_prob": "boundary_prob.tif",
    "distance_pred": "distance_pred.tif",
    "valid": "valid.tif",
}


@dataclass(frozen=True)
class CheckpointDrivenPredictContract:
    """Resolved predict-side contract recovered from train checkpoint metadata."""

    checkpoint_path: Path
    checkpoint_metadata_path: Path
    train_manifest_path: Path | None
    config_used_path: Path | None

    feature_mode: str
    assembled_model_input: str
    in_channels: int
    feature_channel_count: int
    channel_semantics: tuple[str, ...]
    valid_as_input_channel: bool

    normalization: dict[str, Any]
    target_heads: dict[str, Any]
    model_version: str | None
    model_architecture: str | None
    model_encoder_depth: int | None
    model_base_channels: int | None

    required_outputs: tuple[str, ...]
    output_raster_filenames: dict[str, str]


def _normalize_existing_path(path: Any, *, name: str) -> Path:
    if isinstance(path, (str, PathLike)):
        normalized = Path(path)
    else:
        raise ContractError(f"{name} must be path-like, got {type(path).__name__}.")
    if str(normalized).strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    if not normalized.exists():
        raise ContractError(f"{name} does not exist: {normalized}")
    return normalized


def _normalize_optional_path(path: Any, *, name: str) -> Path | None:
    if path is None:
        return None
    return _normalize_existing_path(path, name=name)


def _require_non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise CheckpointMetadataError(f"{name} must be a non-empty string.")
    return value


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise CheckpointMetadataError(f"{name} must be a boolean.")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CheckpointMetadataError(f"{name} must be an integer >= 1.")
    return int(value)


def _normalize_optional_positive_int(
    value: Any,
    *,
    name: str,
    min_value: int = 1,
) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < min_value
    ):
        raise CheckpointMetadataError(
            f"{name} must be an integer >= {min_value} when provided."
        )
    return int(value)


def _normalize_optional_model_contract_fields(
    metadata: Mapping[str, Any],
) -> tuple[str | None, int | None, int | None]:
    model_architecture = metadata.get("model_architecture")
    if model_architecture is not None:
        if not isinstance(model_architecture, str) or model_architecture.strip() == "":
            raise CheckpointMetadataError(
                "checkpoint_metadata.model_architecture must be a non-empty string when provided."
            )

    encoder_depth = _normalize_optional_positive_int(
        metadata.get("encoder_depth"),
        name="checkpoint_metadata.encoder_depth",
        min_value=2,
    )
    base_channels = _normalize_optional_positive_int(
        metadata.get("base_channels"),
        name="checkpoint_metadata.base_channels",
        min_value=1,
    )

    provided_flags = (
        model_architecture is not None,
        encoder_depth is not None,
        base_channels is not None,
    )
    if any(provided_flags) and not all(provided_flags):
        raise CheckpointMetadataError(
            "checkpoint_metadata model architecture fields must be provided together: "
            "'model_architecture', 'encoder_depth', and 'base_channels'."
        )

    return model_architecture, encoder_depth, base_channels


def _require_pair_of_numbers(value: Any, *, name: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise NormalizationContractError(f"{name} must be a 2-item numeric sequence.")
    out: list[float] = []
    for idx, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise NormalizationContractError(
                f"{name}[{idx}] must be numeric, got {type(item).__name__}."
            )
        out.append(float(item))
    return out


def _normalize_normalization_contract(normalization: Any) -> dict[str, Any]:
    if not isinstance(normalization, Mapping):
        raise NormalizationContractError(
            "checkpoint_metadata.normalization must be a mapping/object."
        )

    normalization_name_raw = normalization.get("normalization_name")
    if not isinstance(normalization_name_raw, str) or normalization_name_raw.strip() == "":
        raise NormalizationContractError(
            "checkpoint_metadata.normalization.normalization_name must be a non-empty string."
        )
    normalization_name = normalization_name_raw

    stats_source_raw = normalization.get("stats_source")
    if not isinstance(stats_source_raw, str) or stats_source_raw.strip() == "":
        raise NormalizationContractError(
            "checkpoint_metadata.normalization.stats_source must be a non-empty string."
        )
    stats_source = stats_source_raw
    clip_percentiles = _require_pair_of_numbers(
        normalization.get("clip_percentiles"),
        name="checkpoint_metadata.normalization.clip_percentiles",
    )
    scaling_range = _require_pair_of_numbers(
        normalization.get("scaling_range"),
        name="checkpoint_metadata.normalization.scaling_range",
    )

    return {
        "normalization_name": normalization_name,
        "stats_source": stats_source,
        "clip_percentiles": clip_percentiles,
        "scaling_range": scaling_range,
    }


def _normalize_target_heads(target_heads: Any) -> dict[str, Any]:
    if not isinstance(target_heads, Mapping):
        raise CheckpointMetadataError(
            "checkpoint_metadata.target_heads must be a mapping/object."
        )

    normalized: dict[str, Any] = {}
    for key in _REQUIRED_TARGET_HEADS:
        if key not in target_heads:
            raise CheckpointMetadataError(
                f"checkpoint_metadata.target_heads is missing required head: {key!r}."
            )
        value = target_heads[key]
        if not isinstance(value, Mapping):
            raise CheckpointMetadataError(
                f"checkpoint_metadata.target_heads[{key!r}] must be a mapping/object."
            )
        normalized[key] = dict(value)
    return normalized


def _normalize_channel_semantics(value: Any, *, expected_channels: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CheckpointMetadataError(
            "checkpoint_metadata.channel_semantics must be a sequence of strings."
        )

    semantics = tuple(value)
    if len(semantics) != expected_channels:
        raise ChannelCountError(
            "checkpoint_metadata.channel_semantics length must match in_channels: "
            f"{len(semantics)} != {expected_channels}."
        )
    if not all(isinstance(item, str) and item.strip() != "" for item in semantics):
        raise CheckpointMetadataError(
            "checkpoint_metadata.channel_semantics must contain only non-empty strings."
        )
    return semantics


def _resolve_expected_input_contract(feature_mode: str) -> tuple[str, int, int]:
    if feature_mode not in FEATURE_MODES:
        raise FeatureModeError(
            f"Unsupported checkpoint feature_mode {feature_mode!r}. Expected one of {list(FEATURE_MODES)}."
        )

    assembled = f"{feature_mode}_valid"
    if assembled not in ASSEMBLED_MODEL_INPUTS:
        raise CheckpointMetadataError(
            f"Assembled model input {assembled!r} is not supported by baseline contract."
        )

    feature_channels = CHANNEL_COUNTS[feature_mode]
    final_channels = CHANNEL_COUNTS[assembled]
    return assembled, feature_channels, final_channels


def resolve_checkpoint_predict_contract(
    *,
    checkpoint_path: str | Path,
    checkpoint_metadata_path: str | Path,
    train_manifest_path: str | Path | None = None,
    config_used_path: str | Path | None = None,
) -> CheckpointDrivenPredictContract:
    """Resolve strict predict-time contract from net_train exported metadata.

    Parameters
    ----------
    checkpoint_path:
        Path to the trained checkpoint artifact.
    checkpoint_metadata_path:
        Path to ``checkpoint_metadata.json`` exported by module_net_train.
    train_manifest_path:
        Optional path to ``train_manifest.json`` for provenance linkage.
    config_used_path:
        Optional path to ``config_used.yaml`` for provenance linkage.

    Returns
    -------
    CheckpointDrivenPredictContract
        Fully resolved and validated predict-side contract.

    Raises
    ------
    ContractError subclasses
        On missing/invalid metadata or contract mismatch. No silent fallback is used.
    """

    resolved_checkpoint_path = _normalize_existing_path(checkpoint_path, name="checkpoint_path")
    resolved_metadata_path = _normalize_existing_path(
        checkpoint_metadata_path,
        name="checkpoint_metadata_path",
    )
    resolved_train_manifest_path = _normalize_optional_path(
        train_manifest_path,
        name="train_manifest_path",
    )
    resolved_config_used_path = _normalize_optional_path(
        config_used_path,
        name="config_used_path",
    )

    metadata = read_manifest(resolved_metadata_path)
    schema_name = metadata.get("schema_name")
    if schema_name != _EXPECTED_CHECKPOINT_METADATA_SCHEMA:
        raise CheckpointMetadataError(
            f"checkpoint metadata schema_name must be {_EXPECTED_CHECKPOINT_METADATA_SCHEMA!r}, "
            f"got {schema_name!r}."
        )

    feature_mode = _require_non_empty_string(
        metadata.get("feature_mode"),
        name="checkpoint_metadata.feature_mode",
    )
    assembled_model_input = _require_non_empty_string(
        metadata.get("assembled_model_input"),
        name="checkpoint_metadata.assembled_model_input",
    )
    in_channels = _require_positive_int(
        metadata.get("in_channels"),
        name="checkpoint_metadata.in_channels",
    )
    valid_as_input_channel = _require_bool(
        metadata.get("valid_as_input_channel"),
        name="checkpoint_metadata.valid_as_input_channel",
    )
    if valid_as_input_channel is not True:
        raise ValidPolicyError(
            "checkpoint_metadata.valid_as_input_channel must be True for baseline v1 predict contract."
        )

    expected_assembled, feature_channels, expected_final_channels = _resolve_expected_input_contract(
        feature_mode
    )
    if assembled_model_input != expected_assembled:
        raise ChannelCountError(
            "checkpoint_metadata.assembled_model_input is inconsistent with feature_mode: "
            f"expected {expected_assembled!r}, got {assembled_model_input!r}."
        )
    if in_channels != expected_final_channels:
        raise ChannelCountError(
            "checkpoint_metadata.in_channels is inconsistent with assembled_model_input: "
            f"expected {expected_final_channels}, got {in_channels}."
        )

    channel_semantics = _normalize_channel_semantics(
        metadata.get("channel_semantics"),
        expected_channels=in_channels,
    )
    if channel_semantics[-1] != "valid":
        raise CheckpointMetadataError(
            "checkpoint_metadata.channel_semantics must keep 'valid' as the last channel "
            "for baseline assembled input contract."
        )

    normalization = _normalize_normalization_contract(metadata.get("normalization"))
    target_heads = _normalize_target_heads(metadata.get("target_heads"))

    model_version_raw = metadata.get("model_version")
    if model_version_raw is not None and not isinstance(model_version_raw, str):
        raise CheckpointMetadataError(
            "checkpoint_metadata.model_version must be a string or null."
        )
    model_architecture, model_encoder_depth, model_base_channels = (
        _normalize_optional_model_contract_fields(metadata)
    )

    return CheckpointDrivenPredictContract(
        checkpoint_path=resolved_checkpoint_path,
        checkpoint_metadata_path=resolved_metadata_path,
        train_manifest_path=resolved_train_manifest_path,
        config_used_path=resolved_config_used_path,
        feature_mode=feature_mode,
        assembled_model_input=assembled_model_input,
        in_channels=in_channels,
        feature_channel_count=feature_channels,
        channel_semantics=channel_semantics,
        valid_as_input_channel=True,
        normalization=normalization,
        target_heads=target_heads,
        model_version=model_version_raw,
        model_architecture=model_architecture,
        model_encoder_depth=model_encoder_depth,
        model_base_channels=model_base_channels,
        required_outputs=tuple(REQUIRED_PREDICT_OUTPUTS),
        output_raster_filenames=dict(_OUTPUT_RASTER_FILENAMES),
    )


__all__ = [
    "CheckpointDrivenPredictContract",
    "resolve_checkpoint_predict_contract",
]
