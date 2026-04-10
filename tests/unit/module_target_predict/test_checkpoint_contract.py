"""Unit tests for module_target_predict checkpoint-driven contract resolver."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import (
    ChannelCountError,
    CheckpointMetadataError,
    FeatureModeError,
    NormalizationContractError,
    ValidPolicyError,
)
from ai_fields.common.manifests import write_manifest
from ai_fields.module_target_predict.checkpoint_contract import (
    resolve_checkpoint_predict_contract,
)


def _dummy_checkpoint(path: Path) -> Path:
    path.write_bytes(b"checkpoint-bytes")
    return path


def _base_manifest_top_level() -> dict[str, Any]:
    return {
        "schema_name": "net_train.checkpoint_metadata",
        "schema_version": "v1",
        "module_name": "module_net_train",
        "module_version": None,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "run_id": "net_train_run_001",
        "stage_name": "checkpoint_export",
        "created_at_utc": "2026-04-03T00:00:00Z",
        "status": "success",
    }


def _channel_semantics_for_mode(feature_mode: str) -> list[str]:
    if feature_mode == "raw8":
        return [
            "coastal",
            "blue",
            "green",
            "yellow",
            "red",
            "rededge",
            "nir1",
            "nir2",
            "valid",
        ]
    if feature_mode == "raw8_idx3":
        return [
            "coastal",
            "blue",
            "green",
            "yellow",
            "red",
            "rededge",
            "nir1",
            "nir2",
            "NDVI",
            "SAVI",
            "NDWI",
            "valid",
        ]
    raise RuntimeError(f"unsupported test feature_mode: {feature_mode}")


def _metadata_payload(
    *,
    feature_mode: str = "raw8",
    include_arch_fields: bool = False,
) -> dict[str, Any]:
    assembled = f"{feature_mode}_valid"
    in_channels = 9 if feature_mode == "raw8" else 12
    payload = _base_manifest_top_level()
    payload.update(
        {
            "checkpoint_path": "/tmp/fake/best.ckpt",
            "feature_mode": feature_mode,
            "assembled_model_input": assembled,
            "in_channels": in_channels,
            "channel_semantics": _channel_semantics_for_mode(feature_mode),
            "valid_as_input_channel": True,
            "normalization": {
                "normalization_name": "per_band_robust_percentile",
                "stats_source": "train_norm_stats.json",
                "clip_percentiles": [0.5, 99.5],
                "scaling_range": [0.0, 1.0],
            },
            "target_heads": {
                "extent": {
                    "type": "binary_segmentation",
                    "ignore_label": 255,
                },
                "boundary": {
                    "type": "multiclass_segmentation",
                    "classes": {"0": "background", "1": "skeleton", "2": "buffer"},
                },
                "distance": {
                    "type": "regression",
                    "definition": "unsigned_distance_to_nearest_boundary",
                },
            },
            "model_version": "v1-baseline",
        }
    )
    if include_arch_fields:
        payload.update(
            {
                "model_architecture": "edge_aware_multitask_v1",
                "encoder_depth": 4,
                "base_channels": 8,
            }
        )
    return payload


def _write_checkpoint_metadata(path: Path, payload: dict[str, Any]) -> Path:
    write_manifest(path, payload)
    return path


def test_resolve_checkpoint_contract_happy_path_raw8(tmp_path: Path) -> None:
    checkpoint_path = _dummy_checkpoint(tmp_path / "best.ckpt")
    metadata_path = _write_checkpoint_metadata(
        tmp_path / "checkpoint_metadata.json",
        _metadata_payload(feature_mode="raw8"),
    )
    train_manifest_path = tmp_path / "train_manifest.json"
    train_manifest_path.write_text("{}", encoding="utf-8")
    config_used_path = tmp_path / "config_used.yaml"
    config_used_path.write_text("feature_mode: raw8\n", encoding="utf-8")

    resolved = resolve_checkpoint_predict_contract(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=metadata_path,
        train_manifest_path=train_manifest_path,
        config_used_path=config_used_path,
    )

    assert resolved.feature_mode == "raw8"
    assert resolved.assembled_model_input == "raw8_valid"
    assert resolved.in_channels == 9
    assert resolved.feature_channel_count == 8
    assert resolved.valid_as_input_channel is True
    assert resolved.channel_semantics[-1] == "valid"
    assert resolved.normalization["stats_source"] == "train_norm_stats.json"
    assert resolved.required_outputs == (
        "extent_prob",
        "boundary_prob",
        "distance_pred",
        "valid",
    )
    assert resolved.output_raster_filenames["extent_prob"] == "extent_prob.tif"
    assert resolved.output_raster_filenames["boundary_prob"] == "boundary_prob.tif"
    assert resolved.output_raster_filenames["distance_pred"] == "distance_pred.tif"
    assert resolved.output_raster_filenames["valid"] == "valid.tif"


def test_resolve_checkpoint_contract_happy_path_raw8_idx3(tmp_path: Path) -> None:
    checkpoint_path = _dummy_checkpoint(tmp_path / "best.ckpt")
    metadata_path = _write_checkpoint_metadata(
        tmp_path / "checkpoint_metadata.json",
        _metadata_payload(feature_mode="raw8_idx3"),
    )

    resolved = resolve_checkpoint_predict_contract(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=metadata_path,
    )

    assert resolved.feature_mode == "raw8_idx3"
    assert resolved.assembled_model_input == "raw8_idx3_valid"
    assert resolved.in_channels == 12
    assert resolved.feature_channel_count == 11
    assert resolved.channel_semantics[-4:] == ("NDVI", "SAVI", "NDWI", "valid")


def test_unsupported_feature_mode_raises_explicit_error(tmp_path: Path) -> None:
    checkpoint_path = _dummy_checkpoint(tmp_path / "best.ckpt")
    payload = _metadata_payload(feature_mode="raw8")
    payload["feature_mode"] = "raw11"
    payload["assembled_model_input"] = "raw11_valid"
    metadata_path = _write_checkpoint_metadata(tmp_path / "checkpoint_metadata.json", payload)

    with pytest.raises(FeatureModeError, match="feature_mode"):
        resolve_checkpoint_predict_contract(
            checkpoint_path=checkpoint_path,
            checkpoint_metadata_path=metadata_path,
        )


def test_missing_assembled_contract_field_has_no_silent_fallback(tmp_path: Path) -> None:
    checkpoint_path = _dummy_checkpoint(tmp_path / "best.ckpt")
    payload = _metadata_payload(feature_mode="raw8")
    payload.pop("assembled_model_input")
    metadata_path = _write_checkpoint_metadata(tmp_path / "checkpoint_metadata.json", payload)

    with pytest.raises(CheckpointMetadataError, match="assembled_model_input"):
        resolve_checkpoint_predict_contract(
            checkpoint_path=checkpoint_path,
            checkpoint_metadata_path=metadata_path,
        )


def test_channel_semantics_length_mismatch_raises(tmp_path: Path) -> None:
    checkpoint_path = _dummy_checkpoint(tmp_path / "best.ckpt")
    payload = _metadata_payload(feature_mode="raw8")
    payload["channel_semantics"] = payload["channel_semantics"][:-1]
    metadata_path = _write_checkpoint_metadata(tmp_path / "checkpoint_metadata.json", payload)

    with pytest.raises(ChannelCountError, match="channel_semantics"):
        resolve_checkpoint_predict_contract(
            checkpoint_path=checkpoint_path,
            checkpoint_metadata_path=metadata_path,
        )


def test_valid_as_input_channel_must_be_true(tmp_path: Path) -> None:
    checkpoint_path = _dummy_checkpoint(tmp_path / "best.ckpt")
    payload = _metadata_payload(feature_mode="raw8")
    payload["valid_as_input_channel"] = False
    metadata_path = _write_checkpoint_metadata(tmp_path / "checkpoint_metadata.json", payload)

    with pytest.raises(ValidPolicyError, match="must be True"):
        resolve_checkpoint_predict_contract(
            checkpoint_path=checkpoint_path,
            checkpoint_metadata_path=metadata_path,
        )


def test_missing_normalization_stats_source_raises(tmp_path: Path) -> None:
    checkpoint_path = _dummy_checkpoint(tmp_path / "best.ckpt")
    payload = _metadata_payload(feature_mode="raw8")
    payload = deepcopy(payload)
    payload["normalization"].pop("stats_source")
    metadata_path = _write_checkpoint_metadata(tmp_path / "checkpoint_metadata.json", payload)

    with pytest.raises(NormalizationContractError, match="stats_source"):
        resolve_checkpoint_predict_contract(
            checkpoint_path=checkpoint_path,
            checkpoint_metadata_path=metadata_path,
        )


def test_explicit_model_architecture_fields_are_resolved(tmp_path: Path) -> None:
    checkpoint_path = _dummy_checkpoint(tmp_path / "best.ckpt")
    metadata_path = _write_checkpoint_metadata(
        tmp_path / "checkpoint_metadata.json",
        _metadata_payload(feature_mode="raw8", include_arch_fields=True),
    )

    resolved = resolve_checkpoint_predict_contract(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=metadata_path,
    )

    assert resolved.model_architecture == "edge_aware_multitask_v1"
    assert resolved.model_encoder_depth == 4
    assert resolved.model_base_channels == 8


def test_partial_model_architecture_fields_raise(tmp_path: Path) -> None:
    checkpoint_path = _dummy_checkpoint(tmp_path / "best.ckpt")
    payload = _metadata_payload(feature_mode="raw8")
    payload["model_architecture"] = "edge_aware_multitask_v1"
    metadata_path = _write_checkpoint_metadata(tmp_path / "checkpoint_metadata.json", payload)

    with pytest.raises(CheckpointMetadataError, match="must be provided together"):
        resolve_checkpoint_predict_contract(
            checkpoint_path=checkpoint_path,
            checkpoint_metadata_path=metadata_path,
        )
