"""Unit tests for module_target_predict minimal inference core."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ChannelCountError, ContractError
from ai_fields.common.manifests import write_manifest
from ai_fields.module_net_train.model import EdgeAwareMultitaskNet
from ai_fields.module_target_predict.checkpoint_contract import (
    resolve_checkpoint_predict_contract,
)
from ai_fields.module_target_predict.feature_adapter import resolve_predict_input
from ai_fields.module_target_predict.inference_core import (
    load_predict_model,
    run_predict_forward,
)

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")


def _write_raster(
    path: Path,
    *,
    count: int = 8,
    width: int = 6,
    height: int = 5,
    nodata: float | int | None = None,
) -> Path:
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(100.0, 200.0, 1.0, 1.0)

    data = np.zeros((count, height, width), dtype=np.float32)
    for band in range(count):
        data[band, :, :] = float(band + 1) + np.linspace(0.0, 1.0, width)[None, :]
    if nodata is not None:
        data[0, 0, 0] = float(nodata)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="float32",
        crs="EPSG:32637",
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(data)

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
    stats_source: str = "norm_stats.json",
    model_architecture: str | None = None,
    encoder_depth: int | None = None,
    base_channels: int | None = None,
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
                "stats_source": stats_source,
                "clip_percentiles": [0.5, 99.5],
                "scaling_range": [0.0, 1.0],
            },
            "target_heads": {
                "extent": {"type": "binary_segmentation", "ignore_label": 255},
                "boundary": {
                    "type": "multiclass_segmentation",
                    "classes": {"0": "background", "1": "skeleton", "2": "buffer"},
                },
                "distance": {"type": "regression"},
            },
            "model_version": "v1-baseline",
        }
    )
    if model_architecture is not None:
        payload["model_architecture"] = model_architecture
    if encoder_depth is not None:
        payload["encoder_depth"] = encoder_depth
    if base_channels is not None:
        payload["base_channels"] = base_channels
    return payload


def _write_norm_stats(path: Path, *, channels: int) -> Path:
    band_stats: list[dict[str, float | int]] = []
    for idx in range(channels):
        if idx < 8:
            p_lo = 0.0
            p_hi = 10.0
        else:
            p_lo = -1.0
            p_hi = 1.0
        band_stats.append({"band_idx": idx, "p_lo": p_lo, "p_hi": p_hi})

    payload = {
        "clip_percentiles": [0.5, 99.5],
        "band_stats": band_stats,
        "computed_on": "valid_train_pixels",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_checkpoint_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _prepare_contract_and_input(
    tmp_path: Path,
    *,
    model: torch.nn.Module,
    feature_mode: str = "raw8",
    checkpoint_payload_override: dict[str, Any] | None = None,
    include_arch_fields_in_metadata: bool = False,
    metadata_architecture_override: dict[str, Any] | None = None,
):
    channels = 8 if feature_mode == "raw8" else 11
    raster_path = _write_raster(tmp_path / "scene.tif", count=8, nodata=-9999.0)
    _write_norm_stats(tmp_path / "norm_stats.json", channels=channels)

    checkpoint_path = tmp_path / "best.ckpt"
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_mode": feature_mode,
        "assembled_model_input": f"{feature_mode}_valid",
        "in_channels": 9 if feature_mode == "raw8" else 12,
        "channel_semantics": _channel_semantics_for_mode(feature_mode),
        "valid_as_input_channel": True,
        "epochs_completed": 1,
    }
    if checkpoint_payload_override is not None:
        payload.update(checkpoint_payload_override)
    _write_checkpoint_payload(checkpoint_path, payload)

    metadata_path = tmp_path / "checkpoint_metadata.json"
    metadata_kwargs: dict[str, Any] = {}
    if include_arch_fields_in_metadata:
        metadata_kwargs = {
            "model_architecture": "edge_aware_multitask_v1",
            "encoder_depth": int(getattr(model, "encoder_depth")),
            "base_channels": int(getattr(model, "base_channels")),
        }
    if metadata_architecture_override is not None:
        metadata_kwargs.update(metadata_architecture_override)
    write_manifest(
        metadata_path,
        _metadata_payload(feature_mode=feature_mode, **metadata_kwargs),
    )
    checkpoint_contract = resolve_checkpoint_predict_contract(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=metadata_path,
    )
    predict_input = resolve_predict_input(
        raster_path=raster_path,
        checkpoint_contract=checkpoint_contract,
    )
    return checkpoint_contract, predict_input


def test_happy_path_forward_runs_and_validates_output_contract(tmp_path: Path) -> None:
    model = EdgeAwareMultitaskNet(in_channels=9, encoder_depth=3, base_channels=8)
    checkpoint_contract, predict_input = _prepare_contract_and_input(
        tmp_path,
        model=model,
        feature_mode="raw8",
        include_arch_fields_in_metadata=True,
    )

    loaded = load_predict_model(
        checkpoint_contract=checkpoint_contract,
        device="cpu",
    )
    result = run_predict_forward(
        loaded_model=loaded,
        predict_input=predict_input,
        checkpoint_contract=checkpoint_contract,
    )

    h, w = predict_input.assembled_input_shape[1], predict_input.assembled_input_shape[2]
    assert result.forward_contract_ok is True
    assert result.input_shape == (1, 9, h, w)
    assert result.extent_shape == (1, 1, h, w)
    assert result.boundary_shape == (1, 3, h, w)
    assert result.distance_shape == (1, 1, h, w)
    assert result.aux_count == 1
    assert result.device == "cpu"
    assert np.all((result.extent_prob >= 0.0) & (result.extent_prob <= 1.0))
    assert np.all((result.boundary_prob >= 0.0) & (result.boundary_prob <= 1.0))
    assert np.allclose(
        result.boundary_prob.sum(axis=1),
        np.ones((1, h, w), dtype=np.float32),
        atol=1e-5,
    )
    assert loaded.model_shape_source == "checkpoint_metadata_explicit"


def test_input_channel_mismatch_raises_explicit_error(tmp_path: Path) -> None:
    model = EdgeAwareMultitaskNet(in_channels=9, encoder_depth=3, base_channels=8)
    checkpoint_contract, predict_input = _prepare_contract_and_input(
        tmp_path,
        model=model,
        feature_mode="raw8",
    )

    bad_assembled = predict_input.assembled_input[:-1, :, :]
    bad_shape = (bad_assembled.shape[0], bad_assembled.shape[1], bad_assembled.shape[2])
    bad_input = replace(
        predict_input,
        assembled_input=bad_assembled,
        assembled_input_shape=bad_shape,
    )

    loaded = load_predict_model(
        checkpoint_contract=checkpoint_contract,
        device="cpu",
    )
    with pytest.raises(ChannelCountError, match="channel count"):
        run_predict_forward(
            loaded_model=loaded,
            predict_input=bad_input,
            checkpoint_contract=checkpoint_contract,
        )


def test_output_contract_mismatch_raises_explicit_error(tmp_path: Path) -> None:
    model = EdgeAwareMultitaskNet(in_channels=9, encoder_depth=3, base_channels=8)
    checkpoint_contract, predict_input = _prepare_contract_and_input(
        tmp_path,
        model=model,
        feature_mode="raw8",
    )
    loaded = load_predict_model(
        checkpoint_contract=checkpoint_contract,
        device="cpu",
    )

    class _BrokenBoundaryModel(torch.nn.Module):
        def forward(self, x):  # type: ignore[override]
            b, _, h, w = x.shape
            extent = torch.zeros((b, 1, h, w), dtype=x.dtype, device=x.device)
            boundary = torch.zeros((b, 2, h, w), dtype=x.dtype, device=x.device)
            distance = torch.zeros((b, 1, h, w), dtype=x.dtype, device=x.device)
            return {"extent": extent, "boundary": boundary, "distance": distance, "aux": []}

    broken_loaded = replace(loaded, model=_BrokenBoundaryModel().to(loaded.device))
    with pytest.raises(ChannelCountError, match="boundary output channel count"):
        run_predict_forward(
            loaded_model=broken_loaded,
            predict_input=predict_input,
            checkpoint_contract=checkpoint_contract,
        )


def test_missing_checkpoint_required_key_has_no_silent_fallback(tmp_path: Path) -> None:
    model = EdgeAwareMultitaskNet(in_channels=9, encoder_depth=3, base_channels=8)
    checkpoint_contract, _predict_input = _prepare_contract_and_input(
        tmp_path,
        model=model,
        feature_mode="raw8",
        checkpoint_payload_override={"model_state_dict": None},
    )
    with pytest.raises(ContractError, match="model_state_dict"):
        load_predict_model(
            checkpoint_contract=checkpoint_contract,
            device="cpu",
        )


def test_incompatible_state_dict_raises_explicit_error(tmp_path: Path) -> None:
    model = EdgeAwareMultitaskNet(in_channels=9, encoder_depth=3, base_channels=8)
    state = dict(model.state_dict())
    removed_key = "extent_head.head.1.bias"
    state.pop(removed_key)

    checkpoint_contract, _predict_input = _prepare_contract_and_input(
        tmp_path,
        model=model,
        feature_mode="raw8",
        checkpoint_payload_override={"model_state_dict": state},
    )
    with pytest.raises(ContractError, match="incompatible"):
        load_predict_model(
            checkpoint_contract=checkpoint_contract,
            device="cpu",
        )


def test_legacy_fallback_without_architecture_metadata_still_works(tmp_path: Path) -> None:
    model = EdgeAwareMultitaskNet(in_channels=9, encoder_depth=3, base_channels=8)
    checkpoint_contract, _predict_input = _prepare_contract_and_input(
        tmp_path,
        model=model,
        feature_mode="raw8",
        include_arch_fields_in_metadata=False,
    )

    loaded = load_predict_model(
        checkpoint_contract=checkpoint_contract,
        device="cpu",
    )
    assert loaded.model_shape_source == "legacy_state_dict_fallback"
    assert loaded.encoder_depth == 3
    assert loaded.base_channels == 8


def test_inconsistent_architecture_metadata_fails_explicitly(tmp_path: Path) -> None:
    model = EdgeAwareMultitaskNet(in_channels=9, encoder_depth=3, base_channels=8)
    checkpoint_contract, _predict_input = _prepare_contract_and_input(
        tmp_path,
        model=model,
        feature_mode="raw8",
        include_arch_fields_in_metadata=True,
        metadata_architecture_override={"encoder_depth": 5},
    )

    with pytest.raises(ContractError, match="architecture fields are inconsistent"):
        load_predict_model(
            checkpoint_contract=checkpoint_contract,
            device="cpu",
        )
