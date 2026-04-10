"""Unit tests for module_target_predict feature adapter layer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    NormalizationContractError,
)
from ai_fields.common.manifests import write_manifest
from ai_fields.module_target_predict.checkpoint_contract import (
    resolve_checkpoint_predict_contract,
)
from ai_fields.module_target_predict.feature_adapter import resolve_predict_input

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
        # Keep deterministic non-constant spatial values for robust checks.
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
    return payload


def _write_norm_stats(path: Path, *, channels: int) -> Path:
    band_stats: list[dict[str, float | int]] = []
    for idx in range(channels):
        if idx < 8:
            # Spectral bands in fixture are mostly in [1..9], keep simple clipping.
            p_lo = 0.0
            p_hi = 10.0
        else:
            # Derived indices are bounded near [-1, 1].
            p_lo = -1.0
            p_hi = 1.0
        band_stats.append({"band_idx": idx, "p_lo": p_lo, "p_hi": p_hi})
    # Make band 0 intentionally tight so we can assert exact scaling behavior.
    band_stats[0]["p_lo"] = 1.0
    band_stats[0]["p_hi"] = 2.0

    payload = {
        "clip_percentiles": [0.5, 99.5],
        "band_stats": band_stats,
        "computed_on": "valid_train_pixels",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_checkpoint_contract(
    tmp_path: Path,
    *,
    feature_mode: str = "raw8",
    stats_source: str = "norm_stats.json",
):
    checkpoint_path = tmp_path / "best.ckpt"
    checkpoint_path.write_bytes(b"checkpoint-bytes")
    metadata_path = tmp_path / "checkpoint_metadata.json"
    write_manifest(
        metadata_path,
        _metadata_payload(feature_mode=feature_mode, stats_source=stats_source),
    )
    return resolve_checkpoint_predict_contract(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=metadata_path,
    )


def test_resolve_predict_input_happy_path_raw8(tmp_path: Path) -> None:
    raster_path = _write_raster(tmp_path / "scene_raw8.tif", count=8, nodata=-9999.0)
    _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
    checkpoint_contract = _make_checkpoint_contract(tmp_path, feature_mode="raw8")

    resolved = resolve_predict_input(
        raster_path=raster_path,
        checkpoint_contract=checkpoint_contract,
    )

    assert resolved.feature_mode == "raw8"
    assert resolved.assembled_model_input == "raw8_valid"
    assert resolved.channel_semantics[-1] == "valid"
    assert resolved.raw_feature_shape[0] == 8
    assert resolved.normalized_feature_shape[0] == 8
    assert resolved.assembled_input_shape[0] == 9
    assert resolved.valid_mask_source == "nodata_metadata"
    assert resolved.input_ready_for_model is True
    assert resolved.normalization_summary["stats_origin"].startswith("stats_source_path:")
    # nodata pixel must stay invalid and map to scale_min in normalized stack.
    assert resolved.valid_mask[0, 0] == 0
    assert resolved.normalized_feature_stack[0, 0, 0] == pytest.approx(0.0)
    # valid channel is always appended as last input channel.
    assert resolved.assembled_input[-1, 0, 0] == pytest.approx(0.0)
    assert resolved.assembled_input[-1, 1, 1] == pytest.approx(1.0)


def test_resolve_predict_input_happy_path_raw8_idx3_builds_indices(tmp_path: Path) -> None:
    raster_path = _write_raster(tmp_path / "scene_raw8_idx3.tif", count=8, nodata=-9999.0)
    _write_norm_stats(tmp_path / "norm_stats.json", channels=11)
    checkpoint_contract = _make_checkpoint_contract(tmp_path, feature_mode="raw8_idx3")

    resolved = resolve_predict_input(
        raster_path=raster_path,
        checkpoint_contract=checkpoint_contract,
    )

    assert resolved.feature_mode == "raw8_idx3"
    assert resolved.raw_feature_shape[0] == 11
    assert resolved.assembled_input_shape[0] == 12
    assert resolved.channel_semantics[-4:] == ("NDVI", "SAVI", "NDWI", "valid")
    # Derived channels should be present and non-zero on valid pixels.
    valid_pixel = (2, 2)
    red = resolved.raw_feature_stack[4, valid_pixel[0], valid_pixel[1]]
    nir1 = resolved.raw_feature_stack[6, valid_pixel[0], valid_pixel[1]]
    green = resolved.raw_feature_stack[2, valid_pixel[0], valid_pixel[1]]
    expected_ndvi = (nir1 - red) / (nir1 + red + 1e-6)
    expected_savi = ((nir1 - red) / (nir1 + red + 0.5 + 1e-6)) * 1.5
    expected_ndwi = (green - nir1) / (green + nir1 + 1e-6)

    ndvi = resolved.raw_feature_stack[8, valid_pixel[0], valid_pixel[1]]
    savi = resolved.raw_feature_stack[9, valid_pixel[0], valid_pixel[1]]
    ndwi = resolved.raw_feature_stack[10, valid_pixel[0], valid_pixel[1]]
    assert ndvi == pytest.approx(float(expected_ndvi), rel=1e-6)
    assert savi == pytest.approx(float(expected_savi), rel=1e-6)
    assert ndwi == pytest.approx(float(expected_ndwi), rel=1e-6)
    # Invalid pixels stay explicit non-signal.
    assert resolved.raw_feature_stack[8, 0, 0] == pytest.approx(0.0)
    assert resolved.raw_feature_stack[9, 0, 0] == pytest.approx(0.0)
    assert resolved.raw_feature_stack[10, 0, 0] == pytest.approx(0.0)


def test_missing_normalization_stats_raises_explicit_error(tmp_path: Path) -> None:
    raster_path = _write_raster(tmp_path / "scene_missing_stats.tif", count=8, nodata=-9999.0)
    checkpoint_contract = _make_checkpoint_contract(
        tmp_path,
        feature_mode="raw8",
        stats_source="missing_norm_stats.json",
    )

    with pytest.raises(NormalizationContractError, match="per-band normalization stats"):
        resolve_predict_input(
            raster_path=raster_path,
            checkpoint_contract=checkpoint_contract,
        )


def test_missing_required_channel_semantics_raises_explicit_error(tmp_path: Path) -> None:
    raster_path = _write_raster(tmp_path / "scene_bad_semantics.tif", count=8, nodata=-9999.0)
    _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
    contract = _make_checkpoint_contract(tmp_path, feature_mode="raw8")
    bad_semantics = list(contract.channel_semantics)
    bad_semantics[0] = "blue"  # break canonical semantic order
    bad_contract = replace(contract, channel_semantics=tuple(bad_semantics))

    with pytest.raises(ContractError, match="channel_semantics"):
        resolve_predict_input(
            raster_path=raster_path,
            checkpoint_contract=bad_contract,
        )


def test_raster_checkpoint_band_mismatch_raises_explicit_error(tmp_path: Path) -> None:
    raster_path = _write_raster(tmp_path / "scene_bad_bands.tif", count=7, nodata=-9999.0)
    _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
    checkpoint_contract = _make_checkpoint_contract(tmp_path, feature_mode="raw8")

    with pytest.raises(ChannelCountError, match="band_count"):
        resolve_predict_input(
            raster_path=raster_path,
            checkpoint_contract=checkpoint_contract,
        )
