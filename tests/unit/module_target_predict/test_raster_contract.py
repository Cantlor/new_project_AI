"""Unit tests for module_target_predict raster contract layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    SpatialContractError,
    ValidPolicyError,
)
from ai_fields.common.manifests import write_manifest
from ai_fields.module_target_predict.checkpoint_contract import (
    resolve_checkpoint_predict_contract,
)
from ai_fields.module_target_predict.raster_contract import (
    read_predict_raster_metadata,
    resolve_predict_raster_contract,
    resolve_predict_valid_mask,
)

rasterio = pytest.importorskip("rasterio")


def _write_raster(
    path: Path,
    *,
    count: int = 8,
    width: int = 8,
    height: int = 6,
    nodata: float | int | None = None,
    with_internal_mask: bool = False,
    with_crs: bool = True,
) -> Path:
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    crs = "EPSG:32637" if with_crs else None

    data = np.zeros((count, height, width), dtype=np.float32)
    for band in range(count):
        data[band, :, :] = float(band + 1)

    if nodata is not None:
        # Keep nodata signal explicit in first band for nodata-based valid fallback.
        data[0, 0, 0] = float(nodata)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(data)
        if with_internal_mask:
            mask = np.full((height, width), 255, dtype=np.uint8)
            mask[0, 0] = 0
            mask[1, 1] = 0
            ds.write_mask(mask)

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


def _metadata_payload(*, feature_mode: str = "raw8") -> dict[str, Any]:
    if feature_mode == "raw8":
        assembled = "raw8_valid"
        in_channels = 9
        channel_semantics = [
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
    elif feature_mode == "raw8_idx3":
        assembled = "raw8_idx3_valid"
        in_channels = 12
        channel_semantics = [
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
    else:
        raise RuntimeError(f"Unsupported test feature_mode: {feature_mode}")

    payload = _base_manifest_top_level()
    payload.update(
        {
            "checkpoint_path": "/tmp/fake/best.ckpt",
            "feature_mode": feature_mode,
            "assembled_model_input": assembled,
            "in_channels": in_channels,
            "channel_semantics": channel_semantics,
            "valid_as_input_channel": True,
            "normalization": {
                "normalization_name": "per_band_robust_percentile",
                "stats_source": "train_norm_stats.json",
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


def _make_checkpoint_contract(tmp_path: Path, *, feature_mode: str = "raw8"):
    checkpoint_path = tmp_path / "best.ckpt"
    checkpoint_path.write_bytes(b"checkpoint-bytes")
    metadata_path = tmp_path / "checkpoint_metadata.json"
    write_manifest(metadata_path, _metadata_payload(feature_mode=feature_mode))
    return resolve_checkpoint_predict_contract(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=metadata_path,
    )


def test_read_predict_raster_metadata_happy_path(tmp_path: Path) -> None:
    raster_path = _write_raster(tmp_path / "scene.tif", count=8, nodata=-9999.0)

    meta = read_predict_raster_metadata(raster_path)

    assert meta.raster_path == raster_path
    assert meta.width == 8
    assert meta.height == 6
    assert meta.band_count == 8
    assert meta.crs == "EPSG:32637"
    assert len(meta.transform_gdal) == 6
    assert meta.dtype == "float32"
    assert meta.nodata == pytest.approx(-9999.0)


def test_resolve_valid_mask_prefers_internal_gdal_mask(tmp_path: Path) -> None:
    raster_path = _write_raster(
        tmp_path / "scene_masked.tif",
        count=8,
        nodata=-9999.0,
        with_internal_mask=True,
    )

    resolved = resolve_predict_valid_mask(raster_path=raster_path)

    assert resolved.source == "gdal_valid_data_mask"
    assert resolved.invalid_pixels == 2
    assert resolved.valid_pixels == (8 * 6 - 2)
    assert resolved.valid_mask[0, 0] == 0
    assert resolved.valid_mask[1, 1] == 0


def test_resolve_valid_mask_from_nodata_when_no_internal_mask(tmp_path: Path) -> None:
    raster_path = _write_raster(
        tmp_path / "scene_nodata.tif",
        count=8,
        nodata=-9999.0,
        with_internal_mask=False,
    )

    resolved = resolve_predict_valid_mask(raster_path=raster_path)

    assert resolved.source == "nodata_metadata"
    assert resolved.invalid_pixels == 1
    assert resolved.valid_mask[0, 0] == 0


def test_resolve_valid_mask_missing_semantics_raises(tmp_path: Path) -> None:
    raster_path = _write_raster(
        tmp_path / "scene_no_mask_no_nodata.tif",
        count=8,
        nodata=None,
        with_internal_mask=False,
    )

    with pytest.raises(ValidPolicyError, match="Unable to resolve valid mask semantics"):
        resolve_predict_valid_mask(raster_path=raster_path)


def test_resolve_valid_mask_uses_explicit_override_as_last_resort(tmp_path: Path) -> None:
    raster_path = _write_raster(
        tmp_path / "scene_override.tif",
        count=8,
        nodata=None,
        with_internal_mask=False,
    )
    override = np.ones((6, 8), dtype=np.uint8)
    override[2, 3] = 0

    resolved = resolve_predict_valid_mask(
        raster_path=raster_path,
        explicit_valid_mask_override=override,
    )

    assert resolved.source == "explicit_override"
    assert resolved.invalid_pixels == 1
    assert resolved.valid_mask[2, 3] == 0


def test_resolve_predict_raster_contract_happy_path(tmp_path: Path) -> None:
    checkpoint_contract = _make_checkpoint_contract(tmp_path, feature_mode="raw8")
    raster_path = _write_raster(tmp_path / "scene_ok.tif", count=8, nodata=-9999.0)

    resolved = resolve_predict_raster_contract(
        raster_path=raster_path,
        checkpoint_contract=checkpoint_contract,
    )

    assert resolved.checkpoint_compatible is True
    assert resolved.feature_mode == "raw8"
    assert resolved.assembled_model_input == "raw8_valid"
    assert resolved.in_channels == 9
    assert resolved.required_source_band_count == 8
    assert resolved.valid_mask_resolution.source == "nodata_metadata"


def test_resolve_predict_raster_contract_band_mismatch_raises(tmp_path: Path) -> None:
    checkpoint_contract = _make_checkpoint_contract(tmp_path, feature_mode="raw8")
    raster_path = _write_raster(tmp_path / "scene_bad_bands.tif", count=7, nodata=-9999.0)

    with pytest.raises(ChannelCountError, match="band_count"):
        resolve_predict_raster_contract(
            raster_path=raster_path,
            checkpoint_contract=checkpoint_contract,
        )


def test_missing_spatial_metadata_crs_raises_explicit_error(tmp_path: Path) -> None:
    raster_path = _write_raster(
        tmp_path / "scene_no_crs.tif",
        count=8,
        nodata=-9999.0,
        with_crs=False,
    )

    with pytest.raises(SpatialContractError, match="CRS"):
        read_predict_raster_metadata(raster_path)


def test_no_silent_fallback_missing_raster_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="does not exist"):
        read_predict_raster_metadata(tmp_path / "missing.tif")
