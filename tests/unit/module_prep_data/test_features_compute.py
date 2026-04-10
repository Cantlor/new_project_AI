"""Unit tests for features_compute (stage 03 runtime helper).

Covers: valid mask shape/dtype, raw8 stack (8,H,W), raw8_idx3 stack (11,H,W),
NDVI in [-1,1], output GeoTIFF CRS matches input.

All tests use pytest.importorskip for rasterio dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

rasterio = pytest.importorskip("rasterio")
np = pytest.importorskip("numpy")

from ai_fields.module_prep_data.features_compute import (  # noqa: E402
    FEATURES_COMPUTE_MODE,
    CHANNEL_SEMANTICS,
    build_feature_stack,
    compute_valid_mask,
    compute_and_save_features,
)
from ai_fields.common.errors import ContractError  # noqa: E402


def _make_large_nodata_raster(path: Path) -> Path:
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(
        west=599800.0, south=4399800.0, east=600200.0, north=4400200.0,
        width=16, height=16,
    )
    data = np.full((8, 16, 16), fill_value=1000, dtype=np.uint32)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=16, width=16, count=8,
        dtype="uint32",
        crs=CRS.from_epsg(32637),
        transform=transform,
        nodata=65536,
    ) as ds:
        ds.write(data)
    return path


class TestComputeValidMask:
    def test_valid_mask_shape(self, tiny_8band_raster_path):
        with rasterio.open(tiny_8band_raster_path) as ds:
            mask = compute_valid_mask(ds)
        assert mask.ndim == 2
        assert mask.dtype == "uint8"

    def test_valid_mask_values_binary(self, tiny_8band_raster_path):
        with rasterio.open(tiny_8band_raster_path) as ds:
            mask = compute_valid_mask(ds)
        unique = set(int(v) for v in np.unique(mask))
        assert unique.issubset({0, 1}), f"Non-binary values in valid mask: {unique}"


class TestBuildFeatureStack:
    def test_raw8_shape(self, tiny_8band_raster_path):
        with rasterio.open(tiny_8band_raster_path) as ds:
            stack = build_feature_stack(ds, feature_mode="raw8")
        assert stack.shape[0] == 8, f"Expected 8 channels, got {stack.shape[0]}"
        assert stack.dtype == np.float32

    def test_raw8_idx3_shape(self, tiny_8band_raster_path):
        with rasterio.open(tiny_8band_raster_path) as ds:
            stack = build_feature_stack(ds, feature_mode="raw8_idx3")
        assert stack.shape[0] == 11, f"Expected 11 channels, got {stack.shape[0]}"
        assert stack.dtype == np.float32

    def test_ndvi_in_valid_range(self, tiny_8band_raster_path):
        """NDVI (channel index 8 in raw8_idx3) must be in [-1, 1]."""
        with rasterio.open(tiny_8band_raster_path) as ds:
            stack = build_feature_stack(ds, feature_mode="raw8_idx3")
        ndvi = stack[8]  # 9th channel (0-indexed)
        assert float(ndvi.min()) >= -1.0 - 1e-5, f"NDVI below -1: {ndvi.min()}"
        assert float(ndvi.max()) <= 1.0 + 1e-5, f"NDVI above 1: {ndvi.max()}"


class TestComputeAndSaveFeatures:
    def test_output_files_created(self, tiny_8band_raster_path, tmp_path):
        result = compute_and_save_features(
            raster_path=tiny_8band_raster_path,
            output_dir=tmp_path,
            feature_mode="raw8",
        )
        assert result["img_path"] is not None
        assert result["valid_path"] is not None
        import os
        assert os.path.exists(result["img_path"])
        assert os.path.exists(result["valid_path"])

    def test_output_crs_matches_input(self, tiny_8band_raster_path, tmp_path):
        result = compute_and_save_features(
            raster_path=tiny_8band_raster_path,
            output_dir=tmp_path,
            feature_mode="raw8",
        )
        with rasterio.open(tiny_8band_raster_path) as src:
            expected_crs = src.crs
        with rasterio.open(result["img_path"]) as out:
            assert out.crs == expected_crs

    def test_valid_output_uint8_binary_and_no_nodata(self, tiny_8band_raster_path, tmp_path):
        result = compute_and_save_features(
            raster_path=tiny_8band_raster_path,
            output_dir=tmp_path,
            feature_mode="raw8",
        )
        with rasterio.open(tiny_8band_raster_path) as src:
            expected_crs = src.crs
            expected_transform = src.transform
            expected_shape = (src.height, src.width)
        with rasterio.open(result["valid_path"]) as valid_ds:
            assert valid_ds.count == 1
            assert valid_ds.dtypes[0] == "uint8"
            assert valid_ds.nodata is None
            assert valid_ds.crs == expected_crs
            assert valid_ds.transform == expected_transform
            arr = valid_ds.read(1)
            assert arr.shape == expected_shape
            unique = set(int(v) for v in np.unique(arr))
            assert unique.issubset({0, 1}), f"valid.tif must be binary 0/1, got {unique}"

    def test_valid_output_does_not_inherit_large_source_nodata(self, tmp_path):
        raster_path = _make_large_nodata_raster(tmp_path / "raster_fixtures" / "large_nodata.tif")
        result = compute_and_save_features(
            raster_path=raster_path,
            output_dir=tmp_path / "out",
            feature_mode="raw8",
        )
        with rasterio.open(result["valid_path"]) as valid_ds:
            assert valid_ds.dtypes[0] == "uint8"
            assert valid_ds.nodata is None

    def test_raw8_channel_count(self, tiny_8band_raster_path, tmp_path):
        result = compute_and_save_features(
            raster_path=tiny_8band_raster_path,
            output_dir=tmp_path / "raw8",
            feature_mode="raw8",
        )
        assert result["feature_channel_count"] == 8

    def test_raw8_idx3_channel_count(self, tiny_8band_raster_path, tmp_path):
        result = compute_and_save_features(
            raster_path=tiny_8band_raster_path,
            output_dir=tmp_path / "raw8_idx3",
            feature_mode="raw8_idx3",
        )
        assert result["feature_channel_count"] == 11

    def test_channel_semantics_returned(self, tiny_8band_raster_path, tmp_path):
        result = compute_and_save_features(
            raster_path=tiny_8band_raster_path,
            output_dir=tmp_path,
            feature_mode="raw8",
        )
        assert isinstance(result["channel_semantics"], list)
        assert len(result["channel_semantics"]) == 8

    def test_features_compute_mode(self, tiny_8band_raster_path, tmp_path):
        result = compute_and_save_features(
            raster_path=tiny_8band_raster_path,
            output_dir=tmp_path,
            feature_mode="raw8",
        )
        assert result["features_compute_mode"] == FEATURES_COMPUTE_MODE
