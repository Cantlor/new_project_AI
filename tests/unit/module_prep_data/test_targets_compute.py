"""Unit tests for targets_compute (stage 04 runtime helper).

Covers: extent values ⊆ {0,1,255}, boundary values ⊆ {0,1,2},
distance ≥ 0, shape consistency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

rasterio = pytest.importorskip("rasterio")
np = pytest.importorskip("numpy")
pytest.importorskip("geopandas")
pytest.importorskip("scipy")

from ai_fields.module_prep_data.targets_compute import (  # noqa: E402
    TARGETS_COMPUTE_MODE,
    compute_and_save_targets,
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


class TestComputeAndSaveTargets:
    def test_output_files_created(self, tiny_8band_raster_path, tiny_vector_path, tmp_path):
        result = compute_and_save_targets(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            output_dir=tmp_path,
        )
        import os
        for key in ("extent_path", "boundary_path", "boundary_raw_path", "distance_path", "valid_path"):
            assert result[key] is not None
            assert os.path.exists(result[key]), f"{key} not created: {result[key]}"

    def test_extent_values_subset(self, tiny_8band_raster_path, tiny_vector_path, tmp_path):
        result = compute_and_save_targets(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            output_dir=tmp_path,
        )
        with rasterio.open(result["extent_path"]) as ds:
            data = ds.read(1)
        unique = set(int(v) for v in np.unique(data))
        assert unique.issubset({0, 1, 255}), f"extent has unexpected values: {unique}"

    def test_boundary_values_subset(self, tiny_8band_raster_path, tiny_vector_path, tmp_path):
        result = compute_and_save_targets(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            output_dir=tmp_path,
        )
        with rasterio.open(result["boundary_path"]) as ds:
            data = ds.read(1)
        unique = set(int(v) for v in np.unique(data))
        assert unique.issubset({0, 1, 2}), f"boundary has unexpected values: {unique}"

    def test_distance_non_negative(self, tiny_8band_raster_path, tiny_vector_path, tmp_path):
        result = compute_and_save_targets(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            output_dir=tmp_path,
        )
        with rasterio.open(result["distance_path"]) as ds:
            data = ds.read(1).astype(np.float32)
        assert float(data.min()) >= 0.0, f"distance contains negative values: {data.min()}"

    def test_shape_consistency(self, tiny_8band_raster_path, tiny_vector_path, tmp_path):
        result = compute_and_save_targets(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            output_dir=tmp_path,
        )
        with rasterio.open(tiny_8band_raster_path) as src:
            ref_shape = (src.height, src.width)
        for key in ("extent_path", "boundary_path", "distance_path", "valid_path"):
            with rasterio.open(result[key]) as ds:
                actual = (ds.height, ds.width)
            assert actual == ref_shape, f"{key} shape {actual} != raster shape {ref_shape}"

    def test_targets_compute_mode(self, tiny_8band_raster_path, tiny_vector_path, tmp_path):
        result = compute_and_save_targets(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            output_dir=tmp_path,
        )
        assert result.get("targets_compute_mode") == TARGETS_COMPUTE_MODE

    def test_extent_preserves_spatial_metadata_and_uses_uint8(self, tiny_8band_raster_path, tiny_vector_path, tmp_path):
        result = compute_and_save_targets(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            output_dir=tmp_path,
        )
        with rasterio.open(tiny_8band_raster_path) as src:
            expected_crs = src.crs
            expected_transform = src.transform
            expected_shape = (src.height, src.width)
        with rasterio.open(result["extent_path"]) as extent_ds:
            assert extent_ds.dtypes[0] == "uint8"
            assert extent_ds.nodata is None
            assert extent_ds.crs == expected_crs
            assert extent_ds.transform == expected_transform
            assert (extent_ds.height, extent_ds.width) == expected_shape

    def test_semantic_uint8_targets_do_not_inherit_large_source_nodata(self, tiny_vector_path, tmp_path):
        raster_path = _make_large_nodata_raster(tmp_path / "raster_fixtures" / "large_nodata.tif")
        result = compute_and_save_targets(
            raster_path=raster_path,
            vector_path=tiny_vector_path,
            output_dir=tmp_path / "targets",
        )
        semantic_keys = ("extent_path", "boundary_path", "boundary_raw_path", "valid_path")
        for key in semantic_keys:
            with rasterio.open(result[key]) as ds:
                assert ds.dtypes[0] == "uint8"
                assert ds.nodata is None, f"{key} must not inherit source nodata into uint8 output."
