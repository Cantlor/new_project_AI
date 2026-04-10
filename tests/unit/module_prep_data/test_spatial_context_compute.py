"""Unit tests for spatial_context_compute (stage 02 runtime helper).

All tests use pytest.importorskip to skip gracefully when rasterio/geopandas
are unavailable.  Tests with real GeoTIFF I/O rely on fixtures defined in
conftest.py (tiny_8band_raster_path, tiny_vector_path).
"""

from __future__ import annotations

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("geopandas")

from ai_fields.module_prep_data.spatial_context_compute import (  # noqa: E402
    SPATIAL_COMPUTE_MODE,
    compute_spatial_context,
)
from ai_fields.common.errors import ContractError  # noqa: E402


class TestComputeSpatialContext:
    def test_returns_raster_crs(self, tiny_8band_raster_path, tiny_vector_path):
        result = compute_spatial_context(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            aoi_path=None,
            buffer_m=0.0,
        )
        assert "raster_crs" in result
        assert result["raster_crs"] is not None

    def test_returns_vector_crs(self, tiny_8band_raster_path, tiny_vector_path):
        result = compute_spatial_context(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            aoi_path=None,
            buffer_m=0.0,
        )
        assert "vector_crs" in result

    def test_effective_extent_bounds_without_aoi(self, tiny_8band_raster_path, tiny_vector_path):
        result = compute_spatial_context(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            aoi_path=None,
            buffer_m=0.0,
        )
        bounds = result["effective_extent_bounds"]
        assert isinstance(bounds, list)
        assert len(bounds) == 4
        # xmin < xmax and ymin < ymax
        assert bounds[0] < bounds[2]
        assert bounds[1] < bounds[3]

    def test_aoi_reprojected_is_none_without_aoi(self, tiny_8band_raster_path, tiny_vector_path):
        result = compute_spatial_context(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            aoi_path=None,
            buffer_m=0.0,
        )
        assert result["aoi_reprojected"] is None

    def test_spatial_compute_mode(self, tiny_8band_raster_path, tiny_vector_path):
        result = compute_spatial_context(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            aoi_path=None,
            buffer_m=0.0,
        )
        assert result["spatial_compute_mode"] == SPATIAL_COMPUTE_MODE

    def test_writes_vector_reprojected_artifact_when_output_dir_is_set(
        self, tiny_8band_raster_path, tiny_vector_path, tmp_path
    ):
        result = compute_spatial_context(
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            aoi_path=None,
            buffer_m=0.0,
            output_dir=tmp_path / "spatial",
        )
        assert result["vector_reprojected_path"] is not None
        assert (tmp_path / "spatial" / "vector_in_raster_crs.gpkg").exists()

    def test_missing_raster_raises_contract_error(self, tiny_vector_path, tmp_path):
        nonexistent = tmp_path / "no_such_file.tif"
        with pytest.raises(ContractError):
            compute_spatial_context(
                raster_path=nonexistent,
                vector_path=tiny_vector_path,
                aoi_path=None,
                buffer_m=0.0,
            )

    def test_missing_vector_raises_contract_error(self, tiny_8band_raster_path, tmp_path):
        nonexistent = tmp_path / "no_such_file.gpkg"
        with pytest.raises(ContractError):
            compute_spatial_context(
                raster_path=tiny_8band_raster_path,
                vector_path=nonexistent,
                aoi_path=None,
                buffer_m=0.0,
            )
