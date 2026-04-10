"""Unit tests for module_prep_data input_probe (Phase B runtime slice).

Tests that probe_raster and probe_vector correctly extract minimal metadata
from real GeoTIFF / vector files, and raise ContractError on failures.

Coverage:
  - Happy path: probed metadata matches written values
  - Missing file: ContractError with "does not exist"
  - Unreadable (garbage) file: ContractError

Requires rasterio and fiona; all tests in this module are skipped
automatically if either library is absent.

Source references:
  - module_prep_data.md §6 (01_check_inputs runtime slice)
  - DATA_CONTRACT.md §3.1, §5, §6.2 (valid/NoData contract)
  - TESTING_STRATEGY.md §5 (unit test scope), §9 (negative tests)
"""

from __future__ import annotations

import pytest

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")

from pathlib import Path  # noqa: E402

from ai_fields.common.errors import ContractError  # noqa: E402
from ai_fields.module_prep_data.input_probe import probe_raster, probe_vector  # noqa: E402


# ---------------------------------------------------------------------------
# probe_raster
# ---------------------------------------------------------------------------


class TestProbeRaster:
    def test_returns_expected_metadata(self, tiny_8band_raster_path: Path) -> None:
        """probe_raster returns correct metadata for the synthetic 8-band fixture."""
        meta = probe_raster(tiny_8band_raster_path)

        assert meta["band_count"] == 8
        assert meta["width"] == 16
        assert meta["height"] == 16
        assert meta["crs"] == "EPSG:32637"
        assert meta["nodata"] == 0
        assert meta["dtype"] == "uint16"
        assert meta["readable"] is True

    def test_has_valid_mask_is_bool(self, tiny_8band_raster_path: Path) -> None:
        """has_valid_mask is always a bool (True here because nodata=0 is set)."""
        meta = probe_raster(tiny_8band_raster_path)
        assert isinstance(meta["has_valid_mask"], bool)
        # nodata=0 is set in the fixture, so a mask is implied
        assert meta["has_valid_mask"] is True

    def test_missing_file_raises_contract_error(self, tmp_path: Path) -> None:
        """ContractError is raised when the raster file does not exist."""
        with pytest.raises(ContractError, match="does not exist"):
            probe_raster(tmp_path / "nonexistent.tif")

    def test_garbage_file_raises_contract_error(self, tmp_path: Path) -> None:
        """ContractError is raised when the file is not a valid GeoTIFF."""
        bad = tmp_path / "garbage.tif"
        bad.write_bytes(b"not a geotiff")
        with pytest.raises(ContractError):
            probe_raster(bad)

    def test_directory_raises_contract_error(self, tmp_path: Path) -> None:
        """ContractError is raised when the path points to a directory."""
        with pytest.raises(ContractError, match="regular file"):
            probe_raster(tmp_path)


# ---------------------------------------------------------------------------
# probe_vector
# ---------------------------------------------------------------------------


class TestProbeVector:
    def test_returns_expected_metadata(self, tiny_vector_path: Path) -> None:
        """probe_vector returns correct metadata for the synthetic fixture."""
        meta = probe_vector(tiny_vector_path)

        assert meta["feature_count"] == 1
        assert meta["crs"] == "EPSG:32637"
        assert meta["readable"] is True
        assert "Polygon" in meta["geometry_types"]

    def test_geometry_types_is_sorted_list(self, tiny_vector_path: Path) -> None:
        """geometry_types is always a sorted list of strings."""
        meta = probe_vector(tiny_vector_path)
        assert isinstance(meta["geometry_types"], list)
        assert all(isinstance(t, str) for t in meta["geometry_types"])
        assert meta["geometry_types"] == sorted(meta["geometry_types"])

    def test_missing_file_raises_contract_error(self, tmp_path: Path) -> None:
        """ContractError is raised when the vector file does not exist."""
        with pytest.raises(ContractError, match="does not exist"):
            probe_vector(tmp_path / "nonexistent.gpkg")

    def test_garbage_file_raises_contract_error(self, tmp_path: Path) -> None:
        """ContractError is raised when the file is not a valid vector format."""
        bad = tmp_path / "garbage.gpkg"
        bad.write_bytes(b"not a vector file")
        with pytest.raises(ContractError):
            probe_vector(bad)

    def test_directory_raises_contract_error(self, tmp_path: Path) -> None:
        """ContractError is raised when the path points to a directory."""
        with pytest.raises(ContractError, match="regular file"):
            probe_vector(tmp_path)
