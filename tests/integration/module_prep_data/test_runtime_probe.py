"""Integration tests for 01_check_inputs runtime probe (Phase B).

Verifies that run_check_inputs_stage correctly drives rasterio/fiona to probe
real input files when raster_metadata=None and vector_metadata=None.

Coverage:
  - Happy path: stage succeeds end-to-end with real GeoTIFF + vector fixtures
  - Manifest reflects rasterio_fiona_probe_v1 mode and probed metadata
  - Stage fails gracefully (status="failed") on missing raster or vector
  - Probed metadata (band_count, CRS) appears correctly in manifest

These tests use tiny synthetic fixtures generated in tmp_path (no real
production data required).  They are skipped automatically if rasterio
or fiona is not installed.

Source references:
  - module_prep_data.md §6 (01_check_inputs runtime slice)
  - TESTING_STRATEGY.md §6 (integration tests), §10 (manifest contract)
  - MANIFEST_SCHEMAS.md §7.1 (check_inputs_manifest schema)
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("fiona")

from pathlib import Path  # noqa: E402

from ai_fields.module_prep_data.check_inputs import run_check_inputs_stage  # noqa: E402

_MINIMAL_CONFIG = {"feature_mode": "raw8"}


class TestRunCheckInputsRuntimeProbe:
    def test_stage_succeeds_with_real_files(
        self, tmp_path: Path, tiny_8band_raster_path: Path, tiny_vector_path: Path
    ) -> None:
        """Stage 01 succeeds end-to-end when rasterio/fiona probe is used."""
        result = run_check_inputs_stage(
            output_dir=tmp_path,
            run_id="probe_happy_path",
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            raster_metadata=None,
            vector_metadata=None,
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=True,
            metadata_sidecar_fallback_enabled=False,
        )
        assert result.success, f"Stage failed unexpectedly: {result.blocking_issues}"

    def test_manifest_records_probe_mode(
        self, tmp_path: Path, tiny_8band_raster_path: Path, tiny_vector_path: Path
    ) -> None:
        """Manifest config section records rasterio_fiona_probe_v1 and probe enabled."""
        result = run_check_inputs_stage(
            output_dir=tmp_path,
            run_id="probe_manifest_mode",
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            raster_metadata=None,
            vector_metadata=None,
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=True,
            metadata_sidecar_fallback_enabled=False,
        )
        with result.manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        assert manifest["config"]["runtime_probe_mode"] == "rasterio_fiona_probe_v1"
        assert manifest["config"]["runtime_probe_enabled"] is True

    def test_manifest_input_raster_has_probed_metadata(
        self, tmp_path: Path, tiny_8band_raster_path: Path, tiny_vector_path: Path
    ) -> None:
        """Probed raster metadata (band_count, CRS) appears in manifest input_raster."""
        result = run_check_inputs_stage(
            output_dir=tmp_path,
            run_id="probe_raster_meta",
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            raster_metadata=None,
            vector_metadata=None,
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=True,
            metadata_sidecar_fallback_enabled=False,
        )
        assert result.success
        with result.manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        assert manifest["input_raster"]["count"] == 8
        assert manifest["input_raster"]["crs"] == "EPSG:32637"

    def test_stage_fails_on_missing_raster(
        self, tmp_path: Path, tiny_vector_path: Path
    ) -> None:
        """Stage returns status=failed (not exception) when raster is missing."""
        result = run_check_inputs_stage(
            output_dir=tmp_path,
            run_id="probe_missing_raster",
            raster_path=tmp_path / "nonexistent.tif",
            vector_path=tiny_vector_path,
            raster_metadata=None,
            vector_metadata=None,
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=True,
            metadata_sidecar_fallback_enabled=False,
        )
        assert not result.success
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert len(result.blocking_issues) > 0

    def test_stage_fails_on_missing_vector(
        self, tmp_path: Path, tiny_8band_raster_path: Path
    ) -> None:
        """Stage returns status=failed (not exception) when vector is missing."""
        result = run_check_inputs_stage(
            output_dir=tmp_path,
            run_id="probe_missing_vector",
            raster_path=tiny_8band_raster_path,
            vector_path=tmp_path / "nonexistent.gpkg",
            raster_metadata=None,
            vector_metadata=None,
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=True,
            metadata_sidecar_fallback_enabled=False,
        )
        assert not result.success
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert len(result.blocking_issues) > 0

    def test_explicit_metadata_still_works(
        self, tmp_path: Path, tiny_8band_raster_path: Path, tiny_vector_path: Path
    ) -> None:
        """Backward compat: explicit metadata takes precedence over probing."""
        result = run_check_inputs_stage(
            output_dir=tmp_path,
            run_id="probe_explicit_meta",
            raster_path=tiny_8band_raster_path,
            vector_path=tiny_vector_path,
            raster_metadata={
                "band_count": 8, "crs": "EPSG:32637",
                "has_valid_mask": True, "nodata": None,
                "readable": True, "width": 16, "height": 16, "dtype": "uint16",
            },
            vector_metadata={
                "feature_count": 1, "geometry_types": ["Polygon"],
                "crs": "EPSG:32637", "readable": True,
            },
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=True,
            metadata_sidecar_fallback_enabled=False,
        )
        assert result.success
