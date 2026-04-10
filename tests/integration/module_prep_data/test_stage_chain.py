"""Integration tests for module_prep_data stage chain.

Tests stage-to-stage provenance contract: verifies that output manifests from
one stage are correctly referenced in the provenance of the next stage.

Coverage (TESTING_STRATEGY.md §6, §6.2 prep_data → stubs contract):
  - Stage 01 (check_inputs) → Stage 02 (prepare_spatial_context) provenance link
  - Both stages succeed with valid in-memory metadata (no real GeoTIFF required)
  - Stage 02's manifest references stage 01's manifest via provenance.source_manifest_paths

These tests use in-memory metadata dicts (no filesystem raster/vector reads).
runtime_probe_enabled=False disables filesystem existence checks so that
synthetic paths can be used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.module_prep_data.check_inputs import run_check_inputs_stage
from ai_fields.module_prep_data.prepare_spatial_context import (
    run_prepare_spatial_context_stage,
)

# ---------------------------------------------------------------------------
# Shared metadata fixtures
# ---------------------------------------------------------------------------

_RASTER_META = {
    "crs": "EPSG:32637",
    "band_count": 8,
    "width": 512,
    "height": 512,
    "dtype": "uint16",
    "nodata": 0,
    "readable": True,
}

_VECTOR_META = {
    "crs": "EPSG:32637",
    "feature_count": 10,
    "geometry_types": ["Polygon"],
    "readable": True,
}

_MINIMAL_CONFIG = {"feature_mode": "raw8"}


class TestStage01ToStage02Chain:
    """Verify that stage 01 → stage 02 passes provenance correctly."""

    def test_stage_01_succeeds_with_metadata(self, tmp_path: Path) -> None:
        """Stage 01 completes successfully with in-memory metadata."""
        result = run_check_inputs_stage(
            output_dir=tmp_path,
            run_id="integration_test_run",
            raster_path="/data/scene.tif",
            vector_path="/data/fields.gpkg",
            raster_metadata=_RASTER_META,
            vector_metadata=_VECTOR_META,
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=False,
            metadata_sidecar_fallback_enabled=False,
        )
        assert result.success, f"Stage 01 failed: {result.blocking_issues}"

    def test_stage_02_succeeds_after_stage_01(self, tmp_path: Path) -> None:
        """Stage 02 completes successfully when chained after stage 01."""
        stage01_dir = tmp_path / "stage01"
        stage01_dir.mkdir()

        result_01 = run_check_inputs_stage(
            output_dir=stage01_dir,
            run_id="integration_test_run",
            raster_path="/data/scene.tif",
            vector_path="/data/fields.gpkg",
            raster_metadata=_RASTER_META,
            vector_metadata=_VECTOR_META,
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=False,
            metadata_sidecar_fallback_enabled=False,
        )
        assert result_01.success

        stage02_dir = tmp_path / "stage02"
        stage02_dir.mkdir()

        result_02 = run_prepare_spatial_context_stage(
            output_dir=stage02_dir,
            run_id="integration_test_run",
            raster_path="/data/scene.tif",
            vector_path="/data/fields.gpkg",
            raster_metadata=_RASTER_META,
            vector_metadata=_VECTOR_META,
            config=_MINIMAL_CONFIG,
            source_manifest_path=result_01.manifest_path,
            runtime_compute_enabled=False,
        )
        assert result_02.success, f"Stage 02 failed: {result_02.blocking_issues}"

    def test_stage_02_manifest_references_stage_01_manifest(self, tmp_path: Path) -> None:
        """Stage 02's manifest.provenance.source_manifest_paths contains stage 01's manifest.

        This verifies the cross-stage provenance contract from MANIFEST_SCHEMAS.md §3.3
        and TESTING_STRATEGY.md §10.3 (forensic-ready minimum).
        """
        stage01_dir = tmp_path / "stage01"
        stage01_dir.mkdir()

        result_01 = run_check_inputs_stage(
            output_dir=stage01_dir,
            run_id="integration_test_run",
            raster_path="/data/scene.tif",
            vector_path="/data/fields.gpkg",
            raster_metadata=_RASTER_META,
            vector_metadata=_VECTOR_META,
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=False,
            metadata_sidecar_fallback_enabled=False,
        )
        assert result_01.success

        stage02_dir = tmp_path / "stage02"
        stage02_dir.mkdir()

        result_02 = run_prepare_spatial_context_stage(
            output_dir=stage02_dir,
            run_id="integration_test_run",
            raster_path="/data/scene.tif",
            vector_path="/data/fields.gpkg",
            raster_metadata=_RASTER_META,
            vector_metadata=_VECTOR_META,
            config=_MINIMAL_CONFIG,
            source_manifest_path=result_01.manifest_path,
            runtime_compute_enabled=False,
        )
        assert result_02.success

        # Read stage 02's manifest and verify provenance
        with result_02.manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)

        source_paths = manifest["provenance"]["source_manifest_paths"]
        assert len(source_paths) == 1, (
            f"Expected 1 source manifest path, got {source_paths}"
        )
        assert str(result_01.manifest_path) in source_paths, (
            f"Stage 01 manifest path not in stage 02 provenance: {source_paths}"
        )

    def test_stage_02_manifest_has_required_schema_fields(self, tmp_path: Path) -> None:
        """Stage 02's manifest contains mandatory top-level fields (MANIFEST_SCHEMAS.md §3.1)."""
        stage01_dir = tmp_path / "stage01"
        stage01_dir.mkdir()

        result_01 = run_check_inputs_stage(
            output_dir=stage01_dir,
            run_id="integration_test_run",
            raster_path="/data/scene.tif",
            vector_path="/data/fields.gpkg",
            raster_metadata=_RASTER_META,
            vector_metadata=_VECTOR_META,
            config=_MINIMAL_CONFIG,
            runtime_probe_enabled=False,
            metadata_sidecar_fallback_enabled=False,
        )

        stage02_dir = tmp_path / "stage02"
        stage02_dir.mkdir()

        result_02 = run_prepare_spatial_context_stage(
            output_dir=stage02_dir,
            run_id="integration_test_run",
            raster_path="/data/scene.tif",
            vector_path="/data/fields.gpkg",
            raster_metadata=_RASTER_META,
            vector_metadata=_VECTOR_META,
            config=_MINIMAL_CONFIG,
            source_manifest_path=result_01.manifest_path,
            runtime_compute_enabled=False,
        )

        with result_02.manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)

        required = [
            "schema_name", "schema_version", "module_name", "data_contract_version",
            "run_id", "created_at_utc", "status",
        ]
        for field in required:
            assert field in manifest, f"Required manifest field missing: {field!r}"

        assert manifest["schema_name"] == "prep_data.aoi_manifest"
        assert manifest["module_name"] == "module_prep_data"
        assert manifest["data_contract_version"] == "v1"
        assert manifest["run_id"] == "integration_test_run"
        assert manifest["status"] == "success"
