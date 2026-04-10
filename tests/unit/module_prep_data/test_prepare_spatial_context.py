"""Unit tests for module_prep_data 02_prepare_spatial_context stage skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.module_prep_data import prepare_spatial_context as stage
from ai_fields.module_prep_data.schemas import PrepDataConfig


def _valid_raster_meta(**overrides):
    data = {
        "crs": "EPSG:32642",
        "width": 1024,
        "height": 1024,
    }
    data.update(overrides)
    return data


def _valid_vector_meta(**overrides):
    data = {
        "crs": "EPSG:32642",
        "feature_count": 10,
        "geometry_types": ["Polygon"],
    }
    data.update(overrides)
    return data


def _valid_aoi_meta(**overrides):
    data = {
        "crs": "EPSG:32642",
        "feature_count": 1,
        "geometry_types": ["Polygon"],
        "bounds": [100.0, 200.0, 400.0, 700.0],
    }
    data.update(overrides)
    return data


class TestRunPrepareSpatialContextStage:
    def test_happy_path_without_aoi(self, tmp_path):
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-100",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            runtime_compute_enabled=False,
        )

        assert result.success is True
        assert result.status == "success"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.spatial_context_mode == "full_raster"
        assert result.aoi_present is False
        assert result.resolved_buffer_m is None

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["schema_name"] == "prep_data.aoi_manifest"
        assert manifest_data["stage_name"] == "02_prepare_spatial_context"
        assert manifest_data["status"] == "success"
        assert manifest_data["aoi_present"] is False
        assert manifest_data["aoi_source_path"] is None
        assert manifest_data["buffer_m"] is None
        assert manifest_data["spatial_context_mode"] == "full_raster"
        assert manifest_data["checks"]["contract_checks_passed"] is True
        assert manifest_data["checks"]["crs_compatible"] is True
        assert manifest_data["config"]["input_refs_source"] == "stage_args_transitional"

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["schema_name"] == "prep_data.summary"
        assert summary_data["status"] == "success"
        assert summary_data["spatial_context_mode"] == "full_raster"
        assert summary_data["aoi_present"] is False

    def test_happy_path_with_aoi_and_buffer_policy(self, tmp_path):
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-101",
            config={
                "feature_mode": "raw8",
                "aoi": {
                    "enabled": True,
                    "aoi_path": "aoi.gpkg",
                    "buffer_m": 45.0,
                },
            },
            raster_path="input.tif",
            vector_path="labels.gpkg",
            aoi_path="aoi.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            aoi_metadata=_valid_aoi_meta(),
            source_manifest_path=tmp_path / "check_inputs_manifest.json",
            runtime_compute_enabled=False,
        )

        assert result.success is True
        assert result.status == "success"
        assert result.spatial_context_mode == "aoi_limited"
        assert result.aoi_present is True
        assert result.aoi_policy_enabled is True
        assert result.resolved_buffer_m == 45.0

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["aoi_present"] is True
        assert manifest_data["aoi_source_path"] == "aoi.gpkg"
        assert manifest_data["aoi_source_crs"] == "EPSG:32642"
        assert manifest_data["aoi_target_crs"] == "EPSG:32642"
        assert manifest_data["aoi_reprojected"] is None  # None in skeleton mode; populated by runtime compute
        assert manifest_data["buffer_m"] == 45.0
        assert manifest_data["effective_extent_bounds"] is None
        assert manifest_data["aoi_bounds_metadata_hint"] == [100.0, 200.0, 400.0, 700.0]
        assert manifest_data["spatial_context_mode"] == "aoi_limited"
        assert manifest_data["provenance"]["source_manifest_paths"] == [
            str(tmp_path / "check_inputs_manifest.json")
        ]

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["source_manifest_path"] == str(tmp_path / "check_inputs_manifest.json")

    def test_transformable_crs_mismatch_is_allowed(self, tmp_path):
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-102",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            raster_metadata=_valid_raster_meta(crs="EPSG:32642"),
            vector_metadata=_valid_vector_meta(crs="EPSG:3857"),
            runtime_compute_enabled=False,
        )

        assert result.success is True
        assert result.status == "success"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.checks["contract_checks_passed"] is True
        assert result.checks["crs_compatible"] is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "success"
        assert manifest_data["resolved_contract"]["spatial"]["vector_reprojection_required"] is True
        assert manifest_data["resolved_contract"]["spatial"]["vector_reprojection_applied"] is None
        assert manifest_data["diagnostics"]["errors"] == []
        assert manifest_data["diagnostics"]["warnings"]

    def test_invalid_aoi_contract_returns_failed_result(self, tmp_path):
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-103",
            config={"feature_mode": "raw8"},  # AOI disabled by default
            raster_path="input.tif",
            vector_path="labels.gpkg",
            aoi_path="aoi.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            aoi_metadata=_valid_aoi_meta(),
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.checks["aoi_contract_consistent"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "config": {"feature_mode": "raw8"},
                "config_path": "prep_data.yaml",
            },
            {
                # neither config nor config_path
            },
        ],
    )
    def test_requires_exactly_one_of_config_or_config_path(self, tmp_path, kwargs):
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-103b",
            raster_path="input.tif",
            vector_path="labels.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            **kwargs,
        )
        assert result.status == "failed"
        assert result.error_type == "ContractError"

    @pytest.mark.parametrize(
        "bad_bounds",
        ["1,2,3,4", [1, 2, 3], [1, 2, "x", 4], [1, 2, True, 4]],
    )
    def test_invalid_aoi_bounds_typing_returns_failed_result(self, tmp_path, bad_bounds):
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-103c",
            config={
                "feature_mode": "raw8",
                "aoi": {"enabled": True, "aoi_path": "aoi.gpkg", "buffer_m": 30.0},
            },
            raster_path="input.tif",
            vector_path="labels.gpkg",
            aoi_path="aoi.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            aoi_metadata=_valid_aoi_meta(bounds=bad_bounds),
        )
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["contract_checks_passed"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "config": {"feature_mode": "raw8"},
                "source_manifest_path": 123,  # invalid path-like type
            },
            {
                "config": {
                    "feature_mode": "raw8",
                    "aoi": {"enabled": True, "aoi_path": "aoi.gpkg", "buffer_m": 30.0},
                },
                "aoi_path": "aoi.gpkg",
                "aoi_metadata": _valid_aoi_meta(bounds=[1, 2, "bad", 4]),
            },
        ],
    )
    def test_no_valueerror_or_typeerror_leakage(self, tmp_path, kwargs):
        base_kwargs = {
            "output_dir": tmp_path / "run_artifacts",
            "run_id": "run-103d",
            "raster_path": "input.tif",
            "vector_path": "labels.gpkg",
            "raster_metadata": _valid_raster_meta(),
            "vector_metadata": _valid_vector_meta(),
        }
        base_kwargs.update(kwargs)
        try:
            result = stage.run_prepare_spatial_context_stage(**base_kwargs)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive guard
            pytest.fail(f"Raw exception leaked from stage layer: {type(exc).__name__}: {exc}")
        else:
            assert result.status == "failed"

    def test_stage_reuses_existing_layers_via_composition(self, tmp_path, monkeypatch):
        called = {
            "build_config": 0,
            "validate_input_paths_contract": 0,
            "validate_crs_contract": 0,
            "write_manifest": 0,
            "write_summary": 0,
        }

        def fake_build_config(raw):
            called["build_config"] += 1
            assert raw["feature_mode"] == "raw8"
            return PrepDataConfig(feature_mode="raw8")

        def fake_validate_input_paths_contract(**kwargs):
            called["validate_input_paths_contract"] += 1
            return {
                "raster_path": Path(kwargs["raster_path"]),
                "vector_path": Path(kwargs["vector_path"]),
                "aoi_path": None,
            }

        def fake_validate_crs_contract(**kwargs):
            called["validate_crs_contract"] += 1
            assert kwargs["raster_metadata"]["crs"] == "EPSG:32642"
            return {
                "raster_crs": "EPSG:32642",
                "vector_crs": "EPSG:32642",
                "aoi_crs": None,
            }

        def fake_write_manifest(path, payload):
            called["write_manifest"] += 1
            assert payload["schema_name"] == "prep_data.aoi_manifest"
            assert payload["status"] == "success"

        def fake_write_summary(path, payload):
            called["write_summary"] += 1
            assert payload["schema_name"] == "prep_data.summary"
            assert payload["status"] == "success"

        monkeypatch.setattr(stage.prep_data_config, "build_config", fake_build_config)
        monkeypatch.setattr(
            stage.prep_data_validators,
            "validate_input_paths_contract",
            fake_validate_input_paths_contract,
        )
        monkeypatch.setattr(
            stage.prep_data_validators,
            "validate_crs_contract",
            fake_validate_crs_contract,
        )
        monkeypatch.setattr(stage, "write_manifest", fake_write_manifest)
        monkeypatch.setattr(stage, "write_summary", fake_write_summary)

        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-104",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            runtime_compute_enabled=False,
        )

        assert result.status == "success"
        assert called["build_config"] == 1
        assert called["validate_input_paths_contract"] == 1
        assert called["validate_crs_contract"] == 1
        assert called["write_manifest"] == 1
        assert called["write_summary"] == 1

    def test_runtime_compute_writes_reprojected_vector_artifact(
        self,
        tmp_path,
        tiny_8band_raster_path,
        tiny_vector_path,
    ):
        pytest.importorskip("rasterio")
        pytest.importorskip("geopandas")
        # tiny_vector_path fixture is EPSG:32637. We intentionally report a different
        # vector CRS in metadata to ensure Stage 02 contract allows transformable mismatch
        # and runtime compute materializes internal reprojected artifact.
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-105",
            config={"feature_mode": "raw8"},
            raster_path=str(tiny_8band_raster_path),
            vector_path=str(tiny_vector_path),
            raster_metadata=_valid_raster_meta(crs="EPSG:32637"),
            vector_metadata=_valid_vector_meta(crs="EPSG:3857"),
            runtime_compute_enabled=True,
        )
        assert result.success is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        vector_output_path = manifest_data["vector_output_path"]
        assert isinstance(vector_output_path, str)
        assert Path(vector_output_path).exists()
        assert manifest_data["vector_target_crs"] == "EPSG:32637"
        assert manifest_data["vector_reprojected"] in (True, False)


class TestSpatialContextAOIFields:
    """Tests for new aoi_output_path / effective_extent_bounds fields and
    the populated inputs/outputs artifact lists in the aoi_manifest."""

    def test_no_aoi_skeleton_mode_fields_are_none(self, tmp_path):
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-aoi-001",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            runtime_compute_enabled=False,
        )
        assert result.success is True
        assert result.aoi_output_path is None
        assert result.effective_extent_bounds is None

    def test_aoi_present_skeleton_mode_fields_are_none(self, tmp_path):
        # Even with AOI, skeleton mode (no compute) should not write any file.
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-aoi-002",
            config={
                "feature_mode": "raw8",
                "aoi": {"enabled": True, "aoi_path": "aoi.gpkg", "buffer_m": 30.0},
            },
            raster_path="input.tif",
            vector_path="labels.gpkg",
            aoi_path="aoi.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            aoi_metadata=_valid_aoi_meta(),
            runtime_compute_enabled=False,
        )
        assert result.success is True
        assert result.aoi_output_path is None
        assert result.effective_extent_bounds is None

    def test_no_aoi_manifest_artifacts_are_empty(self, tmp_path):
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-aoi-003",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            runtime_compute_enabled=False,
        )
        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["inputs"]["artifacts"] == []
        assert manifest_data["outputs"]["artifacts"] == []

    def test_aoi_present_skeleton_mode_manifest_artifacts_both_empty(self, tmp_path):
        # inputs.artifacts and outputs.artifacts are populated only during runtime compute.
        # In skeleton mode (runtime_compute_enabled=False), both must remain empty even
        # when AOI is provided — no actual file I/O occurred.
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-aoi-004",
            config={
                "feature_mode": "raw8",
                "aoi": {"enabled": True, "aoi_path": "aoi.gpkg", "buffer_m": 30.0},
            },
            raster_path="input.tif",
            vector_path="labels.gpkg",
            aoi_path="aoi.gpkg",
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
            aoi_metadata=_valid_aoi_meta(),
            runtime_compute_enabled=False,
        )
        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["inputs"]["artifacts"] == []
        assert manifest_data["outputs"]["artifacts"] == []

    def test_runtime_compute_no_aoi_outputs_artifact_has_vector(
        self,
        tmp_path,
        tiny_8band_raster_path,
        tiny_vector_path,
    ):
        pytest.importorskip("rasterio")
        pytest.importorskip("geopandas")
        result = stage.run_prepare_spatial_context_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-aoi-005",
            config={"feature_mode": "raw8"},
            raster_path=str(tiny_8band_raster_path),
            vector_path=str(tiny_vector_path),
            raster_metadata=_valid_raster_meta(crs="EPSG:32637"),
            vector_metadata=_valid_vector_meta(crs="EPSG:3857"),
            runtime_compute_enabled=True,
        )
        assert result.success is True
        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        outputs_artifacts = manifest_data["outputs"]["artifacts"]
        roles = [a["role"] for a in outputs_artifacts]
        assert "vector_in_raster_crs" in roles
        # No AOI → no aoi_resolved_in_raster_crs artifact
        assert "aoi_resolved_in_raster_crs" not in roles
        assert manifest_data["inputs"]["artifacts"] == []
