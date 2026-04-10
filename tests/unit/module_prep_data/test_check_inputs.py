"""Unit tests for module_prep_data 01_check_inputs stage skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.module_prep_data import check_inputs as stage
from ai_fields.module_prep_data.schemas import PrepDataConfig


def _valid_raster_meta(**overrides):
    data = {
        "band_count": 8,
        "crs": "EPSG:32642",
        "has_valid_mask": True,
        "nodata": None,
        "readable": True,
        "width": 1024,
        "height": 1024,
        "dtype": "uint16",
    }
    data.update(overrides)
    return data


def _valid_vector_meta(**overrides):
    data = {
        "feature_count": 10,
        "geometry_types": ["Polygon"],
        "crs": "EPSG:32642",
        "readable": True,
    }
    data.update(overrides)
    return data


def _write_dummy_file(path: Path, content: bytes = b"x") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _make_input_files(tmp_path: Path, *, with_aoi: bool = False) -> dict[str, str]:
    paths = {
        "raster_path": _write_dummy_file(tmp_path / "input.tif", b"raster"),
        "vector_path": _write_dummy_file(tmp_path / "labels.gpkg", b"vector"),
    }
    if with_aoi:
        paths["aoi_path"] = _write_dummy_file(tmp_path / "aoi.gpkg", b"aoi")
    return paths


def _write_sidecar(path_like: str | Path, payload: dict[str, object]) -> str:
    p = Path(path_like)
    sidecar = p.with_name(f"{p.name}.meta.json")
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return str(sidecar)


class TestRunCheckInputsStage:
    def test_happy_path_writes_manifest_and_summary(self, tmp_path):
        input_paths = _make_input_files(tmp_path)
        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-001",
            config={"feature_mode": "raw8"},
            raster_path=input_paths["raster_path"],
            vector_path=input_paths["vector_path"],
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
        )

        assert result.success is True
        assert result.status == "success"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.blocking_issues == ()
        assert result.checks["contract_checks_passed"] is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["schema_name"] == "prep_data.check_inputs_manifest"
        assert manifest_data["status"] == "success"
        assert manifest_data["stage_name"] == "01_check_inputs"
        assert manifest_data["checks"]["band_count_ok"] is True
        assert manifest_data["checks"]["raster_readable"] is True
        assert manifest_data["checks"]["vector_readable"] is True
        assert manifest_data["checks"]["aoi_readable"] is None
        assert manifest_data["config"]["input_refs_source"] == "stage_args_transitional"
        assert "provenance" in manifest_data
        assert "inputs" in manifest_data
        assert "outputs" in manifest_data
        assert "resolved_contract" in manifest_data
        assert "runtime" in manifest_data
        assert manifest_data["provenance"]["source_run_ids"] == []
        assert manifest_data["inputs"]["artifacts"] == []
        assert manifest_data["outputs"]["artifacts"] == []

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["schema_name"] == "prep_data.summary"
        assert summary_data["status"] == "success"
        assert summary_data["contract_checks_passed"] is True
        assert summary_data["input_refs_source"] == "stage_args_transitional"

    def test_supports_config_path(self, tmp_path):
        input_paths = _make_input_files(tmp_path)
        cfg_path = tmp_path / "prep_data.yaml"
        cfg_path.write_text("feature_mode: raw8\n", encoding="utf-8")
        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-002",
            config_path=cfg_path,
            raster_path=input_paths["raster_path"],
            vector_path=input_paths["vector_path"],
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
        )
        assert result.status == "success"
        assert result.success is True

    def test_invalid_config_returns_failed_result_without_raw_traceback(self, tmp_path):
        input_paths = _make_input_files(tmp_path)
        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-003",
            config={"patches": {"patch_size": 512}},  # feature_mode missing
            raster_path=input_paths["raster_path"],
            vector_path=input_paths["vector_path"],
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error_type == "FeatureModeError"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert len(result.blocking_issues) >= 1

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "failed"
        assert manifest_data["diagnostics"]["errors"]
        assert manifest_data["checks"]["band_count_ok"] is None
        assert manifest_data["checks"]["crs_compatible"] is None
        assert manifest_data["checks"]["geometry_validity_ok"] is None
        assert manifest_data["checks"]["nodata_interpretation_resolved"] is None

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["status"] == "failed"
        assert summary_data["blocking_issues"]

    def test_validator_failure_returns_failed_result(self, tmp_path):
        input_paths = _make_input_files(tmp_path)
        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-004",
            config={"feature_mode": "raw8"},
            raster_path=input_paths["raster_path"],
            vector_path=input_paths["vector_path"],
            raster_metadata=_valid_raster_meta(band_count=7),  # invalid by contract
            vector_metadata=_valid_vector_meta(),
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error_type is not None
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.checks["contract_checks_passed"] is False
        assert result.checks["band_count_ok"] is False
        assert result.checks["crs_compatible"] is None
        assert result.checks["nodata_interpretation_resolved"] is None

    def test_transformable_crs_mismatch_does_not_fail(self, tmp_path):
        input_paths = _make_input_files(tmp_path)
        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-004b",
            config={"feature_mode": "raw8"},
            raster_path=input_paths["raster_path"],
            vector_path=input_paths["vector_path"],
            raster_metadata=_valid_raster_meta(crs="EPSG:32642"),
            vector_metadata=_valid_vector_meta(crs="EPSG:3857"),
            runtime_probe_enabled=False,
        )

        assert result.success is True
        assert result.checks["crs_compatible"] is True
        assert result.checks["vector_reprojection_required"] is True
        assert result.checks["reprojection_pending_stage"] == "02_prepare_spatial_context"

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "success"
        assert manifest_data["validator_result"]["crs"]["vector_reprojection_required"] is True
        assert manifest_data["diagnostics"]["warnings"]

    def test_unparseable_crs_still_fails(self, tmp_path):
        input_paths = _make_input_files(tmp_path)
        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-004c",
            config={"feature_mode": "raw8"},
            raster_path=input_paths["raster_path"],
            vector_path=input_paths["vector_path"],
            raster_metadata=_valid_raster_meta(crs="EPSG:32642"),
            vector_metadata=_valid_vector_meta(crs="EPSG:NOT_A_REAL_CODE"),
            runtime_probe_enabled=False,
        )

        assert result.success is False
        assert result.error_type == "SpatialContractError"
        assert result.checks["crs_compatible"] is False

    def test_stage_reuses_existing_layers_via_composition(self, tmp_path, monkeypatch):
        input_paths = _make_input_files(tmp_path)
        called = {
            "build_config": 0,
            "validators": 0,
            "write_manifest": 0,
            "write_summary": 0,
        }

        def fake_build_config(raw):
            called["build_config"] += 1
            assert raw["feature_mode"] == "raw8"
            return PrepDataConfig(feature_mode="raw8")

        def fake_validate_check_inputs_contract(**kwargs):
            called["validators"] += 1
            assert Path(kwargs["raster_path"]) == Path(input_paths["raster_path"])
            return {
                "raster_path": input_paths["raster_path"],
                "vector_path": input_paths["vector_path"],
                "aoi_path": None,
                "band_count": 8,
                "crs": {"raster_crs": "EPSG:32642", "vector_crs": "EPSG:32642", "aoi_crs": None},
                "valid_resolution": {
                    "has_valid_mask": True,
                    "nodata_value": None,
                    "config_override_present": False,
                },
            }

        def fake_write_manifest(path, payload):
            called["write_manifest"] += 1
            assert payload["schema_name"] == "prep_data.check_inputs_manifest"
            assert payload["status"] == "success"

        def fake_write_summary(path, payload):
            called["write_summary"] += 1
            assert payload["schema_name"] == "prep_data.summary"
            assert payload["status"] == "success"

        monkeypatch.setattr(stage.prep_data_config, "build_config", fake_build_config)
        monkeypatch.setattr(
            stage.prep_data_validators,
            "validate_check_inputs_contract",
            fake_validate_check_inputs_contract,
        )
        monkeypatch.setattr(stage, "write_manifest", fake_write_manifest)
        monkeypatch.setattr(stage, "write_summary", fake_write_summary)

        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-005",
            config={"feature_mode": "raw8"},
            raster_path=input_paths["raster_path"],
            vector_path=input_paths["vector_path"],
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
        )

        assert result.status == "success"
        assert called["build_config"] == 1
        assert called["validators"] == 1
        assert called["write_manifest"] == 1
        assert called["write_summary"] == 1

    def test_requires_exactly_one_of_config_or_config_path(self, tmp_path):
        input_paths = _make_input_files(tmp_path)
        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-006",
            config={"feature_mode": "raw8"},
            config_path=tmp_path / "cfg.yaml",
            raster_path=input_paths["raster_path"],
            vector_path=input_paths["vector_path"],
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
        )
        assert result.status == "failed"
        assert result.error_type == "ContractError"

    def test_runtime_probe_missing_file_returns_failed_result(self, tmp_path):
        vector_path = _write_dummy_file(tmp_path / "labels.gpkg", b"vector")
        missing_raster = tmp_path / "missing.tif"
        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-007",
            config={"feature_mode": "raw8"},
            raster_path=str(missing_raster),
            vector_path=vector_path,
            raster_metadata=_valid_raster_meta(),
            vector_metadata=_valid_vector_meta(),
        )

        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["raster_readable"] is False

    def test_runtime_probe_can_load_sidecar_metadata_when_missing(self, tmp_path):
        input_paths = _make_input_files(tmp_path, with_aoi=True)
        _write_sidecar(
            input_paths["raster_path"],
            {
                "band_count": 8,
                "crs": "EPSG:32642",
                "has_valid_mask": True,
                "nodata": None,
                "readable": True,
            },
        )
        _write_sidecar(
            input_paths["vector_path"],
            {
                "feature_count": 4,
                "geometry_types": ["Polygon"],
                "crs": "EPSG:32642",
                "readable": True,
            },
        )
        _write_sidecar(
            input_paths["aoi_path"],
            {
                "feature_count": 1,
                "geometry_types": ["Polygon"],
                "crs": "EPSG:32642",
                "readable": True,
            },
        )

        result = stage.run_check_inputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-008",
            config={
                "feature_mode": "raw8",
                "aoi": {"enabled": True, "aoi_path": input_paths["aoi_path"], "buffer_m": 30.0},
            },
            raster_path=input_paths["raster_path"],
            vector_path=input_paths["vector_path"],
            aoi_path=input_paths["aoi_path"],
            raster_metadata=None,
            vector_metadata=None,
            aoi_metadata=None,
            # Sidecar fallback only applies when runtime probe is disabled; with probe
            # enabled the stage would call rasterio/fiona on these dummy files and fail.
            runtime_probe_enabled=False,
        )

        assert result.status == "success"
        assert result.checks["raster_readable"] is True
        assert result.checks["vector_readable"] is True
        assert result.checks["aoi_readable"] is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["input_raster"]["count"] == 8
        assert manifest_data["input_vectors"]["feature_count"] == 4
        assert manifest_data["input_aoi"]["feature_count"] == 1
        assert manifest_data["config"]["runtime_probe_mode"] == "rasterio_fiona_probe_v1"
        assert manifest_data["config"]["runtime_probe_enabled"] is False

    def test_bad_sidecar_json_returns_failed_result_without_raw_json_error(self, tmp_path):
        input_paths = _make_input_files(tmp_path)
        sidecar_path = Path(f"{input_paths['raster_path']}.meta.json")
        sidecar_path.write_text("{bad json", encoding="utf-8")
        _write_sidecar(
            input_paths["vector_path"],
            {
                "feature_count": 4,
                "geometry_types": ["Polygon"],
                "crs": "EPSG:32642",
            },
        )

        try:
            result = stage.run_check_inputs_stage(
                output_dir=tmp_path / "run_artifacts",
                run_id="run-009",
                config={"feature_mode": "raw8"},
                raster_path=input_paths["raster_path"],
                vector_path=input_paths["vector_path"],
                raster_metadata=None,
                vector_metadata=None,
            )
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            pytest.fail(f"JSONDecodeError leaked from runtime probe layer: {exc}")
        else:
            assert result.status == "failed"
            assert result.error_type == "ContractError"
