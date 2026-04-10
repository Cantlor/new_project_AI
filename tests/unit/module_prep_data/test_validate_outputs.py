"""Unit tests for module_prep_data 07_validate_outputs stage skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.common.constants import REQUIRED_SAMPLE_LAYERS
from ai_fields.module_prep_data import validate_outputs as stage
from ai_fields.module_prep_data.schemas import PrepDataConfig


def _matching_validation_metadata(
    *,
    feature_channel_count: int = 8,
    model_input_channel_count_after_valid: int = 9,
) -> dict[str, object]:
    return {
        "expected_validation_contract_mode": "metadata_snapshot_only",
        "expected_required_dirs": list(REQUIRED_SAMPLE_LAYERS),
        "expected_target_layers": ["extent", "boundary", "distance", "valid"],
        "expected_feature_channel_count": feature_channel_count,
        "expected_model_input_channel_count_after_valid": model_input_channel_count_after_valid,
        "expected_valid_saved_separately": True,
    }


class TestRunValidateOutputsStage:
    def test_happy_path_minimal_validation_contract(self, tmp_path):
        result = stage.run_validate_outputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-700",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
        )

        assert result.success is True
        assert result.status == "success"
        assert result.feature_mode == "raw8"
        assert result.feature_channel_count == 8
        assert result.model_input_channel_count_after_valid == 9
        assert result.validation_runtime_executed is False
        assert result.manifest_path.exists()
        assert result.summary_path.exists()

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["schema_name"] == "prep_data.validate_outputs_manifest"
        assert manifest_data["stage_name"] == "07_validate_outputs"
        assert manifest_data["status"] == "success"
        assert manifest_data["validation_contract_mode"] == "metadata_snapshot_only"
        assert manifest_data["patch_size"] == 512
        assert manifest_data["required_dirs_contract"] == list(REQUIRED_SAMPLE_LAYERS)
        assert manifest_data["target_layers_contract"] == ["extent", "boundary", "distance", "valid"]
        assert manifest_data["feature_mode"] == "raw8"
        assert manifest_data["feature_channel_count_before_valid"] == 8
        assert manifest_data["model_input_channel_count_after_valid"] == 9
        assert manifest_data["validation_runtime_executed"] is False
        assert manifest_data["runtime_checks"] == {
            "shapes_consistency_checked": None,
            "target_value_domains_checked": None,
            "broken_files_checked": None,
            "valid_nodata_consistency_checked": None,
            "boundary_coverage_checked": None,
            "distance_boundary_consistency_checked": None,
        }
        assert manifest_data["checks"]["validation_metadata_consistent"] is None
        assert manifest_data["checks"]["runtime_output_scan_executed"] is False
        assert manifest_data["config"]["input_refs_source"] == "stage_args_transitional"

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["schema_name"] == "prep_data.summary"
        assert summary_data["status"] == "success"
        assert summary_data["validation_contract_mode"] == "metadata_snapshot_only"
        assert summary_data["feature_mode"] == "raw8"
        assert summary_data["patch_size"] == 512
        assert summary_data["feature_channel_count_before_valid"] == 8
        assert summary_data["model_input_channel_count_after_valid"] == 9
        assert summary_data["validation_runtime_executed"] is False
        assert summary_data["validation_metadata_consistent"] is None

    def test_happy_path_with_provenance_link(self, tmp_path):
        result = stage.run_validate_outputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-701",
            config={"feature_mode": "raw8_idx3"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            validation_metadata=_matching_validation_metadata(
                feature_channel_count=11,
                model_input_channel_count_after_valid=12,
            ),
            source_manifest_path=tmp_path / "split_manifest.json",
        )

        assert result.status == "success"
        assert result.feature_mode == "raw8_idx3"
        assert result.feature_channel_count == 11
        assert result.model_input_channel_count_after_valid == 12
        assert result.checks["validation_metadata_consistent"] is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["checks"]["validation_metadata_consistent"] is True
        assert manifest_data["provenance"]["source_manifest_paths"] == [
            str(tmp_path / "split_manifest.json")
        ]

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["source_manifest_path"] == str(tmp_path / "split_manifest.json")

    def test_invalid_validation_hint_returns_failed_result(self, tmp_path):
        result = stage.run_validate_outputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-702",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            validation_metadata={"expected_validation_contract_mode": "runtime_scan_v2"},
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.checks["contract_checks_passed"] is False
        assert result.checks["validation_contract_mode_resolved"] is False
        assert result.checks["validation_metadata_consistent"] is False

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "failed"
        assert manifest_data["diagnostics"]["errors"]

    def test_inconsistent_validation_metadata_typing_returns_failed_result(self, tmp_path):
        result = stage.run_validate_outputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-703",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            validation_metadata={"expected_required_dirs": "img,extent,boundary"},
        )

        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["validation_contract_resolved"] is False
        assert result.checks["dataset_structure_contract_resolved"] is False
        assert result.checks["split_export_contract_resolved"] is False
        assert result.checks["validation_metadata_consistent"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"config": {"feature_mode": "raw8"}, "config_path": "prep_data.yaml"},
            {},
        ],
    )
    def test_requires_exactly_one_of_config_or_config_path(self, tmp_path, kwargs):
        result = stage.run_validate_outputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-704",
            raster_path="input.tif",
            vector_path="labels.gpkg",
            **kwargs,
        )

        assert result.status == "failed"
        assert result.error_type == "ContractError"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "config": {"feature_mode": "raw8"},
                "source_manifest_path": 123,
            },
            {
                "config": {"feature_mode": "raw8"},
                "validation_metadata": ["not-a-mapping"],
            },
        ],
    )
    def test_no_valueerror_or_typeerror_leakage(self, tmp_path, kwargs):
        base_kwargs = {
            "output_dir": tmp_path / "run_artifacts",
            "run_id": "run-705",
            "raster_path": "input.tif",
            "vector_path": "labels.gpkg",
        }
        base_kwargs.update(kwargs)

        try:
            result = stage.run_validate_outputs_stage(**base_kwargs)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive guard
            pytest.fail(f"Raw exception leaked from stage layer: {type(exc).__name__}: {exc}")
        else:
            assert result.status == "failed"

    def test_stage_reuses_existing_layers_via_composition(self, tmp_path, monkeypatch):
        called = {
            "build_config": 0,
            "validate_input_paths_contract": 0,
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

        def fake_write_manifest(path, payload):
            called["write_manifest"] += 1
            assert payload["schema_name"] == "prep_data.validate_outputs_manifest"
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
        monkeypatch.setattr(stage, "write_manifest", fake_write_manifest)
        monkeypatch.setattr(stage, "write_summary", fake_write_summary)

        result = stage.run_validate_outputs_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-706",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
        )

        assert result.status == "success"
        assert called["build_config"] == 1
        assert called["validate_input_paths_contract"] == 1
        assert called["write_manifest"] == 1
        assert called["write_summary"] == 1
