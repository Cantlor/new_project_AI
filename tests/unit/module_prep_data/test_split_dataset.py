"""Unit tests for module_prep_data 06_split_dataset stage skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.common.constants import REQUIRED_SAMPLE_LAYERS
from ai_fields.module_prep_data import split_dataset as stage
from ai_fields.module_prep_data.schemas import PrepDataConfig


def _matching_split_metadata(
    *,
    split_policy: str = "spatial_stratified",
    random_seed: int | None = 42,
    feature_channel_count: int = 8,
) -> dict[str, object]:
    return {
        "expected_split_policy": split_policy,
        "expected_random_seed": random_seed,
        "expected_feature_channel_count": feature_channel_count,
        "expected_export_required_dirs": list(REQUIRED_SAMPLE_LAYERS),
    }


class TestRunSplitDatasetStage:
    def test_happy_path_minimal_split_contract(self, tmp_path):
        result = stage.run_split_dataset_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-600",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
        )

        assert result.success is True
        assert result.status == "success"
        assert result.split_policy == "spatial_stratified"
        assert result.random_seed == 42
        assert result.feature_mode == "raw8"
        assert result.feature_channel_count == 8
        assert result.split_assignment_executed is False
        assert result.export_layout_materialized is False
        assert result.manifest_path.exists()
        assert result.summary_path.exists()

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["schema_name"] == "prep_data.split_manifest"
        assert manifest_data["stage_name"] == "06_split_dataset"
        assert manifest_data["status"] == "success"
        assert manifest_data["split_contract_mode"] == "metadata_snapshot_only"
        assert manifest_data["split_policy"] == "spatial_stratified"
        assert manifest_data["patch_size"] == 512
        assert manifest_data["random_seed"] == 42
        assert manifest_data["feature_mode"] == "raw8"
        assert manifest_data["feature_channel_count"] == 8
        assert manifest_data["channel_semantics"] == []
        assert manifest_data["channel_semantics_status"] == "unresolved"
        assert manifest_data["split_ratio_plan"] is None
        assert manifest_data["split_ratio_plan_status"] == "unresolved"
        assert manifest_data["splits"] == {
            "train": {"sample_count": 0},
            "val": {"sample_count": 0},
            "test": {"sample_count": 0},
        }
        assert manifest_data["split_assignment_executed"] is False
        assert manifest_data["export_layout_materialized"] is False
        assert manifest_data["export_structure"] == {
            "required_dirs": list(REQUIRED_SAMPLE_LAYERS),
            "materialized": False,
        }
        assert manifest_data["checks"]["split_metadata_consistent"] is None
        assert manifest_data["config"]["input_refs_source"] == "stage_args_transitional"

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["schema_name"] == "prep_data.summary"
        assert summary_data["status"] == "success"
        assert summary_data["split_contract_mode"] == "metadata_snapshot_only"
        assert summary_data["split_policy"] == "spatial_stratified"
        assert summary_data["patch_size"] == 512
        assert summary_data["random_seed"] == 42
        assert summary_data["split_assignment_executed"] is False
        assert summary_data["export_layout_materialized"] is False
        assert summary_data["split_metadata_consistent"] is None

    def test_happy_path_with_split_policy_snapshot_and_provenance(self, tmp_path):
        result = stage.run_split_dataset_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-601",
            config={
                "feature_mode": "raw8_idx3",
                "split": {"policy": "random", "random_seed": 777},
            },
            raster_path="input.tif",
            vector_path="labels.gpkg",
            split_metadata=_matching_split_metadata(
                split_policy="random",
                random_seed=777,
                feature_channel_count=11,
            ),
            source_manifest_path=tmp_path / "patches_manifest.json",
        )

        assert result.status == "success"
        assert result.split_policy == "random"
        assert result.random_seed == 777
        assert result.feature_mode == "raw8_idx3"
        assert result.feature_channel_count == 11
        assert result.checks["split_metadata_consistent"] is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["checks"]["split_metadata_consistent"] is True
        assert manifest_data["provenance"]["source_manifest_paths"] == [
            str(tmp_path / "patches_manifest.json")
        ]
        assert manifest_data["export_structure"]["required_dirs"] == list(REQUIRED_SAMPLE_LAYERS)

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["source_manifest_path"] == str(tmp_path / "patches_manifest.json")

    def test_invalid_split_contract_hint_returns_failed_result(self, tmp_path):
        result = stage.run_split_dataset_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-602",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            split_metadata={"expected_split_policy": "unsupported_policy_v2"},
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.checks["contract_checks_passed"] is False
        assert result.checks["split_policy_resolved"] is False
        assert result.checks["split_metadata_consistent"] is False

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "failed"
        assert manifest_data["diagnostics"]["errors"]

    def test_inconsistent_split_metadata_typing_returns_failed_result(self, tmp_path):
        result = stage.run_split_dataset_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-603",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            split_metadata={"expected_export_required_dirs": "img,extent,boundary"},
        )

        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["split_contract_resolved"] is False
        assert result.checks["export_structure_resolved"] is False
        assert result.checks["split_metadata_consistent"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"config": {"feature_mode": "raw8"}, "config_path": "prep_data.yaml"},
            {},
        ],
    )
    def test_requires_exactly_one_of_config_or_config_path(self, tmp_path, kwargs):
        result = stage.run_split_dataset_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-604",
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
                "split_metadata": ["not-a-mapping"],
            },
        ],
    )
    def test_no_valueerror_or_typeerror_leakage(self, tmp_path, kwargs):
        base_kwargs = {
            "output_dir": tmp_path / "run_artifacts",
            "run_id": "run-605",
            "raster_path": "input.tif",
            "vector_path": "labels.gpkg",
        }
        base_kwargs.update(kwargs)

        try:
            result = stage.run_split_dataset_stage(**base_kwargs)
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
            assert payload["schema_name"] == "prep_data.split_manifest"
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

        result = stage.run_split_dataset_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-606",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
        )

        assert result.status == "success"
        assert called["build_config"] == 1
        assert called["validate_input_paths_contract"] == 1
        assert called["write_manifest"] == 1
        assert called["write_summary"] == 1
