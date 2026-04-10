"""Unit tests for module_prep_data 04_prepare_targets stage skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.module_prep_data import prepare_targets as stage
from ai_fields.module_prep_data.schemas import PrepDataConfig


def _make_large_nodata_raster(path: Path) -> Path:
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
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


def _matching_target_metadata() -> dict[str, object]:
    return {
        "expected_target_layers": ["extent", "boundary", "distance", "valid"],
        "expected_boundary_encoding": "background_skeleton_buffer",
        "expected_distance_target": "unsigned_distance_to_boundary",
        "expected_valid_saved_separately": True,
        "expected_boundary_raw_enabled": True,
    }


class TestRunPrepareTargetsStage:
    def test_happy_path_minimal_target_contract(self, tmp_path):
        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-300",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            runtime_compute_enabled=False,
        )

        assert result.success is True
        assert result.status == "success"
        assert result.target_layers == ("extent", "boundary", "distance", "valid")
        assert result.boundary_encoding == "background_skeleton_buffer"
        assert result.distance_target == "unsigned_distance_to_boundary"
        assert result.valid_saved_separately is True
        assert result.boundary_raw_enabled is True
        assert result.manifest_path.exists()
        assert result.summary_path.exists()

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["schema_name"] == "prep_data.targets_manifest"
        assert manifest_data["stage_name"] == "04_prepare_targets"
        assert manifest_data["status"] == "success"
        assert manifest_data["target_contract_mode"] == "metadata_snapshot_only"
        assert manifest_data["target_layers"] == ["extent", "boundary", "distance", "valid"]
        assert manifest_data["boundary_encoding"] == "background_skeleton_buffer"
        assert manifest_data["distance_target"] == "unsigned_distance_to_boundary"
        assert manifest_data["valid_saved_separately"] is True
        assert manifest_data["boundary_raw_policy"]["enabled"] is True
        assert manifest_data["checks"]["target_metadata_consistent"] is None
        assert manifest_data["config"]["input_refs_source"] == "stage_args_transitional"

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["schema_name"] == "prep_data.summary"
        assert summary_data["status"] == "success"
        assert summary_data["target_layers"] == ["extent", "boundary", "distance", "valid"]
        assert summary_data["boundary_encoding"] == "background_skeleton_buffer"
        assert summary_data["distance_target"] == "unsigned_distance_to_boundary"
        assert summary_data["target_metadata_consistent"] is None

    def test_happy_path_with_policy_snapshot_hints(self, tmp_path):
        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-301",
            config={"feature_mode": "raw8_idx3"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            valid_path="valid.tif",
            target_metadata=_matching_target_metadata(),
            source_manifest_path=tmp_path / "features_manifest.json",
            runtime_compute_enabled=False,
        )

        assert result.status == "success"
        assert result.target_layers == ("extent", "boundary", "distance", "valid")
        assert result.boundary_encoding == "background_skeleton_buffer"
        assert result.distance_target == "unsigned_distance_to_boundary"
        assert result.checks["target_metadata_consistent"] is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["input_valid_path"] == "valid.tif"
        assert manifest_data["checks"]["target_metadata_consistent"] is True
        assert manifest_data["provenance"]["source_manifest_paths"] == [
            str(tmp_path / "features_manifest.json")
        ]

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["source_manifest_path"] == str(tmp_path / "features_manifest.json")

    def test_invalid_target_contract_hint_returns_failed_result(self, tmp_path):
        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-302",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            target_metadata={"expected_target_layers": ["extent", "distance", "valid"]},
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.checks["target_contract_resolved"] is False
        assert result.checks["target_layers_resolved"] is False
        assert result.checks["target_metadata_consistent"] is False

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "failed"
        assert manifest_data["diagnostics"]["errors"]

    def test_inconsistent_target_metadata_typing_returns_failed_result(self, tmp_path):
        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-303",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            target_metadata={"expected_boundary_raw_enabled": "yes"},
        )

        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["target_contract_resolved"] is False
        assert result.checks["target_metadata_consistent"] is False

    @pytest.mark.parametrize(
        ("override", "expected_failed_check"),
        [
            (
                {"expected_boundary_encoding": "binary_edges"},
                "boundary_policy_resolved",
            ),
            (
                {"expected_distance_target": "signed_distance_to_boundary"},
                "distance_policy_resolved",
            ),
            (
                {"expected_valid_saved_separately": False},
                "valid_semantics_resolved",
            ),
        ],
    )
    def test_target_metadata_mismatch_sets_specific_failure_check(
        self,
        tmp_path,
        override,
        expected_failed_check,
    ):
        target_metadata = _matching_target_metadata()
        target_metadata.update(override)

        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-303a",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            target_metadata=target_metadata,
        )

        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["target_contract_resolved"] is False
        assert result.checks[expected_failed_check] is False
        assert result.checks["target_metadata_consistent"] is False

    def test_failure_checks_prefer_stable_error_code_mapping(self, tmp_path, monkeypatch):
        def fake_resolve_target_contract(**_kwargs):
            err = stage.ContractError("generic mismatch without field names")
            setattr(
                err,
                stage._ERROR_CODE_ATTR,
                stage._ERR_TARGET_METADATA_EXPECTED_DISTANCE_TARGET_MISMATCH,
            )
            raise err

        monkeypatch.setattr(stage, "_resolve_target_contract", fake_resolve_target_contract)

        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-303b",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
        )

        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert "expected_distance_target" not in (result.error_message or "")
        assert result.checks["distance_policy_resolved"] is False
        assert result.checks["target_metadata_consistent"] is False

    @pytest.mark.parametrize(
        ("raster_path", "vector_path"),
        [
            (123, "labels.gpkg"),
            ("input.tif", []),
        ],
    )
    def test_invalid_stage_boundary_path_types_return_failed_result(
        self,
        tmp_path,
        raster_path,
        vector_path,
    ):
        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-303c",
            config={"feature_mode": "raw8"},
            raster_path=raster_path,
            vector_path=vector_path,
        )

        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["contract_checks_passed"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"config": {"feature_mode": "raw8"}, "config_path": "prep_data.yaml"},
            {},
        ],
    )
    def test_requires_exactly_one_of_config_or_config_path(self, tmp_path, kwargs):
        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-304",
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
                "target_metadata": {"expected_target_layers": "extent"},
            },
        ],
    )
    def test_no_valueerror_or_typeerror_leakage(self, tmp_path, kwargs):
        base_kwargs = {
            "output_dir": tmp_path / "run_artifacts",
            "run_id": "run-305",
            "raster_path": "input.tif",
            "vector_path": "labels.gpkg",
        }
        base_kwargs.update(kwargs)
        try:
            result = stage.run_prepare_targets_stage(**base_kwargs)
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
            assert payload["schema_name"] == "prep_data.targets_manifest"
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

        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-306",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            runtime_compute_enabled=False,
        )
        assert result.status == "success"
        assert called["build_config"] == 1
        assert called["validate_input_paths_contract"] == 1
        assert called["write_manifest"] == 1
        assert called["write_summary"] == 1

    def test_runtime_compute_succeeds_with_large_source_nodata(
        self, tmp_path, tiny_vector_path
    ):
        pytest.importorskip("rasterio")
        pytest.importorskip("geopandas")
        pytest.importorskip("scipy")

        raster_path = _make_large_nodata_raster(tmp_path / "raster_fixtures" / "large_nodata.tif")
        result = stage.run_prepare_targets_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-307",
            config={"feature_mode": "raw8"},
            raster_path=str(raster_path),
            vector_path=str(tiny_vector_path),
            runtime_compute_enabled=True,
        )
        assert result.status == "success"
        assert result.success is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "success"
        assert summary_data["status"] == "success"
        assert manifest_data["extent_output_path"] is not None
