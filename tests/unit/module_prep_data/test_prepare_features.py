"""Unit tests for module_prep_data 03_prepare_features stage skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.module_prep_data import prepare_features as stage
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


class TestRunPrepareFeaturesStage:
    def test_happy_path_raw8(self, tmp_path):
        result = stage.run_prepare_features_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-200",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            valid_path="valid.tif",
            runtime_compute_enabled=False,
        )

        assert result.success is True
        assert result.status == "success"
        assert result.feature_mode == "raw8"
        assert result.feature_channel_count == 8
        assert result.derived_indices == ()
        assert result.assembled_model_input_variants == ("raw8_valid",)
        assert result.valid_saved_separately is True
        assert result.manifest_path.exists()
        assert result.summary_path.exists()

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["schema_name"] == "prep_data.features_manifest"
        assert manifest_data["stage_name"] == "03_prepare_features"
        assert manifest_data["status"] == "success"
        assert manifest_data["feature_mode"] == "raw8"
        assert manifest_data["feature_channel_count"] == 8
        assert manifest_data["derived_indices"] == []
        assert manifest_data["assembled_model_input_variants"] == ["raw8_valid"]
        assert manifest_data["valid_saved_separately"] is True
        assert manifest_data["input_raster_path"] == "input.tif"
        assert manifest_data["input_valid_path"] == "valid.tif"
        assert manifest_data["normalization_plan"]["normalization_name"] == "robust_percentile"
        assert manifest_data["normalization_plan"]["dtype_before_model"] == "float32"
        assert manifest_data["normalization_plan"]["clip_percentiles"] == [0.5, 99.5]
        assert manifest_data["normalization_plan"]["scaling_range"] == [0.0, 1.0]
        assert manifest_data["config"]["input_refs_source"] == "stage_args_transitional"
        assert manifest_data["checks"]["feature_metadata_consistent"] is None
        assert manifest_data["checks"]["channel_semantics_resolved"] is None
        assert manifest_data["channel_semantics_status"] == "unresolved"

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["schema_name"] == "prep_data.summary"
        assert summary_data["status"] == "success"
        assert summary_data["feature_mode"] == "raw8"
        assert summary_data["feature_channel_count"] == 8
        assert summary_data["assembled_model_input_variants"] == ["raw8_valid"]
        assert summary_data["feature_metadata_consistent"] is None
        assert summary_data["channel_semantics_resolved"] is None

    def test_happy_path_raw8_idx3(self, tmp_path):
        result = stage.run_prepare_features_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-201",
            config={"feature_mode": "raw8_idx3"},
            raster_path="input.tif",
            source_manifest_path=tmp_path / "aoi_manifest.json",
            runtime_compute_enabled=False,
        )

        assert result.status == "success"
        assert result.feature_mode == "raw8_idx3"
        assert result.feature_channel_count == 11
        assert result.derived_indices == ("NDVI", "SAVI", "NDWI")
        assert result.assembled_model_input_variants == ("raw8_idx3_valid",)

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["feature_mode"] == "raw8_idx3"
        assert manifest_data["derived_indices"] == ["NDVI", "SAVI", "NDWI"]
        assert manifest_data["assembled_model_input_variants"] == ["raw8_idx3_valid"]
        assert manifest_data["provenance"]["source_manifest_paths"] == [
            str(tmp_path / "aoi_manifest.json")
        ]
        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["source_manifest_path"] == str(tmp_path / "aoi_manifest.json")

    def test_happy_path_with_channel_semantics_hint(self, tmp_path):
        channel_semantics = [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
        ]
        result = stage.run_prepare_features_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-201b",
            config={"feature_mode": "raw8"},
            feature_metadata={"channel_semantics": channel_semantics},
            runtime_compute_enabled=False,
        )
        assert result.status == "success"
        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["channel_semantics"] == channel_semantics
        assert manifest_data["channel_semantics_status"] == "resolved"
        assert manifest_data["checks"]["channel_semantics_resolved"] is True
        assert manifest_data["checks"]["feature_metadata_consistent"] is True

    def test_invalid_feature_mode_returns_failed_result(self, tmp_path):
        result = stage.run_prepare_features_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-202",
            config={"feature_mode": "raw11"},
        )
        assert result.success is False
        assert result.status == "failed"
        assert result.error_type == "FeatureModeError"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.checks["contract_checks_passed"] is False

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "failed"
        assert manifest_data["diagnostics"]["errors"]

    def test_inconsistent_feature_metadata_hint_returns_failed_result(self, tmp_path):
        result = stage.run_prepare_features_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-203",
            config={"feature_mode": "raw8"},
            feature_metadata={"expected_feature_channel_count": 11},
        )
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["feature_metadata_consistent"] is False

    def test_channel_semantics_length_mismatch_returns_failed_result(self, tmp_path):
        result = stage.run_prepare_features_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-203b",
            config={"feature_mode": "raw8"},
            feature_metadata={"channel_semantics": ["B1", "B2"]},
        )
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["channel_semantics_resolved"] is False
        assert result.checks["feature_metadata_consistent"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"config": {"feature_mode": "raw8"}, "config_path": "prep_data.yaml"},
            {},
        ],
    )
    def test_requires_exactly_one_of_config_or_config_path(self, tmp_path, kwargs):
        result = stage.run_prepare_features_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-204",
            **kwargs,
        )
        assert result.status == "failed"
        assert result.error_type == "ContractError"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "config": {"feature_mode": "raw8"},
                "feature_metadata": {"expected_derived_indices": "NDVI"},
            },
            {
                "config": {"feature_mode": "raw8"},
                "source_manifest_path": 123,
            },
        ],
    )
    def test_no_valueerror_or_typeerror_leakage(self, tmp_path, kwargs):
        base_kwargs = {
            "output_dir": tmp_path / "run_artifacts",
            "run_id": "run-205",
        }
        base_kwargs.update(kwargs)
        try:
            result = stage.run_prepare_features_stage(**base_kwargs)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive guard
            pytest.fail(f"Raw exception leaked from stage layer: {type(exc).__name__}: {exc}")
        else:
            assert result.status == "failed"

    def test_stage_reuses_existing_layers_via_composition(self, tmp_path, monkeypatch):
        called = {
            "build_config": 0,
            "write_manifest": 0,
            "write_summary": 0,
        }

        def fake_build_config(raw):
            called["build_config"] += 1
            assert raw["feature_mode"] == "raw8"
            return PrepDataConfig(feature_mode="raw8")

        def fake_write_manifest(path, payload):
            called["write_manifest"] += 1
            assert payload["schema_name"] == "prep_data.features_manifest"
            assert payload["status"] == "success"

        def fake_write_summary(path, payload):
            called["write_summary"] += 1
            assert payload["schema_name"] == "prep_data.summary"
            assert payload["status"] == "success"

        monkeypatch.setattr(stage.prep_data_config, "build_config", fake_build_config)
        monkeypatch.setattr(stage, "write_manifest", fake_write_manifest)
        monkeypatch.setattr(stage, "write_summary", fake_write_summary)

        result = stage.run_prepare_features_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-206",
            config={"feature_mode": "raw8"},
            runtime_compute_enabled=False,
        )
        assert result.status == "success"
        assert called["build_config"] == 1
        assert called["write_manifest"] == 1
        assert called["write_summary"] == 1

    def test_runtime_compute_succeeds_with_large_source_nodata(self, tmp_path):
        pytest.importorskip("rasterio")
        raster_path = _make_large_nodata_raster(tmp_path / "raster_fixtures" / "large_nodata.tif")
        result = stage.run_prepare_features_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-207",
            config={"feature_mode": "raw8"},
            raster_path=str(raster_path),
            runtime_compute_enabled=True,
        )
        assert result.status == "success"
        assert result.success is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "success"
        assert summary_data["status"] == "success"
        assert manifest_data["valid_output_path"] is not None
