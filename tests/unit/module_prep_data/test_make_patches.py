"""Unit tests for module_prep_data 05_make_patches stage skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.common.constants import REQUIRED_SAMPLE_LAYERS
from ai_fields.module_prep_data import make_patches as stage
from ai_fields.module_prep_data.schemas import PrepDataConfig


def _make_large_nodata_raster(path: Path, *, width: int = 512, height: int = 512) -> Path:
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(
        west=599800.0, south=4399800.0, east=600200.0, north=4400200.0,
        width=width, height=height,
    )
    data = np.full((8, height, width), fill_value=1000, dtype=np.uint32)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width, count=8,
        dtype="uint32",
        crs=CRS.from_epsg(32637),
        transform=transform,
        nodata=65536,
    ) as ds:
        ds.write(data)
    return path


def _matching_patch_metadata(*, patch_size: int = 512, sampling_policy: str = "strategic"):
    return {
        "expected_patch_size": patch_size,
        "expected_sampling_policy": sampling_policy,
        "expected_patch_layers": list(REQUIRED_SAMPLE_LAYERS),
        "expected_patch_exports": {
            "img_count": 0,
            "extent_count": 0,
            "boundary_count": 0,
            "distance_count": 0,
            "valid_count": 0,
            "meta_count": 0,
        },
    }


class TestRunMakePatchesStage:
    def test_happy_path_minimal_patching_contract(self, tmp_path):
        result = stage.run_make_patches_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-500",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
        )

        assert result.success is True
        assert result.status == "success"
        assert result.patch_size == 512
        assert result.sampling_policy == "strategic"
        assert result.patch_layers == REQUIRED_SAMPLE_LAYERS
        assert result.written_total == 0
        assert result.patch_runtime_executed is False
        assert result.manifest_path.exists()
        assert result.summary_path.exists()

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["schema_name"] == "prep_data.patches_manifest"
        assert manifest_data["stage_name"] == "05_make_patches"
        assert manifest_data["status"] == "success"
        assert manifest_data["patch_contract_mode"] == "metadata_snapshot_only"
        assert manifest_data["patch_size"] == 512
        assert manifest_data["sampling_policy"] == "strategic"
        assert manifest_data["patch_layers"] == list(REQUIRED_SAMPLE_LAYERS)
        assert manifest_data["written_total"] == 0
        assert manifest_data["written_center"] is None
        assert manifest_data["written_boundary"] is None
        assert manifest_data["written_negative"] is None
        assert manifest_data["shortfall_negative"] is None
        assert manifest_data["patch_exports"] == {
            "img_count": 0,
            "extent_count": 0,
            "boundary_count": 0,
            "distance_count": 0,
            "valid_count": 0,
            "meta_count": 0,
        }
        assert manifest_data["rejection_stats"] == {
            "invalid_ratio_rejects": None,
            "mask_ratio_rejects": None,
            "boundary_quality_rejects": None,
            "duplicate_or_overlap_rejects": None,
        }
        assert manifest_data["patch_runtime_executed"] is False
        assert manifest_data["patch_artifacts_materialized"] is False
        assert manifest_data["patch_metadata_checked"] is False
        assert manifest_data["checks"]["patch_metadata_consistent"] is None
        assert manifest_data["config"]["input_refs_source"] == "stage_args_transitional"

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["schema_name"] == "prep_data.summary"
        assert summary_data["status"] == "success"
        assert summary_data["patch_contract_mode"] == "metadata_snapshot_only"
        assert summary_data["patch_size"] == 512
        assert summary_data["sampling_policy"] == "strategic"
        assert summary_data["written_total"] == 0
        assert summary_data["patch_runtime_executed"] is False
        assert summary_data["patch_metadata_consistent"] is None

    def test_happy_path_with_sampling_policy_snapshot(self, tmp_path):
        result = stage.run_make_patches_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-501",
            config={
                "feature_mode": "raw8_idx3",
                "patches": {"patch_size": 384, "sampling_policy": "random"},
            },
            raster_path="input.tif",
            vector_path="labels.gpkg",
            img_path="img.tif",
            extent_path="extent.tif",
            boundary_path="boundary.tif",
            distance_path="distance.tif",
            valid_path="valid.tif",
            patch_metadata=_matching_patch_metadata(patch_size=384, sampling_policy="random"),
            source_manifest_path=tmp_path / "targets_manifest.json",
            runtime_compute_enabled=False,
        )

        assert result.status == "success"
        assert result.patch_size == 384
        assert result.sampling_policy == "random"
        assert result.patch_layers == REQUIRED_SAMPLE_LAYERS
        assert result.checks["patch_metadata_consistent"] is True

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["input_img_path"] == "img.tif"
        assert manifest_data["input_extent_path"] == "extent.tif"
        assert manifest_data["input_boundary_path"] == "boundary.tif"
        assert manifest_data["input_distance_path"] == "distance.tif"
        assert manifest_data["input_valid_path"] == "valid.tif"
        assert manifest_data["checks"]["patch_metadata_consistent"] is True
        assert manifest_data["provenance"]["source_manifest_paths"] == [
            str(tmp_path / "targets_manifest.json")
        ]

        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary_data["source_manifest_path"] == str(tmp_path / "targets_manifest.json")

    def test_invalid_patching_contract_hint_returns_failed_result(self, tmp_path):
        result = stage.run_make_patches_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-502",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            patch_metadata={"expected_sampling_policy": "unsupported_policy_v2"},
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.manifest_path.exists()
        assert result.summary_path.exists()
        assert result.checks["contract_checks_passed"] is False
        assert result.checks["sampling_policy_resolved"] is False
        assert result.checks["patch_metadata_consistent"] is False

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "failed"
        assert manifest_data["diagnostics"]["errors"]

    def test_inconsistent_patch_metadata_typing_returns_failed_result(self, tmp_path):
        result = stage.run_make_patches_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-503",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
            patch_metadata={"expected_patch_exports": "bad-exports-shape"},
        )

        assert result.status == "failed"
        assert result.error_type == "ContractError"
        assert result.checks["patch_contract_resolved"] is False
        assert result.checks["patch_exports_snapshot_resolved"] is False
        assert result.checks["patch_metadata_consistent"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"config": {"feature_mode": "raw8"}, "config_path": "prep_data.yaml"},
            {},
        ],
    )
    def test_requires_exactly_one_of_config_or_config_path(self, tmp_path, kwargs):
        result = stage.run_make_patches_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-504",
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
                "patch_metadata": ["not-a-mapping"],
            },
        ],
    )
    def test_no_valueerror_or_typeerror_leakage(self, tmp_path, kwargs):
        base_kwargs = {
            "output_dir": tmp_path / "run_artifacts",
            "run_id": "run-505",
            "raster_path": "input.tif",
            "vector_path": "labels.gpkg",
        }
        base_kwargs.update(kwargs)

        try:
            result = stage.run_make_patches_stage(**base_kwargs)
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
            assert payload["schema_name"] == "prep_data.patches_manifest"
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

        result = stage.run_make_patches_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-506",
            config={"feature_mode": "raw8"},
            raster_path="input.tif",
            vector_path="labels.gpkg",
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

        from ai_fields.module_prep_data.features_compute import compute_and_save_features  # noqa: PLC0415
        from ai_fields.module_prep_data.targets_compute import compute_and_save_targets  # noqa: PLC0415

        raster_path = _make_large_nodata_raster(tmp_path / "raster_fixtures" / "large_nodata.tif")
        features = compute_and_save_features(
            raster_path=raster_path,
            output_dir=tmp_path / "features",
            feature_mode="raw8",
        )
        targets = compute_and_save_targets(
            raster_path=raster_path,
            vector_path=tiny_vector_path,
            output_dir=tmp_path / "targets",
            valid_path=features["valid_path"],
        )

        result = stage.run_make_patches_stage(
            output_dir=tmp_path / "run_artifacts",
            run_id="run-507",
            config={"feature_mode": "raw8"},
            raster_path=str(raster_path),
            vector_path=str(tiny_vector_path),
            img_path=str(features["img_path"]),
            extent_path=str(targets["extent_path"]),
            boundary_path=str(targets["boundary_path"]),
            distance_path=str(targets["distance_path"]),
            valid_path=str(targets["valid_path"]),
            runtime_compute_enabled=True,
        )
        assert result.status == "success"
        assert result.success is True
        assert result.patch_runtime_executed is True
        assert result.written_total is not None

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        summary_data = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert manifest_data["status"] == "success"
        assert summary_data["status"] == "success"
