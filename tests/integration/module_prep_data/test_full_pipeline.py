"""Full pipeline integration test: stages 01–07 using runtime-compute fixtures.

Verifies that all 7 stages can run end-to-end with runtime compute enabled,
producing a train-ready dataset with the correct structure and norm_stats.json.

Skipped automatically if rasterio, geopandas, fiona, or scipy are not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

pytest.importorskip("rasterio")
pytest.importorskip("geopandas")
pytest.importorskip("fiona")
pytest.importorskip("scipy")


from ai_fields.module_prep_data.check_inputs import run_check_inputs_stage  # noqa: E402
from ai_fields.module_prep_data.input_probe import probe_raster, probe_vector  # noqa: E402
from ai_fields.module_prep_data.prepare_spatial_context import run_prepare_spatial_context_stage  # noqa: E402
from ai_fields.module_prep_data.prepare_features import run_prepare_features_stage  # noqa: E402
from ai_fields.module_prep_data.prepare_targets import run_prepare_targets_stage  # noqa: E402
from ai_fields.module_prep_data.make_patches import run_make_patches_stage  # noqa: E402
from ai_fields.module_prep_data.split_dataset import run_split_dataset_stage  # noqa: E402
from ai_fields.module_prep_data.validate_outputs import run_validate_outputs_stage  # noqa: E402
from ai_fields.module_prep_data.schemas import PrepDataConfig, PatchesConfig  # noqa: E402


def _make_patch_ready_raster(path: Path, *, width: int = 700, height: int = 700) -> Path:
    import rasterio  # noqa: PLC0415
    from rasterio.crs import CRS  # noqa: PLC0415
    from rasterio.transform import from_bounds  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(
        west=599800.0, south=4399800.0, east=600200.0, north=4400200.0,
        width=width, height=height,
    )
    data = np.full((8, height, width), fill_value=1000, dtype=np.uint16)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width, count=8,
        dtype="uint16",
        crs=CRS.from_epsg(32637),
        transform=transform,
        nodata=0,
    ) as ds:
        ds.write(data)
    return path


def _make_patch_ready_vector(path: Path) -> Path:
    import fiona  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs="EPSG:32637") as dst:
        dst.write({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    (599900.0, 4399900.0),
                    (600100.0, 4399900.0),
                    (600100.0, 4400100.0),
                    (599900.0, 4400100.0),
                    (599900.0, 4399900.0),
                ]],
            },
            "properties": {"id": 1},
        })
    return path


@pytest.fixture
def pipeline_result(tmp_path):
    """Run all 7 stages and return state dict."""
    raster_path = _make_patch_ready_raster(tmp_path / "inputs" / "scene_700.tif")
    vector_path = _make_patch_ready_vector(tmp_path / "inputs" / "labels.gpkg")
    config = PrepDataConfig(feature_mode="raw8", patches=PatchesConfig(patch_size=512))

    # Probe real files once — passed as metadata to stages that require it
    raster_meta = probe_raster(raster_path)
    vector_meta = probe_vector(vector_path)

    # Stage 01
    stage01_out = tmp_path / "stage01"
    stage01_out.mkdir()
    r01 = run_check_inputs_stage(
        run_id="test_full_pipeline_01",
        raster_path=raster_path,
        vector_path=vector_path,
        raster_metadata=raster_meta,
        vector_metadata=vector_meta,
        output_dir=stage01_out,
        config=config,
        runtime_probe_enabled=False,
    )
    assert r01.status == "success", f"Stage 01 failed: {r01.error_message}"

    # Stage 02
    stage02_out = tmp_path / "stage02"
    stage02_out.mkdir()
    r02 = run_prepare_spatial_context_stage(
        run_id="test_full_pipeline_02",
        raster_path=raster_path,
        vector_path=vector_path,
        raster_metadata=raster_meta,
        vector_metadata=vector_meta,
        output_dir=stage02_out,
        config=config,
        source_manifest_path=r01.manifest_path,
        runtime_compute_enabled=True,
    )
    assert r02.status == "success", f"Stage 02 failed: {r02.error_message}"

    # Stage 03
    stage03_out = tmp_path / "stage03"
    stage03_out.mkdir()
    r03 = run_prepare_features_stage(
        run_id="test_full_pipeline_03",
        raster_path=raster_path,
        output_dir=stage03_out,
        config=config,
        runtime_compute_enabled=True,
    )
    assert r03.status == "success", f"Stage 03 failed: {r03.error_message}"

    img_path = stage03_out / "img.tif"
    valid_path = stage03_out / "valid.tif"

    # Stage 04
    stage04_out = tmp_path / "stage04"
    stage04_out.mkdir()
    r04 = run_prepare_targets_stage(
        run_id="test_full_pipeline_04",
        raster_path=raster_path,
        vector_path=vector_path,
        output_dir=stage04_out,
        config=config,
        runtime_compute_enabled=True,
        valid_path=valid_path,
    )
    assert r04.status == "success", f"Stage 04 failed: {r04.error_message}"

    extent_path = stage04_out / "extent.tif"
    boundary_path = stage04_out / "boundary.tif"
    distance_path = stage04_out / "distance.tif"

    # Stage 05 — patches go to output_dir/patches/
    stage05_out = tmp_path / "stage05"
    stage05_out.mkdir()
    r05 = run_make_patches_stage(
        run_id="test_full_pipeline_05",
        raster_path=raster_path,
        vector_path=vector_path,
        img_path=img_path,
        extent_path=extent_path,
        boundary_path=boundary_path,
        distance_path=distance_path,
        valid_path=valid_path,
        output_dir=stage05_out,
        config=config,
        runtime_compute_enabled=True,
    )
    assert r05.status == "success", f"Stage 05 failed: {r05.error_message}"

    patches_subdir = stage05_out / "patches"

    # Stage 06 — dataset goes to output_dir/dataset/
    stage06_out = tmp_path / "stage06"
    stage06_out.mkdir()
    r06 = run_split_dataset_stage(
        run_id="test_full_pipeline_06",
        raster_path=raster_path,
        vector_path=vector_path,
        output_dir=stage06_out,
        config=config,
        runtime_compute_enabled=True,
        patches_dir=patches_subdir,
    )
    assert r06.status == "success", f"Stage 06 failed: {r06.error_message}"

    dataset_dir = stage06_out / "dataset"

    # Stage 07
    stage07_out = tmp_path / "stage07"
    stage07_out.mkdir()
    r07 = run_validate_outputs_stage(
        run_id="test_full_pipeline_07",
        raster_path=raster_path,
        vector_path=vector_path,
        output_dir=stage07_out,
        config=config,
        runtime_compute_enabled=True,
        dataset_dir=dataset_dir,
    )
    assert r07.status == "success", f"Stage 07 failed: {r07.error_message}"

    return {
        "results": (r01, r02, r03, r04, r05, r06, r07),
        "r05": r05,
        "r06": r06,
        "r07": r07,
        "stage03_out": stage03_out,
        "stage04_out": stage04_out,
        "patches_subdir": patches_subdir,
        "dataset_dir": dataset_dir,
    }


class TestFullPipeline:
    def test_all_stages_succeed(self, pipeline_result):
        for i, r in enumerate(pipeline_result["results"], start=1):
            assert r.status == "success", f"Stage {i:02d} did not succeed"

    def test_feature_files_created(self, pipeline_result):
        out = pipeline_result["stage03_out"]
        assert (out / "img.tif").exists(), "img.tif not created"
        assert (out / "valid.tif").exists(), "valid.tif not created"

    def test_target_files_created(self, pipeline_result):
        out = pipeline_result["stage04_out"]
        for fname in ("extent.tif", "boundary.tif", "boundary_raw.tif", "distance.tif"):
            assert (out / fname).exists(), f"{fname} not created"

    def test_dataset_split_dirs_created(self, pipeline_result):
        r05 = pipeline_result["r05"]
        if r05.written_total == 0:
            pytest.skip("No patches written")

        dataset_dir = pipeline_result["dataset_dir"]
        for split in ("train", "val", "test"):
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            for layer in ("img", "extent", "boundary", "distance", "valid", "meta"):
                assert (split_dir / layer).exists(), (
                    f"Missing {split}/{layer} directory in dataset"
                )

    def test_norm_stats_json_exists_with_8_bands(self, pipeline_result):
        r05 = pipeline_result["r05"]
        if r05.written_total == 0:
            pytest.skip("No patches written — norm_stats.json cannot be computed")

        norm_path = pipeline_result["dataset_dir"] / "norm_stats.json"
        assert norm_path.exists(), "norm_stats.json not created"

        norm_stats = json.loads(norm_path.read_text(encoding="utf-8"))
        assert "band_stats" in norm_stats, "norm_stats.json missing 'band_stats' key"
        assert len(norm_stats["band_stats"]) == 8, (
            f"Expected 8 band_stats for raw8, got {len(norm_stats['band_stats'])}"
        )

    def test_stage06_split_counts_sum_to_total(self, pipeline_result):
        r05 = pipeline_result["r05"]
        if r05.written_total == 0:
            pytest.skip("No patches written")

        # Read manifest JSON for split counts (not directly on result object)
        r06 = pipeline_result["r06"]
        manifest = json.loads(r06.manifest_path.read_text(encoding="utf-8"))
        splits = manifest.get("splits", {})
        total = sum(
            splits.get(s, {}).get("sample_count", 0)
            for s in ("train", "val", "test")
        )
        assert total == r05.written_total, (
            f"Split counts ({total}) != patches written ({r05.written_total})"
        )

    def test_stage07_runtime_executed_when_patches_exist(self, pipeline_result):
        r05 = pipeline_result["r05"]
        r07 = pipeline_result["r07"]
        if r05.written_total == 0:
            pytest.skip("No patches to validate")

        assert r07.validation_runtime_executed is True, (
            "Stage 07 should have run runtime validation when dataset_dir provided"
        )

    def test_net_train_dataloader_smoke_has_fixed_patch_shapes(self, pipeline_result):
        pytest.importorskip("torch")
        from torch.utils.data import DataLoader  # noqa: PLC0415
        from ai_fields.module_net_train.dataset import (  # noqa: PLC0415
            FieldsDataset,
            fields_collate_fn,
        )

        train_split_dir = pipeline_result["dataset_dir"] / "train"
        ds = FieldsDataset(train_split_dir, feature_mode="raw8", augment=False)
        if len(ds) < 2:
            pytest.skip("Not enough train samples for DataLoader smoke batch of size 2")

        loader = DataLoader(
            ds,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=fields_collate_fn,
        )
        batch = next(iter(loader))
        assert tuple(batch["image"].shape[-2:]) == (512, 512)
