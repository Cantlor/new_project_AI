"""Unit tests for split_compute (stage 06 runtime helper).

Covers: split sizes ~70/15/15, export dirs created, norm_stats.json band count.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")

from ai_fields.module_prep_data.split_compute import (  # noqa: E402
    SPLIT_COMPUTE_MODE,
    assign_splits,
    load_patch_meta_list,
)
from ai_fields.common.errors import ContractError  # noqa: E402


def _make_large_nodata_raster(path: Path, *, width: int = 700, height: int = 700) -> Path:
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


def _make_meta_list(n: int) -> list[dict]:
    return [{"patch_id": f"patch_{i:06d}", "xoff": i * 10, "yoff": 0} for i in range(n)]


class TestAssignSplits:
    def test_random_split_sizes(self):
        meta = _make_meta_list(100)
        splits = assign_splits(meta, policy="random", random_seed=42)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 100

    def test_random_split_approximately_70_15_15(self):
        meta = _make_meta_list(100)
        splits = assign_splits(meta, policy="random", random_seed=42)
        assert 60 <= len(splits["train"]) <= 80
        assert 5 <= len(splits["val"]) <= 25
        assert 5 <= len(splits["test"]) <= 25

    def test_spatial_stratified_split(self):
        meta = _make_meta_list(50)
        splits = assign_splits(meta, policy="spatial_stratified", random_seed=None)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 50

    def test_reproducible_with_seed(self):
        meta = _make_meta_list(80)
        s1 = assign_splits(meta, policy="random", random_seed=7)
        s2 = assign_splits(meta, policy="random", random_seed=7)
        assert s1["train"] == s2["train"]

    def test_empty_raises_contract_error(self):
        with pytest.raises(ContractError):
            assign_splits([], policy="random", random_seed=42)


class TestLoadPatchMetaList:
    def test_missing_dir_raises_contract_error(self, tmp_path):
        with pytest.raises(ContractError):
            load_patch_meta_list(tmp_path / "nonexistent_dir")

    def test_empty_dir_raises_contract_error(self, tmp_path):
        empty = tmp_path / "empty_patches"
        empty.mkdir()
        with pytest.raises(ContractError):
            load_patch_meta_list(empty)

    def test_loads_meta_files(self, tmp_path):
        patches_dir = tmp_path / "patches"
        patches_dir.mkdir()
        for i in range(3):
            meta = {"patch_id": f"patch_{i:06d}", "xoff": i, "yoff": 0}
            (patches_dir / f"patch_{i:06d}_meta.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
        metas = load_patch_meta_list(patches_dir)
        assert len(metas) == 3
        assert metas[0]["patch_id"] == "patch_000000"


class TestComputeAndSaveSplit:
    """Integration-level test using patches fixtures from conftest."""

    def test_export_dirs_created(self, tiny_vector_path, tmp_path):
        pytest.importorskip("rasterio")
        from ai_fields.module_prep_data.features_compute import compute_and_save_features  # noqa: PLC0415
        from ai_fields.module_prep_data.targets_compute import compute_and_save_targets  # noqa: PLC0415
        from ai_fields.module_prep_data.patches_compute import compute_and_save_patches  # noqa: PLC0415
        from ai_fields.module_prep_data.split_compute import compute_and_save_split  # noqa: PLC0415
        from ai_fields.module_prep_data.schemas import PrepDataConfig  # noqa: PLC0415

        raster_path = _make_large_nodata_raster(tmp_path / "raster_fixtures" / "split_case.tif")
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

        # Generate a few real patches first
        patches_out = tmp_path / "patches_stage"
        config = PrepDataConfig(feature_mode="raw8")
        patch_result = compute_and_save_patches(
            img_path=features["img_path"],
            extent_path=targets["extent_path"],
            boundary_path=targets["boundary_path"],
            distance_path=targets["distance_path"],
            valid_path=targets["valid_path"],
            output_dir=patches_out,
            config=config,
            feature_mode="raw8",
        )

        assert patch_result["written_total"] > 0

        split_out = tmp_path / "split_stage"
        result = compute_and_save_split(
            patches_dir=patch_result["patches_subdir"],
            output_dir=split_out,
            config=config,
        )
        assert result["split_assignment_executed"] is True
        assert result["export_layout_materialized"] is True
        total = result["train_count"] + result["val_count"] + result["test_count"]
        assert total == patch_result["written_total"]

    def test_norm_stats_json_created(self, tiny_vector_path, tmp_path):
        pytest.importorskip("rasterio")
        from ai_fields.module_prep_data.features_compute import compute_and_save_features  # noqa: PLC0415
        from ai_fields.module_prep_data.targets_compute import compute_and_save_targets  # noqa: PLC0415
        from ai_fields.module_prep_data.patches_compute import compute_and_save_patches  # noqa: PLC0415
        from ai_fields.module_prep_data.split_compute import compute_and_save_split  # noqa: PLC0415
        from ai_fields.module_prep_data.schemas import PrepDataConfig  # noqa: PLC0415

        raster_path = _make_large_nodata_raster(tmp_path / "raster_fixtures" / "split_case_norm.tif")
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

        patches_out = tmp_path / "patches_stage2"
        config = PrepDataConfig(feature_mode="raw8")
        patch_result = compute_and_save_patches(
            img_path=features["img_path"],
            extent_path=targets["extent_path"],
            boundary_path=targets["boundary_path"],
            distance_path=targets["distance_path"],
            valid_path=targets["valid_path"],
            output_dir=patches_out,
            config=config,
            feature_mode="raw8",
        )
        assert patch_result["written_total"] > 0

        split_out = tmp_path / "split_stage2"
        result = compute_and_save_split(
            patches_dir=patch_result["patches_subdir"],
            output_dir=split_out,
            config=config,
        )
        norm_path = split_out / "dataset" / "norm_stats.json"
        assert norm_path.exists(), "norm_stats.json not created"
        norm_stats = json.loads(norm_path.read_text())
        assert "band_stats" in norm_stats
        assert len(norm_stats["band_stats"]) == 8  # raw8 → 8 bands

    def test_rejects_malformed_patch_size_before_export(self, tmp_path):
        from rasterio.crs import CRS  # noqa: PLC0415
        from rasterio.transform import from_bounds  # noqa: PLC0415
        from ai_fields.module_prep_data.split_compute import compute_and_save_split  # noqa: PLC0415
        from ai_fields.module_prep_data.schemas import PrepDataConfig  # noqa: PLC0415

        patches_dir = tmp_path / "patches_bad"
        patches_dir.mkdir(parents=True, exist_ok=True)
        pid = "patch_000000"

        transform = from_bounds(0, 0, 338, 512, 338, 512)
        profile = {
            "driver": "GTiff",
            "height": 512,
            "width": 338,
            "count": 1,
            "dtype": "uint8",
            "crs": CRS.from_epsg(32637),
            "transform": transform,
        }
        layers = ("extent", "boundary", "valid")
        for layer in layers:
            with rasterio.open(patches_dir / f"{pid}_{layer}.tif", "w", **profile) as ds:
                ds.write(np.zeros((1, 512, 338), dtype=np.uint8))

        img_profile = dict(profile)
        img_profile["count"] = 8
        img_profile["dtype"] = "float32"
        with rasterio.open(patches_dir / f"{pid}_img.tif", "w", **img_profile) as ds:
            ds.write(np.zeros((8, 512, 338), dtype=np.float32))

        distance_profile = dict(profile)
        distance_profile["count"] = 1
        distance_profile["dtype"] = "float32"
        with rasterio.open(patches_dir / f"{pid}_distance.tif", "w", **distance_profile) as ds:
            ds.write(np.zeros((1, 512, 338), dtype=np.float32))

        (patches_dir / f"{pid}_meta.json").write_text(
            json.dumps({"patch_id": pid, "xoff": 0, "yoff": 0}),
            encoding="utf-8",
        )

        cfg = PrepDataConfig(feature_mode="raw8")
        with pytest.raises(ContractError, match="Patch size contract violation before split export"):
            compute_and_save_split(
                patches_dir=patches_dir,
                output_dir=tmp_path / "split_bad",
                config=cfg,
            )
