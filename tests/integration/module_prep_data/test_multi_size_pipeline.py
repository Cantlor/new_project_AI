"""Integration tests for module_prep_data single-size and multi-size run_pipeline modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("fiona")
pytest.importorskip("geopandas")
pytest.importorskip("scipy")

from ai_fields.module_prep_data.run_pipeline import main as run_prep_pipeline  # noqa: E402
from ai_fields.module_net_train.dataset import list_sample_ids, read_sample  # noqa: E402


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


def _write_config(path: Path, *, feature_mode: str = "raw8", patch_size: int = 512) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = f"""
feature_mode: {feature_mode}
valid_policy:
  nodata_source: metadata_then_config
  compute_before_fill: true
aoi:
  enabled: false
  aoi_path: null
  buffer_m: 30
patches:
  patch_size: {patch_size}
  sampling_policy: strategic
boundary:
  encoding: background_skeleton_buffer
distance:
  target: unsigned_distance_to_boundary
normalization:
  name: robust_percentile
  clip_percentiles: [0.5, 99.5]
  scale_range: [0.0, 1.0]
  stats_computed_on: valid_train_pixels
split:
  policy: spatial_stratified
  random_seed: 42
"""
    path.write_text(cfg.strip() + "\n", encoding="utf-8")
    return path


def _assert_dataset_root_fixed_size(dataset_root: Path, expected_patch_size: int) -> None:
    import rasterio  # noqa: PLC0415

    assert (dataset_root / "train").exists()
    assert (dataset_root / "val").exists()
    assert (dataset_root / "test").exists()
    for split in ("train", "val", "test"):
        img_dir = dataset_root / split / "img"
        for img_path in img_dir.glob("*_img.tif"):
            with rasterio.open(img_path) as ds:
                assert ds.height == expected_patch_size
                assert ds.width == expected_patch_size


class TestRunPipelineMultiSize:
    def test_single_size_mode_still_runs(self, tmp_path: Path) -> None:
        raster = _make_patch_ready_raster(tmp_path / "inputs" / "scene.tif")
        vector = _make_patch_ready_vector(tmp_path / "inputs" / "labels.gpkg")
        config = _write_config(tmp_path / "configs" / "single.yaml", patch_size=512)

        out_root = tmp_path / "runs"
        run_id = "single-size-smoke"
        code = run_prep_pipeline(
            [
                "--config", str(config),
                "--raster", str(raster),
                "--vector", str(vector),
                "--output-dir", str(out_root),
                "--run-id", run_id,
                "--runtime-compute-enabled",
            ]
        )
        assert code == 0

        dataset_root = out_root / run_id / "06_split_dataset" / "dataset"
        _assert_dataset_root_fixed_size(dataset_root, expected_patch_size=512)

    def test_multi_size_mode_exports_separate_fixed_size_datasets(self, tmp_path: Path) -> None:
        raster = _make_patch_ready_raster(tmp_path / "inputs" / "scene.tif")
        vector = _make_patch_ready_vector(tmp_path / "inputs" / "labels.gpkg")
        config = _write_config(tmp_path / "configs" / "multi.yaml", patch_size=512)

        out_root = tmp_path / "runs"
        export_root = tmp_path / "prep_data_for_train"
        run_id = "multi-size-smoke"

        code = run_prep_pipeline(
            [
                "--config", str(config),
                "--raster", str(raster),
                "--vector", str(vector),
                "--output-dir", str(out_root),
                "--run-id", run_id,
                "--runtime-compute-enabled",
                "--patch-sizes", "256,384,512",
                "--multi-size-export-root", str(export_root),
            ]
        )
        assert code == 0

        for size in (256, 384, 512):
            dataset_root = export_root / "raw8" / str(size)
            _assert_dataset_root_fixed_size(dataset_root, expected_patch_size=size)

            train_split_dir = dataset_root / "train"
            sample_ids = list_sample_ids(train_split_dir)
            if sample_ids:
                sample = read_sample(train_split_dir, sample_ids[0], feature_mode="raw8")
                assert sample["img"].shape[1:] == (size, size)
                assert sample["extent"].shape == (size, size)
                assert sample["boundary"].shape == (size, size)
                assert sample["distance"].shape == (size, size)
                assert sample["valid"].shape == (size, size)

            # Per-size stage manifests remain truthful about patch_size.
            split_manifest = out_root / f"{run_id}-ps{size}" / "06_split_dataset" / "split_manifest.json"
            validate_manifest = (
                out_root / f"{run_id}-ps{size}" / "07_validate_outputs" / "validate_outputs_manifest.json"
            )
            split_data = json.loads(split_manifest.read_text(encoding="utf-8"))
            validate_data = json.loads(validate_manifest.read_text(encoding="utf-8"))
            assert split_data["patch_size"] == size
            assert validate_data["patch_size"] == size

        multi_manifest = out_root / f"{run_id}__multi_size" / "multi_size_manifest.json"
        multi_data = json.loads(multi_manifest.read_text(encoding="utf-8"))
        assert multi_data["mode"] == "multi_size"
        assert multi_data["patch_sizes"] == [256, 384, 512]
        assert len(multi_data["runs"]) == 3
        for entry in multi_data["runs"]:
            assert entry["export_dataset_root"]
            assert entry["split_counts"] is not None

    def test_downstream_dataset_loader_smoke_on_one_generated_size(self, tmp_path: Path) -> None:
        raster = _make_patch_ready_raster(tmp_path / "inputs" / "scene.tif")
        vector = _make_patch_ready_vector(tmp_path / "inputs" / "labels.gpkg")
        config = _write_config(tmp_path / "configs" / "multi_loader.yaml", patch_size=512)

        out_root = tmp_path / "runs"
        export_root = tmp_path / "prep_data_for_train"
        run_id = "multi-size-loader-smoke"

        code = run_prep_pipeline(
            [
                "--config", str(config),
                "--raster", str(raster),
                "--vector", str(vector),
                "--output-dir", str(out_root),
                "--run-id", run_id,
                "--runtime-compute-enabled",
                "--patch-sizes", "512",
                "--multi-size-export-root", str(export_root),
            ]
        )
        assert code == 0

        dataset_root = export_root / "raw8" / "512" / "train"
        sample_ids = list_sample_ids(dataset_root)
        if not sample_ids:
            pytest.skip("No train samples produced for loader smoke test.")

        sample = read_sample(dataset_root, sample_ids[0], feature_mode="raw8")
        assert sample["img"].shape[1:] == (512, 512)
        assert sample["extent"].shape == (512, 512)
        assert sample["boundary"].shape == (512, 512)
        assert sample["distance"].shape == (512, 512)
        assert sample["valid"].shape == (512, 512)
