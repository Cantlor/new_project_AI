"""Unit tests for patches_compute (stage 05 runtime helper).

Covers: classify_patch logic, patch shape invariants, meta.json keys.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

rasterio = pytest.importorskip("rasterio")
np = pytest.importorskip("numpy")

from ai_fields.module_prep_data.patches_compute import (  # noqa: E402
    PATCHES_COMPUTE_MODE,
    classify_patch,
    generate_patch_windows,
)


def _make_large_nodata_raster(path: Path, *, width: int = 512, height: int = 512) -> Path:
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


class TestClassifyPatch:
    def test_reject_below_threshold(self):
        assert classify_patch(valid_ratio=0.04, edge_ratio=0.0) is None

    def test_near_invalid(self):
        cls = classify_patch(valid_ratio=0.3, edge_ratio=0.05)
        assert cls == "near_invalid"

    def test_boundary_positive(self):
        cls = classify_patch(valid_ratio=0.8, edge_ratio=0.15)
        assert cls == "boundary_positive"

    def test_hard_negative(self):
        cls = classify_patch(valid_ratio=0.9, edge_ratio=0.0)
        assert cls == "hard_negative"

    def test_center_positive(self):
        cls = classify_patch(valid_ratio=0.9, edge_ratio=0.05)
        assert cls == "center_positive"


class TestGeneratePatchWindows:
    def test_returns_list(self):
        windows = generate_patch_windows(height=64, width=64, patch_size=32)
        assert isinstance(windows, list)
        assert len(windows) > 0

    def test_window_dict_keys(self):
        windows = generate_patch_windows(height=64, width=64, patch_size=32)
        for w in windows:
            assert "xoff" in w
            assert "yoff" in w
            assert "width" in w
            assert "height" in w

    def test_all_windows_within_bounds(self):
        h, w, p = 64, 64, 32
        windows = generate_patch_windows(height=h, width=w, patch_size=p)
        for win in windows:
            assert win["xoff"] >= 0
            assert win["yoff"] >= 0
            assert win["xoff"] + win["width"] <= w
            assert win["yoff"] + win["height"] <= h
            assert win["width"] == p
            assert win["height"] == p

    def test_right_and_bottom_edges_use_full_size_windows(self):
        windows = generate_patch_windows(height=700, width=700, patch_size=512, stride=256)
        offsets = {(w["xoff"], w["yoff"]) for w in windows}
        assert (188, 188) in offsets  # 700 - 512
        for win in windows:
            assert win["width"] == 512
            assert win["height"] == 512

    def test_raster_smaller_than_patch_size_returns_no_windows(self):
        windows = generate_patch_windows(height=16, width=16, patch_size=512)
        assert windows == []


class TestComputeAndSavePatches:
    """Integration-level test using fixture dirs from conftest."""

    def test_small_raster_without_full_windows_raises_contract_error(
        self, tiny_feature_stack_dir, tiny_targets_dir, tmp_path
    ):
        from ai_fields.module_prep_data.patches_compute import compute_and_save_patches  # noqa: PLC0415
        from ai_fields.module_prep_data.schemas import PrepDataConfig  # noqa: PLC0415
        from ai_fields.common.errors import ContractError  # noqa: PLC0415

        config = PrepDataConfig(feature_mode="raw8")
        with pytest.raises(ContractError, match="No full-size patch windows"):
            compute_and_save_patches(
                img_path=tiny_feature_stack_dir / "img.tif",
                extent_path=tiny_targets_dir / "extent.tif",
                boundary_path=tiny_targets_dir / "boundary.tif",
                distance_path=tiny_targets_dir / "distance.tif",
                valid_path=tiny_targets_dir / "valid.tif",
                output_dir=tmp_path,
                config=config,
                feature_mode="raw8",
            )

    def test_meta_json_keys(self, tmp_path, tiny_vector_path):
        from ai_fields.module_prep_data.patches_compute import compute_and_save_patches  # noqa: PLC0415
        from ai_fields.module_prep_data.features_compute import compute_and_save_features  # noqa: PLC0415
        from ai_fields.module_prep_data.targets_compute import compute_and_save_targets  # noqa: PLC0415
        from ai_fields.module_prep_data.schemas import PrepDataConfig  # noqa: PLC0415

        raster_path = _make_large_nodata_raster(
            tmp_path / "raster_fixtures" / "meta_case.tif", width=512, height=512
        )
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

        config = PrepDataConfig(feature_mode="raw8")
        result = compute_and_save_patches(
            img_path=features["img_path"],
            extent_path=targets["extent_path"],
            boundary_path=targets["boundary_path"],
            distance_path=targets["distance_path"],
            valid_path=targets["valid_path"],
            output_dir=tmp_path,
            config=config,
            feature_mode="raw8",
        )
        patches_subdir = result["patches_subdir"]
        meta_files = list(patches_subdir.glob("*_meta.json"))
        if meta_files:
            meta = json.loads(meta_files[0].read_text())
            required_keys = {"patch_id", "feature_mode", "xoff", "yoff", "valid_ratio", "sampling_class"}
            for key in required_keys:
                assert key in meta, f"Missing meta key: {key}"

    def test_semantic_uint8_patch_layers_do_not_inherit_large_source_nodata(
        self, tmp_path, tiny_vector_path
    ):
        from ai_fields.module_prep_data.features_compute import compute_and_save_features  # noqa: PLC0415
        from ai_fields.module_prep_data.targets_compute import compute_and_save_targets  # noqa: PLC0415
        from ai_fields.module_prep_data.patches_compute import compute_and_save_patches  # noqa: PLC0415
        from ai_fields.module_prep_data.schemas import PrepDataConfig  # noqa: PLC0415

        raster_path = _make_large_nodata_raster(
            tmp_path / "raster_fixtures" / "large_nodata.tif", width=700, height=700
        )
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

        config = PrepDataConfig(feature_mode="raw8")
        result = compute_and_save_patches(
            img_path=features["img_path"],
            extent_path=targets["extent_path"],
            boundary_path=targets["boundary_path"],
            distance_path=targets["distance_path"],
            valid_path=targets["valid_path"],
            output_dir=tmp_path / "patches",
            config=config,
            feature_mode="raw8",
        )

        assert result["written_total"] >= 1
        patches_subdir = result["patches_subdir"]
        extent_patch = sorted(patches_subdir.glob("*_extent.tif"))[0]
        boundary_patch = sorted(patches_subdir.glob("*_boundary.tif"))[0]
        valid_patch = sorted(patches_subdir.glob("*_valid.tif"))[0]
        img_patch = Path(str(extent_patch).replace("_extent.tif", "_img.tif"))

        with rasterio.open(img_patch) as img_ds:
            ref_crs = img_ds.crs
            ref_transform = img_ds.transform
            ref_shape = (img_ds.height, img_ds.width)

        with rasterio.open(extent_patch) as ds:
            assert ds.dtypes[0] == "uint8"
            assert ds.nodata is None
            assert ds.crs == ref_crs
            assert ds.transform == ref_transform
            assert (ds.height, ds.width) == ref_shape
            values = set(int(v) for v in np.unique(ds.read(1)))
            assert values.issubset({0, 1, 255})

        with rasterio.open(boundary_patch) as ds:
            assert ds.dtypes[0] == "uint8"
            assert ds.nodata is None
            values = set(int(v) for v in np.unique(ds.read(1)))
            assert values.issubset({0, 1, 2})

        with rasterio.open(valid_patch) as ds:
            assert ds.dtypes[0] == "uint8"
            assert ds.nodata is None
            values = set(int(v) for v in np.unique(ds.read(1)))
            assert values.issubset({0, 1})

    def test_border_windows_do_not_export_partial_size_patches(self, tmp_path, tiny_vector_path):
        from ai_fields.module_prep_data.features_compute import compute_and_save_features  # noqa: PLC0415
        from ai_fields.module_prep_data.targets_compute import compute_and_save_targets  # noqa: PLC0415
        from ai_fields.module_prep_data.patches_compute import compute_and_save_patches  # noqa: PLC0415
        from ai_fields.module_prep_data.schemas import PrepDataConfig  # noqa: PLC0415

        raster_path = _make_large_nodata_raster(
            tmp_path / "raster_fixtures" / "border_case.tif", width=700, height=700
        )
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

        config = PrepDataConfig(feature_mode="raw8")
        result = compute_and_save_patches(
            img_path=features["img_path"],
            extent_path=targets["extent_path"],
            boundary_path=targets["boundary_path"],
            distance_path=targets["distance_path"],
            valid_path=targets["valid_path"],
            output_dir=tmp_path / "patches",
            config=config,
            feature_mode="raw8",
        )

        assert result["written_total"] > 0
        for img_path in result["patches_subdir"].glob("*_img.tif"):
            with rasterio.open(img_path) as ds:
                assert ds.height == 512
                assert ds.width == 512
