"""Unit tests for validate_outputs_compute (stage 07 runtime helper).

Covers: shapes_ok=True/False, domain ok/fail cases.
"""

from __future__ import annotations

import json
import pytest

rasterio = pytest.importorskip("rasterio")
np = pytest.importorskip("numpy")

from ai_fields.module_prep_data.validate_outputs_compute import (  # noqa: E402
    VALIDATE_COMPUTE_MODE,
    check_target_value_domains,
    check_shapes_consistency,
    check_metadata_contract,
    scan_split_dir,
    validate_dataset,
)
from ai_fields.common.errors import ContractError  # noqa: E402
from ai_fields.module_prep_data.schemas import PrepDataConfig  # noqa: E402


def _write_tif(path, array, profile):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        if array.ndim == 2:
            dst.write(array[np.newaxis, ...])
        else:
            dst.write(array)


def _make_tiny_dataset(dataset_dir, split="train", H=16, W=16):
    """Create a minimal train split with 1 patch for testing."""
    import numpy as np  # noqa: PLC0415
    from rasterio.transform import from_bounds  # noqa: PLC0415

    transform = from_bounds(0, 0, W, H, W, H)
    base_profile = {
        "driver": "GTiff", "crs": "EPSG:32637",
        "transform": transform, "width": W, "height": H,
    }

    split_dir = dataset_dir / split
    patch_id = "patch_000000"

    # img (8 bands)
    img_profile = {**base_profile, "count": 8, "dtype": "float32"}
    _write_tif(split_dir / "img" / f"{patch_id}_img.tif",
               np.ones((8, H, W), dtype=np.float32), img_profile)

    # extent {0,1}
    ext_profile = {**base_profile, "count": 1, "dtype": "uint8"}
    _write_tif(split_dir / "extent" / f"{patch_id}_extent.tif",
               np.zeros((H, W), dtype=np.uint8), ext_profile)

    # boundary {0,1,2}
    bnd_profile = {**base_profile, "count": 1, "dtype": "uint8"}
    _write_tif(split_dir / "boundary" / f"{patch_id}_boundary.tif",
               np.zeros((H, W), dtype=np.uint8), bnd_profile)

    # distance ≥ 0
    dist_profile = {**base_profile, "count": 1, "dtype": "float32"}
    _write_tif(split_dir / "distance" / f"{patch_id}_distance.tif",
               np.zeros((H, W), dtype=np.float32), dist_profile)

    # valid {0,1}
    val_profile = {**base_profile, "count": 1, "dtype": "uint8"}
    _write_tif(split_dir / "valid" / f"{patch_id}_valid.tif",
               np.ones((H, W), dtype=np.uint8), val_profile)

    # meta.json
    meta = {
        "patch_id": patch_id,
        "feature_mode": "raw8",
        "feature_channel_count": 8,
        "valid_ratio": 1.0,
        "sampling_class": "center_positive",
        "xoff": 0, "yoff": 0,
    }
    (split_dir / "meta").mkdir(parents=True, exist_ok=True)
    (split_dir / "meta" / f"{patch_id}_meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    return split_dir


class TestCheckShapesConsistency:
    def test_consistent_dataset_returns_true(self, tmp_path):
        _make_tiny_dataset(tmp_path)
        result = check_shapes_consistency(tmp_path)
        assert result["consistent"] is True
        assert result["mismatches"] == []
        assert result["checked_count"] > 0

    def test_missing_split_dir_is_skipped(self, tmp_path):
        # No train split created → checked_count 0 but no error
        result = check_shapes_consistency(tmp_path)
        assert result["checked_count"] == 0
        assert result["consistent"] is True

    def test_expected_patch_size_mismatch_detected(self, tmp_path):
        _make_tiny_dataset(tmp_path, H=16, W=12)
        result = check_shapes_consistency(tmp_path, expected_patch_size=16)
        assert result["consistent"] is False
        assert any("expected (16,16)" in issue for issue in result["mismatches"])


class TestCheckTargetValueDomains:
    def test_valid_domains_ok(self, tmp_path):
        _make_tiny_dataset(tmp_path)
        result = check_target_value_domains(tmp_path, n_samples=5)
        assert result["all_ok"] is True
        assert result["issues"] == []

    def test_invalid_extent_value_detected(self, tmp_path):
        """Inject value 99 into extent — should be flagged."""
        _make_tiny_dataset(tmp_path)
        extent_path = tmp_path / "train" / "extent" / "patch_000000_extent.tif"
        with rasterio.open(extent_path) as ds:
            data = ds.read()
            profile = ds.profile
        data[0, 0, 0] = 99  # invalid
        with rasterio.open(extent_path, "w", **profile) as dst:
            dst.write(data)
        result = check_target_value_domains(tmp_path, n_samples=5)
        assert result["all_ok"] is False
        assert any("extent" in issue for issue in result["issues"])


class TestCheckMetadataContract:
    def test_valid_metadata_ok(self, tmp_path):
        _make_tiny_dataset(tmp_path)
        result = check_metadata_contract(tmp_path, feature_mode="raw8", n_samples=5)
        assert result["ok"] is True
        assert result["issues"] == []

    def test_wrong_feature_mode_detected(self, tmp_path):
        _make_tiny_dataset(tmp_path)
        result = check_metadata_contract(tmp_path, feature_mode="raw8_idx3", n_samples=5)
        assert result["ok"] is False
        assert any("feature_mode" in issue for issue in result["issues"])


class TestValidateDataset:
    def test_validate_dataset_fails_on_patch_size_contract_violation(self, tmp_path):
        _make_tiny_dataset(tmp_path, H=16, W=16)
        cfg = PrepDataConfig(feature_mode="raw8")
        result = validate_dataset(tmp_path, cfg)
        assert result["shapes_ok"] is False
        assert any("expected (512,512)" in issue for issue in result["issues"])
