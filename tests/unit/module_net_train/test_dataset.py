"""Unit tests for module_net_train.dataset.

Covers:
  - assemble_model_input(): pure numpy, no rasterio required
  - list_sample_ids():      requires directory structure (tmp_path)
  - read_sample():          requires rasterio (auto-skipped without it)
  - FieldsDataset:          requires torch (auto-skipped without it)

Contract checks (DATA_CONTRACT.md §7.4, §7.5, §6.4):
  - assembled input channel count = feature_mode channels + 1 (for valid)
  - valid is preserved as a separate key (dual role)
  - unsupported feature_mode raises FeatureModeError
  - channel mismatch raises ChannelCountError
  - missing files raise ContractError
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    FeatureModeError,
)
from ai_fields.common.constants import CHANNEL_COUNTS
from ai_fields.module_net_train.dataset import (
    assemble_model_input,
    list_sample_ids,
    read_sample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_split_dir(tmp_path: Path, feature_mode: str = "raw8", n_samples: int = 2) -> Path:
    """Create a minimal canonical split directory with synthetic GeoTIFF patches.

    Skipped automatically if rasterio is not installed.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    n_ch = CHANNEL_COUNTS[feature_mode]
    split_dir = tmp_path / "train"
    for layer in ("img", "extent", "boundary", "distance", "valid", "meta"):
        (split_dir / layer).mkdir(parents=True)

    transform = from_bounds(0, 0, 1, 1, width=32, height=32)
    crs = CRS.from_epsg(32637)

    for i in range(n_samples):
        pid = f"patch_{i:06d}"

        # img  (C, 32, 32) — non-constant so spatial augmentations are detectable
        img_path = split_dir / "img" / f"{pid}_img.tif"
        rng_fixture = np.random.default_rng(i)
        img_data = rng_fixture.random((n_ch, 32, 32)).astype(np.float32) * 1000.0
        with rasterio.open(
            img_path, "w",
            driver="GTiff", height=32, width=32, count=n_ch,
            dtype="float32", crs=crs, transform=transform,
        ) as ds:
            ds.write(img_data)

        # extent  (1, 32, 32) uint8
        for layer_name, dtype, fill in (
            ("extent",   "uint8",   1),
            ("boundary", "uint8",   1),
            ("valid",    "uint8",   1),
        ):
            p = split_dir / layer_name / f"{pid}_{layer_name}.tif"
            with rasterio.open(
                p, "w",
                driver="GTiff", height=32, width=32, count=1,
                dtype=dtype, crs=crs, transform=transform,
            ) as ds:
                ds.write(np.full((1, 32, 32), fill_value=fill, dtype=np.uint8))

        # distance  (1, 32, 32) float32
        dist_path = split_dir / "distance" / f"{pid}_distance.tif"
        with rasterio.open(
            dist_path, "w",
            driver="GTiff", height=32, width=32, count=1,
            dtype="float32", crs=crs, transform=transform,
        ) as ds:
            ds.write(np.full((1, 32, 32), fill_value=5.0, dtype=np.float32))

        # meta
        meta = {
            "patch_id": pid,
            "feature_mode": feature_mode,
            "feature_channel_count": n_ch,
            "channel_names": [f"band_{j+1}" for j in range(n_ch)],
        }
        (split_dir / "meta" / f"{pid}_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

    return split_dir


# ---------------------------------------------------------------------------
# assemble_model_input — pure numpy, no rasterio required
# ---------------------------------------------------------------------------


class TestAssembleModelInput:
    def test_raw8_produces_9ch(self):
        img = np.zeros((8, 16, 16), dtype=np.float32)
        valid = np.ones((16, 16), dtype=np.uint8)
        result = assemble_model_input(img, valid)
        assert result.shape == (9, 16, 16)
        assert result.dtype == np.float32

    def test_raw8_idx3_produces_12ch(self):
        img = np.zeros((11, 16, 16), dtype=np.float32)
        valid = np.ones((16, 16), dtype=np.uint8)
        result = assemble_model_input(img, valid)
        assert result.shape == (12, 16, 16)

    def test_valid_channel_is_last(self):
        img = np.zeros((8, 4, 4), dtype=np.float32)
        valid = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        # Use 2x2 spatial
        img2 = np.zeros((8, 2, 2), dtype=np.float32)
        result = assemble_model_input(img2, valid)
        # Last channel should equal valid cast to float32
        np.testing.assert_array_equal(result[-1], valid.astype(np.float32))

    def test_valid_values_preserved_as_float(self):
        img = np.ones((8, 4, 4), dtype=np.float32)
        valid = np.zeros((4, 4), dtype=np.uint8)
        valid[0, 0] = 1
        result = assemble_model_input(img, valid)
        assert result[-1, 0, 0] == 1.0
        assert result[-1, 1, 1] == 0.0

    def test_original_img_channels_preserved(self):
        img = np.arange(8 * 4 * 4, dtype=np.float32).reshape(8, 4, 4)
        valid = np.ones((4, 4), dtype=np.uint8)
        result = assemble_model_input(img, valid)
        np.testing.assert_array_equal(result[:8], img)

    def test_wrong_img_ndim_raises(self):
        img = np.zeros((8, 16), dtype=np.float32)  # 2D, wrong
        valid = np.ones((16,), dtype=np.uint8)
        with pytest.raises(ContractError, match="3-D"):
            assemble_model_input(img, valid)

    def test_wrong_valid_ndim_raises(self):
        img = np.zeros((8, 16, 16), dtype=np.float32)
        valid = np.ones((1, 16, 16), dtype=np.uint8)  # 3D, wrong
        with pytest.raises(ContractError, match="2-D"):
            assemble_model_input(img, valid)

    def test_shape_mismatch_raises(self):
        img = np.zeros((8, 16, 16), dtype=np.float32)
        valid = np.ones((8, 8), dtype=np.uint8)  # wrong spatial
        with pytest.raises(ContractError, match="shape"):
            assemble_model_input(img, valid)


# ---------------------------------------------------------------------------
# list_sample_ids
# ---------------------------------------------------------------------------


class TestListSampleIds:
    def test_returns_sorted_ids(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=3)
        ids = list_sample_ids(split_dir)
        assert ids == sorted(ids)
        assert len(ids) == 3

    def test_ids_match_file_stems(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=2)
        ids = list_sample_ids(split_dir)
        for sid in ids:
            assert (split_dir / "img" / f"{sid}_img.tif").exists()

    def test_missing_img_dir_raises(self, tmp_path):
        split_dir = tmp_path / "empty_split"
        split_dir.mkdir()
        with pytest.raises(ContractError, match="img sub-directory not found"):
            list_sample_ids(split_dir)

    def test_empty_img_dir_raises(self, tmp_path):
        split_dir = tmp_path / "empty_split"
        (split_dir / "img").mkdir(parents=True)
        with pytest.raises(ContractError, match="No \\*_img.tif files"):
            list_sample_ids(split_dir)


# ---------------------------------------------------------------------------
# read_sample — requires rasterio
# ---------------------------------------------------------------------------


class TestReadSample:
    def test_returns_expected_keys(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        sample = read_sample(split_dir, ids[0], "raw8")
        for key in ("img", "extent", "boundary", "distance", "valid", "meta", "sample_id"):
            assert key in sample, f"Missing key: {key}"

    def test_img_shape_raw8(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        sample = read_sample(split_dir, ids[0], "raw8")
        assert sample["img"].shape == (8, 32, 32)
        assert sample["img"].dtype == np.float32

    def test_img_shape_raw8_idx3(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8_idx3", n_samples=1)
        ids = list_sample_ids(split_dir)
        sample = read_sample(split_dir, ids[0], "raw8_idx3")
        assert sample["img"].shape == (11, 32, 32)

    def test_target_shapes(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        sample = read_sample(split_dir, ids[0], "raw8")
        for layer in ("extent", "boundary", "distance", "valid"):
            assert sample[layer].shape == (32, 32), f"Wrong shape for {layer}"

    def test_valid_is_uint8(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        sample = read_sample(split_dir, ids[0], "raw8")
        assert sample["valid"].dtype == np.uint8

    def test_distance_is_float32(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        sample = read_sample(split_dir, ids[0], "raw8")
        assert sample["distance"].dtype == np.float32

    def test_meta_parsed_correctly(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        sample = read_sample(split_dir, ids[0], "raw8")
        assert isinstance(sample["meta"], dict)
        assert sample["meta"]["feature_mode"] == "raw8"

    def test_sample_id_preserved(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        sample = read_sample(split_dir, ids[0], "raw8")
        assert sample["sample_id"] == ids[0]

    def test_unknown_feature_mode_raises(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        with pytest.raises(FeatureModeError, match="Unsupported feature_mode"):
            read_sample(split_dir, ids[0], "raw99")

    def test_channel_count_mismatch_raises(self, tmp_path):
        pytest.importorskip("rasterio")
        # Create raw8 files but try to read as raw8_idx3 (expects 11ch)
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        with pytest.raises(ChannelCountError):
            read_sample(split_dir, ids[0], "raw8_idx3")

    def test_missing_file_raises(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        # Remove extent file
        extent_file = split_dir / "extent" / f"{ids[0]}_extent.tif"
        extent_file.unlink()
        with pytest.raises(ContractError, match="Required sample file missing"):
            read_sample(split_dir, ids[0], "raw8")


# ---------------------------------------------------------------------------
# valid dual role — assemble + preserved separately
# ---------------------------------------------------------------------------


class TestValidDualRole:
    """DATA_CONTRACT.md §6.4, DEC-002: valid is BOTH mask AND input channel."""

    def test_assemble_includes_valid_as_last_channel(self):
        img = np.zeros((8, 4, 4), dtype=np.float32)
        valid = np.ones((4, 4), dtype=np.uint8)
        assembled = assemble_model_input(img, valid)
        # valid must be the last channel
        np.testing.assert_array_equal(assembled[-1], valid.astype(np.float32))

    def test_read_sample_valid_is_separate_key(self, tmp_path):
        pytest.importorskip("rasterio")
        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ids = list_sample_ids(split_dir)
        sample = read_sample(split_dir, ids[0], "raw8")
        # valid must be a separate key (not embedded in img)
        assert "valid" in sample
        # img must NOT already contain valid (dataset-side only)
        assert sample["img"].shape[0] == 8  # 8ch, not 9ch


# ---------------------------------------------------------------------------
# FieldsDataset — requires torch
# ---------------------------------------------------------------------------


class TestFieldsDataset:
    def test_dataset_len(self, tmp_path):
        pytest.importorskip("rasterio")
        torch = pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=3)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        assert len(ds) == 3

    def test_dataset_assembled_channels_raw8(self, tmp_path):
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        assert ds.assembled_input_channels == 9

    def test_dataset_assembled_channels_raw8_idx3(self, tmp_path):
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8_idx3", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8_idx3")
        assert ds.assembled_input_channels == 12

    def test_getitem_keys(self, tmp_path):
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        item = ds[0]
        for key in ("image", "extent", "boundary", "distance", "valid", "sample_id", "meta"):
            assert key in item, f"Missing key in sample dict: {key}"

    def test_image_tensor_shape_raw8(self, tmp_path):
        pytest.importorskip("rasterio")
        torch = pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        item = ds[0]
        assert item["image"].shape == (9, 32, 32)

    def test_valid_is_bool_tensor_and_separate(self, tmp_path):
        """DATA_CONTRACT.md §6.4: valid kept as separate bool tensor."""
        pytest.importorskip("rasterio")
        torch = pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        item = ds[0]
        assert item["valid"].dtype == torch.bool
        assert item["valid"].shape == (32, 32)
        # image must be 9ch (8 + valid), not 8ch
        assert item["image"].shape[0] == 9

    def test_targets_are_correct_dtype(self, tmp_path):
        pytest.importorskip("rasterio")
        torch = pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        item = ds[0]
        assert item["extent"].dtype == torch.int64
        assert item["boundary"].dtype == torch.int64
        assert item["distance"].dtype == torch.float32

    def test_unknown_feature_mode_raises(self, tmp_path):
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        with pytest.raises(FeatureModeError):
            FieldsDataset(split_dir, feature_mode="raw99")

    def test_no_torch_raises_contract_error(self, tmp_path, monkeypatch):
        pytest.importorskip("rasterio")
        # Simulate torch not available
        import ai_fields.module_net_train.dataset as ds_module
        monkeypatch.setattr(ds_module, "_TORCH_AVAILABLE", False)
        monkeypatch.setattr(ds_module, "_TorchDataset", object)

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        with pytest.raises(ContractError, match="torch is required"):
            ds_module.FieldsDataset(split_dir, feature_mode="raw8")


# ---------------------------------------------------------------------------
# _apply_spatial_augmentation — pure numpy, no rasterio/torch required
# ---------------------------------------------------------------------------


class TestApplySpatialAugmentation:
    """Tests for _apply_spatial_augmentation (module_net_train.md §11.2)."""

    def _make_arrays(self, h: int = 8, w: int = 8, c: int = 8):
        rng = np.random.default_rng(0)
        img = rng.random((c, h, w)).astype(np.float32)
        extent = rng.integers(0, 2, size=(h, w)).astype(np.int64)
        boundary = rng.integers(0, 3, size=(h, w)).astype(np.int64)
        distance = rng.random((h, w)).astype(np.float32)
        valid = rng.integers(0, 2, size=(h, w)).astype(np.uint8)
        return img, extent, boundary, distance, valid

    def test_output_shapes_preserved(self):
        from ai_fields.module_net_train.dataset import _apply_spatial_augmentation

        img, extent, boundary, distance, valid = self._make_arrays(h=6, w=10)
        aug_img, aug_ext, aug_bnd, aug_dst, aug_val = _apply_spatial_augmentation(
            img=img, extent=extent, boundary=boundary, distance=distance, valid=valid,
            rng=np.random.default_rng(42),
        )
        # Flips preserve shape; rotation of non-square may swap dims
        # We just check that spatial dims are consistent across all outputs.
        assert aug_img.shape[0] == img.shape[0], "channel count must not change"
        h_out, w_out = aug_img.shape[1], aug_img.shape[2]
        assert aug_ext.shape == (h_out, w_out)
        assert aug_bnd.shape == (h_out, w_out)
        assert aug_dst.shape == (h_out, w_out)
        assert aug_val.shape == (h_out, w_out)

    def test_no_augmentation_with_fixed_rng(self):
        """k=0 rotation + no flips → output equals input."""
        from ai_fields.module_net_train.dataset import _apply_spatial_augmentation

        img, extent, boundary, distance, valid = self._make_arrays()

        # Force: hflip=0, vflip=0, k=0 by monkeypatching rng
        class _ZeroRng:
            def integers(self, *_args, **_kwargs):
                return 0

        aug_img, aug_ext, aug_bnd, aug_dst, aug_val = _apply_spatial_augmentation(
            img=img, extent=extent, boundary=boundary, distance=distance, valid=valid,
            rng=_ZeroRng(),  # type: ignore[arg-type]
        )
        np.testing.assert_array_equal(aug_img, img)
        np.testing.assert_array_equal(aug_ext, extent)
        np.testing.assert_array_equal(aug_val, valid)

    def test_consistent_spatial_transform_across_layers(self):
        """Rotation must be applied identically to img and all 2-D layers."""
        from ai_fields.module_net_train.dataset import _apply_spatial_augmentation

        # Use a square array so rotation is easy to verify
        img, extent, boundary, distance, valid = self._make_arrays(h=8, w=8)

        class _Rotate90Rng:
            """Always: hflip=0, vflip=0, k=1 (90° CCW)."""
            _calls = 0

            def integers(self, low, high=None, **_kwargs):
                self._calls += 1
                # calls: hflip, vflip, rotation
                if self._calls <= 2:
                    return 0   # no flip
                return 1       # k=1

        aug_img, aug_ext, aug_bnd, aug_dst, aug_val = _apply_spatial_augmentation(
            img=img, extent=extent, boundary=boundary, distance=distance, valid=valid,
            rng=_Rotate90Rng(),  # type: ignore[arg-type]
        )
        expected_img = np.rot90(img, k=1, axes=(1, 2)).copy()
        expected_ext = np.rot90(extent, k=1, axes=(0, 1)).copy()
        np.testing.assert_array_equal(aug_img, expected_img)
        np.testing.assert_array_equal(aug_ext, expected_ext)

    def test_dataset_augment_flag_changes_output(self, tmp_path):
        """FieldsDataset with augment=True should sometimes return different arrays."""
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds_aug = FieldsDataset(split_dir, feature_mode="raw8", augment=True)
        ds_plain = FieldsDataset(split_dir, feature_mode="raw8", augment=False)

        # With augment=True, after many draws at least one should differ from plain.
        import torch
        plain_img = ds_plain[0]["image"].numpy()
        found_difference = False
        for _ in range(32):
            aug_img = ds_aug[0]["image"].numpy()
            if not np.array_equal(aug_img, plain_img):
                found_difference = True
                break
        assert found_difference, (
            "augment=True produced identical output over 32 draws — augmentation not applied"
        )


# ---------------------------------------------------------------------------
# DataLoader collation safety (num_workers > 0 regression guard)
# ---------------------------------------------------------------------------


class TestDataLoaderCollationSafety:
    """Verify that FieldsDataset sample tensors are safe for DataLoader collation.

    Regression guard for: RuntimeError: Trying to resize storage that is not
    resizable — caused by torch.from_numpy() returning tensors backed by
    numpy-owned (non-resizable) storage, which fails during IPC share_memory_()
    when num_workers > 0.

    These tests verify the fix: torch.tensor(...) is used instead of
    torch.from_numpy(), ensuring all sample tensors have PyTorch-owned storage.
    """

    def test_sample_tensors_are_contiguous(self, tmp_path):
        """All tensor fields returned by __getitem__ must be contiguous."""
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        item = ds[0]
        for key in ("image", "extent", "boundary", "distance", "valid"):
            assert item[key].is_contiguous(), (
                f"Tensor '{key}' is not contiguous — DataLoader collation will fail"
            )

    def test_sample_tensors_have_pytorch_storage(self, tmp_path):
        """Tensors must have PyTorch-owned (resizable) storage, not numpy-backed."""
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        item = ds[0]
        for key in ("image", "extent", "boundary", "distance", "valid"):
            t = item[key]
            # torch.from_numpy returns a tensor whose data_ptr matches the numpy array.
            # torch.tensor(...) always copies, so the storage is PyTorch-owned.
            # We verify by checking that calling share_memory_() does not raise.
            try:
                t.share_memory_()
            except RuntimeError as exc:
                pytest.fail(
                    f"Tensor '{key}' has non-resizable (numpy-backed) storage: {exc}"
                )

    def test_sample_tensors_contiguous_after_no_augmentation(self, tmp_path):
        """Contiguity must hold even when augmentation leaves arrays unmodified
        (the zero-transform path, p=6.25%, previously returned raw rasterio views)."""
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8", augment=False)
        item = ds[0]
        for key in ("image", "extent", "boundary", "distance", "valid"):
            assert item[key].is_contiguous(), (
                f"Tensor '{key}' not contiguous on no-augmentation path"
            )

    def test_dataloader_num_workers_1_fetches_batch(self, tmp_path):
        """DataLoader with num_workers=1 must successfully fetch one batch.

        This is the end-to-end regression guard for the collation bug.
        Skipped on platforms where fork-based multiprocessing is unreliable
        (e.g. some WSL2 environments).
        """
        pytest.importorskip("rasterio")
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=4)
        ds = FieldsDataset(split_dir, feature_mode="raw8")

        loader = DataLoader(ds, batch_size=2, num_workers=1, persistent_workers=False)
        try:
            batch = next(iter(loader))
        except RuntimeError as exc:
            if "resize storage" in str(exc) or "not resizable" in str(exc):
                pytest.fail(
                    f"DataLoader collation crashed with storage error (regression): {exc}"
                )
            # Other RuntimeErrors (e.g. fork-related on this platform) are skipped.
            pytest.skip(f"DataLoader multiprocessing unavailable on this platform: {exc}")

        assert "image" in batch
        assert batch["image"].shape == (2, 9, 32, 32)
        assert batch["image"].dtype == torch.float32

    def test_contract_dtypes_unchanged(self, tmp_path):
        """Fixing storage must not change tensor dtypes (regression guard)."""
        pytest.importorskip("rasterio")
        torch = pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        item = ds[0]
        assert item["image"].dtype    == torch.float32
        assert item["extent"].dtype   == torch.int64
        assert item["boundary"].dtype == torch.int64
        assert item["distance"].dtype == torch.float32
        assert item["valid"].dtype    == torch.bool

    def test_contract_shapes_unchanged(self, tmp_path):
        """Fixing storage must not change tensor shapes (regression guard)."""
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=1)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        item = ds[0]
        assert item["image"].shape    == (9, 32, 32)
        assert item["extent"].shape   == (32, 32)
        assert item["boundary"].shape == (32, 32)
        assert item["distance"].shape == (32, 32)
        assert item["valid"].shape    == (32, 32)


# ---------------------------------------------------------------------------
# fields_collate_fn — batch collation contract
# ---------------------------------------------------------------------------


class TestFieldsCollateFn:
    """Verify fields_collate_fn produces the correct batch structure.

    fields_collate_fn replaces default_collate so that:
      - tensor fields are stacked correctly
      - 'sample_id' is left as list[str] (not a tensor)
      - 'meta' is left as list[dict] (not recursively tensor-ified)

    The 'meta' protection is critical in production: default_collate recurses
    into meta and (a) converts int/float values to tensors via torch.tensor(),
    which are numpy-backed when the call uses torch.as_tensor() on numpy
    scalars, and (b) raises TypeError if any value is None (e.g. source_crs)
    and another sample has a string for the same key.
    """

    def test_stacks_tensor_fields(self, tmp_path):
        """fields_collate_fn must stack the five tensor fields into (B, ...) tensors."""
        pytest.importorskip("rasterio")
        torch = pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset, fields_collate_fn

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=3)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        batch = fields_collate_fn([ds[i] for i in range(3)])

        assert batch["image"].shape    == (3, 9, 32, 32)
        assert batch["extent"].shape   == (3, 32, 32)
        assert batch["boundary"].shape == (3, 32, 32)
        assert batch["distance"].shape == (3, 32, 32)
        assert batch["valid"].shape    == (3, 32, 32)
        assert batch["image"].dtype    == torch.float32
        assert batch["extent"].dtype   == torch.int64
        assert batch["boundary"].dtype == torch.int64
        assert batch["distance"].dtype == torch.float32
        assert batch["valid"].dtype    == torch.bool

    def test_sample_id_is_list_not_tensor(self, tmp_path):
        """fields_collate_fn must leave 'sample_id' as list[str], not a tensor."""
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset, fields_collate_fn

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=2)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        batch = fields_collate_fn([ds[0], ds[1]])

        assert isinstance(batch["sample_id"], list)
        assert len(batch["sample_id"]) == 2
        assert all(isinstance(sid, str) for sid in batch["sample_id"])

    def test_meta_is_list_of_dicts_not_collated(self, tmp_path):
        """fields_collate_fn must leave 'meta' as list[dict], not tensor-ified."""
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset, fields_collate_fn

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=2)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        batch = fields_collate_fn([ds[0], ds[1]])

        assert isinstance(batch["meta"], list)
        assert len(batch["meta"]) == 2
        for m in batch["meta"]:
            assert isinstance(m, dict)

    def test_meta_with_none_values_does_not_crash(self, tmp_path):
        """fields_collate_fn must handle meta dicts where some values are None.

        In production, source_crs can be None for some patches and a string for
        others.  default_collate would raise TypeError on this mixed-type list;
        fields_collate_fn avoids that by not touching meta at all.
        """
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset, fields_collate_fn

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=2)
        ds = FieldsDataset(split_dir, feature_mode="raw8")

        # Patch meta in-memory to simulate production samples where
        # source_crs is None in one sample and a string in another.
        item0 = ds[0]
        item1 = ds[1]
        item0["meta"]["source_crs"] = None
        item1["meta"]["source_crs"] = "EPSG:32637"

        # Must not raise TypeError or any other error.
        batch = fields_collate_fn([item0, item1])
        assert batch["meta"][0]["source_crs"] is None
        assert batch["meta"][1]["source_crs"] == "EPSG:32637"

    def test_collate_fn_batch_storage_is_resizable(self, tmp_path):
        """Stacked tensors produced by fields_collate_fn must have PyTorch-owned
        (resizable) storage — verified by calling share_memory_() without error.
        This is the IPC-safety check for num_workers > 0."""
        pytest.importorskip("rasterio")
        pytest.importorskip("torch")
        from ai_fields.module_net_train.dataset import FieldsDataset, fields_collate_fn

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=2)
        ds = FieldsDataset(split_dir, feature_mode="raw8")
        batch = fields_collate_fn([ds[0], ds[1]])

        for key in ("image", "extent", "boundary", "distance", "valid"):
            try:
                batch[key].share_memory_()
            except RuntimeError as exc:
                pytest.fail(
                    f"Batched tensor '{key}' has non-resizable storage: {exc}"
                )

    def test_dataloader_with_collate_fn_and_augment(self, tmp_path):
        """DataLoader using fields_collate_fn + augment=True must fetch a batch.

        Regression guard for the zero-transform augmentation path (p=6.25%):
        when no flip or rotation is applied, arrays are returned unmodified
        from read_sample().  With augment=True, this path can be hit on any
        iteration.  The collate_fn must handle it safely with num_workers > 0.
        """
        pytest.importorskip("rasterio")
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        from ai_fields.module_net_train.dataset import FieldsDataset, fields_collate_fn

        split_dir = _make_split_dir(tmp_path, "raw8", n_samples=8)
        ds = FieldsDataset(split_dir, feature_mode="raw8", augment=True)

        loader = DataLoader(
            ds,
            batch_size=4,
            num_workers=1,
            collate_fn=fields_collate_fn,
            persistent_workers=False,
        )
        try:
            batch = next(iter(loader))
        except RuntimeError as exc:
            if "resize storage" in str(exc) or "not resizable" in str(exc):
                pytest.fail(
                    f"fields_collate_fn with augment=True crashed with storage error: {exc}"
                )
            pytest.skip(f"DataLoader multiprocessing unavailable on this platform: {exc}")

        assert batch["image"].shape == (4, 9, 32, 32)
        assert batch["image"].dtype == torch.float32
        assert isinstance(batch["sample_id"], list)
        assert isinstance(batch["meta"], list)
