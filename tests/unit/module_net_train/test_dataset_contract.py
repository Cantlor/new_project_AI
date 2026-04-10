"""Unit tests for fixed-size dataset contract checks in module_net_train."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai_fields.common.constants import CHANNEL_COUNTS
from ai_fields.common.errors import ContractError
from ai_fields.module_net_train.run_train import validate_train_ready_dataset_contract


def _make_split(
    root: Path,
    split_name: str,
    *,
    feature_mode: str = "raw8",
    patch_size: int = 32,
    n_samples: int = 2,
) -> Path:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    split_dir = root / split_name
    for layer in ("img", "extent", "boundary", "distance", "valid", "meta"):
        (split_dir / layer).mkdir(parents=True, exist_ok=True)

    channels = CHANNEL_COUNTS[feature_mode]
    transform = from_bounds(0.0, 0.0, float(patch_size), float(patch_size), patch_size, patch_size)
    crs = CRS.from_epsg(32642)

    for idx in range(n_samples):
        sample_id = f"{split_name}_{idx:06d}"
        img = np.full((channels, patch_size, patch_size), fill_value=10.0 + idx, dtype=np.float32)
        with rasterio.open(
            split_dir / "img" / f"{sample_id}_img.tif",
            "w",
            driver="GTiff",
            width=patch_size,
            height=patch_size,
            count=channels,
            dtype="float32",
            transform=transform,
            crs=crs,
        ) as ds:
            ds.write(img)

        for layer in ("extent", "boundary", "valid"):
            layer_data = np.ones((1, patch_size, patch_size), dtype=np.uint8)
            with rasterio.open(
                split_dir / layer / f"{sample_id}_{layer}.tif",
                "w",
                driver="GTiff",
                width=patch_size,
                height=patch_size,
                count=1,
                dtype="uint8",
                transform=transform,
                crs=crs,
            ) as ds:
                ds.write(layer_data)

        distance = np.zeros((1, patch_size, patch_size), dtype=np.float32)
        with rasterio.open(
            split_dir / "distance" / f"{sample_id}_distance.tif",
            "w",
            driver="GTiff",
            width=patch_size,
            height=patch_size,
            count=1,
            dtype="float32",
            transform=transform,
            crs=crs,
        ) as ds:
            ds.write(distance)

        (split_dir / "meta" / f"{sample_id}_meta.json").write_text(
            json.dumps({"sample_id": sample_id, "feature_mode": feature_mode}),
            encoding="utf-8",
        )

    return split_dir


class TestValidateTrainReadyDatasetContract:
    def test_happy_path_resolves_fixed_patch_size(self, tmp_path: Path) -> None:
        train_dir = _make_split(tmp_path / "dataset", "train", patch_size=384)
        val_dir = _make_split(tmp_path / "dataset", "val", patch_size=384)

        result = validate_train_ready_dataset_contract(
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            feature_mode="raw8",
            expected_patch_size=384,
        )

        assert result.patch_size == 384
        assert result.train_sample_count == 2
        assert result.val_sample_count == 2
        assert result.feature_mode == "raw8"
        assert result.expected_channels == 8

    def test_mixed_patch_sizes_are_rejected(self, tmp_path: Path) -> None:
        train_dir = _make_split(tmp_path / "dataset", "train", patch_size=512)
        val_dir = _make_split(tmp_path / "dataset", "val", patch_size=256)

        with pytest.raises(ContractError, match="patch size"):
            validate_train_ready_dataset_contract(
                train_split_dir=train_dir,
                val_split_dir=val_dir,
                feature_mode="raw8",
            )

    def test_missing_required_layer_dir_is_rejected(self, tmp_path: Path) -> None:
        train_dir = _make_split(tmp_path / "dataset", "train", patch_size=256)
        val_dir = _make_split(tmp_path / "dataset", "val", patch_size=256)
        missing_meta_dir = val_dir / "meta"
        for path in missing_meta_dir.glob("*"):
            path.unlink()
        missing_meta_dir.rmdir()

        with pytest.raises(ContractError, match="missing required layer directory"):
            validate_train_ready_dataset_contract(
                train_split_dir=train_dir,
                val_split_dir=val_dir,
                feature_mode="raw8",
            )
