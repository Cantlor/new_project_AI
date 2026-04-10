"""Unit tests for module_target_predict tiled inference engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.module_target_predict.tiled_predict import (
    TiledPredictResult,
    build_gaussian_kernel,
    cleanup_tiled_predict_result,
    generate_tile_offsets,
    run_tiled_predict,
)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import rasterio

    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None  # type: ignore[assignment]
    _RASTERIO_AVAILABLE = False

_requires_torch_rasterio = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _RASTERIO_AVAILABLE),
    reason="torch and rasterio are required",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_raster(
    path: Path,
    *,
    count: int = 8,
    width: int = 6,
    height: int = 5,
    nodata: float | None = -9999.0,
) -> Path:
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    data = np.zeros((count, height, width), dtype=np.float32)
    for b in range(count):
        data[b, :, :] = float(b + 1) + np.linspace(0.0, 1.0, width)[np.newaxis, :]
    if nodata is not None:
        data[0, 0, 0] = float(nodata)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="float32",
        crs="EPSG:32637",
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(data)
    return path


def _write_norm_stats(path: Path, *, channels: int) -> Path:
    band_stats = []
    for idx in range(channels):
        p_lo = 0.0 if idx < 8 else -1.0
        p_hi = 10.0 if idx < 8 else 1.0
        band_stats.append({"band_idx": idx, "p_lo": p_lo, "p_hi": p_hi})
    payload = {
        "clip_percentiles": [0.5, 99.5],
        "band_stats": band_stats,
        "computed_on": "valid_train_pixels",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_norm_stats_mapping(*, channels: int) -> dict[str, Any]:
    band_stats = []
    for idx in range(channels):
        p_lo = 0.0 if idx < 8 else -1.0
        p_hi = 10.0 if idx < 8 else 1.0
        band_stats.append({"band_idx": idx, "p_lo": p_lo, "p_hi": p_hi})
    return {"band_stats": band_stats}


def _make_dummy_contract(
    tmp_path: Path,
    *,
    feature_mode: str = "raw8",
) -> Any:
    """Build a CheckpointDrivenPredictContract directly (no file loading)."""
    from ai_fields.common.constants import DATA_CONTRACT_VERSION
    from ai_fields.module_target_predict.checkpoint_contract import (
        CheckpointDrivenPredictContract,
    )

    in_channels = 9 if feature_mode == "raw8" else 12
    feature_channel_count = 8 if feature_mode == "raw8" else 11
    assembled = f"{feature_mode}_valid"

    if feature_mode == "raw8":
        semantics = (
            "coastal", "blue", "green", "yellow",
            "red", "rededge", "nir1", "nir2", "valid",
        )
    else:
        semantics = (
            "coastal", "blue", "green", "yellow",
            "red", "rededge", "nir1", "nir2",
            "NDVI", "SAVI", "NDWI", "valid",
        )

    # Dummy files (content doesn't matter for tiled_predict; it doesn't re-read them)
    ckpt_path = tmp_path / "dummy.ckpt"
    ckpt_path.write_bytes(b"dummy")
    meta_path = tmp_path / "checkpoint_metadata.json"
    meta_path.write_bytes(b"{}")

    return CheckpointDrivenPredictContract(
        checkpoint_path=ckpt_path,
        checkpoint_metadata_path=meta_path,
        train_manifest_path=None,
        config_used_path=None,
        feature_mode=feature_mode,
        assembled_model_input=assembled,
        in_channels=in_channels,
        feature_channel_count=feature_channel_count,
        channel_semantics=semantics,
        valid_as_input_channel=True,
        normalization={
            "normalization_name": "per_band_robust_percentile",
            "stats_source": "norm_stats.json",
            "clip_percentiles": [0.5, 99.5],
            "scaling_range": [0.0, 1.0],
        },
        target_heads={
            "extent": {"type": "binary_segmentation", "ignore_label": 255},
            "boundary": {"type": "multiclass_segmentation"},
            "distance": {"type": "regression"},
        },
        model_version="v1-baseline",
        model_architecture=None,
        model_encoder_depth=None,
        model_base_channels=None,
        required_outputs=("extent_prob", "boundary_prob", "distance_pred", "valid"),
        output_raster_filenames={
            "extent_prob": "extent_prob.tif",
            "boundary_prob": "boundary_prob.tif",
            "distance_pred": "distance_pred.tif",
            "valid": "valid.tif",
        },
    )


def _make_constant_model(*, in_channels: int, device: str = "cpu") -> Any:
    """Return a LoadedPredictModel wrapping a model that outputs zeros."""

    class _ConstantModel(torch.nn.Module):
        def forward(self, x):  # type: ignore[override]
            b, c, h, w = x.shape
            return {
                "extent": torch.zeros(b, 1, h, w, dtype=x.dtype, device=x.device),
                "boundary": torch.zeros(b, 3, h, w, dtype=x.dtype, device=x.device),
                "distance": torch.zeros(b, 1, h, w, dtype=x.dtype, device=x.device),
                "aux": [],
            }

    from ai_fields.module_target_predict.inference_core import LoadedPredictModel

    return LoadedPredictModel(
        model=_ConstantModel().to(device),
        device=device,
        in_channels=in_channels,
        encoder_depth=3,
        base_channels=8,
        architecture="edge_aware_multitask_v1",
        model_shape_source="legacy_state_dict_fallback",
        aux_head_count=0,
        checkpoint_epochs_completed=None,
    )


# ===========================================================================
# TestGenerateTileOffsets
# ===========================================================================

class TestGenerateTileOffsets:
    def test_single_tile_when_scene_equals_tile_size(self) -> None:
        offsets = generate_tile_offsets(total_size=512, tile_size=512, stride=384)
        assert offsets == [0]

    def test_single_tile_when_scene_smaller_than_tile_size(self) -> None:
        offsets = generate_tile_offsets(total_size=400, tile_size=512, stride=384)
        assert offsets == [0]

    def test_exact_fit_two_tiles(self) -> None:
        # total=8, tile=4, stride=4 → offsets [0, 4]
        offsets = generate_tile_offsets(total_size=8, tile_size=4, stride=4)
        assert offsets == [0, 4]

    def test_last_tile_covers_end(self) -> None:
        # total=10, tile=4, stride=4 → regular [0,4,8]; last_start=10-4=6
        # 8+4=12>10, so regular gives [0,4]; last_start=6 → [0,4,6]
        offsets = generate_tile_offsets(total_size=10, tile_size=4, stride=4)
        assert 0 in offsets
        last = offsets[-1]
        assert last + 4 == 10  # last tile covers the end

    def test_result_is_sorted(self) -> None:
        offsets = generate_tile_offsets(total_size=20, tile_size=5, stride=3)
        assert offsets == sorted(offsets)

    def test_no_duplicates(self) -> None:
        offsets = generate_tile_offsets(total_size=20, tile_size=5, stride=5)
        assert len(offsets) == len(set(offsets))

    def test_full_coverage(self) -> None:
        # Every position in [0, total_size) is covered by at least one tile.
        total_size = 15
        tile_size = 4
        stride = 3
        offsets = generate_tile_offsets(total_size, tile_size, stride)
        covered = np.zeros(total_size, dtype=bool)
        for off in offsets:
            covered[off:off + tile_size] = True
        assert np.all(covered)

    def test_invalid_total_size_raises(self) -> None:
        from ai_fields.common.errors import ContractError

        with pytest.raises(ContractError):
            generate_tile_offsets(total_size=0, tile_size=4, stride=4)

    def test_invalid_tile_size_raises(self) -> None:
        from ai_fields.common.errors import ContractError

        with pytest.raises(ContractError):
            generate_tile_offsets(total_size=10, tile_size=-1, stride=4)

    def test_invalid_stride_raises(self) -> None:
        from ai_fields.common.errors import ContractError

        with pytest.raises(ContractError):
            generate_tile_offsets(total_size=10, tile_size=4, stride=0)


# ===========================================================================
# TestBuildGaussianKernel
# ===========================================================================

class TestBuildGaussianKernel:
    def test_shape(self) -> None:
        kernel = build_gaussian_kernel(8)
        assert kernel.shape == (8, 8)

    def test_dtype_float32(self) -> None:
        kernel = build_gaussian_kernel(8)
        assert kernel.dtype == np.float32

    def test_center_is_maximum(self) -> None:
        kernel = build_gaussian_kernel(8)
        center_r, center_c = 4, 4  # for tile_size=8, center = index 3.5 → peak at [3,3] and [4,4]
        assert kernel[4, 4] == kernel.max()

    def test_all_positive(self) -> None:
        kernel = build_gaussian_kernel(16)
        assert np.all(kernel > 0.0)

    def test_corners_less_than_center(self) -> None:
        kernel = build_gaussian_kernel(16)
        center_val = kernel[8, 8]
        corner_val = kernel[0, 0]
        assert corner_val < center_val

    def test_symmetric(self) -> None:
        kernel = build_gaussian_kernel(12)
        assert np.allclose(kernel, kernel.T, atol=1e-6)
        assert np.allclose(kernel, kernel[::-1, :], atol=1e-6)

    def test_single_pixel(self) -> None:
        kernel = build_gaussian_kernel(1)
        assert kernel.shape == (1, 1)
        assert kernel[0, 0] > 0.0

    def test_invalid_tile_size_raises(self) -> None:
        from ai_fields.common.errors import ContractError

        with pytest.raises(ContractError):
            build_gaussian_kernel(0)


# ===========================================================================
# TestRunTiledPredict
# ===========================================================================

class TestRunTiledPredict:
    @_requires_torch_rasterio
    def test_output_shapes_and_dtype(self, tmp_path: Path) -> None:
        H, W = 5, 6
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)

        valid_mask = np.ones((H, W), dtype=np.uint8)
        norm_stats = _make_norm_stats_mapping(channels=8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=norm_stats,
        )

        assert isinstance(result, TiledPredictResult)
        assert result.extent_prob.shape == (H, W)
        assert result.boundary_prob.shape == (3, H, W)
        assert result.distance_pred.shape == (H, W)
        assert result.valid_mask.shape == (H, W)
        assert result.extent_prob.dtype == np.float32
        assert result.boundary_prob.dtype == np.float32
        assert result.distance_pred.dtype == np.float32
        assert result.valid_mask.dtype == np.uint8

    @_requires_torch_rasterio
    def test_valid_mask_preserved_in_result(self, tmp_path: Path) -> None:
        H, W = 4, 4
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)

        valid_mask = np.ones((H, W), dtype=np.uint8)
        valid_mask[0, 0] = 0

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=_make_norm_stats_mapping(channels=8),
        )

        assert result.valid_mask[0, 0] == 0
        assert int(result.valid_mask[1, 1]) == 1

    @_requires_torch_rasterio
    def test_all_invalid_tile_skipped(self, tmp_path: Path) -> None:
        H, W = 4, 4
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)

        # All pixels invalid → no tiles processed
        valid_mask = np.zeros((H, W), dtype=np.uint8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=_make_norm_stats_mapping(channels=8),
        )

        assert result.tiles_skipped_invalid >= 1
        assert result.tiles_processed == 0
        # Outputs should be zeros where no valid pixels exist
        assert np.all(result.extent_prob == 0.0)

    @_requires_torch_rasterio
    def test_tiles_processed_plus_skipped_equals_total(self, tmp_path: Path) -> None:
        H, W = 5, 6
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)

        valid_mask = np.ones((H, W), dtype=np.uint8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=_make_norm_stats_mapping(channels=8),
        )

        assert result.tiles_processed + result.tiles_skipped_invalid == result.tiles_total

    @_requires_torch_rasterio
    def test_zero_logit_gives_0_5_extent_prob(self, tmp_path: Path) -> None:
        H, W = 4, 4
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)

        valid_mask = np.ones((H, W), dtype=np.uint8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=_make_norm_stats_mapping(channels=8),
        )

        # sigmoid(0) = 0.5 everywhere
        assert np.allclose(result.extent_prob, 0.5, atol=1e-4)

    @_requires_torch_rasterio
    def test_zero_logit_boundary_sums_to_one_per_pixel(self, tmp_path: Path) -> None:
        H, W = 4, 4
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)

        valid_mask = np.ones((H, W), dtype=np.uint8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=_make_norm_stats_mapping(channels=8),
        )

        # softmax([0,0,0]) = [1/3, 1/3, 1/3]; sum across class dim = 1
        class_sums = result.boundary_prob.sum(axis=0)  # (H, W)
        assert np.allclose(class_sums, 1.0, atol=1e-4)

    @_requires_torch_rasterio
    def test_raw8_idx3_mode(self, tmp_path: Path) -> None:
        H, W = 4, 4
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path, feature_mode="raw8_idx3")
        loaded = _make_constant_model(in_channels=12)

        valid_mask = np.ones((H, W), dtype=np.uint8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=_make_norm_stats_mapping(channels=11),
        )

        assert result.feature_mode == "raw8_idx3"
        assert result.extent_prob.shape == (H, W)

    @_requires_torch_rasterio
    def test_normalization_stats_path_resolves(self, tmp_path: Path) -> None:
        H, W = 4, 4
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        stats_path = _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)

        valid_mask = np.ones((H, W), dtype=np.uint8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats_path=stats_path,
        )

        assert result.tiles_processed > 0

    def test_no_torch_raises_contract_error(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from ai_fields.common.errors import ContractError
        import ai_fields.module_target_predict.tiled_predict as tp_module

        with patch.object(tp_module, "_TORCH_AVAILABLE", False):
            with pytest.raises(ContractError, match="torch is required"):
                run_tiled_predict(
                    raster_path=tmp_path / "dummy.tif",
                    checkpoint_contract=None,  # type: ignore[arg-type]
                    loaded_model=None,  # type: ignore[arg-type]
                    valid_mask=np.ones((4, 4), dtype=np.uint8),
                )

    @_requires_torch_rasterio
    def test_scene_height_width_in_result(self, tmp_path: Path) -> None:
        H, W = 5, 7
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)
        valid_mask = np.ones((H, W), dtype=np.uint8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=_make_norm_stats_mapping(channels=8),
        )

        assert result.scene_height == H
        assert result.scene_width == W

    @_requires_torch_rasterio
    def test_blending_field_is_gaussian(self, tmp_path: Path) -> None:
        H, W = 4, 4
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)
        valid_mask = np.ones((H, W), dtype=np.uint8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=_make_norm_stats_mapping(channels=8),
        )

        assert result.blending == "gaussian"

    @_requires_torch_rasterio
    def test_forced_memmap_accumulators_and_cleanup(self, tmp_path: Path) -> None:
        H, W = 8, 8
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W, nodata=-9999.0)
        contract = _make_dummy_contract(tmp_path)
        loaded = _make_constant_model(in_channels=9)
        valid_mask = np.ones((H, W), dtype=np.uint8)

        result = run_tiled_predict(
            raster_path=raster_path,
            checkpoint_contract=contract,
            loaded_model=loaded,
            valid_mask=valid_mask,
            tile_size=4,
            overlap=0.0,
            normalization_stats=_make_norm_stats_mapping(channels=8),
            accumulator_memmap_threshold_bytes=0,
        )

        assert result.temp_work_dir is not None
        work_dir = Path(result.temp_work_dir)
        assert work_dir.exists()
        assert result.extent_prob.shape == (H, W)
        assert result.boundary_prob.shape == (3, H, W)
        assert result.distance_pred.shape == (H, W)

        cleanup_tiled_predict_result(result)
        assert not work_dir.exists()
