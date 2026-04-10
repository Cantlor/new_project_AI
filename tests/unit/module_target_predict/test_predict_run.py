"""Unit tests for module_target_predict output writers and scene orchestrator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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

_requires_rasterio = pytest.mark.skipif(
    not _RASTERIO_AVAILABLE, reason="rasterio required"
)
_requires_torch_rasterio = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _RASTERIO_AVAILABLE),
    reason="torch and rasterio required",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
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


def _write_checkpoint_and_metadata(
    tmp_path: Path,
    *,
    feature_mode: str = "raw8",
) -> tuple[Path, Path]:
    """Write a minimal .ckpt + checkpoint_metadata.json for run_predict_for_scene."""
    from ai_fields.common.constants import DATA_CONTRACT_VERSION
    from ai_fields.common.manifests import write_manifest
    from ai_fields.module_net_train.model import EdgeAwareMultitaskNet

    in_channels = 9 if feature_mode == "raw8" else 12

    if feature_mode == "raw8":
        semantics = [
            "coastal", "blue", "green", "yellow",
            "red", "rededge", "nir1", "nir2", "valid",
        ]
    else:
        semantics = [
            "coastal", "blue", "green", "yellow",
            "red", "rededge", "nir1", "nir2",
            "NDVI", "SAVI", "NDWI", "valid",
        ]

    model = EdgeAwareMultitaskNet(in_channels=in_channels, encoder_depth=3, base_channels=8)
    ckpt_path = tmp_path / "best.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mode": feature_mode,
            "assembled_model_input": f"{feature_mode}_valid",
            "in_channels": in_channels,
            "channel_semantics": semantics,
            "valid_as_input_channel": True,
            "epochs_completed": 1,
        },
        ckpt_path,
    )

    meta_path = tmp_path / "checkpoint_metadata.json"
    write_manifest(
        meta_path,
        {
            "schema_name": "net_train.checkpoint_metadata",
            "schema_version": "v1",
            "module_name": "module_net_train",
            "module_version": None,
            "data_contract_version": DATA_CONTRACT_VERSION,
            "run_id": "net_train_run_001",
            "stage_name": "checkpoint_export",
            "created_at_utc": "2026-04-03T00:00:00Z",
            "status": "success",
            "checkpoint_path": str(ckpt_path),
            "feature_mode": feature_mode,
            "assembled_model_input": f"{feature_mode}_valid",
            "in_channels": in_channels,
            "channel_semantics": semantics,
            "valid_as_input_channel": True,
            "normalization": {
                "normalization_name": "per_band_robust_percentile",
                "stats_source": "norm_stats.json",
                "clip_percentiles": [0.5, 99.5],
                "scaling_range": [0.0, 1.0],
            },
            "target_heads": {
                "extent": {"type": "binary_segmentation", "ignore_label": 255},
                "boundary": {
                    "type": "multiclass_segmentation",
                    "classes": {"0": "background", "1": "skeleton", "2": "buffer"},
                },
                "distance": {"type": "regression"},
            },
            "model_version": "v1-baseline",
            "model_architecture": "edge_aware_multitask_v1",
            "encoder_depth": 3,
            "base_channels": 8,
        },
    )
    return ckpt_path, meta_path


def _make_tiled_result(*, height: int, width: int) -> Any:
    """Build a minimal TiledPredictResult for output writer tests."""
    from ai_fields.module_target_predict.tiled_predict import TiledPredictResult

    return TiledPredictResult(
        extent_prob=np.full((height, width), 0.5, dtype=np.float32),
        boundary_prob=np.full((3, height, width), 1.0 / 3.0, dtype=np.float32),
        distance_pred=np.zeros((height, width), dtype=np.float32),
        valid_mask=np.ones((height, width), dtype=np.uint8),
        scene_height=height,
        scene_width=width,
        tiles_total=4,
        tiles_processed=4,
        tiles_skipped_invalid=0,
        feature_mode="raw8",
        tile_size=4,
        overlap=0.0,
        blending="gaussian",
    )


def _make_raster_metadata(
    tmp_path: Path,
    *,
    height: int = 5,
    width: int = 6,
) -> Any:
    from ai_fields.module_target_predict.raster_contract import read_predict_raster_metadata

    raster_path = _write_raster(tmp_path / "meta_raster.tif", height=height, width=width)
    return read_predict_raster_metadata(raster_path)


# ===========================================================================
# TestWritePredictRasterOutputs
# ===========================================================================

class TestWritePredictRasterOutputs:
    @_requires_rasterio
    def test_all_four_files_created(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import write_predict_raster_outputs

        H, W = 5, 6
        result = _make_tiled_result(height=H, width=W)
        metadata = _make_raster_metadata(tmp_path, height=H, width=W)
        out_dir = tmp_path / "out"

        written = write_predict_raster_outputs(
            output_dir=out_dir, result=result, metadata=metadata
        )

        assert "extent_prob" in written
        assert "boundary_prob" in written
        assert "distance_pred" in written
        assert "valid" in written

        for name, path in written.items():
            assert path.exists(), f"{name} was not created at {path}"

    @_requires_rasterio
    def test_extent_prob_is_single_band_float32(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import write_predict_raster_outputs

        H, W = 5, 6
        result = _make_tiled_result(height=H, width=W)
        metadata = _make_raster_metadata(tmp_path, height=H, width=W)

        written = write_predict_raster_outputs(
            output_dir=tmp_path / "out", result=result, metadata=metadata
        )

        with rasterio.open(written["extent_prob"]) as ds:
            assert ds.count == 1
            assert ds.dtypes[0] == "float32"
            assert ds.height == H
            assert ds.width == W

    @_requires_rasterio
    def test_boundary_prob_is_three_band_float32(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import write_predict_raster_outputs

        H, W = 5, 6
        result = _make_tiled_result(height=H, width=W)
        metadata = _make_raster_metadata(tmp_path, height=H, width=W)

        written = write_predict_raster_outputs(
            output_dir=tmp_path / "out", result=result, metadata=metadata
        )

        with rasterio.open(written["boundary_prob"]) as ds:
            assert ds.count == 3
            assert ds.dtypes[0] == "float32"

    @_requires_rasterio
    def test_valid_tif_is_uint8_with_0_1_values(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import write_predict_raster_outputs

        H, W = 5, 6
        result = _make_tiled_result(height=H, width=W)
        # Set one pixel invalid in result
        valid_with_invalid = result.valid_mask.copy()
        valid_with_invalid[0, 0] = 0
        from ai_fields.module_target_predict.tiled_predict import TiledPredictResult
        result2 = TiledPredictResult(
            **{
                **result.__dict__,
                "valid_mask": valid_with_invalid,
            }
        )
        metadata = _make_raster_metadata(tmp_path, height=H, width=W)

        written = write_predict_raster_outputs(
            output_dir=tmp_path / "out", result=result2, metadata=metadata
        )

        with rasterio.open(written["valid"]) as ds:
            assert ds.count == 1
            assert ds.dtypes[0] == "uint8"
            data = ds.read(1)
            assert set(np.unique(data)).issubset({0, 1})

    @_requires_rasterio
    def test_output_crs_matches_input(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import write_predict_raster_outputs

        H, W = 5, 6
        result = _make_tiled_result(height=H, width=W)
        metadata = _make_raster_metadata(tmp_path, height=H, width=W)

        written = write_predict_raster_outputs(
            output_dir=tmp_path / "out", result=result, metadata=metadata
        )

        with rasterio.open(written["extent_prob"]) as ds:
            assert "32637" in ds.crs.to_string()

    @_requires_rasterio
    def test_values_written_correctly(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import write_predict_raster_outputs

        H, W = 4, 4
        result = _make_tiled_result(height=H, width=W)
        metadata = _make_raster_metadata(tmp_path, height=H, width=W)

        written = write_predict_raster_outputs(
            output_dir=tmp_path / "out", result=result, metadata=metadata
        )

        with rasterio.open(written["extent_prob"]) as ds:
            data = ds.read(1)
            assert np.allclose(data, 0.5, atol=1e-5)


# ===========================================================================
# TestRunPredictForScene
# ===========================================================================

class TestRunPredictForScene:
    @_requires_torch_rasterio
    def test_all_artifacts_created(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import run_predict_for_scene

        H, W = 5, 6
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W)
        norm_stats_path = _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
        ckpt_path, meta_path = _write_checkpoint_and_metadata(tmp_path)
        out_dir = tmp_path / "predict_out"

        run_info = run_predict_for_scene(
            raster_path=raster_path,
            checkpoint_path=ckpt_path,
            checkpoint_metadata_path=meta_path,
            output_dir=out_dir,
            device="cpu",
            tile_size=4,
            overlap=0.0,
            normalization_stats_path=norm_stats_path,
            run_id="test_run_001",
        )

        # All 4 rasters created
        assert Path(run_info["output_paths"]["extent_prob"]).exists()
        assert Path(run_info["output_paths"]["boundary_prob"]).exists()
        assert Path(run_info["output_paths"]["distance_pred"]).exists()
        assert Path(run_info["output_paths"]["valid"]).exists()

        # Manifest and summary created
        assert Path(run_info["manifest_path"]).exists()
        assert Path(run_info["summary_path"]).exists()
        assert Path(run_info["config_path"]).exists()

    @_requires_torch_rasterio
    def test_predict_manifest_required_fields(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import run_predict_for_scene

        H, W = 5, 6
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W)
        norm_stats_path = _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
        ckpt_path, meta_path = _write_checkpoint_and_metadata(tmp_path)
        out_dir = tmp_path / "predict_out"

        run_info = run_predict_for_scene(
            raster_path=raster_path,
            checkpoint_path=ckpt_path,
            checkpoint_metadata_path=meta_path,
            output_dir=out_dir,
            device="cpu",
            tile_size=4,
            overlap=0.0,
            normalization_stats_path=norm_stats_path,
            run_id="test_run_002",
        )

        manifest = json.loads(Path(run_info["manifest_path"]).read_text(encoding="utf-8"))

        assert manifest["schema_name"] == "target_predict.predict_manifest"
        assert manifest["schema_version"] == "v1"
        assert manifest["module_name"] == "module_target_predict"
        assert manifest["run_id"] == "test_run_002"
        assert manifest["status"] == "success"
        assert "data_contract_version" in manifest
        assert "resolved_contract" in manifest
        assert "tiling" in manifest
        assert "runtime" in manifest
        assert "valid_coverage" in manifest

        features = manifest["resolved_contract"]["features"]
        assert features["dataset_feature_mode"] == "raw8"
        assert features["assembled_model_input"] == "raw8_valid"
        assert features["valid_as_input_channel"] is True
        assert features["final_input_channel_count"] == 9

    @_requires_torch_rasterio
    def test_no_postprocess_outputs_written(self, tmp_path: Path) -> None:
        """Thresholding / watershed / polygon artifacts must NOT be written."""
        from ai_fields.module_target_predict.predict_run import run_predict_for_scene

        H, W = 5, 6
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W)
        norm_stats_path = _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
        ckpt_path, meta_path = _write_checkpoint_and_metadata(tmp_path)
        out_dir = tmp_path / "predict_out"

        run_predict_for_scene(
            raster_path=raster_path,
            checkpoint_path=ckpt_path,
            checkpoint_metadata_path=meta_path,
            output_dir=out_dir,
            device="cpu",
            tile_size=4,
            overlap=0.0,
            normalization_stats_path=norm_stats_path,
        )

        tif_files = {p.name for p in out_dir.glob("*.tif")}
        forbidden = {
            "parcel_instance.tif",
            "parcels.tif",
        }
        assert not (tif_files & forbidden), (
            f"Postprocess artifacts must not appear in predict output: {tif_files & forbidden}"
        )

    @_requires_torch_rasterio
    def test_tiles_total_stats_in_return(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import run_predict_for_scene

        H, W = 5, 6
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W)
        norm_stats_path = _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
        ckpt_path, meta_path = _write_checkpoint_and_metadata(tmp_path)

        run_info = run_predict_for_scene(
            raster_path=raster_path,
            checkpoint_path=ckpt_path,
            checkpoint_metadata_path=meta_path,
            output_dir=tmp_path / "out",
            device="cpu",
            tile_size=4,
            overlap=0.0,
            normalization_stats_path=norm_stats_path,
        )

        assert "tiles_total" in run_info
        assert "tiles_processed" in run_info
        assert "tiles_skipped_invalid" in run_info
        assert run_info["tiles_processed"] + run_info["tiles_skipped_invalid"] == run_info["tiles_total"]

    @_requires_torch_rasterio
    def test_device_resolved_is_cpu(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import run_predict_for_scene

        H, W = 5, 6
        raster_path = _write_raster(tmp_path / "scene.tif", height=H, width=W)
        norm_stats_path = _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
        ckpt_path, meta_path = _write_checkpoint_and_metadata(tmp_path)

        run_info = run_predict_for_scene(
            raster_path=raster_path,
            checkpoint_path=ckpt_path,
            checkpoint_metadata_path=meta_path,
            output_dir=tmp_path / "out",
            device="cpu",
            tile_size=4,
            overlap=0.0,
            normalization_stats_path=norm_stats_path,
        )

        assert run_info["device_resolved"] == "cpu"

    @_requires_torch_rasterio
    def test_manifest_includes_train_run_id_when_train_manifest_provided(self, tmp_path: Path) -> None:
        from ai_fields.module_target_predict.predict_run import run_predict_for_scene
        from ai_fields.common.constants import DATA_CONTRACT_VERSION
        from ai_fields.common.manifests import write_manifest

        raster_path = _write_raster(tmp_path / "scene.tif", height=5, width=6)
        norm_stats_path = _write_norm_stats(tmp_path / "norm_stats.json", channels=8)
        ckpt_path, meta_path = _write_checkpoint_and_metadata(tmp_path)
        train_manifest_path = tmp_path / "train_manifest.json"
        write_manifest(
            train_manifest_path,
            {
                "schema_name": "net_train.train_manifest",
                "schema_version": "v1",
                "module_name": "module_net_train",
                "module_version": None,
                "data_contract_version": DATA_CONTRACT_VERSION,
                "run_id": "net-train-run-xyz",
                "stage_name": "train",
                "created_at_utc": "2026-04-08T00:00:00Z",
                "status": "success",
            },
        )

        run_info = run_predict_for_scene(
            raster_path=raster_path,
            checkpoint_path=ckpt_path,
            checkpoint_metadata_path=meta_path,
            train_manifest_path=train_manifest_path,
            output_dir=tmp_path / "out",
            device="cpu",
            tile_size=4,
            overlap=0.0,
            normalization_stats_path=norm_stats_path,
        )

        manifest = json.loads(Path(run_info["manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["checkpoint"]["train_run_id"] == "net-train-run-xyz"
