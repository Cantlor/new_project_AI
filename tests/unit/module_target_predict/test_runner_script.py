"""Smoke tests for canonical shell runner tools/run_module_target_predict.sh."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.manifests import write_manifest

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import rasterio
    from rasterio.transform import from_origin

    _RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    rasterio = None  # type: ignore[assignment]
    from_origin = None  # type: ignore[assignment]
    _RASTERIO_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "tools" / "run_module_target_predict.sh"


def _run_runner(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_minimal_raster(path: Path) -> Path:
    if not _RASTERIO_AVAILABLE:
        pytest.skip("rasterio required")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones((8, 6, 7), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=7,
        height=6,
        count=8,
        dtype="float32",
        crs="EPSG:32642",
        transform=from_origin(500_000.0, 4_600_000.0, 10.0, 10.0),
        nodata=-9999.0,
    ) as ds:
        ds.write(data)
    return path


def _write_norm_stats(path: Path, *, channels: int = 8) -> Path:
    band_stats = [{"band_idx": idx, "p_lo": 0.0, "p_hi": 2.0} for idx in range(channels)]
    payload = {
        "band_stats": band_stats,
        "clip_percentiles": [0.5, 99.5],
        "computed_on": "valid_train_pixels",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_checkpoint_bundle(run_dir: Path, *, feature_mode: str = "raw8") -> dict[str, Path]:
    if not _TORCH_AVAILABLE:
        pytest.skip("torch required")

    from ai_fields.module_net_train.model import EdgeAwareMultitaskNet

    run_dir.mkdir(parents=True, exist_ok=True)
    in_channels = 9 if feature_mode == "raw8" else 12
    if feature_mode == "raw8":
        semantics = [
            "coastal",
            "blue",
            "green",
            "yellow",
            "red",
            "rededge",
            "nir1",
            "nir2",
            "valid",
        ]
    else:
        semantics = [
            "coastal",
            "blue",
            "green",
            "yellow",
            "red",
            "rededge",
            "nir1",
            "nir2",
            "NDVI",
            "SAVI",
            "NDWI",
            "valid",
        ]

    norm_stats_path = _write_norm_stats(run_dir / "norm_stats.json", channels=8 if feature_mode == "raw8" else 11)
    checkpoint_path = run_dir / "best.ckpt"
    model = EdgeAwareMultitaskNet(in_channels=in_channels, encoder_depth=3, base_channels=8)
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
        checkpoint_path,
    )

    checkpoint_metadata_path = run_dir / "checkpoint_metadata.json"
    metadata_payload: dict[str, Any] = {
        "schema_name": "net_train.checkpoint_metadata",
        "schema_version": "v1",
        "module_name": "module_net_train",
        "module_version": None,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "run_id": "net_train_for_predict_test",
        "stage_name": "checkpoint_export",
        "created_at_utc": "2026-04-08T00:00:00Z",
        "status": "success",
        "checkpoint_path": str(checkpoint_path),
        "feature_mode": feature_mode,
        "assembled_model_input": f"{feature_mode}_valid",
        "in_channels": in_channels,
        "channel_semantics": semantics,
        "valid_as_input_channel": True,
        "normalization": {
            "normalization_name": "per_band_robust_percentile",
            "stats_source": str(norm_stats_path),
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
    }
    write_manifest(checkpoint_metadata_path, metadata_payload)

    train_manifest_path = run_dir / "train_manifest.json"
    write_manifest(
        train_manifest_path,
        {
            "schema_name": "net_train.train_manifest",
            "schema_version": "v1",
            "module_name": "module_net_train",
            "module_version": None,
            "data_contract_version": DATA_CONTRACT_VERSION,
            "run_id": "net_train_for_predict_test",
            "stage_name": "train",
            "created_at_utc": "2026-04-08T00:00:00Z",
            "status": "success",
        },
    )
    config_used_path = run_dir / "config_used.yaml"
    config_used_path.write_text("feature_mode: raw8\n", encoding="utf-8")

    return {
        "checkpoint": checkpoint_path,
        "checkpoint_metadata": checkpoint_metadata_path,
        "train_manifest": train_manifest_path,
        "config_used": config_used_path,
    }


class TestRunModuleTargetPredictScript:
    def test_dry_run_resolves_sibling_train_artifacts(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "train_run"
        checkpoint = run_dir / "best.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"ckpt")
        (run_dir / "checkpoint_metadata.json").write_text("{}", encoding="utf-8")
        (run_dir / "train_manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "config_used.yaml").write_text("x: 1\n", encoding="utf-8")
        raster = tmp_path / "scene.tif"
        raster.write_bytes(b"raster")

        result = _run_runner(
            "--checkpoint",
            str(checkpoint),
            "--input-raster",
            str(raster),
            "--output-dir",
            str(tmp_path / "predict_runs"),
            "--run-id",
            "predict-run-test",
            "--dry-run",
        )

        assert result.returncode == 0, result.stderr
        assert "[INFO] Checkpoint metadata:" in result.stdout
        assert str(run_dir / "checkpoint_metadata.json") in result.stdout
        assert "[INFO] Train manifest:" in result.stdout
        assert str(run_dir / "train_manifest.json") in result.stdout
        assert "[INFO] Config used:" in result.stdout
        assert str(run_dir / "config_used.yaml") in result.stdout
        assert "[INFO] Entry point: ai_fields.module_target_predict.predict_run.run_predict_for_scene" in result.stdout

    def test_missing_checkpoint_metadata_fails_clearly(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "train_run" / "best.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"ckpt")
        raster = tmp_path / "scene.tif"
        raster.write_bytes(b"raster")

        result = _run_runner(
            "--checkpoint",
            str(checkpoint),
            "--input-raster",
            str(raster),
            "--dry-run",
        )

        assert result.returncode != 0
        assert "Could not resolve checkpoint metadata automatically" in result.stderr

    def test_aoi_flag_fails_explicitly_for_current_runtime(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "train_run" / "best.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"ckpt")
        (checkpoint.parent / "checkpoint_metadata.json").write_text("{}", encoding="utf-8")
        raster = tmp_path / "scene.tif"
        raster.write_bytes(b"raster")
        aoi = tmp_path / "aoi.geojson"
        aoi.write_text("{}", encoding="utf-8")

        result = _run_runner(
            "--checkpoint",
            str(checkpoint),
            "--input-raster",
            str(raster),
            "--aoi",
            str(aoi),
            "--dry-run",
        )

        assert result.returncode != 0
        assert "AOI inference mode is not implemented" in result.stderr

    @pytest.mark.skipif(
        not (_TORCH_AVAILABLE and _RASTERIO_AVAILABLE),
        reason="torch+rasterio required for runner smoke test",
    )
    def test_smoke_run_creates_predict_artifacts(self, tmp_path: Path) -> None:
        bundle = _write_checkpoint_bundle(tmp_path / "train_run")
        raster = _write_minimal_raster(tmp_path / "scene.tif")
        output_root = tmp_path / "predict_runs"
        run_id = "predict-smoke-001"

        result = _run_runner(
            "--checkpoint",
            str(bundle["checkpoint"]),
            "--input-raster",
            str(raster),
            "--output-dir",
            str(output_root),
            "--run-id",
            run_id,
            "--device-override",
            "cpu",
            "--tile-size",
            "4",
            "--overlap",
            "0.0",
            "--no-progress",
        )

        assert result.returncode == 0, result.stderr
        run_dir = output_root / run_id
        assert (run_dir / "extent_prob.tif").exists()
        assert (run_dir / "boundary_prob.tif").exists()
        assert (run_dir / "distance_pred.tif").exists()
        assert (run_dir / "valid.tif").exists()
        assert (run_dir / "predict_manifest.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "config_used.yaml").exists()
