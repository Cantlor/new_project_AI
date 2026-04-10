"""Smoke tests for canonical shell runner tools/run_module_postprocess_vectorize.sh."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.manifests import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "tools" / "run_module_postprocess_vectorize.sh"


def _run_runner(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _prepare_predict_run_dir_for_dry_run(base_dir: Path) -> Path:
    run_dir = base_dir / "predict_run"
    _touch(run_dir / "extent_prob.tif")
    _touch(run_dir / "boundary_prob.tif")
    _touch(run_dir / "distance_pred.tif")
    _touch(run_dir / "valid.tif")
    return run_dir


class TestRunModulePostprocessVectorizeScript:
    def test_predict_run_dir_mode_resolves_inputs_in_dry_run(self, tmp_path: Path) -> None:
        predict_run_dir = _prepare_predict_run_dir_for_dry_run(tmp_path)
        _touch(predict_run_dir / "predict_manifest.json")

        result = _run_runner(
            "--predict-run-dir",
            str(predict_run_dir),
            "--output-dir",
            str(tmp_path / "post_runs"),
            "--run-id",
            "post-run-test",
            "--dry-run",
        )

        assert result.returncode == 0, result.stderr
        assert "[INFO] Input mode: predict-run-dir" in result.stdout
        assert f"[INFO] Predict run dir: {predict_run_dir}" in result.stdout
        assert f"[INFO] extent_prob: {predict_run_dir / 'extent_prob.tif'}" in result.stdout
        assert f"[INFO] boundary_prob: {predict_run_dir / 'boundary_prob.tif'}" in result.stdout
        assert "[INFO] Entry point: ai_fields.module_postprocess_vectorize.run_postprocess.run_postprocess_for_scene" in result.stdout

    def test_explicit_mode_works_in_dry_run(self, tmp_path: Path) -> None:
        extent = _touch(tmp_path / "extent_prob.tif")
        boundary = _touch(tmp_path / "boundary_prob.tif")
        distance = _touch(tmp_path / "distance_pred.tif")
        valid = _touch(tmp_path / "valid.tif")

        result = _run_runner(
            "--extent-prob",
            str(extent),
            "--boundary-prob",
            str(boundary),
            "--distance-pred",
            str(distance),
            "--valid",
            str(valid),
            "--output-dir",
            str(tmp_path / "post_runs"),
            "--run-id",
            "post-explicit",
            "--dry-run",
        )

        assert result.returncode == 0, result.stderr
        assert "[INFO] Input mode: explicit raster paths" in result.stdout

    def test_workers_override_is_reflected_in_dry_run_logs(self, tmp_path: Path) -> None:
        predict_run_dir = _prepare_predict_run_dir_for_dry_run(tmp_path)

        result = _run_runner(
            "--predict-run-dir",
            str(predict_run_dir),
            "--workers",
            "4",
            "--dry-run",
        )

        assert result.returncode == 0, result.stderr
        assert "num_workers=4" in result.stdout

    def test_ambiguous_input_mode_fails_clearly(self, tmp_path: Path) -> None:
        predict_run_dir = _prepare_predict_run_dir_for_dry_run(tmp_path)
        extent = _touch(tmp_path / "other_extent.tif")

        result = _run_runner(
            "--predict-run-dir",
            str(predict_run_dir),
            "--extent-prob",
            str(extent),
            "--dry-run",
        )

        assert result.returncode != 0
        assert "Ambiguous input mode" in result.stderr

    def test_missing_required_predict_artifact_fails_clearly(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "predict_run"
        _touch(run_dir / "extent_prob.tif")
        _touch(run_dir / "distance_pred.tif")
        _touch(run_dir / "valid.tif")

        result = _run_runner(
            "--predict-run-dir",
            str(run_dir),
            "--dry-run",
        )

        assert result.returncode != 0
        assert f"Missing required raster input: {run_dir / 'boundary_prob.tif'}" in result.stderr

    @pytest.mark.skipif(
        pytest.importorskip("rasterio", reason="rasterio required") is None,
        reason="rasterio not available",
    )
    @pytest.mark.skipif(
        pytest.importorskip("scipy", reason="scipy required") is None,
        reason="scipy not available",
    )
    @pytest.mark.skipif(
        pytest.importorskip("fiona", reason="fiona required") is None,
        reason="fiona not available",
    )
    @pytest.mark.skipif(
        pytest.importorskip("shapely", reason="shapely required") is None,
        reason="shapely not available",
    )
    def test_smoke_run_creates_expected_postprocess_artifacts(self, tmp_path: Path) -> None:
        import rasterio
        from rasterio.transform import from_origin

        def _write_raster(path: Path, *, array: np.ndarray) -> Path:
            path.parent.mkdir(parents=True, exist_ok=True)
            count, height, width = array.shape
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=count,
                dtype=str(array.dtype),
                crs="EPSG:32637",
                transform=from_origin(100.0, 200.0, 1.0, 1.0),
            ) as ds:
                ds.write(array)
            return path

        h, w = 6, 8
        extent = np.full((1, h, w), 0.9, dtype=np.float32)

        boundary_presence = np.full((h, w), 0.15, dtype=np.float32)
        boundary_presence[:, w // 2] = 0.8
        boundary = np.zeros((3, h, w), dtype=np.float32)
        boundary[1] = boundary_presence * 0.6
        boundary[2] = boundary_presence * 0.4
        boundary[0] = 1.0 - boundary_presence

        distance = np.ones((1, h, w), dtype=np.float32)
        distance[0, :, : w // 2] = 2.5

        valid = np.ones((1, h, w), dtype=np.uint8)
        valid[0, 0, 0] = 0
        valid[0, 1, 1] = 0

        predict_run_dir = tmp_path / "predict_run"
        extent_path = _write_raster(predict_run_dir / "extent_prob.tif", array=extent)
        boundary_path = _write_raster(predict_run_dir / "boundary_prob.tif", array=boundary)
        distance_path = _write_raster(predict_run_dir / "distance_pred.tif", array=distance)
        valid_path = _write_raster(predict_run_dir / "valid.tif", array=valid)

        predict_manifest_path = predict_run_dir / "predict_manifest.json"
        write_manifest(
            predict_manifest_path,
            {
                "schema_name": "target_predict.predict_manifest",
                "schema_version": "v1",
                "module_name": "module_target_predict",
                "module_version": None,
                "data_contract_version": DATA_CONTRACT_VERSION,
                "run_id": "predict_smoke_001",
                "stage_name": "predict_scene",
                "created_at_utc": "2026-04-08T00:00:00Z",
                "status": "success",
                "output_paths": {
                    "extent_prob": str(extent_path),
                    "boundary_prob": str(boundary_path),
                    "distance_pred": str(distance_path),
                    "valid": str(valid_path),
                },
            },
        )

        output_root = tmp_path / "postprocess_runs"
        run_id = "post-smoke-001"
        result = _run_runner(
            "--predict-run-dir",
            str(predict_run_dir),
            "--output-dir",
            str(output_root),
            "--run-id",
            run_id,
            "--no-progress",
        )

        assert result.returncode == 0, result.stderr
        run_dir = output_root / run_id
        assert (run_dir / "parcel_instance.tif").exists()
        assert (run_dir / "parcels.gpkg").exists()
        assert (run_dir / "postprocess_manifest.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "config_used.yaml").exists()

        manifest = json.loads((run_dir / "postprocess_manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_predict_manifest_path"] == str(predict_manifest_path)
        assert manifest["source_predict_run_id"] == "predict_smoke_001"
