"""Unit tests for module_eval Stage E.5 visual diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from ai_fields.module_eval.visual_diagnostics import (
    VisualDiagnosticsResult,
    write_visual_diagnostics,
)

rasterio = pytest.importorskip("rasterio")

try:
    import matplotlib as _mpl  # noqa: F401
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

_needs_matplotlib = pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not installed")


def _write_raster(
    path: Path,
    *,
    array: np.ndarray,
    crs: str = "EPSG:32637",
) -> Path:
    from rasterio.transform import from_origin

    transform = from_origin(100.0, 200.0, 1.0, 1.0)
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
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(array)
    return path


def _make_minimal_contract(tmp_path: Path):
    """Minimal 8×8 contract with all required raster inputs."""
    from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract

    h, w = 8, 8

    gt_extent = np.zeros((1, h, w), dtype=np.uint8)
    gt_extent[0, 2:6, 2:6] = 1

    gt_boundary = np.zeros((1, h, w), dtype=np.uint8)
    gt_boundary[0, 2, 2:6] = 1  # skeleton row

    gt_valid = np.ones((1, h, w), dtype=np.uint8)

    pred_extent = np.zeros((1, h, w), dtype=np.float32)
    pred_extent[0, 2:6, 2:6] = 0.85

    pred_boundary = np.zeros((3, h, w), dtype=np.float32)
    pred_boundary[0] = 1.0  # all background (valid simplex: ch0+ch1+ch2=1)
    pred_boundary[0, 2, 2:6] = 0.3  # reduce background near skeleton
    pred_boundary[1, 2, 2:6] = 0.7  # skeleton channel

    pred_distance = np.ones((1, h, w), dtype=np.float32)
    pred_valid = gt_valid.copy()

    paths = {
        "gt_extent_path": _write_raster(tmp_path / "gt_extent.tif", array=gt_extent),
        "gt_boundary_path": _write_raster(tmp_path / "gt_boundary.tif", array=gt_boundary),
        "gt_valid_path": _write_raster(tmp_path / "gt_valid.tif", array=gt_valid),
        "pred_extent_prob_path": _write_raster(tmp_path / "extent_prob.tif", array=pred_extent),
        "pred_boundary_prob_path": _write_raster(tmp_path / "boundary_prob.tif", array=pred_boundary),
        "pred_distance_pred_path": _write_raster(tmp_path / "distance_pred.tif", array=pred_distance),
        "pred_valid_path": _write_raster(tmp_path / "valid.tif", array=pred_valid),
    }
    return resolve_evaluation_input_contract(**paths)


@_needs_matplotlib
class TestWriteVisualDiagnosticsHappyPath:
    def test_returns_not_skipped(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        result = write_visual_diagnostics(
            tmp_path / "run",
            run_id="test_run",
            eval_mode="end_to_end",
            input_contract=contract,
        )
        assert result.skipped is False
        assert result.skip_reason is None

    def test_extent_overlay_png_created(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        result = write_visual_diagnostics(
            tmp_path / "run",
            run_id="test_run",
            eval_mode="end_to_end",
            input_contract=contract,
        )
        assert result.extent_overlay_path is not None
        assert result.extent_overlay_path.exists()
        assert result.extent_overlay_path.suffix == ".png"

    def test_boundary_heatmap_png_created(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        result = write_visual_diagnostics(
            tmp_path / "run",
            run_id="test_run",
            eval_mode="end_to_end",
            input_contract=contract,
        )
        assert result.boundary_heatmap_path is not None
        assert result.boundary_heatmap_path.exists()
        assert result.boundary_heatmap_path.suffix == ".png"

    def test_diagnostics_index_json_created(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        result = write_visual_diagnostics(
            tmp_path / "run",
            run_id="test_run",
            eval_mode="end_to_end",
            input_contract=contract,
        )
        assert result.diagnostics_index_path is not None
        assert result.diagnostics_index_path.exists()

    def test_diagnostics_index_schema_name(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        result = write_visual_diagnostics(
            tmp_path / "run",
            run_id="test_run",
            eval_mode="end_to_end",
            input_contract=contract,
        )
        payload = json.loads(result.diagnostics_index_path.read_text(encoding="utf-8"))
        assert payload["schema_name"] == "eval.diagnostics_index"

    def test_diagnostics_index_has_two_image_entries(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        result = write_visual_diagnostics(
            tmp_path / "run",
            run_id="test_run",
            eval_mode="end_to_end",
            input_contract=contract,
        )
        payload = json.loads(result.diagnostics_index_path.read_text(encoding="utf-8"))
        assert len(payload["images"]) == 2
        filenames = {img["filename"] for img in payload["images"]}
        assert filenames == {"extent_overlay.png", "boundary_heatmap.png"}

    def test_diagnostics_index_run_id(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        result = write_visual_diagnostics(
            tmp_path / "run",
            run_id="my_run_42",
            eval_mode="end_to_end",
            input_contract=contract,
        )
        payload = json.loads(result.diagnostics_index_path.read_text(encoding="utf-8"))
        assert payload["run_id"] == "my_run_42"

    def test_output_in_diagnostics_subdir(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "run_output"
        contract = _make_minimal_contract(tmp_path)
        result = write_visual_diagnostics(
            out_dir,
            run_id="r1",
            eval_mode="test",
            input_contract=contract,
        )
        assert result.extent_overlay_path.parent.name == "diagnostics"
        assert result.boundary_heatmap_path.parent.name == "diagnostics"


class TestWriteVisualDiagnosticsSkipped:
    def test_skipped_when_matplotlib_unavailable(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        with patch(
            "ai_fields.module_eval.visual_diagnostics._require_matplotlib",
            return_value=(None, None, None),
        ):
            result = write_visual_diagnostics(
                tmp_path / "run",
                run_id="r1",
                eval_mode="test",
                input_contract=contract,
            )
        assert result.skipped is True
        assert result.skip_reason is not None
        assert result.extent_overlay_path is None
        assert result.boundary_heatmap_path is None
        assert result.diagnostics_index_path is None

    def test_skipped_result_does_not_create_files(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        out_dir = tmp_path / "run"
        with patch(
            "ai_fields.module_eval.visual_diagnostics._require_matplotlib",
            return_value=(None, None, None),
        ):
            write_visual_diagnostics(
                out_dir,
                run_id="r1",
                eval_mode="test",
                input_contract=contract,
            )
        diag_dir = out_dir / "diagnostics"
        assert not diag_dir.exists() or not any(diag_dir.glob("*.png"))

    def test_internal_generation_failure_returns_skipped(self, tmp_path: Path) -> None:
        contract = _make_minimal_contract(tmp_path)
        with patch(
            "ai_fields.module_eval.visual_diagnostics._require_matplotlib",
            return_value=(object(), object(), object()),
        ), patch(
            "ai_fields.module_eval.visual_diagnostics._read_single_band_preview",
            side_effect=MemoryError("synthetic_memory_pressure"),
        ):
            result = write_visual_diagnostics(
                tmp_path / "run",
                run_id="r1",
                eval_mode="test",
                input_contract=contract,
            )
        assert result.skipped is True
        assert result.skip_reason is not None
        assert "MemoryError" in result.skip_reason


class TestWriteVisualDiagnosticsContract:
    def test_invalid_input_contract_raises(self, tmp_path: Path) -> None:
        from ai_fields.common.errors import ContractError

        with pytest.raises(ContractError):
            write_visual_diagnostics(
                tmp_path,
                run_id="r1",
                eval_mode="test",
                input_contract="not_a_contract",  # type: ignore[arg-type]
            )
