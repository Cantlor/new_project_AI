"""Stage E.5 minimal visual diagnostics for module_eval.

Generates PNG diagnostic images (module_eval §14.2, §19.4):
  - extent_overlay.png  — grayscale pred_extent_prob + GT extent contour overlay
  - boundary_heatmap.png — pred_boundary_prob non-background channel heatmap + GT boundary
  - diagnostics_index.json — machine-readable index of generated images

Requires matplotlib.  If matplotlib is unavailable, write_visual_diagnostics() returns
None with an explanatory skip_reason rather than raising ContractError, since visual
diagnostics are not critical for contract correctness (spec §19.4 lists them as mandatory
for "full baseline protocol" but not as a hard prerequisite for metrics or provenance).

Out of scope:
- merge/split case galleries (§14.2 extension),
- top-K worst scenes (multi-scene run),
- comparison before/after panels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError
from ai_fields.module_eval.input_contract import EvaluationInputContractResult

_CMAP_PRED = "viridis"
_CMAP_HEAT = "hot"
_DIAGNOSTIC_MAX_PREVIEW_PIXELS = 4_000_000


@dataclass(frozen=True)
class VisualDiagnosticsResult:
    """Result from Stage E.5 visual diagnostics generation."""

    extent_overlay_path: Path | None
    boundary_heatmap_path: Path | None
    diagnostics_index_path: Path | None
    skipped: bool
    skip_reason: str | None


def _require_matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return matplotlib, plt, mpatches
    except ImportError:
        return None, None, None


def _require_rasterio() -> tuple[Any, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for visual diagnostics.") from exc
    return rasterio, rasterio.errors


def _resolve_preview_shape(*, height: int, width: int, max_pixels: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ContractError(f"Invalid raster dimensions for diagnostics preview: {height}x{width}.")
    total = int(height) * int(width)
    if total <= max_pixels:
        return height, width
    scale = float(np.sqrt(float(max_pixels) / float(total)))
    out_h = max(1, int(np.floor(height * scale)))
    out_w = max(1, int(np.floor(width * scale)))
    return out_h, out_w


def _read_single_band_preview(
    path: Path,
    *,
    role: str,
    max_pixels: int,
    categorical: bool,
) -> np.ndarray:
    rasterio, rasterio_errors = _require_rasterio()
    try:
        with rasterio.open(path) as ds:
            out_h, out_w = _resolve_preview_shape(
                height=int(ds.height),
                width=int(ds.width),
                max_pixels=max_pixels,
            )
            resampling = (
                rasterio.enums.Resampling.nearest
                if categorical
                else rasterio.enums.Resampling.average
            )
            arr = ds.read(
                1,
                out_shape=(out_h, out_w),
                resampling=resampling,
            )
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read {role}: {path} ({exc})") from exc
    if arr.ndim != 2:
        raise ContractError(f"{role} preview must be 2D, got shape={arr.shape}.")
    return arr.astype(np.float32)


def _read_three_band_preview(path: Path, *, role: str, max_pixels: int) -> np.ndarray:
    rasterio, rasterio_errors = _require_rasterio()
    try:
        with rasterio.open(path) as ds:
            out_h, out_w = _resolve_preview_shape(
                height=int(ds.height),
                width=int(ds.width),
                max_pixels=max_pixels,
            )
            arr = ds.read(
                (1, 2, 3),
                out_shape=(3, out_h, out_w),
                resampling=rasterio.enums.Resampling.average,
            )
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read {role}: {path} ({exc})") from exc
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ContractError(f"{role} must be 3-band, got shape={arr.shape}.")
    return arr.astype(np.float32)


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _write_extent_overlay(
    path: Path,
    *,
    pred_extent_prob: np.ndarray,
    gt_extent: np.ndarray,
    valid: np.ndarray,
    plt: Any,
) -> None:
    """Write extent overlay: pred probability as heatmap, GT boundary as contour."""
    pred_norm = _safe_normalize(pred_extent_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_title("Extent: pred probability (heatmap) + GT boundary (contour)", fontsize=9)

    im = ax.imshow(pred_norm, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pred_extent_prob")

    # GT extent boundary contour
    gt_bin = (gt_extent > 0).astype(np.uint8)
    if gt_bin.max() > 0:
        ax.contour(gt_bin, levels=[0.5], colors=["white"], linewidths=1.5, linestyles="--")

    # Invalid mask overlay (hatching)
    invalid_mask = (valid < 0.5).astype(np.uint8)
    if invalid_mask.max() > 0:
        ax.contourf(
            invalid_mask, levels=[0.5, 1.5],
            hatches=["//"], colors=["none"], alpha=0.0,
        )

    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _write_boundary_heatmap(
    path: Path,
    *,
    pred_boundary_prob: np.ndarray,
    gt_boundary: np.ndarray,
    valid: np.ndarray,
    plt: Any,
) -> None:
    """Write boundary heatmap: non-background boundary prob + GT skeleton contour."""
    non_background_prob = pred_boundary_prob[1] + pred_boundary_prob[2]  # skeleton + buffer
    valid_mask = valid > 0.5

    display = non_background_prob.copy()
    display[~valid_mask] = 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_title("Boundary: pred non-background prob + GT skeleton (contour)", fontsize=9)

    im = ax.imshow(display, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pred boundary (non-bg) prob")

    # GT skeleton contour (class 1)
    gt_skeleton = (gt_boundary == 1).astype(np.uint8)
    if gt_skeleton.max() > 0:
        ax.contour(gt_skeleton, levels=[0.5], colors=["cyan"], linewidths=1.2)

    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_visual_diagnostics(
    output_dir: Path,
    *,
    run_id: str,
    eval_mode: str,
    input_contract: EvaluationInputContractResult,
) -> VisualDiagnosticsResult:
    """Generate and write Stage E.5 visual diagnostics.

    Returns a VisualDiagnosticsResult with skipped=True if matplotlib is unavailable,
    rather than raising ContractError — visual diagnostics are not required for
    contract correctness or provenance completeness.
    """
    if not isinstance(input_contract, EvaluationInputContractResult):
        raise ContractError(
            "input_contract must be EvaluationInputContractResult."
        )

    matplotlib, plt, mpatches = _require_matplotlib()
    if plt is None:
        return VisualDiagnosticsResult(
            extent_overlay_path=None,
            boundary_heatmap_path=None,
            diagnostics_index_path=None,
            skipped=True,
            skip_reason="matplotlib not available",
        )

    try:
        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        extent_overlay_path = diag_dir / "extent_overlay.png"
        boundary_heatmap_path = diag_dir / "boundary_heatmap.png"
        index_path = diag_dir / "diagnostics_index.json"

        # Use bounded-size previews for large scenes to avoid diagnostics-driven OOM.
        valid = _read_single_band_preview(
            input_contract.gt_valid.path,
            role="gt_valid",
            max_pixels=_DIAGNOSTIC_MAX_PREVIEW_PIXELS,
            categorical=True,
        )

        _write_extent_overlay(
            extent_overlay_path,
            pred_extent_prob=_read_single_band_preview(
                input_contract.pred_extent_prob.path,
                role="pred_extent_prob",
                max_pixels=_DIAGNOSTIC_MAX_PREVIEW_PIXELS,
                categorical=False,
            ),
            gt_extent=_read_single_band_preview(
                input_contract.gt_extent.path,
                role="gt_extent",
                max_pixels=_DIAGNOSTIC_MAX_PREVIEW_PIXELS,
                categorical=True,
            ),
            valid=valid,
            plt=plt,
        )

        _write_boundary_heatmap(
            boundary_heatmap_path,
            pred_boundary_prob=_read_three_band_preview(
                input_contract.pred_boundary_prob.path,
                role="pred_boundary_prob",
                max_pixels=_DIAGNOSTIC_MAX_PREVIEW_PIXELS,
            ),
            gt_boundary=_read_single_band_preview(
                input_contract.gt_boundary.path,
                role="gt_boundary",
                max_pixels=_DIAGNOSTIC_MAX_PREVIEW_PIXELS,
                categorical=True,
            ).astype(np.int32),
            valid=valid,
            plt=plt,
        )

        index_payload: dict[str, Any] = {
            "schema_name": "eval.diagnostics_index",
            "schema_version": "v1",
            "run_id": run_id,
            "eval_mode": eval_mode,
            "stage_scope": "stage_e5_visual_diagnostics",
            "preview_max_pixels": _DIAGNOSTIC_MAX_PREVIEW_PIXELS,
            "images": [
                {
                    "filename": "extent_overlay.png",
                    "path": str(extent_overlay_path),
                    "description": (
                        "pred_extent_prob as RdYlGn heatmap with GT extent boundary contour (white dashed). "
                        "GT invalid pixels shown with hatching."
                    ),
                },
                {
                    "filename": "boundary_heatmap.png",
                    "path": str(boundary_heatmap_path),
                    "description": (
                        "pred non-background boundary probability as hot heatmap, "
                        "GT skeleton class contour shown in cyan."
                    ),
                },
            ],
        }
        try:
            index_path.write_text(
                json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            raise ContractError(f"Failed to write diagnostics_index.json: {exc}") from exc

        return VisualDiagnosticsResult(
            extent_overlay_path=extent_overlay_path,
            boundary_heatmap_path=boundary_heatmap_path,
            diagnostics_index_path=index_path,
            skipped=False,
            skip_reason=None,
        )
    except Exception as exc:
        return VisualDiagnosticsResult(
            extent_overlay_path=None,
            boundary_heatmap_path=None,
            diagnostics_index_path=None,
            skipped=True,
            skip_reason=f"visual diagnostics generation failed: {type(exc).__name__}: {exc}",
        )


__all__ = [
    "VisualDiagnosticsResult",
    "write_visual_diagnostics",
]
