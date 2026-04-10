"""Stage C minimal boundary metrics for module_eval.

Scope of this layer is intentionally narrow:
- compute boundary precision/recall/F1 over valid pixels;
- compute baseline BDE (boundary displacement error) in projected units when possible;
- run only on top of Stage A resolved EvaluationInputContractResult.

Out of scope:
- object/structure metrics,
- report/comparison framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError, ValidPolicyError
from ai_fields.module_eval.input_contract import EvaluationInputContractResult

_EPS = 1e-9
_BOUNDARY_SUM_ATOL = 5e-3
_STREAM_WINDOW_SIZE = 1024
_BDE_MAX_COORD_POINTS = 25_000_000


@dataclass(frozen=True)
class BoundaryEvaluationPolicy:
    """Explicit policy for Stage C boundary metrics."""

    prediction_interpretation: str
    gt_interpretation: str
    threshold_provenance: str

    non_background_prob_threshold: float = 0.5
    bde_enabled: bool = True
    bde_unit_policy: str = "meters_if_projected_else_pixels"
    empty_boundary_handling: str = "explicit_error"


@dataclass(frozen=True)
class BoundaryMetricsResult:
    """Stage C result contract for minimal boundary metrics."""

    boundary_f1: float
    boundary_precision: float
    boundary_recall: float
    boundary_bde: float | None
    boundary_bde_units: str | None

    tp: int
    fp: int
    fn: int

    effective_pixels: int
    valid_pixels: int
    invalid_pixels: int
    gt_boundary_positive_pixels: int
    pred_boundary_positive_pixels: int

    prediction_interpretation: str
    gt_interpretation: str
    threshold_provenance: str
    non_background_prob_threshold: float
    bde_enabled: bool
    stage_scope: str


def _require_rasterio() -> tuple[Any, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for module_eval Stage C.") from exc
    return rasterio, rasterio.errors


def _require_scipy_spatial() -> Any:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise ContractError("scipy is required to compute BDE in module_eval Stage C.") from exc
    return cKDTree


def _validate_policy(policy: BoundaryEvaluationPolicy) -> BoundaryEvaluationPolicy:
    if not isinstance(policy, BoundaryEvaluationPolicy):
        raise ContractError("policy must be BoundaryEvaluationPolicy for Stage C.")

    if policy.prediction_interpretation not in {
        "argmax_non_background",
        "argmax_skeleton_only",
        "threshold_non_background_prob",
    }:
        raise ContractError(
            "Unsupported policy.prediction_interpretation. Expected one of "
            "{'argmax_non_background','argmax_skeleton_only','threshold_non_background_prob'}."
        )
    if policy.gt_interpretation not in {"non_background", "skeleton_only"}:
        raise ContractError(
            "Unsupported policy.gt_interpretation. Expected one of "
            "{'non_background','skeleton_only'}."
        )
    if not isinstance(policy.threshold_provenance, str) or policy.threshold_provenance.strip() == "":
        raise ContractError("policy.threshold_provenance must be a non-empty string.")

    threshold = float(policy.non_background_prob_threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ContractError(
            "policy.non_background_prob_threshold must be in [0,1], "
            f"got {policy.non_background_prob_threshold!r}."
        )

    if policy.bde_unit_policy not in {"meters_if_projected_else_pixels"}:
        raise ContractError(
            "Unsupported policy.bde_unit_policy. "
            "Expected 'meters_if_projected_else_pixels'."
        )
    if policy.empty_boundary_handling not in {"explicit_error", "zero_if_both_empty"}:
        raise ContractError(
            "Unsupported policy.empty_boundary_handling. "
            "Expected one of {'explicit_error','zero_if_both_empty'}."
        )
    return policy


def _iter_windows(*, height: int, width: int, window_size: int = _STREAM_WINDOW_SIZE):
    rasterio, _ = _require_rasterio()
    Window = rasterio.windows.Window
    for row_off in range(0, height, window_size):
        win_h = min(window_size, height - row_off)
        for col_off in range(0, width, window_size):
            win_w = min(window_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def _read_window(ds: Any, indexes: int | tuple[int, ...], *, window: Any, role: str) -> np.ndarray:
    arr = ds.read(indexes, window=window)
    if not np.isfinite(arr).all():
        raise ContractError(f"{role} contains non-finite values.")
    return arr


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= _EPS:
        return 0.0
    return float(numerator / denominator)


def _boundary_prob_to_pred_positive(
    boundary_prob: np.ndarray,
    *,
    policy: BoundaryEvaluationPolicy,
) -> np.ndarray:
    if policy.prediction_interpretation == "argmax_non_background":
        return boundary_prob.argmax(axis=0) != 0
    if policy.prediction_interpretation == "argmax_skeleton_only":
        return boundary_prob.argmax(axis=0) == 1
    if policy.prediction_interpretation == "threshold_non_background_prob":
        non_background_prob = boundary_prob[1] + boundary_prob[2]
        return non_background_prob >= float(policy.non_background_prob_threshold)
    raise ContractError(f"Unsupported prediction_interpretation={policy.prediction_interpretation!r}.")


def _gt_boundary_to_positive(
    gt_boundary: np.ndarray,
    *,
    policy: BoundaryEvaluationPolicy,
) -> np.ndarray:
    if policy.gt_interpretation == "non_background":
        return gt_boundary != 0
    if policy.gt_interpretation == "skeleton_only":
        return gt_boundary == 1
    raise ContractError(f"Unsupported gt_interpretation={policy.gt_interpretation!r}.")


def _resolve_bde_spacing_and_units(
    *,
    input_contract: EvaluationInputContractResult,
    policy: BoundaryEvaluationPolicy,
) -> tuple[tuple[float, float], str]:
    if policy.bde_unit_policy != "meters_if_projected_else_pixels":
        raise ContractError(f"Unsupported bde_unit_policy={policy.bde_unit_policy!r}.")

    rasterio, _ = _require_rasterio()
    try:
        crs = rasterio.crs.CRS.from_user_input(input_contract.common_crs)
    except Exception as exc:
        raise ContractError(f"Failed to parse common CRS for BDE: {exc}") from exc

    if crs.is_projected:
        gt = input_contract.common_transform_gdal
        x_size = abs(float(gt[1]))
        y_size = abs(float(gt[5]))
        if x_size <= _EPS or y_size <= _EPS:
            raise ContractError(
                "Cannot compute BDE in projected units: invalid pixel size from transform."
            )
        return (y_size, x_size), "meters"

    return (1.0, 1.0), "pixels"


def _guard_bde_coordinate_count(*, pred_count: int, gt_count: int) -> None:
    total = int(pred_count) + int(gt_count)
    if total <= _BDE_MAX_COORD_POINTS:
        return
    approx_bytes = total * 2 * np.dtype(np.float64).itemsize
    raise ContractError(
        "Stage C BDE coordinate set exceeds memory safety guard: "
        f"pred_positive={pred_count}, gt_positive={gt_count}, "
        f"total_points={total}, approx_coordinate_bytes={approx_bytes}, "
        f"max_points={_BDE_MAX_COORD_POINTS}."
    )


def _scale_pixel_coordinates(
    coords_rc: np.ndarray,
    *,
    sampling: tuple[float, float],
) -> np.ndarray:
    # Coordinates are in (row, col). Apply anisotropic sampling to compute physical distance.
    scaled = np.asarray(coords_rc, dtype=np.float64)
    scaled[:, 0] *= float(sampling[0])
    scaled[:, 1] *= float(sampling[1])
    return scaled


def _compute_bde_from_coords(
    *,
    pred_coords_rc: np.ndarray,
    gt_coords_rc: np.ndarray,
    input_contract: EvaluationInputContractResult,
    policy: BoundaryEvaluationPolicy,
) -> tuple[float, str]:
    pred_count = int(pred_coords_rc.shape[0])
    gt_count = int(gt_coords_rc.shape[0])

    if pred_count == 0 or gt_count == 0:
        if policy.empty_boundary_handling == "zero_if_both_empty" and pred_count == 0 and gt_count == 0:
            _, units = _resolve_bde_spacing_and_units(input_contract=input_contract, policy=policy)
            return 0.0, units
        raise ContractError(
            "BDE cannot be computed with empty boundary sets under current policy: "
            f"pred_positive={pred_count}, gt_positive={gt_count}."
        )

    _guard_bde_coordinate_count(pred_count=pred_count, gt_count=gt_count)

    sampling, units = _resolve_bde_spacing_and_units(
        input_contract=input_contract,
        policy=policy,
    )
    cKDTree = _require_scipy_spatial()

    pred_xy = _scale_pixel_coordinates(pred_coords_rc, sampling=sampling)
    gt_xy = _scale_pixel_coordinates(gt_coords_rc, sampling=sampling)

    gt_tree = cKDTree(gt_xy)
    pred_tree = cKDTree(pred_xy)
    pred_to_gt = float(gt_tree.query(pred_xy, workers=1)[0].mean())
    gt_to_pred = float(pred_tree.query(gt_xy, workers=1)[0].mean())
    bde = 0.5 * (pred_to_gt + gt_to_pred)
    return bde, units


def compute_boundary_metrics(
    *,
    input_contract: EvaluationInputContractResult,
    policy: BoundaryEvaluationPolicy,
) -> BoundaryMetricsResult:
    """Compute Stage C minimal boundary metrics on top of Stage A contract."""
    if not isinstance(input_contract, EvaluationInputContractResult):
        raise ContractError(
            "input_contract must be EvaluationInputContractResult from Stage A."
        )
    _validate_policy(policy)

    if input_contract.track_readiness.boundary_ready is not True:
        reason = (
            input_contract.track_readiness.boundary_reason
            if input_contract.track_readiness.boundary_reason is not None
            else "unknown reason"
        )
        raise ContractError(
            f"Boundary metrics cannot run because boundary track is not ready: {reason}."
        )

    expected_shape = (int(input_contract.common_height), int(input_contract.common_width))
    expected_h, expected_w = expected_shape

    tp = fp = fn = 0
    effective_pixels = 0
    valid_pixels = 0
    invalid_pixels = 0
    gt_boundary_positive_pixels = 0
    pred_boundary_positive_pixels = 0
    min_prob = float("inf")
    max_prob = float("-inf")

    gt_coords_chunks: list[np.ndarray] = []
    pred_coords_chunks: list[np.ndarray] = []
    gt_coord_points = 0
    pred_coord_points = 0

    rasterio, rasterio_errors = _require_rasterio()
    try:
        with (
            rasterio.open(input_contract.gt_boundary.path) as gt_boundary_ds,
            rasterio.open(input_contract.gt_valid.path) as gt_valid_ds,
            rasterio.open(input_contract.pred_boundary_prob.path) as pred_boundary_ds,
        ):
            if gt_boundary_ds.count != 1:
                raise ContractError(
                    f"gt_boundary must be single-band, got count={gt_boundary_ds.count}."
                )
            if gt_valid_ds.count != 1:
                raise ContractError(f"gt_valid must be single-band, got count={gt_valid_ds.count}.")
            if pred_boundary_ds.count != 3:
                raise ContractError(
                    f"pred_boundary_prob must be 3-band, got count={pred_boundary_ds.count}."
                )
            for ds, role in (
                (gt_boundary_ds, "gt_boundary"),
                (gt_valid_ds, "gt_valid"),
                (pred_boundary_ds, "pred_boundary_prob"),
            ):
                if (ds.height, ds.width) != (expected_h, expected_w):
                    raise ContractError(
                        f"{role} shape mismatch: expected {expected_shape}, got {(ds.height, ds.width)}."
                    )
            if not np.issubdtype(np.dtype(gt_boundary_ds.dtypes[0]), np.integer):
                raise ContractError("gt_boundary must be integer-encoded labels for Stage C.")

            for window in _iter_windows(height=expected_h, width=expected_w):
                gt_boundary = _read_window(
                    gt_boundary_ds,
                    1,
                    window=window,
                    role="gt_boundary",
                )
                gt_valid = _read_window(gt_valid_ds, 1, window=window, role="gt_valid")
                pred_boundary_prob = _read_window(
                    pred_boundary_ds,
                    (1, 2, 3),
                    window=window,
                    role="pred_boundary_prob",
                ).astype(np.float32)

                unique_gt_boundary = np.unique(gt_boundary)
                if not np.all(np.isin(unique_gt_boundary, [0, 1, 2])):
                    raise ContractError(
                        "gt_boundary must use baseline labels {0,1,2}, "
                        f"got unique values {unique_gt_boundary.tolist()}."
                    )

                unique_valid = np.unique(gt_valid)
                if not np.all(np.isin(unique_valid, [0, 1])):
                    raise ValidPolicyError(
                        "gt_valid must be binary {0,1} for Stage C boundary metrics, "
                        f"got unique values {unique_valid.tolist()}."
                    )

                min_prob = min(min_prob, float(pred_boundary_prob.min()))
                max_prob = max(max_prob, float(pred_boundary_prob.max()))
                if min_prob < -_EPS or max_prob > 1.0 + _EPS:
                    raise ContractError(
                        "pred_boundary_prob must be probability-like in [0,1] for Stage C."
                    )

                valid_mask = gt_valid > 0
                valid_count = int(valid_mask.sum())
                valid_pixels += valid_count
                invalid_pixels += int(valid_mask.size - valid_count)
                effective_pixels += valid_count

                if valid_count > 0:
                    sums_valid = pred_boundary_prob.sum(axis=0)[valid_mask]
                    if not np.all(np.abs(sums_valid - 1.0) <= _BOUNDARY_SUM_ATOL):
                        raise ContractError(
                            "pred_boundary_prob on valid pixels must represent a 3-class probability simplex."
                        )

                gt_positive = _gt_boundary_to_positive(gt_boundary, policy=policy) & valid_mask
                pred_positive = (
                    _boundary_prob_to_pred_positive(pred_boundary_prob, policy=policy)
                    & valid_mask
                )

                gt_boundary_positive_pixels += int(gt_positive.sum())
                pred_boundary_positive_pixels += int(pred_positive.sum())

                tp += int(np.logical_and(gt_positive, pred_positive).sum())
                fp += int(np.logical_and(~gt_positive & valid_mask, pred_positive).sum())
                fn += int(np.logical_and(gt_positive, ~pred_positive & valid_mask).sum())

                if policy.bde_enabled:
                    row_off = int(window.row_off)
                    col_off = int(window.col_off)
                    gt_coords = np.argwhere(gt_positive)
                    pred_coords = np.argwhere(pred_positive)
                    if gt_coords.size > 0:
                        gt_coords[:, 0] += row_off
                        gt_coords[:, 1] += col_off
                        gt_coords_chunks.append(gt_coords.astype(np.int32))
                        gt_coord_points += int(gt_coords.shape[0])
                    if pred_coords.size > 0:
                        pred_coords[:, 0] += row_off
                        pred_coords[:, 1] += col_off
                        pred_coords_chunks.append(pred_coords.astype(np.int32))
                        pred_coord_points += int(pred_coords.shape[0])
                    _guard_bde_coordinate_count(
                        pred_count=pred_coord_points,
                        gt_count=gt_coord_points,
                    )
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read Stage C rasters: {exc}") from exc

    if valid_pixels <= 0:
        raise ValidPolicyError("gt_valid contains zero valid pixels for Stage C.")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    boundary_bde: float | None = None
    boundary_bde_units: str | None = None
    if bool(policy.bde_enabled):
        pred_coords = (
            np.vstack(pred_coords_chunks).astype(np.int32)
            if pred_coords_chunks
            else np.zeros((0, 2), dtype=np.int32)
        )
        gt_coords = (
            np.vstack(gt_coords_chunks).astype(np.int32)
            if gt_coords_chunks
            else np.zeros((0, 2), dtype=np.int32)
        )
        boundary_bde, boundary_bde_units = _compute_bde_from_coords(
            pred_coords_rc=pred_coords,
            gt_coords_rc=gt_coords,
            input_contract=input_contract,
            policy=policy,
        )

    return BoundaryMetricsResult(
        boundary_f1=f1,
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_bde=boundary_bde,
        boundary_bde_units=boundary_bde_units,
        tp=tp,
        fp=fp,
        fn=fn,
        effective_pixels=effective_pixels,
        valid_pixels=valid_pixels,
        invalid_pixels=invalid_pixels,
        gt_boundary_positive_pixels=gt_boundary_positive_pixels,
        pred_boundary_positive_pixels=pred_boundary_positive_pixels,
        prediction_interpretation=policy.prediction_interpretation,
        gt_interpretation=policy.gt_interpretation,
        threshold_provenance=policy.threshold_provenance,
        non_background_prob_threshold=float(policy.non_background_prob_threshold),
        bde_enabled=bool(policy.bde_enabled),
        stage_scope="stage_c_boundary_metrics_only",
    )


def build_boundary_metrics_summary(result: BoundaryMetricsResult) -> dict[str, Any]:
    """Build a compact machine-readable summary for Stage C outputs."""
    if not isinstance(result, BoundaryMetricsResult):
        raise ContractError("result must be BoundaryMetricsResult.")
    return {
        "stage_scope": result.stage_scope,
        "metrics": {
            "boundary_f1": result.boundary_f1,
            "boundary_precision": result.boundary_precision,
            "boundary_recall": result.boundary_recall,
            "boundary_bde": result.boundary_bde,
            "boundary_bde_units": result.boundary_bde_units,
        },
        "counts": {
            "tp": result.tp,
            "fp": result.fp,
            "fn": result.fn,
            "effective_pixels": result.effective_pixels,
            "valid_pixels": result.valid_pixels,
            "invalid_pixels": result.invalid_pixels,
            "gt_boundary_positive_pixels": result.gt_boundary_positive_pixels,
            "pred_boundary_positive_pixels": result.pred_boundary_positive_pixels,
        },
        "policy": {
            "prediction_interpretation": result.prediction_interpretation,
            "gt_interpretation": result.gt_interpretation,
            "threshold_provenance": result.threshold_provenance,
            "non_background_prob_threshold": result.non_background_prob_threshold,
            "bde_enabled": result.bde_enabled,
        },
    }


__all__ = [
    "BoundaryEvaluationPolicy",
    "BoundaryMetricsResult",
    "build_boundary_metrics_summary",
    "compute_boundary_metrics",
]
