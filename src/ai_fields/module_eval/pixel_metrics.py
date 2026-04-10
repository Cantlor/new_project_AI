"""Stage B minimal global/pixel metrics for module_eval.

Scope of this layer is intentionally narrow:
- compute extent raster pixel/global metrics only;
- run only on top of Stage A resolved EvaluationInputContractResult;
- enforce explicit threshold policy and valid/ignore-aware masking.

Out of scope:
- boundary metrics,
- object/structure metrics,
- reporting/comparison framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError, ValidPolicyError
from ai_fields.module_eval.input_contract import EvaluationInputContractResult

_EPS = 1e-9
_STREAM_WINDOW_SIZE = 1024


@dataclass(frozen=True)
class PixelBinarizationPolicy:
    """Explicit threshold policy for Stage B pixel/global metrics."""

    extent_prob_threshold: float
    threshold_provenance: str

    positive_gt_label: int = 1
    ignore_gt_label: int = 255
    prediction_rule: str = "gte"


@dataclass(frozen=True)
class GlobalPixelMetricsResult:
    """Minimal Stage B result contract for extent pixel/global metrics."""

    iou: float
    f1: float
    precision: float
    recall: float

    tp: int
    fp: int
    fn: int
    tn: int

    effective_pixels: int
    ignored_pixels: int
    invalid_pixels: int
    valid_pixels: int

    threshold: float
    threshold_provenance: str
    prediction_rule: str
    stage_scope: str


def _require_rasterio() -> tuple[Any, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for module_eval Stage B.") from exc
    return rasterio, rasterio.errors


def _validate_policy(policy: PixelBinarizationPolicy) -> PixelBinarizationPolicy:
    if not isinstance(policy, PixelBinarizationPolicy):
        raise ContractError("policy must be PixelBinarizationPolicy for Stage B.")

    threshold = float(policy.extent_prob_threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ContractError(
            "policy.extent_prob_threshold must be in [0,1], "
            f"got {policy.extent_prob_threshold!r}."
        )
    if not isinstance(policy.threshold_provenance, str) or policy.threshold_provenance.strip() == "":
        raise ContractError("policy.threshold_provenance must be a non-empty string.")

    if policy.prediction_rule not in {"gte", "gt"}:
        raise ContractError(
            "Unsupported policy.prediction_rule. Expected one of {'gte','gt'}."
        )

    if (
        isinstance(policy.positive_gt_label, bool)
        or not isinstance(policy.positive_gt_label, int)
        or policy.positive_gt_label < 0
    ):
        raise ContractError("policy.positive_gt_label must be a non-negative integer.")
    if (
        isinstance(policy.ignore_gt_label, bool)
        or not isinstance(policy.ignore_gt_label, int)
        or policy.ignore_gt_label < 0
    ):
        raise ContractError("policy.ignore_gt_label must be a non-negative integer.")
    if policy.positive_gt_label == policy.ignore_gt_label:
        raise ContractError("policy.positive_gt_label must differ from policy.ignore_gt_label.")

    return policy


def _iter_windows(*, height: int, width: int, window_size: int = _STREAM_WINDOW_SIZE):
    rasterio, _ = _require_rasterio()
    Window = rasterio.windows.Window
    for row_off in range(0, height, window_size):
        win_h = min(window_size, height - row_off)
        for col_off in range(0, width, window_size):
            win_w = min(window_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def _read_window(
    ds: Any,
    *,
    window: Any,
    role: str,
) -> np.ndarray:
    arr = ds.read(1, window=window)
    if not np.isfinite(arr).all():
        raise ContractError(f"{role} contains non-finite values.")
    return arr


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= _EPS:
        return 0.0
    return float(numerator / denominator)


def compute_global_pixel_metrics(
    *,
    input_contract: EvaluationInputContractResult,
    policy: PixelBinarizationPolicy,
) -> GlobalPixelMetricsResult:
    """Compute minimal Stage B global/pixel metrics for extent track only."""
    if not isinstance(input_contract, EvaluationInputContractResult):
        raise ContractError(
            "input_contract must be EvaluationInputContractResult from Stage A."
        )
    _validate_policy(policy)

    if input_contract.track_readiness.pixel_ready is not True:
        reason = (
            input_contract.track_readiness.pixel_reason
            if input_contract.track_readiness.pixel_reason is not None
            else "unknown reason"
        )
        raise ContractError(
            f"Pixel/global metrics cannot run because pixel track is not ready: {reason}."
        )

    expected_shape = (int(input_contract.common_height), int(input_contract.common_width))
    expected_h, expected_w = expected_shape

    tp = fp = fn = tn = 0
    effective_pixels = 0
    ignored_pixels = 0
    valid_pixels = 0
    invalid_pixels = 0
    pred_min = float("inf")
    pred_max = float("-inf")

    rasterio, rasterio_errors = _require_rasterio()
    try:
        with (
            rasterio.open(input_contract.gt_extent.path) as gt_extent_ds,
            rasterio.open(input_contract.gt_valid.path) as gt_valid_ds,
            rasterio.open(input_contract.pred_extent_prob.path) as pred_extent_ds,
        ):
            for ds, role in (
                (gt_extent_ds, "gt_extent"),
                (gt_valid_ds, "gt_valid"),
                (pred_extent_ds, "pred_extent_prob"),
            ):
                if ds.count != 1:
                    raise ContractError(f"{role} must be single-band, got count={ds.count}.")
                if (ds.height, ds.width) != (expected_h, expected_w):
                    raise ContractError(
                        f"{role} shape mismatch: expected {expected_shape}, "
                        f"got {(ds.height, ds.width)}."
                    )

            if not np.issubdtype(np.dtype(gt_extent_ds.dtypes[0]), np.integer):
                raise ContractError("gt_extent must be integer-encoded labels for Stage B metrics.")

            for window in _iter_windows(height=expected_h, width=expected_w):
                gt_extent = _read_window(gt_extent_ds, window=window, role="gt_extent")
                gt_valid = _read_window(gt_valid_ds, window=window, role="gt_valid")
                pred_extent_prob = _read_window(
                    pred_extent_ds,
                    window=window,
                    role="pred_extent_prob",
                ).astype(np.float32)

                pred_min = min(pred_min, float(pred_extent_prob.min()))
                pred_max = max(pred_max, float(pred_extent_prob.max()))
                if pred_min < -_EPS or pred_max > 1.0 + _EPS:
                    raise ContractError(
                        "pred_extent_prob must be probability-like in [0,1] for Stage B metrics."
                    )

                unique_valid = np.unique(gt_valid)
                if not np.all(np.isin(unique_valid, [0, 1])):
                    raise ValidPolicyError(
                        "gt_valid must be binary {0,1} for Stage B metrics, "
                        f"got unique values {unique_valid.tolist()}."
                    )

                valid01 = gt_valid > 0
                eval_mask = valid01 & (gt_extent != int(policy.ignore_gt_label))

                valid_pixels += int(valid01.sum())
                invalid_pixels += int(valid01.size - int(valid01.sum()))
                ignored_pixels += int((gt_extent == int(policy.ignore_gt_label)).sum())
                effective_pixels += int(eval_mask.sum())

                gt_positive = (gt_extent == int(policy.positive_gt_label)) & eval_mask
                if policy.prediction_rule == "gte":
                    pred_positive = (pred_extent_prob >= float(policy.extent_prob_threshold)) & eval_mask
                else:
                    pred_positive = (pred_extent_prob > float(policy.extent_prob_threshold)) & eval_mask

                tp += int(np.logical_and(gt_positive, pred_positive).sum())
                fp += int(np.logical_and(~gt_positive & eval_mask, pred_positive).sum())
                fn += int(np.logical_and(gt_positive, ~pred_positive & eval_mask).sum())
                tn += int(np.logical_and(~gt_positive & eval_mask, ~pred_positive & eval_mask).sum())
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read Stage B rasters: {exc}") from exc

    if effective_pixels <= 0:
        raise ContractError(
            "No effective pixels available after valid/ignore masking for Stage B metrics."
        )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)

    return GlobalPixelMetricsResult(
        iou=iou,
        f1=f1,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        effective_pixels=effective_pixels,
        ignored_pixels=ignored_pixels,
        invalid_pixels=invalid_pixels,
        valid_pixels=valid_pixels,
        threshold=float(policy.extent_prob_threshold),
        threshold_provenance=policy.threshold_provenance,
        prediction_rule=policy.prediction_rule,
        stage_scope="stage_b_global_pixel_metrics_only",
    )


def build_pixel_metrics_summary(result: GlobalPixelMetricsResult) -> dict[str, Any]:
    """Build a compact machine-readable summary for Stage B outputs."""
    if not isinstance(result, GlobalPixelMetricsResult):
        raise ContractError("result must be GlobalPixelMetricsResult.")
    return {
        "stage_scope": result.stage_scope,
        "metrics": {
            "extent_iou": result.iou,
            "extent_f1": result.f1,
            "extent_precision": result.precision,
            "extent_recall": result.recall,
        },
        "counts": {
            "tp": result.tp,
            "fp": result.fp,
            "fn": result.fn,
            "tn": result.tn,
            "effective_pixels": result.effective_pixels,
            "valid_pixels": result.valid_pixels,
            "invalid_pixels": result.invalid_pixels,
            "ignored_pixels": result.ignored_pixels,
        },
        "threshold_policy": {
            "extent_prob_threshold": result.threshold,
            "threshold_provenance": result.threshold_provenance,
            "prediction_rule": result.prediction_rule,
        },
    }


__all__ = [
    "GlobalPixelMetricsResult",
    "PixelBinarizationPolicy",
    "build_pixel_metrics_summary",
    "compute_global_pixel_metrics",
]
