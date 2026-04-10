"""Stage B.5 distance auxiliary metrics for module_eval.

Computes MAE and RMSE for distance_pred vs gt_distance (module_eval §6.3).
Only runs when gt_distance is available in the input contract; skipped otherwise.

Out of scope:
- pixel / boundary / object/structure metrics,
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
_STREAM_WINDOW_SIZE = 1024


@dataclass(frozen=True)
class DistanceEvaluationPolicy:
    """Policy for Stage B.5 distance auxiliary metrics."""

    threshold_provenance: str
    # Whether to raise an error when gt_distance is absent, or silently skip.
    absent_gt_policy: str = "skip"  # "skip" | "error"


@dataclass(frozen=True)
class DistanceMetricsResult:
    """Stage B.5 result contract for distance auxiliary metrics."""

    mae: float
    rmse: float

    valid_pixels: int
    invalid_pixels: int
    effective_pixels: int

    threshold_provenance: str
    stage_scope: str
    skipped: bool  # True when gt_distance was absent and policy is "skip"
    skip_reason: str | None


def _require_rasterio() -> tuple[Any, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for module_eval Stage B.5.") from exc
    return rasterio, rasterio.errors


def _iter_windows(*, height: int, width: int, window_size: int = _STREAM_WINDOW_SIZE):
    rasterio, _ = _require_rasterio()
    Window = rasterio.windows.Window
    for row_off in range(0, height, window_size):
        win_h = min(window_size, height - row_off)
        for col_off in range(0, width, window_size):
            win_w = min(window_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def _read_window(ds: Any, *, window: Any, role: str) -> np.ndarray:
    arr = ds.read(1, window=window)
    if not np.isfinite(arr).all():
        raise ContractError(f"{role} contains non-finite values.")
    return arr


def _validate_policy(policy: DistanceEvaluationPolicy) -> None:
    if not isinstance(policy, DistanceEvaluationPolicy):
        raise ContractError("policy must be DistanceEvaluationPolicy for Stage B.5.")
    if not isinstance(policy.threshold_provenance, str) or policy.threshold_provenance.strip() == "":
        raise ContractError("policy.threshold_provenance must be a non-empty string.")
    if policy.absent_gt_policy not in {"skip", "error"}:
        raise ContractError(
            "policy.absent_gt_policy must be 'skip' or 'error', "
            f"got {policy.absent_gt_policy!r}."
        )


_SKIPPED_RESULT = DistanceMetricsResult(
    mae=float("nan"),
    rmse=float("nan"),
    valid_pixels=0,
    invalid_pixels=0,
    effective_pixels=0,
    threshold_provenance="",
    stage_scope="stage_b5_distance_metrics_skipped",
    skipped=True,
    skip_reason="gt_distance not available in input contract",
)


def compute_distance_metrics(
    *,
    input_contract: EvaluationInputContractResult,
    policy: DistanceEvaluationPolicy,
) -> DistanceMetricsResult:
    """Compute Stage B.5 distance auxiliary metrics (MAE, RMSE).

    Skips gracefully if gt_distance is absent and policy.absent_gt_policy == 'skip'.
    Raises ContractError if policy.absent_gt_policy == 'error' and gt_distance is absent.
    """
    if not isinstance(input_contract, EvaluationInputContractResult):
        raise ContractError(
            "input_contract must be EvaluationInputContractResult from Stage A."
        )
    _validate_policy(policy)

    if input_contract.gt_distance is None:
        if policy.absent_gt_policy == "error":
            raise ContractError(
                "gt_distance is absent from input contract but "
                "DistanceEvaluationPolicy.absent_gt_policy == 'error'."
            )
        return _SKIPPED_RESULT

    expected_shape = (int(input_contract.common_height), int(input_contract.common_width))
    expected_h, expected_w = expected_shape

    valid_pixels = 0
    invalid_pixels = 0
    sum_abs_error = 0.0
    sum_sq_error = 0.0

    rasterio, rasterio_errors = _require_rasterio()
    try:
        with (
            rasterio.open(input_contract.gt_distance.path) as gt_distance_ds,
            rasterio.open(input_contract.pred_distance_pred.path) as pred_distance_ds,
            rasterio.open(input_contract.gt_valid.path) as gt_valid_ds,
        ):
            for ds, role in (
                (gt_distance_ds, "gt_distance"),
                (pred_distance_ds, "pred_distance_pred"),
                (gt_valid_ds, "gt_valid"),
            ):
                if ds.count != 1:
                    raise ContractError(f"{role} must be single-band, got count={ds.count}.")
                if (ds.height, ds.width) != (expected_h, expected_w):
                    raise ContractError(
                        f"{role} shape mismatch: expected {expected_shape}, got {(ds.height, ds.width)}."
                    )

            for window in _iter_windows(height=expected_h, width=expected_w):
                gt_distance = _read_window(
                    gt_distance_ds,
                    window=window,
                    role="gt_distance",
                ).astype(np.float32)
                pred_distance = _read_window(
                    pred_distance_ds,
                    window=window,
                    role="pred_distance_pred",
                ).astype(np.float32)
                gt_valid = _read_window(gt_valid_ds, window=window, role="gt_valid")

                unique_valid = np.unique(gt_valid)
                if not np.all(np.isin(unique_valid, [0, 1])):
                    raise ValidPolicyError(
                        "gt_valid must be binary {0,1} for Stage B.5 distance metrics, "
                        f"got unique values {unique_valid.tolist()}."
                    )

                valid_mask = gt_valid > 0
                valid_count = int(valid_mask.sum())
                valid_pixels += valid_count
                invalid_pixels += int(valid_mask.size - valid_count)

                if valid_count == 0:
                    continue

                diff = (pred_distance - gt_distance)[valid_mask].astype(np.float64)
                sum_abs_error += float(np.abs(diff).sum())
                sum_sq_error += float((diff ** 2).sum())
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read Stage B.5 rasters: {exc}") from exc

    if valid_pixels == 0:
        raise ValidPolicyError("gt_valid contains zero valid pixels for Stage B.5 distance metrics.")

    mae = float(sum_abs_error / float(valid_pixels))
    rmse = float(np.sqrt(sum_sq_error / float(valid_pixels)))

    return DistanceMetricsResult(
        mae=mae,
        rmse=rmse,
        valid_pixels=valid_pixels,
        invalid_pixels=invalid_pixels,
        effective_pixels=valid_pixels,
        threshold_provenance=policy.threshold_provenance,
        stage_scope="stage_b5_distance_metrics",
        skipped=False,
        skip_reason=None,
    )


def build_distance_metrics_summary(result: DistanceMetricsResult) -> dict[str, Any]:
    """Build a compact machine-readable summary for Stage B.5 outputs."""
    if not isinstance(result, DistanceMetricsResult):
        raise ContractError("result must be DistanceMetricsResult.")
    return {
        "stage_scope": result.stage_scope,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "metrics": {
            "mae": result.mae if not result.skipped else None,
            "rmse": result.rmse if not result.skipped else None,
        },
        "counts": {
            "valid_pixels": result.valid_pixels,
            "invalid_pixels": result.invalid_pixels,
            "effective_pixels": result.effective_pixels,
        },
        "policy": {
            "threshold_provenance": result.threshold_provenance,
        },
    }


__all__ = [
    "DistanceEvaluationPolicy",
    "DistanceMetricsResult",
    "build_distance_metrics_summary",
    "compute_distance_metrics",
]
