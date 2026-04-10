"""Minimal pairwise comparison computation layer for module_eval.

This layer is intentionally narrow:
- consumes resolved comparison input-contract from Stage comparison-contract;
- computes metric deltas only for comparable metric groups;
- keeps non-comparable groups explicit (no fake deltas);
- exposes explicit delta direction/orientation semantics.

Out of scope:
- leaderboard/ranking engine,
- dashboard/reporting framework,
- multi-run table manager.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError
from ai_fields.module_eval.comparison_contract import EvalComparisonInputContractResult

_EXPECTED_SUMMARY_SCHEMA = "eval.summary"
_DELTA_DIRECTION = "right_minus_left"
_COMPARABLE_GROUPS = ("pixel", "boundary", "object_structure")

_REQUIRED_METRICS: dict[str, tuple[str, ...]] = {
    "pixel": ("extent_iou", "extent_f1", "extent_precision", "extent_recall"),
    "boundary": ("boundary_f1", "boundary_precision", "boundary_recall"),
    "object_structure": ("goc", "guc", "gtc"),
}
_OPTIONAL_METRICS: dict[str, tuple[str, ...]] = {
    "pixel": (),
    "boundary": ("boundary_bde",),
    "object_structure": ("normalized_gtc",),
}
_METRIC_ORIENTATION: dict[str, str] = {
    "extent_iou": "higher_is_better",
    "extent_f1": "higher_is_better",
    "extent_precision": "higher_is_better",
    "extent_recall": "higher_is_better",
    "boundary_f1": "higher_is_better",
    "boundary_precision": "higher_is_better",
    "boundary_recall": "higher_is_better",
    "boundary_bde": "lower_is_better",
    "goc": "lower_is_better",
    "guc": "lower_is_better",
    "gtc": "lower_is_better",
    "normalized_gtc": "lower_is_better",
}


@dataclass(frozen=True)
class PairwiseEvalComparisonResult:
    """Pairwise delta result for already validated comparison input-contract."""

    left_run_id: str
    right_run_id: str
    delta_direction: str
    comparison_status: str
    ready_for_pairwise_compare: bool
    partially_ready: bool

    comparable_metric_groups: tuple[str, ...]
    non_comparable_metric_groups: dict[str, str]
    per_group_metric_deltas: dict[str, dict[str, float]]
    per_group_metric_orientation: dict[str, dict[str, str]]
    skipped_metric_reasons: dict[str, dict[str, str]]

    policy_compatibility_summary: dict[str, Any]
    provenance_compatibility_summary: dict[str, Any]
    stage_coverage_compatibility_summary: dict[str, Any]


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping/object.")
    return dict(value)


def _as_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric, got {type(value).__name__}.")
    return float(value)


def _load_eval_summary(path: Path, *, side: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{side}: summary.json is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ContractError(f"{side}: failed reading summary.json: {exc}") from exc

    summary = _require_mapping(payload, name=f"{side}.summary")
    schema_name = summary.get("schema_name")
    if schema_name != _EXPECTED_SUMMARY_SCHEMA:
        raise ContractError(
            f"{side}: summary schema_name must be {_EXPECTED_SUMMARY_SCHEMA!r}, got {schema_name!r}."
        )
    _require_mapping(summary.get("metric_summary"), name=f"{side}.summary.metric_summary")
    return summary


def _extract_group_metrics(
    summary: dict[str, Any],
    *,
    side: str,
    group: str,
) -> dict[str, Any]:
    metric_summary = _require_mapping(
        summary.get("metric_summary"),
        name=f"{side}.summary.metric_summary",
    )
    group_payload = _require_mapping(
        metric_summary.get(group),
        name=f"{side}.summary.metric_summary.{group}",
    )
    return _require_mapping(
        group_payload.get("metrics"),
        name=f"{side}.summary.metric_summary.{group}.metrics",
    )


def _compute_group_deltas(
    *,
    group: str,
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    if group not in _COMPARABLE_GROUPS:
        raise ContractError(f"Unsupported comparable metric group {group!r}.")

    left_metrics = _extract_group_metrics(left_summary, side="left", group=group)
    right_metrics = _extract_group_metrics(right_summary, side="right", group=group)

    deltas: dict[str, float] = {}
    orientations: dict[str, str] = {}
    skipped: dict[str, str] = {}

    for metric_name in _REQUIRED_METRICS[group]:
        if metric_name not in left_metrics or metric_name not in right_metrics:
            raise ContractError(
                f"Comparable group {group!r} requires metric {metric_name!r} on both sides."
            )
        left_val = _as_float(
            left_metrics[metric_name],
            name=f"left.summary.metric_summary.{group}.metrics.{metric_name}",
        )
        right_val = _as_float(
            right_metrics[metric_name],
            name=f"right.summary.metric_summary.{group}.metrics.{metric_name}",
        )
        deltas[f"{metric_name}_delta"] = right_val - left_val
        orientations[metric_name] = _METRIC_ORIENTATION[metric_name]

    for metric_name in _OPTIONAL_METRICS[group]:
        has_left = metric_name in left_metrics
        has_right = metric_name in right_metrics
        if not has_left or not has_right:
            skipped[metric_name] = "metric missing on one or both sides"
            continue

        left_raw = left_metrics[metric_name]
        right_raw = right_metrics[metric_name]
        if left_raw is None or right_raw is None:
            skipped[metric_name] = "metric is null on one or both sides"
            continue

        left_val = _as_float(
            left_raw,
            name=f"left.summary.metric_summary.{group}.metrics.{metric_name}",
        )
        right_val = _as_float(
            right_raw,
            name=f"right.summary.metric_summary.{group}.metrics.{metric_name}",
        )
        deltas[f"{metric_name}_delta"] = right_val - left_val
        orientations[metric_name] = _METRIC_ORIENTATION[metric_name]

    return deltas, orientations, skipped


def compute_pairwise_eval_comparison(
    *,
    comparison_contract: EvalComparisonInputContractResult,
) -> PairwiseEvalComparisonResult:
    """Compute pairwise metric deltas for comparable groups only."""
    if not isinstance(comparison_contract, EvalComparisonInputContractResult):
        raise ContractError(
            "comparison_contract must be EvalComparisonInputContractResult."
        )

    comparable_groups = tuple(comparison_contract.comparable_metric_groups)
    non_comparable_groups = dict(comparison_contract.non_comparable_metric_groups)

    ready = bool(comparison_contract.ready_for_pairwise_compare)
    partial = bool(comparison_contract.partially_ready or (not ready and len(comparable_groups) > 0))
    if ready:
        status = "ready"
    elif partial:
        status = "partial"
    else:
        status = "not_ready"

    if len(comparable_groups) == 0:
        return PairwiseEvalComparisonResult(
            left_run_id=comparison_contract.left_run_id,
            right_run_id=comparison_contract.right_run_id,
            delta_direction=_DELTA_DIRECTION,
            comparison_status=status,
            ready_for_pairwise_compare=ready,
            partially_ready=partial,
            comparable_metric_groups=comparable_groups,
            non_comparable_metric_groups=non_comparable_groups,
            per_group_metric_deltas={},
            per_group_metric_orientation={},
            skipped_metric_reasons={},
            policy_compatibility_summary=dict(comparison_contract.policy_compatibility_summary),
            provenance_compatibility_summary=dict(comparison_contract.provenance_compatibility_summary),
            stage_coverage_compatibility_summary=dict(comparison_contract.stage_coverage_compatibility_summary),
        )

    left_summary = _load_eval_summary(comparison_contract.left_summary_path, side="left")
    right_summary = _load_eval_summary(comparison_contract.right_summary_path, side="right")

    per_group_deltas: dict[str, dict[str, float]] = {}
    per_group_orientation: dict[str, dict[str, str]] = {}
    skipped_metric_reasons: dict[str, dict[str, str]] = {}

    for group in comparable_groups:
        deltas, orientations, skipped = _compute_group_deltas(
            group=group,
            left_summary=left_summary,
            right_summary=right_summary,
        )
        per_group_deltas[group] = deltas
        per_group_orientation[group] = orientations
        if len(skipped) > 0:
            skipped_metric_reasons[group] = skipped

    return PairwiseEvalComparisonResult(
        left_run_id=comparison_contract.left_run_id,
        right_run_id=comparison_contract.right_run_id,
        delta_direction=_DELTA_DIRECTION,
        comparison_status=status,
        ready_for_pairwise_compare=ready,
        partially_ready=partial,
        comparable_metric_groups=comparable_groups,
        non_comparable_metric_groups=non_comparable_groups,
        per_group_metric_deltas=per_group_deltas,
        per_group_metric_orientation=per_group_orientation,
        skipped_metric_reasons=skipped_metric_reasons,
        policy_compatibility_summary=dict(comparison_contract.policy_compatibility_summary),
        provenance_compatibility_summary=dict(comparison_contract.provenance_compatibility_summary),
        stage_coverage_compatibility_summary=dict(comparison_contract.stage_coverage_compatibility_summary),
    )


def build_pairwise_comparison_summary(
    result: PairwiseEvalComparisonResult,
) -> dict[str, Any]:
    """Build compact machine-readable summary for pairwise comparison deltas."""
    if not isinstance(result, PairwiseEvalComparisonResult):
        raise ContractError("result must be PairwiseEvalComparisonResult.")

    return {
        "left_run_id": result.left_run_id,
        "right_run_id": result.right_run_id,
        "delta_direction": result.delta_direction,
        "comparison_status": result.comparison_status,
        "ready_for_pairwise_compare": result.ready_for_pairwise_compare,
        "partially_ready": result.partially_ready,
        "comparable_metric_groups": list(result.comparable_metric_groups),
        "non_comparable_metric_groups": dict(result.non_comparable_metric_groups),
        "metric_deltas": dict(result.per_group_metric_deltas),
        "metric_orientation": dict(result.per_group_metric_orientation),
        "skipped_metric_reasons": dict(result.skipped_metric_reasons),
    }


__all__ = [
    "PairwiseEvalComparisonResult",
    "build_pairwise_comparison_summary",
    "compute_pairwise_eval_comparison",
]

