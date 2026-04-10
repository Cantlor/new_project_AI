"""Minimal comparison artifact/report export layer for module_eval.

This layer is intentionally narrow:
- consumes resolved comparison-input contract and pairwise delta result;
- writes reproducible comparison artifacts;
- keeps provenance/policy/compatibility visibility explicit.

Out of scope:
- leaderboard/ranking engine,
- dashboard/report portal,
- cross-run benchmark manager.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import write_summary
from ai_fields.module_eval.comparison_contract import EvalComparisonInputContractResult
from ai_fields.module_eval.pairwise_comparison import (
    PairwiseEvalComparisonResult,
    build_pairwise_comparison_summary,
)

_COMPARISON_REPORT_SCHEMA = "eval.comparison_report"
_COMPARISON_SUMMARY_SCHEMA = "eval.comparison_summary"
_COMPARISON_DELTA_TABLE_SCHEMA = "eval.comparison_delta_table"
_DELTA_DIRECTION = "right_minus_left"

_PAIRWISE_SUMMARY_REQUIRED_FIELDS = (
    "left_run_id",
    "right_run_id",
    "delta_direction",
    "comparison_status",
    "ready_for_pairwise_compare",
    "partially_ready",
    "comparable_metric_groups",
    "non_comparable_metric_groups",
    "metric_deltas",
    "metric_orientation",
    "skipped_metric_reasons",
)
_COMPARISON_REPORT_REQUIRED_FIELDS = (
    "schema_name",
    "schema_version",
    "module_name",
    "data_contract_version",
    "run_id",
    "status",
    "comparison_mode",
    "left_run",
    "right_run",
    "comparison_contract",
    "pairwise_result",
    "policy_compatibility_summary",
    "provenance_compatibility_summary",
    "stage_coverage_compatibility_summary",
    "artifacts",
    "stage_coverage",
)
_COMPARISON_SUMMARY_REQUIRED_FIELDS = (
    "schema_name",
    "status",
    "comparison_mode",
    "run_id",
    "left_run_id",
    "right_run_id",
    "delta_direction",
    "comparison_status",
    "comparable_metric_groups",
    "non_comparable_metric_groups",
    "metric_delta_counts",
    "policy_compatibility_summary",
    "provenance_compatibility_summary",
    "artifacts",
)
_COMPARISON_DELTA_TABLE_REQUIRED_FIELDS = (
    "schema_name",
    "status",
    "comparison_mode",
    "run_id",
    "left_run_id",
    "right_run_id",
    "delta_direction",
    "comparison_status",
    "comparable_metric_groups",
    "non_comparable_metric_groups",
    "rows",
    "provenance_links",
    "policy_compatibility_summary",
    "provenance_compatibility_summary",
    "stage_coverage_compatibility_summary",
    "stage_honesty",
)
_COMPARISON_DELTA_TABLE_ROW_REQUIRED_FIELDS = (
    "metric_group",
    "metric_name",
    "delta_name",
    "delta_value",
    "delta_direction",
    "orientation",
    "comparable",
)


@dataclass(frozen=True)
class PairwiseComparisonExportArtifacts:
    """Paths for minimal comparison export artifacts."""

    output_dir: Path
    comparison_report_path: Path
    comparison_summary_path: Path
    comparison_delta_table_path: Path
    config_used_path: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_output_dir(path: Any) -> Path:
    if not isinstance(path, (str, PathLike)):
        raise ContractError(
            f"output_dir must be path-like (str/Path), got {type(path).__name__}."
        )
    resolved = Path(path)
    if str(resolved).strip() == "":
        raise ContractError("output_dir must be a non-empty path-like value.")
    return resolved


def _require_non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{name} must be a non-empty string.")
    return value


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping/object.")
    return dict(value)


def _require_fields(
    payload: Mapping[str, Any],
    *,
    artifact_name: str,
    fields: tuple[str, ...],
) -> None:
    missing = [field for field in fields if field not in payload]
    if len(missing) > 0:
        raise ContractError(
            f"{artifact_name} payload is missing required fields: {sorted(missing)}."
        )


def _validate_pairwise_summary_payload(pairwise_summary: Mapping[str, Any]) -> None:
    _require_fields(
        pairwise_summary,
        artifact_name="pairwise_summary",
        fields=_PAIRWISE_SUMMARY_REQUIRED_FIELDS,
    )
    if pairwise_summary.get("delta_direction") != _DELTA_DIRECTION:
        raise ContractError(
            f"pairwise_summary.delta_direction must be {_DELTA_DIRECTION!r}."
        )


def _validate_report_payload(payload: Mapping[str, Any]) -> None:
    _require_fields(
        payload,
        artifact_name="comparison_report",
        fields=_COMPARISON_REPORT_REQUIRED_FIELDS,
    )
    if payload.get("schema_name") != _COMPARISON_REPORT_SCHEMA:
        raise ContractError(
            f"comparison_report schema_name must be {_COMPARISON_REPORT_SCHEMA!r}."
        )
    if payload.get("comparison_mode") != "pairwise":
        raise ContractError("comparison_report.comparison_mode must be 'pairwise'.")
    pairwise_result = _require_mapping(
        payload.get("pairwise_result"),
        name="comparison_report.pairwise_result",
    )
    if pairwise_result.get("delta_direction") != _DELTA_DIRECTION:
        raise ContractError(
            f"comparison_report.pairwise_result.delta_direction must be {_DELTA_DIRECTION!r}."
        )


def _validate_summary_payload(payload: Mapping[str, Any]) -> None:
    _require_fields(
        payload,
        artifact_name="comparison_summary",
        fields=_COMPARISON_SUMMARY_REQUIRED_FIELDS,
    )
    if payload.get("schema_name") != _COMPARISON_SUMMARY_SCHEMA:
        raise ContractError(
            f"comparison_summary schema_name must be {_COMPARISON_SUMMARY_SCHEMA!r}."
        )
    if payload.get("comparison_mode") != "pairwise":
        raise ContractError("comparison_summary.comparison_mode must be 'pairwise'.")
    if payload.get("delta_direction") != _DELTA_DIRECTION:
        raise ContractError(
            f"comparison_summary.delta_direction must be {_DELTA_DIRECTION!r}."
        )


def _validate_delta_table_payload(payload: Mapping[str, Any]) -> None:
    _require_fields(
        payload,
        artifact_name="comparison_delta_table",
        fields=_COMPARISON_DELTA_TABLE_REQUIRED_FIELDS,
    )
    if payload.get("schema_name") != _COMPARISON_DELTA_TABLE_SCHEMA:
        raise ContractError(
            f"comparison_delta_table schema_name must be {_COMPARISON_DELTA_TABLE_SCHEMA!r}."
        )
    if payload.get("comparison_mode") != "pairwise":
        raise ContractError("comparison_delta_table.comparison_mode must be 'pairwise'.")
    if payload.get("delta_direction") != _DELTA_DIRECTION:
        raise ContractError(
            f"comparison_delta_table.delta_direction must be {_DELTA_DIRECTION!r}."
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ContractError("comparison_delta_table.rows must be a list.")
    for idx, row_raw in enumerate(rows):
        row = _require_mapping(
            row_raw,
            name=f"comparison_delta_table.rows[{idx}]",
        )
        _require_fields(
            row,
            artifact_name=f"comparison_delta_table.rows[{idx}]",
            fields=_COMPARISON_DELTA_TABLE_ROW_REQUIRED_FIELDS,
        )
        if row.get("delta_direction") != _DELTA_DIRECTION:
            raise ContractError(
                f"comparison_delta_table.rows[{idx}].delta_direction must be {_DELTA_DIRECTION!r}."
            )
        if not isinstance(row.get("comparable"), bool):
            raise ContractError(
                f"comparison_delta_table.rows[{idx}].comparable must be bool."
            )


def _write_config_used(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ContractError("PyYAML is required to write comparison config_used.yaml.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            yaml.safe_dump(dict(payload), sort_keys=True, allow_unicode=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ContractError(f"Failed to write comparison config_used.yaml at {path}: {exc}") from exc


def _validate_inputs(
    *,
    comparison_contract: EvalComparisonInputContractResult,
    pairwise_result: PairwiseEvalComparisonResult,
) -> None:
    if not isinstance(comparison_contract, EvalComparisonInputContractResult):
        raise ContractError(
            "comparison_contract must be EvalComparisonInputContractResult."
        )
    if not isinstance(pairwise_result, PairwiseEvalComparisonResult):
        raise ContractError("pairwise_result must be PairwiseEvalComparisonResult.")

    if pairwise_result.left_run_id != comparison_contract.left_run_id:
        raise ContractError(
            "left_run_id mismatch between comparison contract and pairwise result."
        )
    if pairwise_result.right_run_id != comparison_contract.right_run_id:
        raise ContractError(
            "right_run_id mismatch between comparison contract and pairwise result."
        )
    if pairwise_result.delta_direction != _DELTA_DIRECTION:
        raise ContractError(
            f"Unsupported delta_direction in pairwise_result; expected {_DELTA_DIRECTION!r}."
        )

    contract_groups = tuple(comparison_contract.comparable_metric_groups)
    result_groups = tuple(pairwise_result.comparable_metric_groups)
    if contract_groups != result_groups:
        raise ContractError(
            "comparable_metric_groups mismatch between comparison contract and pairwise result."
        )

    contract_non_comparable = dict(comparison_contract.non_comparable_metric_groups)
    result_non_comparable = dict(pairwise_result.non_comparable_metric_groups)
    if contract_non_comparable != result_non_comparable:
        raise ContractError(
            "non_comparable_metric_groups mismatch between comparison contract and pairwise result."
        )

    for group in result_groups:
        if group not in pairwise_result.per_group_metric_deltas:
            raise ContractError(
                f"pairwise_result is missing delta payload for comparable group {group!r}."
            )
        if group not in pairwise_result.per_group_metric_orientation:
            raise ContractError(
                f"pairwise_result is missing orientation payload for comparable group {group!r}."
            )


def _status_to_report_status(comparison_status: str) -> str:
    if comparison_status == "ready":
        return "success"
    if comparison_status == "partial":
        return "partial"
    if comparison_status == "not_ready":
        return "failed"
    raise ContractError(f"Unsupported comparison_status {comparison_status!r}.")


def build_comparison_delta_table(
    *,
    comparison_run_id: str,
    comparison_contract: EvalComparisonInputContractResult,
    pairwise_result: PairwiseEvalComparisonResult,
    comparison_report_path: Path,
    comparison_summary_path: Path,
    comparison_config_used_path: Path,
) -> dict[str, Any]:
    """Build machine-readable pairwise delta table payload."""
    rows: list[dict[str, Any]] = []
    for group in pairwise_result.comparable_metric_groups:
        group_deltas = pairwise_result.per_group_metric_deltas.get(group)
        group_orientation = pairwise_result.per_group_metric_orientation.get(group)
        if not isinstance(group_deltas, Mapping):
            raise ContractError(
                f"pairwise_result is missing delta table mapping for comparable group {group!r}."
            )
        if not isinstance(group_orientation, Mapping):
            raise ContractError(
                f"pairwise_result is missing orientation mapping for comparable group {group!r}."
            )

        for delta_key, delta_value_raw in group_deltas.items():
            if not isinstance(delta_key, str) or not delta_key.endswith("_delta"):
                raise ContractError(
                    f"Unexpected delta key {delta_key!r} in comparable group {group!r}."
                )
            metric_name = delta_key[: -len("_delta")]
            if metric_name not in group_orientation:
                raise ContractError(
                    f"Missing orientation for metric {metric_name!r} in group {group!r}."
                )
            if isinstance(delta_value_raw, bool) or not isinstance(delta_value_raw, (int, float)):
                raise ContractError(
                    f"Delta value for {group}.{delta_key} must be numeric."
                )

            rows.append(
                {
                    "metric_group": group,
                    "metric_name": metric_name,
                    "delta_name": delta_key,
                    "delta_value": float(delta_value_raw),
                    "delta_direction": pairwise_result.delta_direction,
                    "orientation": str(group_orientation[metric_name]),
                    "comparable": True,
                }
            )

    return {
        "schema_name": _COMPARISON_DELTA_TABLE_SCHEMA,
        "status": _status_to_report_status(pairwise_result.comparison_status),
        "comparison_mode": "pairwise",
        "run_id": comparison_run_id,
        "left_run_id": comparison_contract.left_run_id,
        "right_run_id": comparison_contract.right_run_id,
        "delta_direction": pairwise_result.delta_direction,
        "comparison_status": pairwise_result.comparison_status,
        "comparable_metric_groups": list(pairwise_result.comparable_metric_groups),
        "non_comparable_metric_groups": dict(pairwise_result.non_comparable_metric_groups),
        "rows": rows,
        "provenance_links": {
            "left_eval_manifest_path": str(comparison_contract.left_eval_manifest_path),
            "right_eval_manifest_path": str(comparison_contract.right_eval_manifest_path),
            "left_summary_path": str(comparison_contract.left_summary_path),
            "right_summary_path": str(comparison_contract.right_summary_path),
            "left_config_used_path": str(comparison_contract.left_config_used_path),
            "right_config_used_path": str(comparison_contract.right_config_used_path),
            "comparison_report_path": str(comparison_report_path),
            "comparison_summary_path": str(comparison_summary_path),
            "comparison_config_used_path": str(comparison_config_used_path),
        },
        "policy_compatibility_summary": dict(pairwise_result.policy_compatibility_summary),
        "provenance_compatibility_summary": dict(pairwise_result.provenance_compatibility_summary),
        "stage_coverage_compatibility_summary": dict(
            pairwise_result.stage_coverage_compatibility_summary
        ),
        "stage_honesty": {
            "not_implemented": {
                "leaderboard": True,
                "ranking_engine": True,
                "dashboard_framework": True,
                "multi_run_benchmark_manager": True,
            }
        },
    }


def export_pairwise_comparison_artifacts(
    *,
    output_dir: str | Path,
    comparison_run_id: str,
    comparison_contract: EvalComparisonInputContractResult,
    pairwise_result: PairwiseEvalComparisonResult,
    extra_effective_config: Mapping[str, Any] | None = None,
) -> PairwiseComparisonExportArtifacts:
    """Write minimal comparison artifacts for pairwise eval results."""
    comparison_run_id = _require_non_empty_string(
        comparison_run_id,
        name="comparison_run_id",
    )
    _validate_inputs(
        comparison_contract=comparison_contract,
        pairwise_result=pairwise_result,
    )

    export_dir = _normalize_output_dir(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    comparison_report_path = export_dir / "comparison_report.json"
    comparison_summary_path = export_dir / "comparison_summary.json"
    comparison_delta_table_path = export_dir / "comparison_delta_table.json"
    config_used_path = export_dir / "config_used.yaml"
    created_at_utc = _utc_now_iso()

    effective_config: dict[str, Any] = {
        "module_name": "module_eval",
        "run_id": comparison_run_id,
        "comparison_mode": "pairwise",
        "stage_coverage": {
            "comparison_input_contract": True,
            "pairwise_comparison_computation": True,
            "comparison_artifact_export": True,
            "not_implemented": {
                "leaderboard": True,
                "ranking_engine": True,
                "dashboard_framework": True,
                "cross_run_benchmark_manager": True,
            },
        },
        "left_run": {
            "run_id": comparison_contract.left_run_id,
            "run_dir": str(comparison_contract.left_run_dir),
            "eval_mode": comparison_contract.left_eval_mode,
            "eval_manifest_path": str(comparison_contract.left_eval_manifest_path),
            "summary_path": str(comparison_contract.left_summary_path),
            "config_used_path": str(comparison_contract.left_config_used_path),
        },
        "right_run": {
            "run_id": comparison_contract.right_run_id,
            "run_dir": str(comparison_contract.right_run_dir),
            "eval_mode": comparison_contract.right_eval_mode,
            "eval_manifest_path": str(comparison_contract.right_eval_manifest_path),
            "summary_path": str(comparison_contract.right_summary_path),
            "config_used_path": str(comparison_contract.right_config_used_path),
        },
        "effective_policy_contract": {
            "delta_direction": pairwise_result.delta_direction,
            "delta_semantics": "delta = right - left",
            "partial_comparability_policy": "compute_deltas_only_for_comparable_groups",
            "orientation_policy": "explicit_metric_orientation_metadata",
            "no_silent_fallback": True,
        },
        "contract_summary": {
            "comparable_metric_groups": list(comparison_contract.comparable_metric_groups),
            "non_comparable_metric_groups": dict(comparison_contract.non_comparable_metric_groups),
            "ready_for_pairwise_compare": bool(comparison_contract.ready_for_pairwise_compare),
            "partially_ready": bool(comparison_contract.partially_ready),
            "global_blockers": list(comparison_contract.global_blockers),
        },
    }
    if extra_effective_config is not None:
        if not isinstance(extra_effective_config, Mapping):
            raise ContractError("extra_effective_config must be a mapping/object.")
        effective_config["extra_effective_config"] = dict(extra_effective_config)

    _write_config_used(config_used_path, effective_config)

    report_status = _status_to_report_status(pairwise_result.comparison_status)
    warnings: list[str] = []
    if pairwise_result.comparison_status == "partial":
        warnings.append("comparison is partial: some metric groups are non-comparable.")
    if pairwise_result.comparison_status == "not_ready":
        warnings.append("comparison is not ready: no comparable metric groups available.")

    pairwise_summary = build_pairwise_comparison_summary(pairwise_result)
    _validate_pairwise_summary_payload(pairwise_summary)

    report_payload: dict[str, Any] = {
        "schema_name": _COMPARISON_REPORT_SCHEMA,
        "schema_version": "v1",
        "module_name": "module_eval",
        "module_version": None,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "run_id": comparison_run_id,
        "stage_name": "export_pairwise_comparison_artifacts",
        "created_at_utc": created_at_utc,
        "status": report_status,
        "comparison_mode": "pairwise",
        "left_run": {
            "run_id": comparison_contract.left_run_id,
            "run_dir": str(comparison_contract.left_run_dir),
            "eval_mode": comparison_contract.left_eval_mode,
            "eval_manifest_path": str(comparison_contract.left_eval_manifest_path),
            "summary_path": str(comparison_contract.left_summary_path),
            "config_used_path": str(comparison_contract.left_config_used_path),
        },
        "right_run": {
            "run_id": comparison_contract.right_run_id,
            "run_dir": str(comparison_contract.right_run_dir),
            "eval_mode": comparison_contract.right_eval_mode,
            "eval_manifest_path": str(comparison_contract.right_eval_manifest_path),
            "summary_path": str(comparison_contract.right_summary_path),
            "config_used_path": str(comparison_contract.right_config_used_path),
        },
        "comparison_contract": {
            "comparable_metric_groups": list(comparison_contract.comparable_metric_groups),
            "non_comparable_metric_groups": dict(comparison_contract.non_comparable_metric_groups),
            "global_blockers": list(comparison_contract.global_blockers),
            "ready_for_pairwise_compare": bool(comparison_contract.ready_for_pairwise_compare),
            "partially_ready": bool(comparison_contract.partially_ready),
        },
        "pairwise_result": {
            "delta_direction": pairwise_summary["delta_direction"],
            "comparison_status": pairwise_summary["comparison_status"],
            "metric_deltas": pairwise_summary["metric_deltas"],
            "metric_orientation": pairwise_summary["metric_orientation"],
            "skipped_metric_reasons": pairwise_summary["skipped_metric_reasons"],
        },
        "policy_compatibility_summary": dict(pairwise_result.policy_compatibility_summary),
        "provenance_compatibility_summary": dict(pairwise_result.provenance_compatibility_summary),
        "stage_coverage_compatibility_summary": dict(
            pairwise_result.stage_coverage_compatibility_summary
        ),
        "artifacts": {
            "comparison_report_path": str(comparison_report_path),
            "comparison_summary_path": str(comparison_summary_path),
            "comparison_delta_table_path": str(comparison_delta_table_path),
            "config_used_path": str(config_used_path),
        },
        "stage_coverage": {
            "comparison_input_contract": True,
            "pairwise_comparison_computation": True,
            "comparison_artifact_export": True,
            "not_implemented": {
                "leaderboard": True,
                "ranking_engine": True,
                "dashboard_framework": True,
                "cross_run_benchmark_manager": True,
            },
        },
        "warnings": warnings,
    }
    _validate_report_payload(report_payload)
    write_summary(comparison_report_path, report_payload)

    summary_payload: dict[str, Any] = {
        "schema_name": _COMPARISON_SUMMARY_SCHEMA,
        "status": report_status,
        "comparison_mode": "pairwise",
        "run_id": comparison_run_id,
        "left_run_id": comparison_contract.left_run_id,
        "right_run_id": comparison_contract.right_run_id,
        "delta_direction": pairwise_result.delta_direction,
        "comparison_status": pairwise_result.comparison_status,
        "comparable_metric_groups": list(pairwise_result.comparable_metric_groups),
        "non_comparable_metric_groups": dict(pairwise_result.non_comparable_metric_groups),
        "metric_delta_counts": {
            group: len(metrics)
            for group, metrics in pairwise_result.per_group_metric_deltas.items()
        },
        "policy_compatibility_summary": dict(pairwise_result.policy_compatibility_summary),
        "provenance_compatibility_summary": dict(pairwise_result.provenance_compatibility_summary),
        "warnings": warnings,
        "key_notes": [
            "Comparison report is pairwise and contract-first.",
            "Leaderboard/ranking/dashboard are intentionally out of scope.",
        ],
        "artifacts": {
            "comparison_report_path": str(comparison_report_path),
            "comparison_delta_table_path": str(comparison_delta_table_path),
            "config_used_path": str(config_used_path),
        },
    }
    _validate_summary_payload(summary_payload)
    write_summary(comparison_summary_path, summary_payload)

    delta_table_payload = build_comparison_delta_table(
        comparison_run_id=comparison_run_id,
        comparison_contract=comparison_contract,
        pairwise_result=pairwise_result,
        comparison_report_path=comparison_report_path,
        comparison_summary_path=comparison_summary_path,
        comparison_config_used_path=config_used_path,
    )
    _validate_delta_table_payload(delta_table_payload)
    write_summary(comparison_delta_table_path, delta_table_payload)

    return PairwiseComparisonExportArtifacts(
        output_dir=export_dir,
        comparison_report_path=comparison_report_path,
        comparison_summary_path=comparison_summary_path,
        comparison_delta_table_path=comparison_delta_table_path,
        config_used_path=config_used_path,
    )


__all__ = [
    "PairwiseComparisonExportArtifacts",
    "build_comparison_delta_table",
    "export_pairwise_comparison_artifacts",
]
