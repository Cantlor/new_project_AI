"""Minimal run-level pairwise comparison orchestration for module_eval.

This layer intentionally composes already-implemented comparison stages:
  Stage A: comparison input-contract resolution
  Stage B: pairwise comparison delta computation
  Stage C: comparison artifact/report export

Out of scope:
- leaderboard/ranking engine,
- dashboard/reporting portal,
- multi-run benchmark manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError
from ai_fields.common.paths import get_run_dir
from ai_fields.common.progress import progress_bar
from ai_fields.module_eval.comparison_contract import (
    EvalComparisonInputContractResult,
    resolve_eval_comparison_contract,
)
from ai_fields.module_eval.comparison_export import (
    PairwiseComparisonExportArtifacts,
    export_pairwise_comparison_artifacts,
)
from ai_fields.module_eval.pairwise_comparison import (
    PairwiseEvalComparisonResult,
    compute_pairwise_eval_comparison,
)


@dataclass(frozen=True)
class PairwiseComparisonRunInputs:
    """Input artifact references for one pairwise comparison run.

    Either run dirs can be provided (recommended), or explicit manifest/summary/config paths.
    Path requirements are validated by `resolve_eval_comparison_contract`.
    """

    left_run_dir: str | Path | None = None
    right_run_dir: str | Path | None = None

    left_eval_manifest_path: str | Path | None = None
    right_eval_manifest_path: str | Path | None = None
    left_summary_path: str | Path | None = None
    right_summary_path: str | Path | None = None
    left_config_used_path: str | Path | None = None
    right_config_used_path: str | Path | None = None


@dataclass(frozen=True)
class PairwiseComparisonRunResult:
    """Run-level result contract for a single pairwise comparison."""

    run_id: str
    run_dir: Path
    comparison_status: str
    ready_for_next_stage: bool

    comparison_report_path: Path
    comparison_summary_path: Path
    comparison_delta_table_path: Path
    config_used_path: Path

    left_run_id: str
    right_run_id: str
    comparable_metric_groups: tuple[str, ...]
    non_comparable_metric_groups: dict[str, str]

    comparison_contract: EvalComparisonInputContractResult
    pairwise_result: PairwiseEvalComparisonResult
    export_artifacts: PairwiseComparisonExportArtifacts


def _require_non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{name} must be a non-empty string.")
    return value


def _resolve_run_dir(*, run_id: str, output_dir: str | PathLike[str] | None) -> Path:
    if output_dir is None:
        return get_run_dir("module_eval", run_id)
    if not isinstance(output_dir, (str, PathLike)):
        raise ContractError(
            f"output_dir must be path-like when provided, got {type(output_dir).__name__}."
        )
    resolved = Path(output_dir)
    if str(resolved).strip() == "":
        raise ContractError("output_dir must be non-empty when provided.")
    return resolved


def run_pairwise_comparison(
    *,
    run_id: str,
    inputs: PairwiseComparisonRunInputs,
    output_dir: str | PathLike[str] | None = None,
    extra_effective_config: dict[str, Any] | None = None,
    progress_enabled: bool | None = None,
) -> PairwiseComparisonRunResult:
    """Run one fail-fast single-pair comparison by composing Stage A->B->C."""
    run_id = _require_non_empty_string(run_id, name="run_id")
    if not isinstance(inputs, PairwiseComparisonRunInputs):
        raise ContractError("inputs must be PairwiseComparisonRunInputs.")

    run_dir = _resolve_run_dir(run_id=run_id, output_dir=output_dir)

    with progress_bar(
        total=3,
        desc="compare: stages",
        unit="stage",
        progress_enabled=progress_enabled,
        leave=False,
    ) as bar:
        # Stage A: comparison input-contract
        bar.set_postfix(stage="A: input contract")
        comparison_contract = resolve_eval_comparison_contract(
            left_run_dir=inputs.left_run_dir,
            right_run_dir=inputs.right_run_dir,
            left_eval_manifest_path=inputs.left_eval_manifest_path,
            right_eval_manifest_path=inputs.right_eval_manifest_path,
            left_summary_path=inputs.left_summary_path,
            right_summary_path=inputs.right_summary_path,
            left_config_used_path=inputs.left_config_used_path,
            right_config_used_path=inputs.right_config_used_path,
        )
        bar.update(1)

        # Stage B: pairwise deltas
        bar.set_postfix(stage="B: pairwise deltas")
        pairwise_result = compute_pairwise_eval_comparison(
            comparison_contract=comparison_contract,
        )
        bar.update(1)

        # Stage C: comparison artifact export
        bar.set_postfix(stage="C: export artifacts")
        artifacts = export_pairwise_comparison_artifacts(
            output_dir=run_dir,
            comparison_run_id=run_id,
            comparison_contract=comparison_contract,
            pairwise_result=pairwise_result,
            extra_effective_config=extra_effective_config,
        )
        bar.update(1)

    ready_for_next_stage = pairwise_result.comparison_status in {"ready", "partial"}
    return PairwiseComparisonRunResult(
        run_id=run_id,
        run_dir=artifacts.output_dir,
        comparison_status=pairwise_result.comparison_status,
        ready_for_next_stage=ready_for_next_stage,
        comparison_report_path=artifacts.comparison_report_path,
        comparison_summary_path=artifacts.comparison_summary_path,
        comparison_delta_table_path=artifacts.comparison_delta_table_path,
        config_used_path=artifacts.config_used_path,
        left_run_id=comparison_contract.left_run_id,
        right_run_id=comparison_contract.right_run_id,
        comparable_metric_groups=tuple(comparison_contract.comparable_metric_groups),
        non_comparable_metric_groups=dict(comparison_contract.non_comparable_metric_groups),
        comparison_contract=comparison_contract,
        pairwise_result=pairwise_result,
        export_artifacts=artifacts,
    )


__all__ = [
    "PairwiseComparisonRunInputs",
    "PairwiseComparisonRunResult",
    "run_pairwise_comparison",
]
