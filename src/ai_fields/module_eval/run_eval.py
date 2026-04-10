"""Minimal run-level eval orchestration path for module_eval.

This layer intentionally composes already-implemented stages:
  Stage A: input contract resolution
  Stage B: global/pixel metrics
  Stage B.5: distance auxiliary metrics
  Stage C: boundary metrics
  Stage D: object/structure metrics
  Stage D.5: bucketed object metrics
  Stage E: eval artifact export

Out of scope:
- comparison engine,
- cross-run benchmarking,
- dashboard/reporting framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError
from ai_fields.common.paths import get_run_dir
from ai_fields.common.progress import progress_bar
from ai_fields.module_eval.boundary_metrics import (
    BoundaryEvaluationPolicy,
    build_boundary_metrics_summary,
    compute_boundary_metrics,
)
from ai_fields.module_eval.bucketed_eval import (
    BucketSizePolicy,
    build_bucketed_eval_summary,
    compute_bucketed_object_metrics,
)
from ai_fields.module_eval.visual_diagnostics import VisualDiagnosticsResult
from ai_fields.module_eval.distance_metrics import (
    DistanceEvaluationPolicy,
    build_distance_metrics_summary,
    compute_distance_metrics,
)
from ai_fields.module_eval.export import export_eval_artifacts
from ai_fields.module_eval.input_contract import (
    EvaluationInputContractResult,
    resolve_evaluation_input_contract,
)
from ai_fields.module_eval.object_metrics import (
    ObjectMatchingPolicy,
    build_object_structure_metrics_summary,
    compute_object_structure_metrics,
)
from ai_fields.module_eval.pixel_metrics import (
    PixelBinarizationPolicy,
    build_pixel_metrics_summary,
    compute_global_pixel_metrics,
)


@dataclass(frozen=True)
class EvalRunInputs:
    """Input artifact paths for one eval run (single evaluation case/scene)."""

    gt_extent_path: str | Path
    gt_boundary_path: str | Path
    gt_valid_path: str | Path
    pred_extent_prob_path: str | Path
    pred_boundary_prob_path: str | Path
    pred_distance_pred_path: str | Path
    pred_valid_path: str | Path

    gt_distance_path: str | Path | None = None
    gt_parcels_path: str | Path | None = None
    post_parcel_instance_path: str | Path | None = None
    post_parcels_gpkg_path: str | Path | None = None
    predict_manifest_path: str | Path | None = None
    postprocess_manifest_path: str | Path | None = None


@dataclass(frozen=True)
class EvalRunPolicies:
    """Effective policy contract for Stage B/C/D/B.5/D.5 within one eval run."""

    eval_mode: str
    pixel_policy: PixelBinarizationPolicy
    boundary_policy: BoundaryEvaluationPolicy
    object_policy: ObjectMatchingPolicy
    distance_policy: DistanceEvaluationPolicy | None = None
    bucket_policy: BucketSizePolicy | None = None


@dataclass(frozen=True)
class EvalRunResult:
    """Run-level result contract for minimal eval orchestration."""

    run_id: str
    eval_mode: str
    run_dir: Path

    eval_manifest_path: Path
    summary_path: Path
    config_used_path: Path
    error_taxonomy_path: Path

    input_contract: EvaluationInputContractResult
    pixel_metrics_summary: dict[str, Any]
    boundary_metrics_summary: dict[str, Any]
    object_metrics_summary: dict[str, Any]
    distance_metrics_summary: dict[str, Any]
    bucketed_metrics_summary: dict[str, Any] | None
    visual_diagnostics: VisualDiagnosticsResult | None

    ready_for_next_stage: bool


_SUPPORTED_EVAL_MODES = frozenset(
    {
        "raster",
        "vector",
        "raster_only",
        "vector_only",
        "end_to_end",
        "end-to-end",
        "end_to_end_single_scene",
    }
)


def _require_non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{name} must be a non-empty string.")
    return value


def _validate_eval_mode(eval_mode: str) -> str:
    if eval_mode in _SUPPORTED_EVAL_MODES:
        return eval_mode
    if eval_mode == "comparison":
        raise ContractError(
            "run_eval does not support eval_mode='comparison'; "
            "use run_pairwise_comparison for comparison mode."
        )
    allowed = ", ".join(sorted(_SUPPORTED_EVAL_MODES))
    raise ContractError(
        f"Unsupported eval_mode {eval_mode!r}. Allowed values: {allowed}."
    )


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


def run_eval(
    *,
    run_id: str,
    inputs: EvalRunInputs,
    policies: EvalRunPolicies,
    output_dir: str | PathLike[str] | None = None,
    extra_source_manifest_paths: list[str | Path] | None = None,
    extra_effective_config: dict[str, Any] | None = None,
    progress_enabled: bool | None = None,
) -> EvalRunResult:
    """Run one fail-fast eval case by composing Stages A->B->C->D->E."""
    run_id = _require_non_empty_string(run_id, name="run_id")
    if not isinstance(inputs, EvalRunInputs):
        raise ContractError("inputs must be EvalRunInputs.")
    if not isinstance(policies, EvalRunPolicies):
        raise ContractError("policies must be EvalRunPolicies.")

    eval_mode = _validate_eval_mode(
        _require_non_empty_string(policies.eval_mode, name="policies.eval_mode")
    )
    run_dir = _resolve_run_dir(run_id=run_id, output_dir=output_dir)

    with progress_bar(
        total=5,
        desc="eval: stages",
        unit="stage",
        progress_enabled=progress_enabled,
        leave=False,
    ) as bar:
        # Stage A
        bar.set_postfix(stage="A: input contract")
        input_contract = resolve_evaluation_input_contract(
            gt_extent_path=inputs.gt_extent_path,
            gt_boundary_path=inputs.gt_boundary_path,
            gt_valid_path=inputs.gt_valid_path,
            pred_extent_prob_path=inputs.pred_extent_prob_path,
            pred_boundary_prob_path=inputs.pred_boundary_prob_path,
            pred_distance_pred_path=inputs.pred_distance_pred_path,
            pred_valid_path=inputs.pred_valid_path,
            gt_distance_path=inputs.gt_distance_path,
            gt_parcels_path=inputs.gt_parcels_path,
            post_parcel_instance_path=inputs.post_parcel_instance_path,
            post_parcels_gpkg_path=inputs.post_parcels_gpkg_path,
            predict_manifest_path=inputs.predict_manifest_path,
            postprocess_manifest_path=inputs.postprocess_manifest_path,
        )
        bar.update(1)

        # Stage B
        bar.set_postfix(stage="B: pixel+distance metrics")
        pixel_result = compute_global_pixel_metrics(
            input_contract=input_contract,
            policy=policies.pixel_policy,
        )
        pixel_summary = build_pixel_metrics_summary(pixel_result)

        # Stage B.5: distance auxiliary metrics (conditional on gt_distance availability)
        _distance_policy = policies.distance_policy or DistanceEvaluationPolicy(
            threshold_provenance=policies.pixel_policy.threshold_provenance,
            absent_gt_policy="skip",
        )
        distance_result = compute_distance_metrics(
            input_contract=input_contract,
            policy=_distance_policy,
        )
        distance_summary = build_distance_metrics_summary(distance_result)
        bar.update(1)

        # Stage C
        bar.set_postfix(stage="C: boundary metrics")
        boundary_result = compute_boundary_metrics(
            input_contract=input_contract,
            policy=policies.boundary_policy,
        )
        boundary_summary = build_boundary_metrics_summary(boundary_result)
        bar.update(1)

        # Stage D
        bar.set_postfix(stage="D: object metrics")
        object_result = compute_object_structure_metrics(
            input_contract=input_contract,
            policy=policies.object_policy,
        )
        object_summary = build_object_structure_metrics_summary(object_result)

        # Stage D.5: bucketed evaluation (optional; skipped if object track not ready or no bucket_policy)
        bucketed_result = None
        bucketed_summary = None
        if (
            policies.bucket_policy is not None
            and input_contract.track_readiness.object_structure_ready
        ):
            bucketed_result = compute_bucketed_object_metrics(
                input_contract=input_contract,
                object_policy=policies.object_policy,
                bucket_policy=policies.bucket_policy,
            )
            bucketed_summary = build_bucketed_eval_summary(bucketed_result)
        bar.update(1)

        # Stage E
        bar.set_postfix(stage="E: export artifacts")
        artifacts = export_eval_artifacts(
            output_dir=run_dir,
            run_id=run_id,
            eval_mode=eval_mode,
            input_contract=input_contract,
            pixel_result=pixel_result,
            boundary_result=boundary_result,
            object_result=object_result,
            distance_result=distance_result,
            bucketed_result=bucketed_result,
            pixel_policy=policies.pixel_policy,
            boundary_policy=policies.boundary_policy,
            object_policy=policies.object_policy,
            extra_source_manifest_paths=extra_source_manifest_paths,
            extra_effective_config=extra_effective_config,
        )
        bar.update(1)

    return EvalRunResult(
        run_id=run_id,
        eval_mode=eval_mode,
        run_dir=artifacts.run_dir,
        eval_manifest_path=artifacts.eval_manifest_path,
        summary_path=artifacts.summary_path,
        config_used_path=artifacts.config_used_path,
        error_taxonomy_path=artifacts.error_taxonomy_path,
        input_contract=input_contract,
        pixel_metrics_summary=pixel_summary,
        boundary_metrics_summary=boundary_summary,
        object_metrics_summary=object_summary,
        distance_metrics_summary=distance_summary,
        bucketed_metrics_summary=bucketed_summary,
        visual_diagnostics=artifacts.visual_diagnostics,
        ready_for_next_stage=True,
    )


__all__ = [
    "EvalRunInputs",
    "EvalRunPolicies",
    "EvalRunResult",
    "run_eval",
    "BucketSizePolicy",
    "DistanceEvaluationPolicy",
]
