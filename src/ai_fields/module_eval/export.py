"""Stage E artifact export layer for module_eval.

This module writes the eval artifact set:
  - eval_manifest.json
  - summary.json
  - config_used.yaml
  - error_taxonomy.json
  - metrics_aggregate.json
  - scenes_included.json
  - scenes_excluded.json
  - source_runs.json
  - metrics_by_bucket.json (when bucketed eval is enabled)
  - diagnostics/*.png + diagnostics_index.json (when visuals are enabled and available)

It is intentionally narrow and contract-first:
  - consumes resolved Stage A input contract and Stage B/C/D metric results;
  - preserves provenance and policy transparency;
  - avoids comparison engine / cross-run benchmarking framework.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import read_manifest, write_manifest, write_summary
from ai_fields.module_eval.boundary_metrics import (
    BoundaryEvaluationPolicy,
    BoundaryMetricsResult,
    build_boundary_metrics_summary,
)
from ai_fields.module_eval.bucketed_eval import (
    BucketedEvalResult,
    build_bucketed_eval_summary,
    write_bucketed_eval_artifact,
)
from ai_fields.module_eval.visual_diagnostics import (
    VisualDiagnosticsResult,
    write_visual_diagnostics,
)
from ai_fields.module_eval.distance_metrics import (
    DistanceMetricsResult,
    build_distance_metrics_summary,
)
from ai_fields.module_eval.input_contract import EvaluationInputContractResult
from ai_fields.module_eval.object_metrics import (
    ObjectMatchingPolicy,
    ObjectStructureMetricsResult,
    build_object_structure_metrics_summary,
)
from ai_fields.module_eval.pixel_metrics import (
    GlobalPixelMetricsResult,
    PixelBinarizationPolicy,
    build_pixel_metrics_summary,
)

_EVAL_MANIFEST_SCHEMA = "eval.eval_manifest"
_EVAL_SUMMARY_SCHEMA = "eval.summary"
_EPS = 1e-9


@dataclass(frozen=True)
class EvalExportArtifacts:
    """Paths of Stage E artifacts for module_eval."""

    run_dir: Path
    eval_manifest_path: Path
    summary_path: Path
    config_used_path: Path
    error_taxonomy_path: Path
    metrics_by_bucket_path: Path | None
    visual_diagnostics: VisualDiagnosticsResult | None
    metrics_aggregate_path: Path | None = None
    scenes_included_path: Path | None = None
    scenes_excluded_path: Path | None = None
    source_runs_path: Path | None = None


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


def _policy_as_dict(policy: Any, *, name: str) -> dict[str, Any]:
    if is_dataclass(policy):
        return asdict(policy)
    if isinstance(policy, Mapping):
        return dict(policy)
    raise ContractError(f"{name} must be a dataclass instance or mapping/object.")


def _write_config_used(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ContractError("PyYAML is required to write config_used.yaml.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            yaml.safe_dump(dict(payload), sort_keys=True, allow_unicode=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ContractError(f"Failed to write config_used.yaml at {path}: {exc}") from exc


def _write_error_taxonomy(
    path: Path,
    *,
    run_id: str,
    eval_mode: str,
    object_result: "ObjectStructureMetricsResult",
    boundary_result: "BoundaryMetricsResult",
) -> None:
    """Write error_taxonomy.json (module_eval §11, §14.1).

    Taxonomy classes (§11.1):
        split_error          — GT objects split into multiple predicted parcels
        merge_error          — GT objects merged with neighboring GT objects
        missed_parcel        — GT objects with no matching predicted parcel
        spurious_parcel      — predicted parcels with no matching GT object
        invalid_area_artifact — GT objects excluded due to zero valid pixels
        boundary_shift       — mean boundary displacement error (BDE)
    """
    payload: dict[str, Any] = {
        "schema_name": "eval.error_taxonomy",
        "schema_version": "v1",
        "run_id": run_id,
        "eval_mode": eval_mode,
        "taxonomy": {
            "split_error": {
                "count": int(object_result.split_gt_count),
                "description": "GT objects that were split into two or more predicted parcels",
            },
            "merge_error": {
                "count": int(object_result.merged_gt_count),
                "description": "GT objects that were merged with at least one neighboring GT object",
            },
            "missed_parcel": {
                "count": int(object_result.unmatched_gt_count),
                "description": "GT objects with no matching predicted parcel",
            },
            "spurious_parcel": {
                "count": int(object_result.spurious_pred_count),
                "description": "Predicted parcels with no matching GT object",
            },
            "invalid_area_artifact": {
                "count": int(object_result.gt_excluded_zero_valid_count),
                "description": "GT objects excluded from evaluation due to zero valid pixels",
            },
            "boundary_shift": {
                "value": boundary_result.boundary_bde,
                "units": boundary_result.boundary_bde_units,
                "description": "Mean Boundary Displacement Error (BDE); None if BDE was disabled",
            },
        },
        "object_counts": {
            "gt_total": int(object_result.gt_object_count),
            "pred_total": int(object_result.pred_object_count),
            "matched_gt": int(object_result.matched_gt_count),
            "matched_pred": int(object_result.matched_pred_count),
        },
        "threshold_provenance": object_result.threshold_provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"Failed to write error_taxonomy.json at {path}: {exc}") from exc


def _normalize_extra_manifest_paths(paths: Any) -> list[Path]:
    if paths is None:
        return []
    if (
        not isinstance(paths, Sequence)
        or isinstance(paths, (str, bytes))
    ):
        raise ContractError("extra_source_manifest_paths must be a sequence of paths.")

    out: list[Path] = []
    for idx, raw in enumerate(paths):
        if not isinstance(raw, (str, PathLike)):
            raise ContractError(
                "extra_source_manifest_paths items must be path-like, "
                f"got {type(raw).__name__} at index {idx}."
            )
        p = Path(raw)
        if str(p).strip() == "":
            raise ContractError(
                f"extra_source_manifest_paths[{idx}] must be non-empty path-like."
            )
        if not p.exists() or not p.is_file():
            raise ContractError(
                f"extra_source_manifest_paths[{idx}] does not exist as file: {p}"
            )
        out.append(p)
    return out


def _validate_stage_inputs(
    *,
    input_contract: EvaluationInputContractResult,
    pixel_result: GlobalPixelMetricsResult,
    boundary_result: BoundaryMetricsResult,
    object_result: ObjectStructureMetricsResult,
    pixel_policy: PixelBinarizationPolicy,
    boundary_policy: BoundaryEvaluationPolicy,
    object_policy: ObjectMatchingPolicy,
) -> None:
    if not isinstance(input_contract, EvaluationInputContractResult):
        raise ContractError("input_contract must be EvaluationInputContractResult.")
    if not isinstance(pixel_result, GlobalPixelMetricsResult):
        raise ContractError("pixel_result must be GlobalPixelMetricsResult.")
    if not isinstance(boundary_result, BoundaryMetricsResult):
        raise ContractError("boundary_result must be BoundaryMetricsResult.")
    if not isinstance(object_result, ObjectStructureMetricsResult):
        raise ContractError("object_result must be ObjectStructureMetricsResult.")
    if not isinstance(pixel_policy, PixelBinarizationPolicy):
        raise ContractError("pixel_policy must be PixelBinarizationPolicy.")
    if not isinstance(boundary_policy, BoundaryEvaluationPolicy):
        raise ContractError("boundary_policy must be BoundaryEvaluationPolicy.")
    if not isinstance(object_policy, ObjectMatchingPolicy):
        raise ContractError("object_policy must be ObjectMatchingPolicy.")

    if input_contract.track_readiness.pixel_ready is not True:
        raise ContractError("Stage E export requires pixel track to be ready.")
    if input_contract.track_readiness.boundary_ready is not True:
        raise ContractError("Stage E export requires boundary track to be ready.")
    if input_contract.track_readiness.object_structure_ready is not True:
        raise ContractError("Stage E export requires object/structure track to be ready.")

    if abs(float(pixel_result.threshold) - float(pixel_policy.extent_prob_threshold)) > _EPS:
        raise ContractError(
            "pixel_result threshold/policy mismatch: "
            f"{pixel_result.threshold} vs {pixel_policy.extent_prob_threshold}."
        )
    if pixel_result.threshold_provenance != pixel_policy.threshold_provenance:
        raise ContractError(
            "pixel_result threshold_provenance/policy mismatch: "
            f"{pixel_result.threshold_provenance!r} vs {pixel_policy.threshold_provenance!r}."
        )
    if pixel_result.prediction_rule != pixel_policy.prediction_rule:
        raise ContractError(
            "pixel_result prediction_rule/policy mismatch: "
            f"{pixel_result.prediction_rule!r} vs {pixel_policy.prediction_rule!r}."
        )

    if boundary_result.prediction_interpretation != boundary_policy.prediction_interpretation:
        raise ContractError(
            "boundary_result prediction_interpretation/policy mismatch: "
            f"{boundary_result.prediction_interpretation!r} vs "
            f"{boundary_policy.prediction_interpretation!r}."
        )
    if boundary_result.gt_interpretation != boundary_policy.gt_interpretation:
        raise ContractError(
            "boundary_result gt_interpretation/policy mismatch: "
            f"{boundary_result.gt_interpretation!r} vs {boundary_policy.gt_interpretation!r}."
        )
    if boundary_result.threshold_provenance != boundary_policy.threshold_provenance:
        raise ContractError(
            "boundary_result threshold_provenance/policy mismatch: "
            f"{boundary_result.threshold_provenance!r} vs {boundary_policy.threshold_provenance!r}."
        )
    if boundary_result.bde_enabled is not bool(boundary_policy.bde_enabled):
        raise ContractError(
            "boundary_result bde_enabled/policy mismatch: "
            f"{boundary_result.bde_enabled!r} vs {boundary_policy.bde_enabled!r}."
        )

    if abs(float(object_result.min_iou_threshold) - float(object_policy.min_iou_threshold)) > _EPS:
        raise ContractError(
            "object_result min_iou_threshold/policy mismatch: "
            f"{object_result.min_iou_threshold} vs {object_policy.min_iou_threshold}."
        )
    if (
        abs(
            float(object_result.min_overlap_gt_threshold)
            - float(object_policy.min_overlap_gt_threshold)
        )
        > _EPS
    ):
        raise ContractError(
            "object_result min_overlap_gt_threshold/policy mismatch: "
            f"{object_result.min_overlap_gt_threshold} vs {object_policy.min_overlap_gt_threshold}."
        )
    if (
        abs(
            float(object_result.min_overlap_pred_threshold)
            - float(object_policy.min_overlap_pred_threshold)
        )
        > _EPS
    ):
        raise ContractError(
            "object_result min_overlap_pred_threshold/policy mismatch: "
            f"{object_result.min_overlap_pred_threshold} vs {object_policy.min_overlap_pred_threshold}."
        )
    if object_result.match_rule != object_policy.match_rule:
        raise ContractError(
            "object_result match_rule/policy mismatch: "
            f"{object_result.match_rule!r} vs {object_policy.match_rule!r}."
        )
    if object_result.threshold_provenance != object_policy.threshold_provenance:
        raise ContractError(
            "object_result threshold_provenance/policy mismatch: "
            f"{object_result.threshold_provenance!r} vs {object_policy.threshold_provenance!r}."
        )


def _threshold_provenance_mode(
    *,
    pixel_policy: PixelBinarizationPolicy,
    boundary_policy: BoundaryEvaluationPolicy,
    object_policy: ObjectMatchingPolicy,
) -> str:
    values = {
        str(pixel_policy.threshold_provenance),
        str(boundary_policy.threshold_provenance),
        str(object_policy.threshold_provenance),
    }
    if len(values) == 1:
        return next(iter(values))
    return "mixed_explicit_threshold_provenance"


def _readiness_as_dict(input_contract: EvaluationInputContractResult) -> dict[str, Any]:
    r = input_contract.track_readiness
    return {
        "pixel_ready": bool(r.pixel_ready),
        "boundary_ready": bool(r.boundary_ready),
        "object_structure_ready": bool(r.object_structure_ready),
        "pixel_reason": r.pixel_reason,
        "boundary_reason": r.boundary_reason,
        "object_structure_reason": r.object_structure_reason,
    }


def _extract_aoi_policy_from_postprocess_manifest(
    postprocess_manifest_path: "Path | None",
) -> "dict[str, Any] | None":
    """Read aoi_policy from upstream postprocess manifest for provenance.

    Returns a structured dict or None. Best-effort: if the manifest exists
    but has no aoi_policy, returns None. Does NOT raise on read failures —
    eval should not block on missing optional upstream provenance.
    """
    if postprocess_manifest_path is None:
        return None
    try:
        m = read_manifest(postprocess_manifest_path)
        resolved = m.get("resolved_policy")
        if not isinstance(resolved, dict):
            return None
        aoi = resolved.get("aoi_policy")
        if aoi is None:
            return None
        if isinstance(aoi, dict):
            result: dict[str, Any] = {
                "source": "upstream_postprocess_manifest",
                "postprocess_manifest_path": str(postprocess_manifest_path),
            }
            result.update(aoi)
            mode = aoi.get("mode", "unknown")
            applied = aoi.get("suppression_applied", False)
            result["aoi_policy_summary"] = (
                f"postprocess aoi_policy: mode={mode}, suppression_applied={applied}"
            )
            return result
    except Exception:
        pass  # best-effort; don't block eval over optional provenance read
    return None


def export_eval_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    eval_mode: str,
    input_contract: EvaluationInputContractResult,
    pixel_result: GlobalPixelMetricsResult,
    boundary_result: BoundaryMetricsResult,
    object_result: ObjectStructureMetricsResult,
    distance_result: DistanceMetricsResult | None = None,
    bucketed_result: BucketedEvalResult | None = None,
    generate_visual_diagnostics: bool = True,
    pixel_policy: PixelBinarizationPolicy,
    boundary_policy: BoundaryEvaluationPolicy,
    object_policy: ObjectMatchingPolicy,
    extra_source_manifest_paths: Sequence[str | Path] | None = None,
    extra_effective_config: Mapping[str, Any] | None = None,
) -> EvalExportArtifacts:
    """Write Stage E eval artifacts for Stage A-D results."""
    run_id = _require_non_empty_string(run_id, name="run_id")
    eval_mode = _require_non_empty_string(eval_mode, name="eval_mode")
    run_dir = _normalize_output_dir(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _validate_stage_inputs(
        input_contract=input_contract,
        pixel_result=pixel_result,
        boundary_result=boundary_result,
        object_result=object_result,
        pixel_policy=pixel_policy,
        boundary_policy=boundary_policy,
        object_policy=object_policy,
    )

    extra_manifest_paths = _normalize_extra_manifest_paths(extra_source_manifest_paths)

    created_at_utc = _utc_now_iso()
    config_used_path = run_dir / "config_used.yaml"
    manifest_path = run_dir / "eval_manifest.json"
    summary_path = run_dir / "summary.json"
    error_taxonomy_path = run_dir / "error_taxonomy.json"
    metrics_by_bucket_path = run_dir / "metrics_by_bucket.json" if bucketed_result is not None else None
    metrics_aggregate_path = run_dir / "metrics_aggregate.json"
    scenes_included_path = run_dir / "scenes_included.json"
    scenes_excluded_path = run_dir / "scenes_excluded.json"
    source_runs_path = run_dir / "source_runs.json"

    source_manifest_paths: list[str] = []
    if input_contract.predict_manifest_path is not None:
        source_manifest_paths.append(str(input_contract.predict_manifest_path))
    if input_contract.postprocess_manifest_path is not None:
        source_manifest_paths.append(str(input_contract.postprocess_manifest_path))
    source_manifest_paths.extend(str(p) for p in extra_manifest_paths)

    source_run_ids = list(input_contract.source_run_ids)

    threshold_provenance = _threshold_provenance_mode(
        pixel_policy=pixel_policy,
        boundary_policy=boundary_policy,
        object_policy=object_policy,
    )

    pixel_summary = build_pixel_metrics_summary(pixel_result)
    boundary_summary = build_boundary_metrics_summary(boundary_result)
    object_summary = build_object_structure_metrics_summary(object_result)
    distance_summary = (
        build_distance_metrics_summary(distance_result)
        if distance_result is not None
        else None
    )

    effective_config: dict[str, Any] = {
        "module_name": "module_eval",
        "run_id": run_id,
        "eval_mode": eval_mode,
        "stage_coverage": {
            "stage_a_input_contract": True,
            "stage_b_pixel_metrics": True,
            "stage_c_boundary_metrics": True,
            "stage_d_object_metrics": True,
            "stage_e_artifact_export": True,
            "error_taxonomy": True,
            "stage_b5_distance_metrics": True,
            "stage_not_implemented": {
                "comparison_engine": True,
                "cross_run_benchmarking": True,
                "dashboard_reporting_framework": True,
                **({"bucketed_evaluation": True} if bucketed_result is None else {}),
            },
        },
        "inputs": {
            "gt_extent_path": str(input_contract.gt_extent.path),
            "gt_boundary_path": str(input_contract.gt_boundary.path),
            "gt_valid_path": str(input_contract.gt_valid.path),
            "gt_distance_path": (
                str(input_contract.gt_distance.path)
                if input_contract.gt_distance is not None
                else None
            ),
            "gt_parcels_path": (
                str(input_contract.gt_parcels.path)
                if input_contract.gt_parcels is not None
                else None
            ),
            "pred_extent_prob_path": str(input_contract.pred_extent_prob.path),
            "pred_boundary_prob_path": str(input_contract.pred_boundary_prob.path),
            "pred_distance_pred_path": str(input_contract.pred_distance_pred.path),
            "pred_valid_path": str(input_contract.pred_valid.path),
            "post_parcel_instance_path": (
                str(input_contract.post_parcel_instance.path)
                if input_contract.post_parcel_instance is not None
                else None
            ),
            "post_parcels_gpkg_path": (
                str(input_contract.post_parcels_vector.path)
                if input_contract.post_parcels_vector is not None
                else None
            ),
        },
        "provenance": {
            "source_run_ids": source_run_ids,
            "source_manifest_paths": source_manifest_paths,
        },
        "effective_policy_contract": {
            "pixel_policy": _policy_as_dict(pixel_policy, name="pixel_policy"),
            "boundary_policy": _policy_as_dict(boundary_policy, name="boundary_policy"),
            "object_matching_policy": _policy_as_dict(object_policy, name="object_policy"),
            "threshold_provenance_mode": threshold_provenance,
        },
        "readiness": _readiness_as_dict(input_contract),
    }
    if extra_effective_config is not None:
        if not isinstance(extra_effective_config, Mapping):
            raise ContractError("extra_effective_config must be a mapping/object when provided.")
        effective_config["extra_effective_config"] = dict(extra_effective_config)

    _write_config_used(config_used_path, effective_config)

    _write_error_taxonomy(
        error_taxonomy_path,
        run_id=run_id,
        eval_mode=eval_mode,
        object_result=object_result,
        boundary_result=boundary_result,
    )

    if bucketed_result is not None and metrics_by_bucket_path is not None:
        write_bucketed_eval_artifact(
            metrics_by_bucket_path,
            run_id=run_id,
            eval_mode=eval_mode,
            result=bucketed_result,
        )

    bucketed_summary = (
        build_bucketed_eval_summary(bucketed_result)
        if bucketed_result is not None
        else None
    )

    # Stage E.5: visual diagnostics (non-blocking — skipped if matplotlib unavailable)
    visual_diag: VisualDiagnosticsResult | None = None
    if generate_visual_diagnostics:
        visual_diag = write_visual_diagnostics(
            run_dir,
            run_id=run_id,
            eval_mode=eval_mode,
            input_contract=input_contract,
        )
    visual_diagnostics_generated = bool(
        generate_visual_diagnostics
        and visual_diag is not None
        and not visual_diag.skipped
    )
    summary_warnings: list[str] = []
    if generate_visual_diagnostics and visual_diag is not None and visual_diag.skipped:
        reason = visual_diag.skip_reason or "unknown reason"
        summary_warnings.append(f"visual diagnostics skipped: {reason}")

    # Ranking summary — module_eval.md §16.2 formula
    # ranking_score = 0.4 * extent_f1 + 0.3 * boundary_f1 + 0.3 * (1 - normalized_gtc)
    _extent_f1 = pixel_summary.get("metrics", {}).get("extent_f1")
    _boundary_f1 = boundary_summary.get("metrics", {}).get("boundary_f1")
    _normalized_gtc = object_summary.get("metrics", {}).get("normalized_gtc")
    if (
        _extent_f1 is not None
        and _boundary_f1 is not None
        and _normalized_gtc is not None
    ):
        _ranking_score = 0.4 * float(_extent_f1) + 0.3 * float(_boundary_f1) + 0.3 * (1.0 - float(_normalized_gtc))
        ranking_summary: dict[str, Any] | None = {
            "ranking_score": round(_ranking_score, 6),
            "formula": "0.4 * extent_f1 + 0.3 * boundary_f1 + 0.3 * (1 - normalized_gtc)",
            "components": {
                "extent_f1": _extent_f1,
                "boundary_f1": _boundary_f1,
                "normalized_gtc": _normalized_gtc,
            },
        }
    else:
        ranking_summary = None

    # Write reporting artifacts (§14.1 mandatory reporting protocol)
    metrics_aggregate_payload: dict[str, Any] = {
        "schema_name": "eval.metrics_aggregate",
        "schema_version": "v1",
        "run_id": run_id,
        "eval_mode": eval_mode,
        "created_at_utc": created_at_utc,
        "pixel": pixel_summary,
        "boundary": boundary_summary,
        "object_structure": object_summary,
        "distance": distance_summary,
        "bucketed": bucketed_summary,
        "ranking_summary": ranking_summary,
    }
    with open(metrics_aggregate_path, "w", encoding="utf-8") as _fh:
        json.dump(metrics_aggregate_payload, _fh, indent=2, default=str)

    scenes_included_payload: dict[str, Any] = {
        "schema_name": "eval.scenes_included",
        "schema_version": "v1",
        "run_id": run_id,
        "eval_mode": eval_mode,
        "created_at_utc": created_at_utc,
        "scene_count": 1,
        "scenes": [run_id],
        "selection_policy": "single_scene_explicit_inputs",
    }
    with open(scenes_included_path, "w", encoding="utf-8") as _fh:
        json.dump(scenes_included_payload, _fh, indent=2, default=str)

    scenes_excluded_payload: dict[str, Any] = {
        "schema_name": "eval.scenes_excluded",
        "schema_version": "v1",
        "run_id": run_id,
        "eval_mode": eval_mode,
        "created_at_utc": created_at_utc,
        "excluded_count": 0,
        "excluded_scenes": [],
        "exclusion_policy": "single_scene_explicit_inputs_no_exclusions",
    }
    with open(scenes_excluded_path, "w", encoding="utf-8") as _fh:
        json.dump(scenes_excluded_payload, _fh, indent=2, default=str)

    source_runs_payload: dict[str, Any] = {
        "schema_name": "eval.source_runs",
        "schema_version": "v1",
        "run_id": run_id,
        "eval_mode": eval_mode,
        "created_at_utc": created_at_utc,
        "source_run_ids": source_run_ids,
        "source_manifest_paths": source_manifest_paths,
    }
    with open(source_runs_path, "w", encoding="utf-8") as _fh:
        json.dump(source_runs_payload, _fh, indent=2, default=str)

    manifest_payload: dict[str, Any] = {
        "schema_name": _EVAL_MANIFEST_SCHEMA,
        "schema_version": "v1",
        "module_name": "module_eval",
        "module_version": None,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "run_id": run_id,
        "stage_name": "export_eval_artifacts",
        "created_at_utc": created_at_utc,
        "status": "success",
        "eval_mode": eval_mode,
        "source_run_ids": source_run_ids,
        "source_manifest_paths": source_manifest_paths,
        "gt_sources": {
            "gt_extent_path": str(input_contract.gt_extent.path),
            "gt_boundary_path": str(input_contract.gt_boundary.path),
            "gt_valid_path": str(input_contract.gt_valid.path),
            "gt_distance_path": (
                str(input_contract.gt_distance.path)
                if input_contract.gt_distance is not None
                else None
            ),
            "gt_parcels_path": (
                str(input_contract.gt_parcels.path)
                if input_contract.gt_parcels is not None
                else None
            ),
        },
        "prediction_sources": {
            "pred_extent_prob_path": str(input_contract.pred_extent_prob.path),
            "pred_boundary_prob_path": str(input_contract.pred_boundary_prob.path),
            "pred_distance_pred_path": str(input_contract.pred_distance_pred.path),
            "pred_valid_path": str(input_contract.pred_valid.path),
        },
        "postprocess_sources": {
            "post_parcel_instance_path": (
                str(input_contract.post_parcel_instance.path)
                if input_contract.post_parcel_instance is not None
                else None
            ),
            "post_parcels_gpkg_path": (
                str(input_contract.post_parcels_vector.path)
                if input_contract.post_parcels_vector is not None
                else None
            ),
        },
        "scene_selection": {
            "policy": "single_scene_explicit_inputs",
            "resolved_scene_list": [run_id],
            "excluded_scenes": [],
        },
        "valid_aoi_policy": {
            "valid_policy": "strict_binary_valid_mask_gt_pred_consistency_required",
            "aoi_policy": _extract_aoi_policy_from_postprocess_manifest(
                input_contract.postprocess_manifest_path
            ),
        },
        "thresholds": {
            "raster_binarization": {
                "extent_prob_threshold": float(pixel_policy.extent_prob_threshold),
                "prediction_rule": pixel_policy.prediction_rule,
                "positive_gt_label": int(pixel_policy.positive_gt_label),
                "ignore_gt_label": int(pixel_policy.ignore_gt_label),
            },
            "vector_matching": {
                "min_iou_threshold": float(object_policy.min_iou_threshold),
                "min_overlap_gt_threshold": float(object_policy.min_overlap_gt_threshold),
                "min_overlap_pred_threshold": float(object_policy.min_overlap_pred_threshold),
                "match_rule": object_policy.match_rule,
            },
            "threshold_provenance": threshold_provenance,
            "threshold_provenance_details": {
                "pixel": pixel_policy.threshold_provenance,
                "boundary": boundary_policy.threshold_provenance,
                "object_structure": object_policy.threshold_provenance,
            },
        },
        "metrics_enabled": {
            "pixel": list(input_contract.output_contract.pixel_metrics),
            "boundary": list(input_contract.output_contract.boundary_metrics),
            "object_structure": list(input_contract.output_contract.object_structure_metrics),
            "distance": (
                ["distance_mae", "distance_rmse"]
                if distance_result is not None and not distance_result.skipped
                else []
            ),
        },
        "readiness": _readiness_as_dict(input_contract),
        "resolved_input_contract": {
            "common_width": int(input_contract.common_width),
            "common_height": int(input_contract.common_height),
            "common_crs": input_contract.common_crs,
            "common_transform_gdal": list(input_contract.common_transform_gdal),
            "semantic_compatibility_summary": dict(input_contract.semantic_compatibility_summary),
        },
        "policy_details": {
            "pixel_policy": _policy_as_dict(pixel_policy, name="pixel_policy"),
            "boundary_policy": _policy_as_dict(boundary_policy, name="boundary_policy"),
            "object_matching_policy": _policy_as_dict(object_policy, name="object_policy"),
        },
        "metric_summary": {
            "pixel": pixel_summary,
            "boundary": boundary_summary,
            "object_structure": object_summary,
            "distance": distance_summary,
            "bucketed": bucketed_summary,
        },
        "comparison": {
            "enabled": False,
            "comparison_pairs": None,
        },
        "runtime": {
            "device_requested": None,
            "device_resolved": None,
            "parallel_mode": None,
            "scene_fallback_mode": None,
        },
        "artifacts": {
            "config_used_path": str(config_used_path),
            "summary_path": str(summary_path),
            "eval_manifest_path": str(manifest_path),
            "error_taxonomy_path": str(error_taxonomy_path),
            "metrics_by_bucket_path": (
                str(metrics_by_bucket_path) if metrics_by_bucket_path is not None else None
            ),
            "metrics_aggregate_path": str(metrics_aggregate_path),
            "scenes_included_path": str(scenes_included_path),
            "scenes_excluded_path": str(scenes_excluded_path),
            "source_runs_path": str(source_runs_path),
        },
        "stage_coverage": {
            "stage_a": True,
            "stage_b": True,
            "stage_b5_distance_metrics": True,
            "stage_c": True,
            "stage_d": True,
            "stage_d5_bucketed_eval": bucketed_result is not None,
            "stage_e": True,
            "stage_e5_visual_diagnostics": visual_diagnostics_generated,
            "error_taxonomy": True,
            "not_implemented": {
                "comparison_engine": True,
                "cross_run_benchmarking": True,
                "dashboard_reporting_framework": True,
                **({"bucketed_evaluation": True} if bucketed_result is None else {}),
            },
        },
    }
    write_manifest(manifest_path, manifest_payload)

    summary_payload: dict[str, Any] = {
        "schema_name": _EVAL_SUMMARY_SCHEMA,
        "status": "success",
        "run_id": run_id,
        "module_name": "module_eval",
        "eval_mode": eval_mode,
        "scene_count": 1,
        "availability_readiness": _readiness_as_dict(input_contract),
        "metric_summary": {
            "pixel": pixel_summary,
            "boundary": boundary_summary,
            "object_structure": object_summary,
            "distance": distance_summary,
            "bucketed": bucketed_summary,
        },
        "ranking_summary": ranking_summary,
        "policy_summary": {
            "threshold_provenance_mode": threshold_provenance,
            "pixel_threshold_provenance": pixel_policy.threshold_provenance,
            "boundary_threshold_provenance": boundary_policy.threshold_provenance,
            "object_matching_threshold_provenance": object_policy.threshold_provenance,
            "boundary_prediction_interpretation": boundary_policy.prediction_interpretation,
            "object_matching_rule": object_policy.match_rule,
        },
        "stage_coverage": {
            "implemented": (
                (
                    ["A", "B", "B5_distance_metrics", "C", "D", "D5_bucketed_eval", "E"]
                    if bucketed_result is not None
                    else ["A", "B", "B5_distance_metrics", "C", "D", "E"]
                )
                + (["E5_visual_diagnostics"] if visual_diagnostics_generated else [])
                + ["error_taxonomy"]
            ),
            "not_implemented": [
                "comparison_engine",
                "cross_run_benchmarking",
                "dashboard_reporting_framework",
                *( ["bucketed_evaluation"] if bucketed_result is None else []),
            ],
        },
        "source_run_ids": source_run_ids,
        "source_manifest_paths": source_manifest_paths,
        "warnings": summary_warnings,
        "key_notes": [
            "Stage E records provenance and effective policy contract for Stages A-D.",
            "Comparison framework is intentionally out of scope at this stage.",
        ],
    }
    write_summary(summary_path, summary_payload)

    return EvalExportArtifacts(
        run_dir=run_dir,
        eval_manifest_path=manifest_path,
        summary_path=summary_path,
        config_used_path=config_used_path,
        error_taxonomy_path=error_taxonomy_path,
        metrics_by_bucket_path=metrics_by_bucket_path,
        visual_diagnostics=visual_diag,
        metrics_aggregate_path=metrics_aggregate_path,
        scenes_included_path=scenes_included_path,
        scenes_excluded_path=scenes_excluded_path,
        source_runs_path=source_runs_path,
    )


__all__ = [
    "EvalExportArtifacts",
    "export_eval_artifacts",
]
