"""Minimal comparison input-contract layer for module_eval.

This layer is intentionally narrow:
- reads two eval artifact sets (manifest/summary/config_used);
- validates whether pairwise comparison is fair/possible;
- exposes explicit comparable and non-comparable metric groups.

Out of scope:
- metric deltas/tables,
- ranking/leaderboard,
- dashboard/reporting framework.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import read_manifest

_EXPECTED_MANIFEST_SCHEMA = "eval.eval_manifest"
_EXPECTED_SUMMARY_SCHEMA = "eval.summary"
_GROUPS = ("pixel", "boundary", "object_structure")
_STAGE_KEY_BY_GROUP = {
    "pixel": "stage_b",
    "boundary": "stage_c",
    "object_structure": "stage_d",
}
_EPS = 1e-9


@dataclass(frozen=True)
class _ResolvedEvalArtifacts:
    side: str
    run_dir: Path
    eval_manifest_path: Path
    summary_path: Path
    config_used_path: Path

    manifest: dict[str, Any]
    summary: dict[str, Any]
    config_used: dict[str, Any]


@dataclass(frozen=True)
class EvalComparisonInputContractResult:
    """Resolved pairwise comparison input contract for next compare stage."""

    left_run_id: str
    right_run_id: str
    left_run_dir: Path
    right_run_dir: Path
    left_eval_manifest_path: Path
    right_eval_manifest_path: Path
    left_summary_path: Path
    right_summary_path: Path
    left_config_used_path: Path
    right_config_used_path: Path

    left_eval_mode: str
    right_eval_mode: str

    comparable_metric_groups: tuple[str, ...]
    non_comparable_metric_groups: dict[str, str]

    policy_compatibility_summary: dict[str, Any]
    provenance_compatibility_summary: dict[str, Any]
    stage_coverage_compatibility_summary: dict[str, Any]

    global_blockers: tuple[str, ...]
    ready_for_pairwise_compare: bool
    partially_ready: bool


def _normalize_existing_file(path: Any, *, name: str) -> Path:
    if not isinstance(path, (str, PathLike)):
        raise ContractError(f"{name} must be path-like, got {type(path).__name__}.")
    resolved = Path(path)
    if str(resolved).strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    if not resolved.exists() or not resolved.is_file():
        raise ContractError(f"{name} does not exist as file: {resolved}")
    return resolved


def _normalize_dir(path: Any, *, name: str) -> Path:
    if not isinstance(path, (str, PathLike)):
        raise ContractError(f"{name} must be path-like, got {type(path).__name__}.")
    resolved = Path(path)
    if str(resolved).strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    if not resolved.exists() or not resolved.is_dir():
        raise ContractError(f"{name} does not exist as directory: {resolved}")
    return resolved


def _require_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{name} must be a non-empty string.")
    return value


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping/object.")
    return dict(value)


def _read_summary(path: Path, *, side: str) -> dict[str, Any]:
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
    _require_mapping(summary.get("stage_coverage"), name=f"{side}.summary.stage_coverage")
    return summary


def _read_config_used(path: Path, *, side: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ContractError("PyYAML is required to read eval config_used.yaml.") from exc

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"{side}: failed reading config_used.yaml: {exc}") from exc

    cfg = _require_mapping(payload, name=f"{side}.config_used")
    policy = _require_mapping(
        cfg.get("effective_policy_contract"),
        name=f"{side}.config_used.effective_policy_contract",
    )
    _require_mapping(policy.get("pixel_policy"), name=f"{side}.config_used.pixel_policy")
    _require_mapping(policy.get("boundary_policy"), name=f"{side}.config_used.boundary_policy")
    _require_mapping(
        policy.get("object_matching_policy"),
        name=f"{side}.config_used.object_matching_policy",
    )
    return cfg


def _resolve_paths(
    *,
    side: str,
    run_dir: str | Path | None,
    eval_manifest_path: str | Path | None,
    summary_path: str | Path | None,
    config_used_path: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    if run_dir is not None:
        resolved_run_dir = _normalize_dir(run_dir, name=f"{side}_run_dir")
        manifest = (
            _normalize_existing_file(eval_manifest_path, name=f"{side}_eval_manifest_path")
            if eval_manifest_path is not None
            else _normalize_existing_file(
                resolved_run_dir / "eval_manifest.json",
                name=f"{side}_eval_manifest_path",
            )
        )
        summary = (
            _normalize_existing_file(summary_path, name=f"{side}_summary_path")
            if summary_path is not None
            else _normalize_existing_file(
                resolved_run_dir / "summary.json",
                name=f"{side}_summary_path",
            )
        )
        config_used = (
            _normalize_existing_file(config_used_path, name=f"{side}_config_used_path")
            if config_used_path is not None
            else _normalize_existing_file(
                resolved_run_dir / "config_used.yaml",
                name=f"{side}_config_used_path",
            )
        )
        return resolved_run_dir, manifest, summary, config_used

    # No run_dir provided: all three artifact paths must be explicit.
    if eval_manifest_path is None or summary_path is None or config_used_path is None:
        raise ContractError(
            f"{side}: either run_dir or all explicit paths "
            "(eval_manifest_path, summary_path, config_used_path) must be provided."
        )
    manifest = _normalize_existing_file(eval_manifest_path, name=f"{side}_eval_manifest_path")
    summary = _normalize_existing_file(summary_path, name=f"{side}_summary_path")
    config_used = _normalize_existing_file(config_used_path, name=f"{side}_config_used_path")
    return manifest.parent, manifest, summary, config_used


def _resolve_artifacts(
    *,
    side: str,
    run_dir: str | Path | None,
    eval_manifest_path: str | Path | None,
    summary_path: str | Path | None,
    config_used_path: str | Path | None,
) -> _ResolvedEvalArtifacts:
    resolved_run_dir, manifest_path, resolved_summary_path, resolved_config_used_path = _resolve_paths(
        side=side,
        run_dir=run_dir,
        eval_manifest_path=eval_manifest_path,
        summary_path=summary_path,
        config_used_path=config_used_path,
    )

    manifest = read_manifest(manifest_path)
    schema_name = manifest.get("schema_name")
    if schema_name != _EXPECTED_MANIFEST_SCHEMA:
        raise ContractError(
            f"{side}: eval_manifest schema_name must be {_EXPECTED_MANIFEST_SCHEMA!r}, got {schema_name!r}."
        )
    dc_version = _require_str(
        manifest.get("data_contract_version"),
        name=f"{side}.manifest.data_contract_version",
    )
    if dc_version != DATA_CONTRACT_VERSION:
        raise ContractError(
            f"{side}: data_contract_version mismatch: expected {DATA_CONTRACT_VERSION!r}, got {dc_version!r}."
        )
    _require_mapping(manifest.get("metrics_enabled"), name=f"{side}.manifest.metrics_enabled")
    _require_mapping(manifest.get("stage_coverage"), name=f"{side}.manifest.stage_coverage")
    _require_mapping(manifest.get("thresholds"), name=f"{side}.manifest.thresholds")

    summary = _read_summary(resolved_summary_path, side=side)
    config = _read_config_used(resolved_config_used_path, side=side)
    return _ResolvedEvalArtifacts(
        side=side,
        run_dir=resolved_run_dir,
        eval_manifest_path=manifest_path,
        summary_path=resolved_summary_path,
        config_used_path=resolved_config_used_path,
        manifest=manifest,
        summary=summary,
        config_used=config,
    )


def _as_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric.")
    return float(value)


def _float_eq(left: Any, right: Any) -> bool:
    return abs(float(left) - float(right)) <= _EPS


def _run_group_available(run: _ResolvedEvalArtifacts, group: str) -> tuple[bool, str | None]:
    metrics_enabled = _require_mapping(
        run.manifest.get("metrics_enabled"),
        name=f"{run.side}.manifest.metrics_enabled",
    )
    stage_coverage = _require_mapping(
        run.manifest.get("stage_coverage"),
        name=f"{run.side}.manifest.stage_coverage",
    )
    metric_summary = _require_mapping(
        run.summary.get("metric_summary"),
        name=f"{run.side}.summary.metric_summary",
    )

    stage_key = _STAGE_KEY_BY_GROUP[group]
    stage_value = stage_coverage.get(stage_key)
    if stage_value is not True:
        return False, f"{run.side}:{group} unavailable because {stage_key} is not true"

    enabled = metrics_enabled.get(group)
    if not isinstance(enabled, list) or len(enabled) == 0:
        return False, f"{run.side}:{group} unavailable because metrics_enabled.{group} is empty"

    if group not in metric_summary:
        return False, f"{run.side}:{group} missing in summary.metric_summary"

    return True, None


def _compare_pixel_policy(left: _ResolvedEvalArtifacts, right: _ResolvedEvalArtifacts) -> tuple[bool, str]:
    l = _require_mapping(
        _require_mapping(
            left.config_used.get("effective_policy_contract"),
            name="left.config_used.effective_policy_contract",
        ).get("pixel_policy"),
        name="left.config_used.effective_policy_contract.pixel_policy",
    )
    r = _require_mapping(
        _require_mapping(
            right.config_used.get("effective_policy_contract"),
            name="right.config_used.effective_policy_contract",
        ).get("pixel_policy"),
        name="right.config_used.effective_policy_contract.pixel_policy",
    )

    keys = (
        "extent_prob_threshold",
        "prediction_rule",
        "positive_gt_label",
        "ignore_gt_label",
        "threshold_provenance",
    )
    for key in keys:
        if key not in l or key not in r:
            raise ContractError(f"pixel policy missing key {key!r} for pairwise comparison.")

    if not _float_eq(_as_float(l["extent_prob_threshold"], name="left.pixel_threshold"), _as_float(r["extent_prob_threshold"], name="right.pixel_threshold")):
        return False, "pixel threshold mismatch"
    if l["prediction_rule"] != r["prediction_rule"]:
        return False, "pixel prediction_rule mismatch"
    if int(l["positive_gt_label"]) != int(r["positive_gt_label"]):
        return False, "pixel positive_gt_label mismatch"
    if int(l["ignore_gt_label"]) != int(r["ignore_gt_label"]):
        return False, "pixel ignore_gt_label mismatch"
    if str(l["threshold_provenance"]) != str(r["threshold_provenance"]):
        return False, "pixel threshold_provenance mismatch"
    return True, "compatible"


def _compare_boundary_policy(left: _ResolvedEvalArtifacts, right: _ResolvedEvalArtifacts) -> tuple[bool, str]:
    l = _require_mapping(
        _require_mapping(
            left.config_used.get("effective_policy_contract"),
            name="left.config_used.effective_policy_contract",
        ).get("boundary_policy"),
        name="left.config_used.effective_policy_contract.boundary_policy",
    )
    r = _require_mapping(
        _require_mapping(
            right.config_used.get("effective_policy_contract"),
            name="right.config_used.effective_policy_contract",
        ).get("boundary_policy"),
        name="right.config_used.effective_policy_contract.boundary_policy",
    )

    required = (
        "prediction_interpretation",
        "gt_interpretation",
        "threshold_provenance",
        "non_background_prob_threshold",
        "bde_enabled",
    )
    for key in required:
        if key not in l or key not in r:
            raise ContractError(f"boundary policy missing key {key!r} for pairwise comparison.")

    if l["prediction_interpretation"] != r["prediction_interpretation"]:
        return False, "boundary prediction_interpretation mismatch"
    if l["gt_interpretation"] != r["gt_interpretation"]:
        return False, "boundary gt_interpretation mismatch"
    if str(l["threshold_provenance"]) != str(r["threshold_provenance"]):
        return False, "boundary threshold_provenance mismatch"
    if not _float_eq(
        _as_float(l["non_background_prob_threshold"], name="left.boundary_threshold"),
        _as_float(r["non_background_prob_threshold"], name="right.boundary_threshold"),
    ):
        return False, "boundary non_background_prob_threshold mismatch"
    if bool(l["bde_enabled"]) is not bool(r["bde_enabled"]):
        return False, "boundary bde_enabled mismatch"
    return True, "compatible"


def _compare_object_policy(left: _ResolvedEvalArtifacts, right: _ResolvedEvalArtifacts) -> tuple[bool, str]:
    l = _require_mapping(
        _require_mapping(
            left.config_used.get("effective_policy_contract"),
            name="left.config_used.effective_policy_contract",
        ).get("object_matching_policy"),
        name="left.config_used.effective_policy_contract.object_matching_policy",
    )
    r = _require_mapping(
        _require_mapping(
            right.config_used.get("effective_policy_contract"),
            name="right.config_used.effective_policy_contract",
        ).get("object_matching_policy"),
        name="right.config_used.effective_policy_contract.object_matching_policy",
    )

    required = (
        "min_iou_threshold",
        "min_overlap_gt_threshold",
        "min_overlap_pred_threshold",
        "match_rule",
        "threshold_provenance",
    )
    for key in required:
        if key not in l or key not in r:
            raise ContractError(f"object matching policy missing key {key!r} for pairwise comparison.")

    if not _float_eq(
        _as_float(l["min_iou_threshold"], name="left.min_iou_threshold"),
        _as_float(r["min_iou_threshold"], name="right.min_iou_threshold"),
    ):
        return False, "object min_iou_threshold mismatch"
    if not _float_eq(
        _as_float(l["min_overlap_gt_threshold"], name="left.min_overlap_gt_threshold"),
        _as_float(r["min_overlap_gt_threshold"], name="right.min_overlap_gt_threshold"),
    ):
        return False, "object min_overlap_gt_threshold mismatch"
    if not _float_eq(
        _as_float(l["min_overlap_pred_threshold"], name="left.min_overlap_pred_threshold"),
        _as_float(r["min_overlap_pred_threshold"], name="right.min_overlap_pred_threshold"),
    ):
        return False, "object min_overlap_pred_threshold mismatch"
    if str(l["match_rule"]) != str(r["match_rule"]):
        return False, "object match_rule mismatch"
    if str(l["threshold_provenance"]) != str(r["threshold_provenance"]):
        return False, "object threshold_provenance mismatch"
    return True, "compatible"


def resolve_eval_comparison_contract(
    *,
    left_run_dir: str | Path | None = None,
    right_run_dir: str | Path | None = None,
    left_eval_manifest_path: str | Path | None = None,
    right_eval_manifest_path: str | Path | None = None,
    left_summary_path: str | Path | None = None,
    right_summary_path: str | Path | None = None,
    left_config_used_path: str | Path | None = None,
    right_config_used_path: str | Path | None = None,
) -> EvalComparisonInputContractResult:
    """Resolve pairwise comparison readiness from two eval artifact sets."""
    left = _resolve_artifacts(
        side="left",
        run_dir=left_run_dir,
        eval_manifest_path=left_eval_manifest_path,
        summary_path=left_summary_path,
        config_used_path=left_config_used_path,
    )
    right = _resolve_artifacts(
        side="right",
        run_dir=right_run_dir,
        eval_manifest_path=right_eval_manifest_path,
        summary_path=right_summary_path,
        config_used_path=right_config_used_path,
    )

    left_run_id = _require_str(left.manifest.get("run_id"), name="left.manifest.run_id")
    right_run_id = _require_str(right.manifest.get("run_id"), name="right.manifest.run_id")
    left_eval_mode = _require_str(left.manifest.get("eval_mode"), name="left.manifest.eval_mode")
    right_eval_mode = _require_str(right.manifest.get("eval_mode"), name="right.manifest.eval_mode")

    global_blockers: list[str] = []
    if left_eval_mode != right_eval_mode:
        global_blockers.append(
            f"eval_mode mismatch: left={left_eval_mode!r}, right={right_eval_mode!r}"
        )

    comparable: list[str] = []
    non_comparable: dict[str, str] = {}
    policy_summary: dict[str, Any] = {}
    stage_summary: dict[str, Any] = {}

    for group in _GROUPS:
        left_available, left_reason = _run_group_available(left, group)
        right_available, right_reason = _run_group_available(right, group)
        stage_summary[group] = {
            "left_available": left_available,
            "right_available": right_available,
            "left_reason": left_reason,
            "right_reason": right_reason,
        }

        if global_blockers:
            non_comparable[group] = "; ".join(global_blockers)
            policy_summary[group] = {"compatible": False, "reason": "; ".join(global_blockers)}
            continue

        if not left_available or not right_available:
            reasons = [r for r in (left_reason, right_reason) if r is not None]
            non_comparable[group] = " / ".join(reasons) if reasons else "group unavailable"
            policy_summary[group] = {"compatible": False, "reason": non_comparable[group]}
            continue

        if group == "pixel":
            is_ok, reason = _compare_pixel_policy(left, right)
        elif group == "boundary":
            is_ok, reason = _compare_boundary_policy(left, right)
        else:
            is_ok, reason = _compare_object_policy(left, right)

        policy_summary[group] = {"compatible": bool(is_ok), "reason": reason}
        if is_ok:
            comparable.append(group)
        else:
            non_comparable[group] = reason

    left_source_manifest_paths = tuple(str(p) for p in left.manifest.get("source_manifest_paths", []))
    right_source_manifest_paths = tuple(str(p) for p in right.manifest.get("source_manifest_paths", []))
    left_source_run_ids = tuple(str(v) for v in left.manifest.get("source_run_ids", []))
    right_source_run_ids = tuple(str(v) for v in right.manifest.get("source_run_ids", []))
    shared_manifest_paths = tuple(sorted(set(left_source_manifest_paths).intersection(right_source_manifest_paths)))
    shared_source_run_ids = tuple(sorted(set(left_source_run_ids).intersection(right_source_run_ids)))

    provenance_summary = {
        "left_source_run_ids": left_source_run_ids,
        "right_source_run_ids": right_source_run_ids,
        "shared_source_run_ids": shared_source_run_ids,
        "left_source_manifest_paths": left_source_manifest_paths,
        "right_source_manifest_paths": right_source_manifest_paths,
        "shared_source_manifest_paths": shared_manifest_paths,
        "source_run_ids_match": set(left_source_run_ids) == set(right_source_run_ids),
        "source_manifest_paths_match": set(left_source_manifest_paths) == set(right_source_manifest_paths),
    }

    ready = len(global_blockers) == 0 and len(non_comparable) == 0 and len(comparable) == len(_GROUPS)
    partial = len(global_blockers) == 0 and len(comparable) > 0 and len(non_comparable) > 0

    return EvalComparisonInputContractResult(
        left_run_id=left_run_id,
        right_run_id=right_run_id,
        left_run_dir=left.run_dir,
        right_run_dir=right.run_dir,
        left_eval_manifest_path=left.eval_manifest_path,
        right_eval_manifest_path=right.eval_manifest_path,
        left_summary_path=left.summary_path,
        right_summary_path=right.summary_path,
        left_config_used_path=left.config_used_path,
        right_config_used_path=right.config_used_path,
        left_eval_mode=left_eval_mode,
        right_eval_mode=right_eval_mode,
        comparable_metric_groups=tuple(comparable),
        non_comparable_metric_groups=non_comparable,
        policy_compatibility_summary=policy_summary,
        provenance_compatibility_summary=provenance_summary,
        stage_coverage_compatibility_summary=stage_summary,
        global_blockers=tuple(global_blockers),
        ready_for_pairwise_compare=ready,
        partially_ready=partial,
    )


def build_comparison_readiness_summary(
    result: EvalComparisonInputContractResult,
) -> dict[str, Any]:
    """Build compact machine-readable summary for comparison readiness."""
    if not isinstance(result, EvalComparisonInputContractResult):
        raise ContractError("result must be EvalComparisonInputContractResult.")
    return {
        "left_run_id": result.left_run_id,
        "right_run_id": result.right_run_id,
        "left_eval_mode": result.left_eval_mode,
        "right_eval_mode": result.right_eval_mode,
        "comparable_metric_groups": list(result.comparable_metric_groups),
        "non_comparable_metric_groups": dict(result.non_comparable_metric_groups),
        "policy_compatibility_summary": dict(result.policy_compatibility_summary),
        "provenance_compatibility_summary": dict(result.provenance_compatibility_summary),
        "stage_coverage_compatibility_summary": dict(result.stage_coverage_compatibility_summary),
        "global_blockers": list(result.global_blockers),
        "ready_for_pairwise_compare": bool(result.ready_for_pairwise_compare),
        "partially_ready": bool(result.partially_ready),
    }


__all__ = [
    "EvalComparisonInputContractResult",
    "build_comparison_readiness_summary",
    "resolve_eval_comparison_contract",
]

