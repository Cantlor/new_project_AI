"""Unit tests for module_eval minimal pairwise comparison computation layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_eval.comparison_contract import EvalComparisonInputContractResult
from ai_fields.module_eval.pairwise_comparison import (
    build_pairwise_comparison_summary,
    compute_pairwise_eval_comparison,
)


def _write_eval_summary(path: Path, *, metric_summary: dict[str, Any]) -> Path:
    payload = {
        "schema_name": "eval.summary",
        "status": "success",
        "run_id": path.parent.name,
        "module_name": "module_eval",
        "eval_mode": "end_to_end_single_scene",
        "scene_count": 1,
        "metric_summary": metric_summary,
        "stage_coverage": {"implemented": ["A", "B", "C", "D", "E"]},
        "warnings": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _default_metric_summary(
    *,
    pixel_offset: float = 0.0,
    boundary_offset: float = 0.0,
    object_offset: float = 0.0,
    boundary_bde: float | None = 2.0,
    normalized_gtc: float | None = 0.3,
) -> dict[str, Any]:
    return {
        "pixel": {
            "metrics": {
                "extent_iou": 0.60 + pixel_offset,
                "extent_f1": 0.72 + pixel_offset,
                "extent_precision": 0.70 + pixel_offset,
                "extent_recall": 0.75 + pixel_offset,
            }
        },
        "boundary": {
            "metrics": {
                "boundary_f1": 0.52 + boundary_offset,
                "boundary_precision": 0.50 + boundary_offset,
                "boundary_recall": 0.56 + boundary_offset,
                "boundary_bde": boundary_bde,
            }
        },
        "object_structure": {
            "metrics": {
                "goc": 0.20 + object_offset,
                "guc": 0.16 + object_offset,
                "gtc": 0.36 + object_offset,
                "normalized_gtc": normalized_gtc,
            }
        },
    }


def _make_contract(
    *,
    tmp_path: Path,
    comparable_groups: tuple[str, ...],
    non_comparable_groups: dict[str, str],
    left_metrics: dict[str, Any],
    right_metrics: dict[str, Any],
    ready: bool,
    partial: bool,
) -> EvalComparisonInputContractResult:
    left_run_dir = tmp_path / "left_run"
    right_run_dir = tmp_path / "right_run"
    left_manifest = left_run_dir / "eval_manifest.json"
    right_manifest = right_run_dir / "eval_manifest.json"
    left_cfg = left_run_dir / "config_used.yaml"
    right_cfg = right_run_dir / "config_used.yaml"
    left_summary = _write_eval_summary(left_run_dir / "summary.json", metric_summary=left_metrics)
    right_summary = _write_eval_summary(right_run_dir / "summary.json", metric_summary=right_metrics)

    left_manifest.write_text("{}", encoding="utf-8")
    right_manifest.write_text("{}", encoding="utf-8")
    left_cfg.write_text("{}", encoding="utf-8")
    right_cfg.write_text("{}", encoding="utf-8")

    return EvalComparisonInputContractResult(
        left_run_id="left_eval_run",
        right_run_id="right_eval_run",
        left_run_dir=left_run_dir,
        right_run_dir=right_run_dir,
        left_eval_manifest_path=left_manifest,
        right_eval_manifest_path=right_manifest,
        left_summary_path=left_summary,
        right_summary_path=right_summary,
        left_config_used_path=left_cfg,
        right_config_used_path=right_cfg,
        left_eval_mode="end_to_end_single_scene",
        right_eval_mode="end_to_end_single_scene",
        comparable_metric_groups=comparable_groups,
        non_comparable_metric_groups=dict(non_comparable_groups),
        policy_compatibility_summary={},
        provenance_compatibility_summary={},
        stage_coverage_compatibility_summary={},
        global_blockers=(),
        ready_for_pairwise_compare=ready,
        partially_ready=partial,
    )


def test_compute_pairwise_eval_comparison_happy_path(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path,
        comparable_groups=("pixel", "boundary", "object_structure"),
        non_comparable_groups={},
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(
            pixel_offset=0.02,
            boundary_offset=0.03,
            object_offset=-0.04,
            boundary_bde=1.7,
            normalized_gtc=0.24,
        ),
        ready=True,
        partial=False,
    )

    result = compute_pairwise_eval_comparison(comparison_contract=contract)

    assert result.comparison_status == "ready"
    assert result.delta_direction == "right_minus_left"
    assert result.per_group_metric_deltas["pixel"]["extent_iou_delta"] == pytest.approx(0.02)
    assert result.per_group_metric_deltas["boundary"]["boundary_f1_delta"] == pytest.approx(0.03)
    assert result.per_group_metric_deltas["object_structure"]["gtc_delta"] == pytest.approx(-0.04)
    assert result.per_group_metric_deltas["boundary"]["boundary_bde_delta"] == pytest.approx(-0.3)
    assert result.per_group_metric_orientation["boundary"]["boundary_bde"] == "lower_is_better"

    summary = build_pairwise_comparison_summary(result)
    assert summary["comparison_status"] == "ready"
    assert summary["delta_direction"] == "right_minus_left"


def test_compute_pairwise_eval_comparison_partial_comparability(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path,
        comparable_groups=("pixel", "boundary"),
        non_comparable_groups={"object_structure": "object match_rule mismatch"},
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(pixel_offset=0.01, boundary_offset=0.02),
        ready=False,
        partial=True,
    )

    result = compute_pairwise_eval_comparison(comparison_contract=contract)

    assert result.comparison_status == "partial"
    assert set(result.per_group_metric_deltas) == {"pixel", "boundary"}
    assert "object_structure" not in result.per_group_metric_deltas
    assert result.non_comparable_metric_groups["object_structure"] == "object match_rule mismatch"


def test_compute_pairwise_eval_comparison_policy_mismatch_carry_through(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path,
        comparable_groups=("boundary", "object_structure"),
        non_comparable_groups={"pixel": "pixel threshold mismatch"},
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(boundary_offset=0.01, object_offset=-0.02),
        ready=False,
        partial=True,
    )

    result = compute_pairwise_eval_comparison(comparison_contract=contract)

    assert "pixel" not in result.per_group_metric_deltas
    assert result.non_comparable_metric_groups["pixel"] == "pixel threshold mismatch"
    assert result.per_group_metric_deltas["object_structure"]["goc_delta"] == pytest.approx(-0.02)


def test_compute_pairwise_eval_comparison_orientation_transparency(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path,
        comparable_groups=("boundary", "object_structure"),
        non_comparable_groups={"pixel": "pixel threshold mismatch"},
        left_metrics=_default_metric_summary(boundary_bde=2.5, normalized_gtc=0.40),
        right_metrics=_default_metric_summary(boundary_bde=2.0, normalized_gtc=0.31),
        ready=False,
        partial=True,
    )

    result = compute_pairwise_eval_comparison(comparison_contract=contract)
    orientation = result.per_group_metric_orientation

    assert orientation["boundary"]["boundary_f1"] == "higher_is_better"
    assert orientation["boundary"]["boundary_bde"] == "lower_is_better"
    assert orientation["object_structure"]["gtc"] == "lower_is_better"
    assert orientation["object_structure"]["normalized_gtc"] == "lower_is_better"


def test_compute_pairwise_eval_comparison_missing_required_metric_fails(tmp_path: Path) -> None:
    left = _default_metric_summary()
    right = _default_metric_summary(pixel_offset=0.01)
    del right["pixel"]["metrics"]["extent_f1"]

    contract = _make_contract(
        tmp_path=tmp_path,
        comparable_groups=("pixel",),
        non_comparable_groups={"boundary": "not requested", "object_structure": "not requested"},
        left_metrics=left,
        right_metrics=right,
        ready=False,
        partial=True,
    )

    with pytest.raises(ContractError, match="requires metric 'extent_f1'"):
        compute_pairwise_eval_comparison(comparison_contract=contract)


def test_compute_pairwise_eval_comparison_not_ready_returns_explicit_status(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path,
        comparable_groups=(),
        non_comparable_groups={
            "pixel": "eval_mode mismatch",
            "boundary": "eval_mode mismatch",
            "object_structure": "eval_mode mismatch",
        },
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(pixel_offset=0.1),
        ready=False,
        partial=False,
    )

    result = compute_pairwise_eval_comparison(comparison_contract=contract)
    assert result.comparison_status == "not_ready"
    assert result.per_group_metric_deltas == {}
    assert result.non_comparable_metric_groups["pixel"] == "eval_mode mismatch"

