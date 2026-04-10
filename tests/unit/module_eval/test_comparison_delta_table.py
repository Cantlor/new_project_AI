"""Unit tests for module_eval comparison delta table artifact layer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_eval.comparison_contract import EvalComparisonInputContractResult
from ai_fields.module_eval.comparison_export import export_pairwise_comparison_artifacts
from ai_fields.module_eval.pairwise_comparison import compute_pairwise_eval_comparison


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


def _metric_summary(*, pixel: float, boundary: float, gtc: float, bde: float) -> dict[str, Any]:
    return {
        "pixel": {
            "metrics": {
                "extent_iou": pixel,
                "extent_f1": pixel + 0.1,
                "extent_precision": pixel + 0.05,
                "extent_recall": pixel + 0.15,
            }
        },
        "boundary": {
            "metrics": {
                "boundary_f1": boundary,
                "boundary_precision": boundary + 0.08,
                "boundary_recall": boundary + 0.1,
                "boundary_bde": bde,
            }
        },
        "object_structure": {
            "metrics": {
                "goc": gtc - 0.2,
                "guc": 0.2,
                "gtc": gtc,
                "normalized_gtc": gtc,
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


def test_comparison_delta_table_happy_path_content(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel", "boundary", "object_structure"),
        non_comparable_groups={},
        left_metrics=_metric_summary(pixel=0.60, boundary=0.50, gtc=0.36, bde=2.3),
        right_metrics=_metric_summary(pixel=0.64, boundary=0.56, gtc=0.30, bde=1.9),
        ready=True,
        partial=False,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)
    artifacts = export_pairwise_comparison_artifacts(
        output_dir=tmp_path / "compare_run",
        comparison_run_id="compare_run_001",
        comparison_contract=contract,
        pairwise_result=pairwise,
    )

    table = json.loads(artifacts.comparison_delta_table_path.read_text(encoding="utf-8"))
    assert table["schema_name"] == "eval.comparison_delta_table"
    assert table["delta_direction"] == "right_minus_left"
    assert table["rows"]

    # one concrete check for correctness and orientation transparency
    rows = {(r["metric_group"], r["metric_name"]): r for r in table["rows"]}
    assert rows[("pixel", "extent_iou")]["delta_value"] == pytest.approx(0.04)
    assert rows[("boundary", "boundary_bde")]["delta_value"] == pytest.approx(-0.4)
    assert rows[("boundary", "boundary_bde")]["orientation"] == "lower_is_better"


def test_comparison_delta_table_preserves_non_comparable_groups(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel",),
        non_comparable_groups={
            "boundary": "boundary prediction_interpretation mismatch",
            "object_structure": "object match_rule mismatch",
        },
        left_metrics=_metric_summary(pixel=0.60, boundary=0.5, gtc=0.36, bde=2.2),
        right_metrics=_metric_summary(pixel=0.62, boundary=0.52, gtc=0.35, bde=2.1),
        ready=False,
        partial=True,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)
    artifacts = export_pairwise_comparison_artifacts(
        output_dir=tmp_path / "compare_run",
        comparison_run_id="compare_run_partial",
        comparison_contract=contract,
        pairwise_result=pairwise,
    )

    table = json.loads(artifacts.comparison_delta_table_path.read_text(encoding="utf-8"))
    assert table["comparison_status"] == "partial"
    assert table["non_comparable_metric_groups"]["boundary"] == "boundary prediction_interpretation mismatch"
    assert table["non_comparable_metric_groups"]["object_structure"] == "object match_rule mismatch"
    assert all(r["metric_group"] == "pixel" for r in table["rows"])


def test_comparison_delta_table_has_provenance_links(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel",),
        non_comparable_groups={"boundary": "off", "object_structure": "off"},
        left_metrics=_metric_summary(pixel=0.6, boundary=0.5, gtc=0.36, bde=2.2),
        right_metrics=_metric_summary(pixel=0.62, boundary=0.5, gtc=0.36, bde=2.1),
        ready=False,
        partial=True,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)
    artifacts = export_pairwise_comparison_artifacts(
        output_dir=tmp_path / "compare_run",
        comparison_run_id="compare_run_provenance",
        comparison_contract=contract,
        pairwise_result=pairwise,
    )

    table = json.loads(artifacts.comparison_delta_table_path.read_text(encoding="utf-8"))
    links = table["provenance_links"]
    assert links["left_summary_path"] == str(contract.left_summary_path)
    assert links["right_summary_path"] == str(contract.right_summary_path)
    assert links["comparison_report_path"] == str(artifacts.comparison_report_path)
    assert links["comparison_summary_path"] == str(artifacts.comparison_summary_path)
    assert links["comparison_config_used_path"] == str(artifacts.config_used_path)


def test_comparison_delta_table_fail_fast_on_missing_orientation(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel",),
        non_comparable_groups={"boundary": "off", "object_structure": "off"},
        left_metrics=_metric_summary(pixel=0.60, boundary=0.5, gtc=0.36, bde=2.2),
        right_metrics=_metric_summary(pixel=0.62, boundary=0.5, gtc=0.36, bde=2.1),
        ready=False,
        partial=True,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)
    broken = replace(
        pairwise,
        per_group_metric_orientation={"pixel": {}},
    )

    with pytest.raises(ContractError, match="Missing orientation for metric 'extent_iou'"):
        export_pairwise_comparison_artifacts(
            output_dir=tmp_path / "compare_run",
            comparison_run_id="compare_run_broken",
            comparison_contract=contract,
            pairwise_result=broken,
        )

