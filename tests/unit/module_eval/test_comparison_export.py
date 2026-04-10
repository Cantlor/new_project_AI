"""Unit tests for module_eval minimal comparison artifact/report export layer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_eval.comparison_contract import EvalComparisonInputContractResult
from ai_fields.module_eval.comparison_export import (
    export_pairwise_comparison_artifacts,
)
from ai_fields.module_eval.pairwise_comparison import (
    compute_pairwise_eval_comparison,
)

yaml = pytest.importorskip("yaml")


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
                "extent_iou": 0.61 + pixel_offset,
                "extent_f1": 0.73 + pixel_offset,
                "extent_precision": 0.71 + pixel_offset,
                "extent_recall": 0.76 + pixel_offset,
            }
        },
        "boundary": {
            "metrics": {
                "boundary_f1": 0.54 + boundary_offset,
                "boundary_precision": 0.52 + boundary_offset,
                "boundary_recall": 0.57 + boundary_offset,
                "boundary_bde": boundary_bde,
            }
        },
        "object_structure": {
            "metrics": {
                "goc": 0.19 + object_offset,
                "guc": 0.17 + object_offset,
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
        policy_compatibility_summary={"pixel": {"compatible": True, "reason": "compatible"}},
        provenance_compatibility_summary={
            "left_source_run_ids": ("pred_left",),
            "right_source_run_ids": ("pred_right",),
        },
        stage_coverage_compatibility_summary={
            "pixel": {"left_available": True, "right_available": True}
        },
        global_blockers=(),
        ready_for_pairwise_compare=ready,
        partially_ready=partial,
    )


def test_export_pairwise_comparison_artifacts_happy_path(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel", "boundary", "object_structure"),
        non_comparable_groups={},
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(
            pixel_offset=0.03,
            boundary_offset=0.04,
            object_offset=-0.02,
            boundary_bde=1.8,
            normalized_gtc=0.27,
        ),
        ready=True,
        partial=False,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)

    artifacts = export_pairwise_comparison_artifacts(
        output_dir=tmp_path / "comparison_run",
        comparison_run_id="comparison_run_001",
        comparison_contract=contract,
        pairwise_result=pairwise,
    )

    assert artifacts.comparison_report_path.exists()
    assert artifacts.comparison_summary_path.exists()
    assert artifacts.comparison_delta_table_path.exists()
    assert artifacts.config_used_path.exists()

    report = json.loads(artifacts.comparison_report_path.read_text(encoding="utf-8"))
    assert report["schema_name"] == "eval.comparison_report"
    assert report["schema_version"] == "v1"
    assert report["pairwise_result"]["delta_direction"] == "right_minus_left"
    assert report["pairwise_result"]["metric_deltas"]["pixel"]["extent_iou_delta"] == pytest.approx(0.03)
    assert report["pairwise_result"]["metric_orientation"]["boundary"]["boundary_bde"] == "lower_is_better"
    assert report["stage_coverage"]["not_implemented"]["leaderboard"] is True
    assert {
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
    }.issubset(set(report))

    summary = json.loads(artifacts.comparison_summary_path.read_text(encoding="utf-8"))
    assert summary["schema_name"] == "eval.comparison_summary"
    assert summary["comparison_status"] == "ready"
    assert summary["metric_delta_counts"]["pixel"] >= 4
    assert {
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
    }.issubset(set(summary))

    delta_table = json.loads(artifacts.comparison_delta_table_path.read_text(encoding="utf-8"))
    assert delta_table["schema_name"] == "eval.comparison_delta_table"
    assert delta_table["delta_direction"] == "right_minus_left"
    assert delta_table["rows"]
    assert {
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
    }.issubset(set(delta_table))

    cfg = yaml.safe_load(artifacts.config_used_path.read_text(encoding="utf-8"))
    assert cfg["effective_policy_contract"]["delta_direction"] == "right_minus_left"
    assert cfg["effective_policy_contract"]["no_silent_fallback"] is True


def test_export_pairwise_comparison_artifacts_partial_comparability(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel", "boundary"),
        non_comparable_groups={"object_structure": "object match_rule mismatch"},
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(pixel_offset=0.01, boundary_offset=0.02),
        ready=False,
        partial=True,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)

    artifacts = export_pairwise_comparison_artifacts(
        output_dir=tmp_path / "comparison_run",
        comparison_run_id="comparison_run_partial",
        comparison_contract=contract,
        pairwise_result=pairwise,
    )
    report = json.loads(artifacts.comparison_report_path.read_text(encoding="utf-8"))

    assert report["status"] == "partial"
    assert report["comparison_contract"]["non_comparable_metric_groups"]["object_structure"] == "object match_rule mismatch"
    assert "object_structure" not in report["pairwise_result"]["metric_deltas"]
    assert "partial" in report["warnings"][0]

    delta_table = json.loads(artifacts.comparison_delta_table_path.read_text(encoding="utf-8"))
    assert set(delta_table["comparable_metric_groups"]) == {"pixel", "boundary"}
    assert delta_table["non_comparable_metric_groups"]["object_structure"] == "object match_rule mismatch"


def test_export_pairwise_comparison_artifacts_mismatched_result_fails(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel",),
        non_comparable_groups={"boundary": "off", "object_structure": "off"},
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(pixel_offset=0.01),
        ready=False,
        partial=True,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)
    broken_pairwise = replace(pairwise, left_run_id="different_left_id")

    with pytest.raises(ContractError, match="left_run_id mismatch"):
        export_pairwise_comparison_artifacts(
            output_dir=tmp_path / "comparison_run",
            comparison_run_id="comparison_run_fail",
            comparison_contract=contract,
            pairwise_result=broken_pairwise,
        )


def test_export_pairwise_comparison_artifacts_provenance_continuity(tmp_path: Path) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel",),
        non_comparable_groups={"boundary": "off", "object_structure": "off"},
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(pixel_offset=0.02),
        ready=False,
        partial=True,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)

    artifacts = export_pairwise_comparison_artifacts(
        output_dir=tmp_path / "comparison_run",
        comparison_run_id="comparison_run_provenance",
        comparison_contract=contract,
        pairwise_result=pairwise,
    )
    report = json.loads(artifacts.comparison_report_path.read_text(encoding="utf-8"))

    assert report["left_run"]["run_id"] == "left_eval_run"
    assert report["right_run"]["run_id"] == "right_eval_run"
    assert report["left_run"]["summary_path"] == str(contract.left_summary_path)
    assert report["right_run"]["summary_path"] == str(contract.right_summary_path)
    assert "provenance_compatibility_summary" in report


def test_export_pairwise_comparison_artifacts_fail_fast_on_incomplete_pairwise_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel",),
        non_comparable_groups={"boundary": "off", "object_structure": "off"},
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(pixel_offset=0.02),
        ready=False,
        partial=True,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)

    def _broken_summary(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "left_run_id": "left_eval_run",
            "right_run_id": "right_eval_run",
            "delta_direction": "right_minus_left",
            "comparison_status": "partial",
            "ready_for_pairwise_compare": False,
            "partially_ready": True,
            "comparable_metric_groups": ["pixel"],
            "non_comparable_metric_groups": {"boundary": "off", "object_structure": "off"},
            "metric_deltas": {"pixel": {"extent_iou_delta": 0.02}},
            # metric_orientation intentionally missing
            "skipped_metric_reasons": {},
        }

    monkeypatch.setattr(
        "ai_fields.module_eval.comparison_export.build_pairwise_comparison_summary",
        _broken_summary,
    )

    with pytest.raises(ContractError, match="pairwise_summary payload is missing required fields"):
        export_pairwise_comparison_artifacts(
            output_dir=tmp_path / "comparison_run",
            comparison_run_id="comparison_run_broken_pairwise_summary",
            comparison_contract=contract,
            pairwise_result=pairwise,
        )


def test_export_pairwise_comparison_artifacts_fail_fast_on_incomplete_delta_table_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _make_contract(
        tmp_path=tmp_path / "contract",
        comparable_groups=("pixel",),
        non_comparable_groups={"boundary": "off", "object_structure": "off"},
        left_metrics=_default_metric_summary(),
        right_metrics=_default_metric_summary(pixel_offset=0.01),
        ready=False,
        partial=True,
    )
    pairwise = compute_pairwise_eval_comparison(comparison_contract=contract)

    def _broken_table(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "schema_name": "eval.comparison_delta_table",
            "status": "partial",
            "comparison_mode": "pairwise",
            "run_id": "comparison_run_broken",
            "left_run_id": "left_eval_run",
            "right_run_id": "right_eval_run",
            "delta_direction": "right_minus_left",
            "comparison_status": "partial",
            "comparable_metric_groups": ["pixel"],
            "non_comparable_metric_groups": {"boundary": "off", "object_structure": "off"},
            # rows intentionally missing
            "provenance_links": {},
            "policy_compatibility_summary": {},
            "provenance_compatibility_summary": {},
            "stage_coverage_compatibility_summary": {},
            "stage_honesty": {"not_implemented": {}},
        }

    monkeypatch.setattr(
        "ai_fields.module_eval.comparison_export.build_comparison_delta_table",
        _broken_table,
    )

    with pytest.raises(
        ContractError,
        match="comparison_delta_table payload is missing required fields",
    ):
        export_pairwise_comparison_artifacts(
            output_dir=tmp_path / "comparison_run",
            comparison_run_id="comparison_run_broken_delta_table",
            comparison_contract=contract,
            pairwise_result=pairwise,
        )
