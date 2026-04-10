"""Unit tests for module_eval minimal pairwise comparison run-level orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import write_manifest
from ai_fields.module_eval.run_compare import (
    PairwiseComparisonRunInputs,
    run_pairwise_comparison,
)

yaml = pytest.importorskip("yaml")


def _write_eval_artifact_set(
    run_dir: Path,
    *,
    run_id: str,
    eval_mode: str = "end_to_end_single_scene",
    pixel_threshold: float = 0.5,
    object_match_rule: str = "iou_or_overlap",
    include_all_groups: bool = True,
    remove_pixel_metric: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_enabled = {
        "pixel": ["extent_iou", "extent_f1", "extent_precision", "extent_recall"],
        "boundary": ["boundary_f1", "boundary_precision", "boundary_recall", "boundary_bde"],
        "object_structure": ["goc", "guc", "gtc"],
    }
    stage_coverage_manifest = {
        "stage_a": True,
        "stage_b": True,
        "stage_c": True,
        "stage_d": include_all_groups,
        "stage_e": True,
        "not_implemented": {
            "comparison_engine": True,
            "cross_run_benchmarking": True,
            "dashboard_reporting_framework": True,
        },
    }
    if not include_all_groups:
        metrics_enabled["object_structure"] = []

    write_manifest(
        run_dir / "eval_manifest.json",
        {
            "schema_name": "eval.eval_manifest",
            "schema_version": "v1",
            "module_name": "module_eval",
            "module_version": None,
            "data_contract_version": DATA_CONTRACT_VERSION,
            "run_id": run_id,
            "stage_name": "export_eval_artifacts",
            "created_at_utc": "2026-04-06T00:00:00Z",
            "status": "success",
            "eval_mode": eval_mode,
            "source_run_ids": [f"pred_{run_id}", f"post_{run_id}"],
            "source_manifest_paths": [f"/tmp/{run_id}/predict_manifest.json"],
            "metrics_enabled": metrics_enabled,
            "stage_coverage": stage_coverage_manifest,
            "thresholds": {
                "raster_binarization": {"extent_prob_threshold": float(pixel_threshold)},
                "vector_matching": {"match_rule": object_match_rule},
                "threshold_provenance": "mixed_explicit_threshold_provenance",
            },
        },
    )

    pixel_metrics: dict[str, Any] = {
        "extent_iou": 0.61,
        "extent_f1": 0.73,
        "extent_precision": 0.71,
        "extent_recall": 0.76,
    }
    if remove_pixel_metric is not None:
        pixel_metrics.pop(remove_pixel_metric, None)

    summary_payload = {
        "schema_name": "eval.summary",
        "status": "success",
        "run_id": run_id,
        "module_name": "module_eval",
        "eval_mode": eval_mode,
        "scene_count": 1,
        "metric_summary": {
            "pixel": {"metrics": pixel_metrics},
            "boundary": {
                "metrics": {
                    "boundary_f1": 0.54,
                    "boundary_precision": 0.52,
                    "boundary_recall": 0.57,
                    "boundary_bde": 2.1,
                }
            },
            "object_structure": {
                "metrics": {
                    "goc": 0.19,
                    "guc": 0.17,
                    "gtc": 0.36,
                    "normalized_gtc": 0.36,
                }
            },
        },
        "stage_coverage": {
            "implemented": ["A", "B", "C", "D", "E"],
            "not_implemented": ["comparison_engine"],
        },
        "warnings": [],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    config_used_payload = {
        "module_name": "module_eval",
        "run_id": run_id,
        "effective_policy_contract": {
            "pixel_policy": {
                "extent_prob_threshold": float(pixel_threshold),
                "prediction_rule": "gte",
                "positive_gt_label": 1,
                "ignore_gt_label": 255,
                "threshold_provenance": "eval_pixel_threshold_v1",
            },
            "boundary_policy": {
                "prediction_interpretation": "argmax_non_background",
                "gt_interpretation": "non_background",
                "threshold_provenance": "eval_boundary_policy_v1",
                "non_background_prob_threshold": 0.5,
                "bde_enabled": True,
            },
            "object_matching_policy": {
                "min_iou_threshold": 0.2,
                "min_overlap_gt_threshold": 0.2,
                "min_overlap_pred_threshold": 0.2,
                "match_rule": object_match_rule,
                "threshold_provenance": "eval_object_matching_v1",
            },
        },
    }
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config_used_payload, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )
    return run_dir


def test_run_pairwise_comparison_happy_path(tmp_path: Path) -> None:
    left_dir = _write_eval_artifact_set(tmp_path / "left_run", run_id="left_eval_run")
    right_dir = _write_eval_artifact_set(tmp_path / "right_run", run_id="right_eval_run")

    run_dir = tmp_path / "runs" / "module_eval" / "compare_run_001"
    result = run_pairwise_comparison(
        run_id="compare_run_001",
        inputs=PairwiseComparisonRunInputs(
            left_run_dir=left_dir,
            right_run_dir=right_dir,
        ),
        output_dir=run_dir,
    )

    assert result.run_dir == run_dir
    assert result.comparison_status == "ready"
    assert result.ready_for_next_stage is True
    assert set(result.comparable_metric_groups) == {"pixel", "boundary", "object_structure"}
    assert result.non_comparable_metric_groups == {}

    assert result.comparison_report_path == run_dir / "comparison_report.json"
    assert result.comparison_summary_path == run_dir / "comparison_summary.json"
    assert result.comparison_delta_table_path == run_dir / "comparison_delta_table.json"
    assert result.config_used_path == run_dir / "config_used.yaml"
    assert result.comparison_report_path.exists()
    assert result.comparison_summary_path.exists()
    assert result.comparison_delta_table_path.exists()
    assert result.config_used_path.exists()


def test_run_pairwise_comparison_fail_fast_on_broken_contract(tmp_path: Path) -> None:
    left_dir = _write_eval_artifact_set(tmp_path / "left_run", run_id="left_eval_run")
    _ = _write_eval_artifact_set(tmp_path / "right_run", run_id="right_eval_run")

    broken_right_dir = tmp_path / "right_run"
    (broken_right_dir / "summary.json").unlink()

    with pytest.raises(ContractError, match="right_summary_path does not exist"):
        run_pairwise_comparison(
            run_id="compare_run_fail_contract",
            inputs=PairwiseComparisonRunInputs(
                left_run_dir=left_dir,
                right_run_dir=broken_right_dir,
            ),
            output_dir=tmp_path / "runs" / "module_eval" / "compare_run_fail_contract",
        )


def test_run_pairwise_comparison_fail_fast_on_pairwise_stage(tmp_path: Path) -> None:
    left_dir = _write_eval_artifact_set(tmp_path / "left_run", run_id="left_eval_run")
    right_dir = _write_eval_artifact_set(
        tmp_path / "right_run",
        run_id="right_eval_run",
        remove_pixel_metric="extent_f1",
    )

    with pytest.raises(ContractError, match="requires metric 'extent_f1'"):
        run_pairwise_comparison(
            run_id="compare_run_fail_pairwise",
            inputs=PairwiseComparisonRunInputs(
                left_run_dir=left_dir,
                right_run_dir=right_dir,
            ),
            output_dir=tmp_path / "runs" / "module_eval" / "compare_run_fail_pairwise",
        )


def test_run_pairwise_comparison_fail_fast_on_export_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    left_dir = _write_eval_artifact_set(tmp_path / "left_run", run_id="left_eval_run")
    right_dir = _write_eval_artifact_set(tmp_path / "right_run", run_id="right_eval_run")

    def _boom(**_: Any):
        raise ContractError("forced export failure")

    monkeypatch.setattr("ai_fields.module_eval.run_compare.export_pairwise_comparison_artifacts", _boom)

    with pytest.raises(ContractError, match="forced export failure"):
        run_pairwise_comparison(
            run_id="compare_run_fail_export",
            inputs=PairwiseComparisonRunInputs(
                left_run_dir=left_dir,
                right_run_dir=right_dir,
            ),
            output_dir=tmp_path / "runs" / "module_eval" / "compare_run_fail_export",
        )


def test_run_pairwise_comparison_partial_propagation_and_provenance(tmp_path: Path) -> None:
    left_dir = _write_eval_artifact_set(
        tmp_path / "left_run",
        run_id="left_eval_run",
        object_match_rule="iou_or_overlap",
    )
    right_dir = _write_eval_artifact_set(
        tmp_path / "right_run",
        run_id="right_eval_run",
        object_match_rule="iou_only",
    )

    run_dir = tmp_path / "runs" / "module_eval" / "compare_run_partial"
    result = run_pairwise_comparison(
        run_id="compare_run_partial",
        inputs=PairwiseComparisonRunInputs(
            left_run_dir=left_dir,
            right_run_dir=right_dir,
        ),
        output_dir=run_dir,
    )

    assert result.comparison_status == "partial"
    assert result.ready_for_next_stage is True
    assert set(result.comparable_metric_groups) == {"pixel", "boundary"}
    assert "object_structure" in result.non_comparable_metric_groups
    assert "object match_rule mismatch" in result.non_comparable_metric_groups["object_structure"]

    report = json.loads(result.comparison_report_path.read_text(encoding="utf-8"))
    assert report["left_run"]["run_id"] == "left_eval_run"
    assert report["right_run"]["run_id"] == "right_eval_run"
    assert report["comparison_contract"]["non_comparable_metric_groups"]["object_structure"] == "object match_rule mismatch"
    assert "object_structure" not in report["pairwise_result"]["metric_deltas"]
