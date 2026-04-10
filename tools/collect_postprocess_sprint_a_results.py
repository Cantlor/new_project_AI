#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BASELINE_PREDICT_RUN_ID = "target-predict-20260408T133539Z"
BASELINE_POSTPROCESS_RUN_ID = "postprocess-vectorize-20260409T092257Z"
BASELINE_EVAL_RUN_ID = "eval-20260410T070540Z"
EXPERIMENTS = ("A1", "A2", "A3", "A4")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metrics(eval_run_id: str, eval_root: Path) -> dict[str, Any]:
    run_dir = eval_root / eval_run_id
    summary = _load_json(run_dir / "summary.json")
    if summary is None:
        return {
            "status": "missing",
            "extent_f1": None,
            "boundary_f1": None,
            "boundary_bde": None,
            "goc": None,
            "guc": None,
            "gtc": None,
            "ranking_score": None,
            "spurious_pred_count": None,
            "merge_error": None,
            "split_error": None,
        }

    error_tax = _load_json(run_dir / "error_taxonomy.json") or {}

    metric_summary = summary.get("metric_summary", {})
    pixel = metric_summary.get("pixel", {}).get("metrics", {})
    boundary = metric_summary.get("boundary", {}).get("metrics", {})
    object_metrics = metric_summary.get("object_structure", {}).get("metrics", {})
    object_counts = metric_summary.get("object_structure", {}).get("counts", {})

    ranking = summary.get("ranking_summary", {}) or {}
    taxonomy = error_tax.get("taxonomy", {}) if isinstance(error_tax, dict) else {}

    split_error = (
        (taxonomy.get("split_error") or {}).get("count")
        if isinstance(taxonomy, dict)
        else None
    )
    merge_error = (
        (taxonomy.get("merge_error") or {}).get("count")
        if isinstance(taxonomy, dict)
        else None
    )
    spurious_pred_count = (
        (taxonomy.get("spurious_parcel") or {}).get("count")
        if isinstance(taxonomy, dict)
        else None
    )

    # Fallbacks from summary counts if error_taxonomy missing.
    if split_error is None:
        split_error = object_counts.get("split_gt_count")
    if merge_error is None:
        merge_error = object_counts.get("merged_gt_count")
    if spurious_pred_count is None:
        spurious_pred_count = object_counts.get("spurious_pred_count")

    return {
        "status": str(summary.get("status", "unknown")),
        "extent_f1": pixel.get("extent_f1"),
        "boundary_f1": boundary.get("boundary_f1"),
        "boundary_bde": boundary.get("boundary_bde"),
        "goc": object_metrics.get("goc"),
        "guc": object_metrics.get("guc"),
        "gtc": object_metrics.get("gtc"),
        "ranking_score": ranking.get("ranking_score"),
        "spurious_pred_count": spurious_pred_count,
        "merge_error": merge_error,
        "split_error": split_error,
    }


def _row(
    *,
    experiment_id: str,
    predict_run_id: str,
    postprocess_run_id: str,
    eval_run_id: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "predict_run_id": predict_run_id,
        "postprocess_run_id": postprocess_run_id,
        "eval_run_id": eval_run_id,
        "status": metrics["status"],
        "spurious_pred_count": metrics["spurious_pred_count"],
        "merge_error": metrics["merge_error"],
        "split_error": metrics["split_error"],
        "boundary_f1": metrics["boundary_f1"],
        "boundary_bde": metrics["boundary_bde"],
        "goc": metrics["goc"],
        "guc": metrics["guc"],
        "gtc": metrics["gtc"],
        "ranking_score": metrics["ranking_score"],
        "extent_f1": metrics["extent_f1"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Sprint A postprocess experiment metrics vs reused A0 baseline.",
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="Batch ID used by tools/run_postprocess_sprint_a.sh",
    )
    parser.add_argument(
        "--eval-root",
        default="runs/module_eval",
        help="Root directory with eval run folders (default: runs/module_eval)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: runs/experiment_logs/sprint_a/<batch-id>/POSTPROCESS_EXPERIMENT_SUMMARY.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    eval_root = Path(args.eval_root)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else Path("runs/experiment_logs/sprint_a") / args.batch_id / "POSTPROCESS_EXPERIMENT_SUMMARY.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    a0_metrics = _extract_metrics(BASELINE_EVAL_RUN_ID, eval_root)
    rows.append(
        _row(
            experiment_id="A0",
            predict_run_id=BASELINE_PREDICT_RUN_ID,
            postprocess_run_id=BASELINE_POSTPROCESS_RUN_ID,
            eval_run_id=BASELINE_EVAL_RUN_ID,
            metrics=a0_metrics,
        )
    )

    for exp in EXPERIMENTS:
        exp_lower = exp.lower()
        pp_run_id = f"postprocess-sprint-a-{args.batch_id}-{exp_lower}"
        eval_run_id = f"eval-sprint-a-{args.batch_id}-{exp_lower}"
        metrics = _extract_metrics(eval_run_id, eval_root)
        rows.append(
            _row(
                experiment_id=exp,
                predict_run_id=BASELINE_PREDICT_RUN_ID,
                postprocess_run_id=pp_run_id,
                eval_run_id=eval_run_id,
                metrics=metrics,
            )
        )

    fieldnames = [
        "experiment_id",
        "predict_run_id",
        "postprocess_run_id",
        "eval_run_id",
        "status",
        "spurious_pred_count",
        "merge_error",
        "split_error",
        "boundary_f1",
        "boundary_bde",
        "goc",
        "guc",
        "gtc",
        "ranking_score",
        "extent_f1",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] wrote: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
