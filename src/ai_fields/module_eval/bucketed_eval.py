"""Stage D.5 bucketed object/structure evaluation for module_eval.

Implements mandatory bucketed evaluation (module_eval §10, §20.5).

Architectural decision on bucket thresholds (§21.2 — previously open question):
  Baseline v1 uses area-based size buckets:
  - In projected CRS (area in m²):   small < 5000, medium [5000, 50000), large ≥ 50000
  - In pixel space (pixel count):     small < 200,  medium [200, 2000),   large ≥ 2000

  These correspond to ~3 m/px resolution fields:
  - small: scattered/fragment plots (< ~0.5 ha)
  - medium: typical agricultural parcels (0.5–5 ha)
  - large: large consolidated fields (> 5 ha)

  Thresholds are fully configurable via BucketSizePolicy; defaults match the above decision.

Output: metrics_by_bucket.json

Out of scope:
- shape-complexity or boundary-difficulty bucketing (future §10.2 extensions),
- cross-run bucket comparison,
- report/comparison framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError
from ai_fields.module_eval.input_contract import EvaluationInputContractResult
from ai_fields.module_eval.object_metrics import ObjectMatchingPolicy, _validate_policy

_EPS = 1e-9
_RELATION_EDGE_MAX_COUNT = 50_000_000

# ---------------------------------------------------------------------------
# Default bucket thresholds (architectural decision §21.2 baseline v1)
# ---------------------------------------------------------------------------

_DEFAULT_PIXEL_THRESHOLDS = (200, 2000)   # (small_max, medium_max) in pixels
_DEFAULT_M2_THRESHOLDS = (5_000, 50_000)  # (small_max, medium_max) in m²


@dataclass(frozen=True)
class BucketSizePolicy:
    """Configuration for area-based size bucketing.

    Attributes
    ----------
    threshold_provenance:
        Identifier of who calibrated / chose these thresholds.
    use_projected_area:
        If True and CRS is projected, compute areas in m²; otherwise use pixel counts.
    small_max_pixels:
        Upper bound (exclusive) for 'small' bucket in pixel count.
    medium_max_pixels:
        Upper bound (exclusive) for 'medium' bucket in pixel count.
    small_max_m2:
        Upper bound (exclusive) for 'small' bucket in m² (used when use_projected_area=True).
    medium_max_m2:
        Upper bound (exclusive) for 'medium' bucket in m² (used when use_projected_area=True).
    """

    threshold_provenance: str
    use_projected_area: bool = True
    small_max_pixels: int = _DEFAULT_PIXEL_THRESHOLDS[0]
    medium_max_pixels: int = _DEFAULT_PIXEL_THRESHOLDS[1]
    small_max_m2: float = float(_DEFAULT_M2_THRESHOLDS[0])
    medium_max_m2: float = float(_DEFAULT_M2_THRESHOLDS[1])


@dataclass(frozen=True)
class BucketMetrics:
    """GOC/GUC/GTC metrics for one bucket."""

    bucket_name: str
    gt_object_count: int
    split_gt_count: int
    merged_gt_count: int
    unmatched_gt_count: int
    goc: float
    guc: float
    gtc: float
    normalized_gtc: float


@dataclass(frozen=True)
class BucketedEvalResult:
    """Stage D.5 result contract for bucketed object/structure metrics."""

    buckets: list[BucketMetrics]
    area_unit: str  # "pixels" or "m2"
    pixel_size_m2: float | None  # pixel area in m² when area_unit == "m2"
    bucket_thresholds: dict[str, Any]
    threshold_provenance: str
    stage_scope: str


def _require_rasterio() -> tuple[Any, Any, Any]:
    try:
        import rasterio
        import rasterio.errors
        import rasterio.features
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for module_eval Stage D.5.") from exc
    return rasterio, rasterio.errors, rasterio.features


def _require_affine() -> Any:
    try:
        from affine import Affine
    except ImportError as exc:  # pragma: no cover
        raise ContractError("affine is required for module_eval Stage D.5.") from exc
    return Affine



def _validate_bucket_policy(policy: BucketSizePolicy) -> None:
    if not isinstance(policy, BucketSizePolicy):
        raise ContractError("policy must be BucketSizePolicy.")
    if not isinstance(policy.threshold_provenance, str) or policy.threshold_provenance.strip() == "":
        raise ContractError("policy.threshold_provenance must be a non-empty string.")
    if policy.small_max_pixels <= 0 or policy.medium_max_pixels <= policy.small_max_pixels:
        raise ContractError(
            "BucketSizePolicy pixel thresholds must satisfy 0 < small_max < medium_max."
        )
    if policy.small_max_m2 <= 0 or policy.medium_max_m2 <= policy.small_max_m2:
        raise ContractError(
            "BucketSizePolicy m2 thresholds must satisfy 0 < small_max < medium_max."
        )


def _resolve_area_unit_and_pixel_size(
    *,
    input_contract: EvaluationInputContractResult,
    policy: BucketSizePolicy,
) -> tuple[str, float | None]:
    """Return (area_unit, pixel_size_m2|None)."""
    if not policy.use_projected_area:
        return "pixels", None

    rasterio, _, _ = _require_rasterio()
    try:
        crs = rasterio.crs.CRS.from_user_input(input_contract.common_crs)
    except Exception as exc:
        raise ContractError(f"Failed to parse common CRS for bucketed eval: {exc}") from exc

    if crs.is_projected:
        gt = input_contract.common_transform_gdal
        x_size = abs(float(gt[1]))
        y_size = abs(float(gt[5]))
        if x_size <= _EPS or y_size <= _EPS:
            raise ContractError(
                "Cannot compute projected area for bucketed eval: invalid pixel size."
            )
        return "m2", x_size * y_size

    return "pixels", None


def _assign_bucket(
    area: float,
    *,
    area_unit: str,
    policy: BucketSizePolicy,
) -> str:
    if area_unit == "m2":
        if area < policy.small_max_m2:
            return "small"
        if area < policy.medium_max_m2:
            return "medium"
        return "large"
    # pixels
    if area < policy.small_max_pixels:
        return "small"
    if area < policy.medium_max_pixels:
        return "medium"
    return "large"


def _safe_div(num: float, den: float) -> float:
    if den <= _EPS:
        return 0.0
    return float(num / den)


def _guard_relation_edge_count(*, edge_count: int) -> None:
    if edge_count <= _RELATION_EDGE_MAX_COUNT:
        return
    approx_bytes = edge_count * np.dtype(np.int32).itemsize
    raise ContractError(
        "Bucketed eval sparse relation edges exceed memory safety guard: "
        f"edge_count={edge_count}, approx_index_bytes={approx_bytes}, "
        f"max_edges={_RELATION_EDGE_MAX_COUNT}."
    )


def _compute_bucket_metrics(
    *,
    bucket_name: str,
    gt_indices: list[int],
    gt_to_pred_relations: list[set[int]],
) -> BucketMetrics:
    """Compute GOC/GUC/GTC for a subset of GT objects identified by gt_indices."""
    n = len(gt_indices)
    if n == 0:
        return BucketMetrics(
            bucket_name=bucket_name,
            gt_object_count=0,
            split_gt_count=0,
            merged_gt_count=0,
            unmatched_gt_count=0,
            goc=0.0,
            guc=0.0,
            gtc=0.0,
            normalized_gtc=0.0,
        )

    gt_match_counts: list[int] = []
    pred_match_counts: dict[int, int] = {}
    for gi in gt_indices:
        matches = gt_to_pred_relations[gi]
        gt_match_counts.append(len(matches))
        for pi in matches:
            pred_match_counts[pi] = pred_match_counts.get(pi, 0) + 1

    merged_pred_indices = {pi for pi, c in pred_match_counts.items() if c >= 2}
    split_gt_count = sum(1 for c in gt_match_counts if c >= 2)
    matched_gt_count = sum(1 for c in gt_match_counts if c >= 1)
    merged_gt_count = sum(
        1 for gi in gt_indices if any((pi in merged_pred_indices) for pi in gt_to_pred_relations[gi])
    )
    unmatched_gt_count = int(n - matched_gt_count)

    goc = _safe_div(split_gt_count, n)
    guc = _safe_div(merged_gt_count, n)
    gtc = goc + guc

    return BucketMetrics(
        bucket_name=bucket_name,
        gt_object_count=n,
        split_gt_count=split_gt_count,
        merged_gt_count=merged_gt_count,
        unmatched_gt_count=unmatched_gt_count,
        goc=goc,
        guc=guc,
        gtc=gtc,
        normalized_gtc=min(1.0, gtc),
    )


def compute_bucketed_object_metrics(
    *,
    input_contract: EvaluationInputContractResult,
    object_policy: ObjectMatchingPolicy,
    bucket_policy: BucketSizePolicy,
) -> BucketedEvalResult:
    """Compute Stage D.5 bucketed GOC/GUC/GTC per object-size bucket.

    Re-reads the same rasters and vector files as compute_object_structure_metrics(),
    partitions GT objects by area bucket, and computes per-bucket metrics.

    Memory policy: uses window-based rasterization (same as object_metrics.py) — no
    full-scene masks are stored for any geometry.  Candidate pairs are pruned via
    STRtree before windowed intersection is computed.
    """
    if not isinstance(input_contract, EvaluationInputContractResult):
        raise ContractError(
            "input_contract must be EvaluationInputContractResult from Stage A."
        )
    _validate_policy(object_policy)
    _validate_bucket_policy(bucket_policy)

    if input_contract.track_readiness.object_structure_ready is not True:
        reason = (
            input_contract.track_readiness.object_structure_reason or "unknown reason"
        )
        raise ContractError(
            f"Bucketed eval cannot run because object track is not ready: {reason}."
        )

    if input_contract.gt_parcels is None or input_contract.post_parcels_vector is None:
        raise ContractError(
            "Bucketed eval requires both gt_parcels and post_parcels_vector."
        )

    # Import window-based helpers from object_metrics (avoids full-scene mask storage)
    from ai_fields.module_eval.object_metrics import (
        _build_effective_objects,
        _build_spatial_index,
        _intersection_area_on_valid,
        _query_candidate_indices,
        _read_valid_binary,
        _read_vector_geometries,
        _relation_ok,
    )

    Affine = _require_affine()

    out_shape = (int(input_contract.common_height), int(input_contract.common_width))
    transform = Affine.from_gdal(*input_contract.common_transform_gdal)

    # Read valid mask (uint8 0/1; bool-compatible for logical_and)
    valid_mask = _read_valid_binary(
        input_contract.gt_valid.path, expected_shape=out_shape
    )

    # Read geometries as Shapely shapes (required by windowed helpers)
    gt_geoms_all = _read_vector_geometries(
        input_contract.gt_parcels.path, role="gt_parcels"
    )
    pred_geoms_all = _read_vector_geometries(
        input_contract.post_parcels_vector.path, role="post_parcels_vector"
    )

    # Build effective objects via windowed rasterization — only stores (geometry, area)
    # per object, not full-scene masks.
    gt_objects, _ = _build_effective_objects(
        geometries=gt_geoms_all,
        valid_mask=valid_mask,
        out_shape=out_shape,
        transform=transform,
    )
    pred_objects, _ = _build_effective_objects(
        geometries=pred_geoms_all,
        valid_mask=valid_mask,
        out_shape=out_shape,
        transform=transform,
    )

    gt_count = len(gt_objects)
    pred_count = len(pred_objects)

    if gt_count == 0 or pred_count == 0:
        # Return empty buckets — consistent with object_metrics empty_object_handling
        empty_buckets = [
            BucketMetrics(bucket_name=name, gt_object_count=0, split_gt_count=0,
                          merged_gt_count=0, unmatched_gt_count=0,
                          goc=0.0, guc=0.0, gtc=0.0, normalized_gtc=0.0)
            for name in ("small", "medium", "large")
        ]
        area_unit, pixel_size_m2 = _resolve_area_unit_and_pixel_size(
            input_contract=input_contract, policy=bucket_policy,
        )
        return BucketedEvalResult(
            buckets=empty_buckets,
            area_unit=area_unit,
            pixel_size_m2=pixel_size_m2,
            bucket_thresholds=_bucket_thresholds_dict(bucket_policy, area_unit),
            threshold_provenance=bucket_policy.threshold_provenance,
            stage_scope="stage_d5_bucketed_eval",
        )

    gt_pixel_areas = np.array([obj.area for obj in gt_objects], dtype=np.int64)

    area_unit, pixel_size_m2 = _resolve_area_unit_and_pixel_size(
        input_contract=input_contract, policy=bucket_policy,
    )

    if area_unit == "m2" and pixel_size_m2 is not None:
        gt_areas_for_bucket = gt_pixel_areas.astype(np.float64) * pixel_size_m2
    else:
        gt_areas_for_bucket = gt_pixel_areas.astype(np.float64)

    # Build spatial index on pred geometries for O(log N) candidate pruning
    pred_geoms = [obj.geometry for obj in pred_objects]
    pred_areas = np.array([obj.area for obj in pred_objects], dtype=np.int64)
    pred_tree, pred_wkb_to_index = _build_spatial_index(pred_geoms)

    # Build sparse relations using windowed intersection — no dense O(N*M) relation matrix.
    gt_to_pred_relations: list[set[int]] = [set() for _ in range(gt_count)]
    relation_edge_count = 0
    for gi, gt_obj in enumerate(gt_objects):
        gt_area = gt_obj.area
        candidate_indices = _query_candidate_indices(
            tree=pred_tree,
            wkb_to_index=pred_wkb_to_index,
            geometry=gt_obj.geometry,
        )
        for pi in candidate_indices:
            pred_area = int(pred_areas[pi])
            inter = _intersection_area_on_valid(
                gt_geometry=gt_obj.geometry,
                pred_geometry=pred_objects[pi].geometry,
                valid_mask=valid_mask,
                out_shape=out_shape,
                transform=transform,
            )
            if inter <= 0:
                continue
            union = gt_area + pred_area - inter
            iou = _safe_div(inter, union)
            overlap_gt = _safe_div(inter, gt_area)
            overlap_pred = _safe_div(inter, pred_area)
            relation_ok = _relation_ok(
                iou=iou,
                overlap_gt=overlap_gt,
                overlap_pred=overlap_pred,
                policy=object_policy,
            )
            if relation_ok:
                gt_to_pred_relations[gi].add(int(pi))
                relation_edge_count += 1
                _guard_relation_edge_count(edge_count=relation_edge_count)

    # Partition GT objects by bucket
    bucket_to_gt_indices: dict[str, list[int]] = {"small": [], "medium": [], "large": []}
    for gi, area in enumerate(gt_areas_for_bucket):
        bucket = _assign_bucket(float(area), area_unit=area_unit, policy=bucket_policy)
        bucket_to_gt_indices[bucket].append(gi)

    buckets = [
        _compute_bucket_metrics(
            bucket_name=name,
            gt_indices=bucket_to_gt_indices[name],
            gt_to_pred_relations=gt_to_pred_relations,
        )
        for name in ("small", "medium", "large")
    ]

    return BucketedEvalResult(
        buckets=buckets,
        area_unit=area_unit,
        pixel_size_m2=pixel_size_m2,
        bucket_thresholds=_bucket_thresholds_dict(bucket_policy, area_unit),
        threshold_provenance=bucket_policy.threshold_provenance,
        stage_scope="stage_d5_bucketed_eval",
    )


def _bucket_thresholds_dict(policy: BucketSizePolicy, area_unit: str) -> dict[str, Any]:
    if area_unit == "m2":
        return {
            "area_unit": "m2",
            "small_max": policy.small_max_m2,
            "medium_max": policy.medium_max_m2,
        }
    return {
        "area_unit": "pixels",
        "small_max": policy.small_max_pixels,
        "medium_max": policy.medium_max_pixels,
    }


def write_bucketed_eval_artifact(
    path: Path,
    *,
    run_id: str,
    eval_mode: str,
    result: BucketedEvalResult,
) -> None:
    """Write metrics_by_bucket.json (module_eval §14.1)."""
    payload: dict[str, Any] = {
        "schema_name": "eval.metrics_by_bucket",
        "schema_version": "v1",
        "run_id": run_id,
        "eval_mode": eval_mode,
        "stage_scope": result.stage_scope,
        "area_unit": result.area_unit,
        "pixel_size_m2": result.pixel_size_m2,
        "bucket_thresholds": result.bucket_thresholds,
        "threshold_provenance": result.threshold_provenance,
        "buckets": [
            {
                "bucket_name": b.bucket_name,
                "gt_object_count": b.gt_object_count,
                "split_gt_count": b.split_gt_count,
                "merged_gt_count": b.merged_gt_count,
                "unmatched_gt_count": b.unmatched_gt_count,
                "goc": b.goc,
                "guc": b.guc,
                "gtc": b.gtc,
                "normalized_gtc": b.normalized_gtc,
            }
            for b in result.buckets
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"Failed to write metrics_by_bucket.json at {path}: {exc}") from exc


def build_bucketed_eval_summary(result: BucketedEvalResult) -> dict[str, Any]:
    """Build compact summary for bucketed eval (for embedding in eval manifest/summary)."""
    return {
        "stage_scope": result.stage_scope,
        "area_unit": result.area_unit,
        "bucket_thresholds": result.bucket_thresholds,
        "buckets": {
            b.bucket_name: {
                "gt_object_count": b.gt_object_count,
                "goc": b.goc,
                "guc": b.guc,
                "gtc": b.gtc,
            }
            for b in result.buckets
        },
    }


__all__ = [
    "BucketSizePolicy",
    "BucketMetrics",
    "BucketedEvalResult",
    "build_bucketed_eval_summary",
    "compute_bucketed_object_metrics",
    "write_bucketed_eval_artifact",
]
