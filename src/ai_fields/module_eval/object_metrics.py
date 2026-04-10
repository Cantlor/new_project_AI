"""Stage D minimal object/structure metrics for module_eval.

Scope of this layer is intentionally narrow:
- compute baseline object/structure metrics (GOC, GUC, GTC);
- use explicit overlap/IoU matching policy;
- run only on top of Stage A resolved EvaluationInputContractResult.

Out of scope:
- boundary/pixel metrics,
- report/comparison framework,
- eval runtime orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError, ValidPolicyError
from ai_fields.module_eval.input_contract import EvaluationInputContractResult

_EPS = 1e-9


@dataclass(frozen=True)
class ObjectMatchingPolicy:
    """Explicit matching policy for Stage D object/structure metrics."""

    threshold_provenance: str

    min_iou_threshold: float = 0.1
    min_overlap_gt_threshold: float = 0.1
    min_overlap_pred_threshold: float = 0.1
    match_rule: str = "iou_or_overlap"

    empty_object_handling: str = "explicit_error"


@dataclass(frozen=True)
class ObjectStructureMetricsResult:
    """Stage D result contract for minimal object/structure metrics."""

    goc: float
    guc: float
    gtc: float
    normalized_gtc: float

    gt_object_count: int
    pred_object_count: int

    matched_gt_count: int
    matched_pred_count: int
    split_gt_count: int
    merged_gt_count: int
    unmatched_gt_count: int
    spurious_pred_count: int

    gt_excluded_zero_valid_count: int
    pred_excluded_zero_valid_count: int

    threshold_provenance: str
    min_iou_threshold: float
    min_overlap_gt_threshold: float
    min_overlap_pred_threshold: float
    match_rule: str
    stage_scope: str


def _require_rasterio() -> tuple[Any, Any, Any]:
    try:
        import rasterio
        import rasterio.errors
        import rasterio.features
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for module_eval Stage D.") from exc
    return rasterio, rasterio.errors, rasterio.features


def _require_affine() -> Any:
    try:
        from affine import Affine
    except ImportError as exc:  # pragma: no cover
        raise ContractError("affine is required for module_eval Stage D.") from exc
    return Affine


def _require_fiona() -> Any:
    try:
        import fiona
    except ImportError as exc:  # pragma: no cover
        raise ContractError("fiona is required for module_eval Stage D.") from exc
    return fiona


def _require_shapely() -> tuple[Any, Any, Any]:
    try:
        from shapely.geometry import mapping, shape
        from shapely.strtree import STRtree
    except ImportError as exc:  # pragma: no cover
        raise ContractError("shapely is required for module_eval Stage D.") from exc
    return mapping, shape, STRtree


def _validate_policy(policy: ObjectMatchingPolicy) -> ObjectMatchingPolicy:
    if not isinstance(policy, ObjectMatchingPolicy):
        raise ContractError("policy must be ObjectMatchingPolicy for Stage D.")
    if not isinstance(policy.threshold_provenance, str) or policy.threshold_provenance.strip() == "":
        raise ContractError("policy.threshold_provenance must be a non-empty string.")

    for name, value in (
        ("min_iou_threshold", policy.min_iou_threshold),
        ("min_overlap_gt_threshold", policy.min_overlap_gt_threshold),
        ("min_overlap_pred_threshold", policy.min_overlap_pred_threshold),
    ):
        f = float(value)
        if f < 0.0 or f > 1.0:
            raise ContractError(f"policy.{name} must be in [0,1], got {value!r}.")

    if policy.match_rule not in {
        "iou_or_overlap",
        "iou_and_overlap",
        "iou_only",
        "overlap_only",
    }:
        raise ContractError(
            "Unsupported policy.match_rule. Expected one of "
            "{'iou_or_overlap','iou_and_overlap','iou_only','overlap_only'}."
        )

    if policy.empty_object_handling not in {"explicit_error", "zero_if_both_empty"}:
        raise ContractError(
            "Unsupported policy.empty_object_handling. "
            "Expected one of {'explicit_error','zero_if_both_empty'}."
        )
    return policy


def _read_valid_binary(path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    rasterio, rasterio_errors, _ = _require_rasterio()
    try:
        with rasterio.open(path) as ds:
            arr = ds.read()
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read gt_valid raster values: {path} ({exc})") from exc

    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ContractError(f"gt_valid must be single-band, got shape={arr.shape}.")
    valid = arr[0]
    if valid.shape != expected_shape:
        raise ContractError(
            f"gt_valid shape mismatch: expected {expected_shape}, got {valid.shape}."
        )
    if not np.isfinite(valid).all():
        raise ContractError("gt_valid contains non-finite values.")

    unique = np.unique(valid)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValidPolicyError(
            "gt_valid must be binary {0,1} for Stage D object metrics, "
            f"got unique values {unique.tolist()}."
        )
    return (valid > 0).astype(np.uint8)


def _read_vector_geometries(path: Path, *, role: str) -> list[Any]:
    fiona = _require_fiona()
    _, shape, _ = _require_shapely()
    out: list[Any] = []
    with fiona.open(path) as src:
        for feat in src:
            geom = feat.get("geometry")
            if geom is None:
                continue
            shp = shape(geom)
            if shp.is_empty:
                continue
            out.append(shp)
    if not out:
        raise ContractError(f"{role} vector source contains zero usable geometries.")
    return out


def _bounds_to_window(
    *,
    bounds: tuple[float, float, float, float],
    out_shape: tuple[int, int],
    transform: Any,
) -> tuple[int, int, int, int] | None:
    rasterio, _, _ = _require_rasterio()
    left, bottom, right, top = bounds
    if not np.isfinite([left, bottom, right, top]).all():
        return None
    if right <= left or top <= bottom:
        return None

    try:
        window = rasterio.windows.from_bounds(
            left,
            bottom,
            right,
            top,
            transform=transform,
        )
    except Exception:
        return None

    row0 = int(np.floor(window.row_off))
    col0 = int(np.floor(window.col_off))
    row1 = int(np.ceil(window.row_off + window.height))
    col1 = int(np.ceil(window.col_off + window.width))

    h, w = out_shape
    row0 = max(0, min(h, row0))
    row1 = max(0, min(h, row1))
    col0 = max(0, min(w, col0))
    col1 = max(0, min(w, col1))
    if row1 <= row0 or col1 <= col0:
        return None
    return row0, row1, col0, col1


def _rasterize_geometry_window(
    *,
    geometry: Any,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    transform: Any,
) -> np.ndarray:
    _, _, rfeatures = _require_rasterio()
    mapping, _, _ = _require_shapely()
    Affine = _require_affine()
    window_transform = transform * Affine.translation(col0, row0)
    arr = rfeatures.rasterize(
        [(mapping(geometry), 1)],
        out_shape=(row1 - row0, col1 - col0),
        transform=window_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    return arr.astype(bool)


@dataclass(frozen=True)
class _EffectiveObject:
    geometry: Any
    area: int


def _build_effective_objects(
    *,
    geometries: list[Any],
    valid_mask: np.ndarray,
    out_shape: tuple[int, int],
    transform: Any,
) -> tuple[list[_EffectiveObject], int]:
    objects: list[_EffectiveObject] = []
    excluded = 0
    for geom in geometries:
        window = _bounds_to_window(
            bounds=geom.bounds,
            out_shape=out_shape,
            transform=transform,
        )
        if window is None:
            excluded += 1
            continue
        row0, row1, col0, col1 = window
        geom_mask = _rasterize_geometry_window(
            geometry=geom,
            row0=row0,
            row1=row1,
            col0=col0,
            col1=col1,
            transform=transform,
        )
        valid_sub = valid_mask[row0:row1, col0:col1]
        area = int(np.logical_and(geom_mask, valid_sub).sum())
        if area <= 0:
            excluded += 1
            continue
        objects.append(_EffectiveObject(geometry=geom, area=area))
    return objects, excluded


def _build_spatial_index(geometries: list[Any]) -> tuple[Any, dict[bytes, int]]:
    _, _, STRtree = _require_shapely()
    tree = STRtree(geometries)
    # Fallback lookup for STRtree variants returning geometry objects.
    wkb_to_index = {geom.wkb: idx for idx, geom in enumerate(geometries)}
    return tree, wkb_to_index


def _query_candidate_indices(
    *,
    tree: Any,
    wkb_to_index: dict[bytes, int],
    geometry: Any,
) -> tuple[int, ...]:
    raw = tree.query(geometry)
    if raw is None:
        return ()

    try:
        count = len(raw)
    except TypeError:
        return ()
    if count == 0:
        return ()

    first = raw[0]
    if isinstance(first, (int, np.integer)):
        return tuple(int(v) for v in raw.tolist())

    indices: list[int] = []
    for candidate in raw:
        idx = wkb_to_index.get(candidate.wkb)
        if idx is not None:
            indices.append(int(idx))
    return tuple(indices)


def _intersection_area_on_valid(
    *,
    gt_geometry: Any,
    pred_geometry: Any,
    valid_mask: np.ndarray,
    out_shape: tuple[int, int],
    transform: Any,
) -> int:
    gt_bounds = gt_geometry.bounds
    pred_bounds = pred_geometry.bounds
    bounds = (
        max(gt_bounds[0], pred_bounds[0]),
        max(gt_bounds[1], pred_bounds[1]),
        min(gt_bounds[2], pred_bounds[2]),
        min(gt_bounds[3], pred_bounds[3]),
    )
    window = _bounds_to_window(bounds=bounds, out_shape=out_shape, transform=transform)
    if window is None:
        return 0

    row0, row1, col0, col1 = window
    valid_sub = valid_mask[row0:row1, col0:col1]
    if not bool(np.any(valid_sub)):
        return 0

    gt_mask = _rasterize_geometry_window(
        geometry=gt_geometry,
        row0=row0,
        row1=row1,
        col0=col0,
        col1=col1,
        transform=transform,
    )
    pred_mask = _rasterize_geometry_window(
        geometry=pred_geometry,
        row0=row0,
        row1=row1,
        col0=col0,
        col1=col1,
        transform=transform,
    )
    return int(np.logical_and(np.logical_and(gt_mask, pred_mask), valid_sub).sum())


def _relation_ok(
    *,
    iou: float,
    overlap_gt: float,
    overlap_pred: float,
    policy: ObjectMatchingPolicy,
) -> bool:
    iou_ok = iou >= float(policy.min_iou_threshold)
    overlap_ok = (
        overlap_gt >= float(policy.min_overlap_gt_threshold)
        and overlap_pred >= float(policy.min_overlap_pred_threshold)
    )

    if policy.match_rule == "iou_or_overlap":
        return bool(iou_ok or overlap_ok)
    if policy.match_rule == "iou_and_overlap":
        return bool(iou_ok and overlap_ok)
    if policy.match_rule == "iou_only":
        return bool(iou_ok)
    if policy.match_rule == "overlap_only":
        return bool(overlap_ok)
    raise ContractError(f"Unsupported match_rule={policy.match_rule!r}.")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= _EPS:
        return 0.0
    return float(numerator / denominator)


def compute_object_structure_metrics(
    *,
    input_contract: EvaluationInputContractResult,
    policy: ObjectMatchingPolicy,
) -> ObjectStructureMetricsResult:
    """Compute Stage D minimal object/structure metrics (GOC, GUC, GTC)."""
    if not isinstance(input_contract, EvaluationInputContractResult):
        raise ContractError(
            "input_contract must be EvaluationInputContractResult from Stage A."
        )
    _validate_policy(policy)

    if input_contract.track_readiness.object_structure_ready is not True:
        reason = (
            input_contract.track_readiness.object_structure_reason
            if input_contract.track_readiness.object_structure_reason is not None
            else "unknown reason"
        )
        raise ContractError(
            "Object/structure metrics cannot run because object track is not ready: "
            f"{reason}."
        )

    if input_contract.gt_parcels is None or input_contract.post_parcels_vector is None:
        raise ContractError(
            "Object/structure metrics require both gt_parcels and post_parcels_vector."
        )

    out_shape = (int(input_contract.common_height), int(input_contract.common_width))
    valid01 = _read_valid_binary(input_contract.gt_valid.path, expected_shape=out_shape)
    valid_mask = valid01 == 1

    Affine = _require_affine()
    transform = Affine.from_gdal(*input_contract.common_transform_gdal)

    gt_geoms = _read_vector_geometries(input_contract.gt_parcels.path, role="gt_parcels")
    pred_geoms = _read_vector_geometries(
        input_contract.post_parcels_vector.path, role="post_parcels"
    )

    gt_objects, gt_excluded = _build_effective_objects(
        geometries=gt_geoms,
        valid_mask=valid_mask,
        out_shape=out_shape,
        transform=transform,
    )
    pred_objects, pred_excluded = _build_effective_objects(
        geometries=pred_geoms,
        valid_mask=valid_mask,
        out_shape=out_shape,
        transform=transform,
    )

    gt_count = len(gt_objects)
    pred_count = len(pred_objects)
    if gt_count == 0 or pred_count == 0:
        if (
            policy.empty_object_handling == "zero_if_both_empty"
            and gt_count == 0
            and pred_count == 0
        ):
            return ObjectStructureMetricsResult(
                goc=0.0,
                guc=0.0,
                gtc=0.0,
                normalized_gtc=0.0,
                gt_object_count=0,
                pred_object_count=0,
                matched_gt_count=0,
                matched_pred_count=0,
                split_gt_count=0,
                merged_gt_count=0,
                unmatched_gt_count=0,
                spurious_pred_count=0,
                gt_excluded_zero_valid_count=int(gt_excluded),
                pred_excluded_zero_valid_count=int(pred_excluded),
                threshold_provenance=policy.threshold_provenance,
                min_iou_threshold=float(policy.min_iou_threshold),
                min_overlap_gt_threshold=float(policy.min_overlap_gt_threshold),
                min_overlap_pred_threshold=float(policy.min_overlap_pred_threshold),
                match_rule=policy.match_rule,
                stage_scope="stage_d_object_structure_metrics_only",
            )
        raise ContractError(
            "No effective objects remain after valid-domain filtering: "
            f"gt_effective={gt_count}, pred_effective={pred_count}."
        )

    gt_areas = np.asarray([obj.area for obj in gt_objects], dtype=np.int64)
    pred_areas = np.asarray([obj.area for obj in pred_objects], dtype=np.int64)

    pred_tree, pred_wkb_to_index = _build_spatial_index(
        [obj.geometry for obj in pred_objects]
    )
    gt_match_counts = np.zeros(gt_count, dtype=np.int32)
    pred_match_counts = np.zeros(pred_count, dtype=np.int32)
    gt_matched_pred_indices: list[list[int]] = [[] for _ in range(gt_count)]

    for gi in range(gt_count):
        gt_geometry = gt_objects[gi].geometry
        gt_area = int(gt_areas[gi])
        candidate_pred_indices = _query_candidate_indices(
            tree=pred_tree,
            wkb_to_index=pred_wkb_to_index,
            geometry=gt_geometry,
        )
        for pi in candidate_pred_indices:
            pred_area = int(pred_areas[pi])
            inter = _intersection_area_on_valid(
                gt_geometry=gt_geometry,
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
                policy=policy,
            )
            if not relation_ok:
                continue
            gt_match_counts[gi] += 1
            pred_match_counts[pi] += 1
            gt_matched_pred_indices[gi].append(int(pi))

    split_gt = gt_match_counts >= 2
    merged_pred = pred_match_counts >= 2
    merged_gt = np.zeros(gt_count, dtype=bool)
    if bool(np.any(merged_pred)):
        for gi, matched_preds in enumerate(gt_matched_pred_indices):
            if not matched_preds:
                continue
            merged_gt[gi] = any(bool(merged_pred[pi]) for pi in matched_preds)

    matched_gt = gt_match_counts >= 1
    matched_pred = pred_match_counts >= 1

    split_gt_count = int(split_gt.sum())
    merged_gt_count = int(merged_gt.sum())
    matched_gt_count = int(matched_gt.sum())
    matched_pred_count = int(matched_pred.sum())
    unmatched_gt_count = int(gt_count - matched_gt_count)
    spurious_pred_count = int(pred_count - matched_pred_count)

    goc = _safe_div(split_gt_count, gt_count)
    guc = _safe_div(merged_gt_count, gt_count)
    gtc = goc + guc
    normalized_gtc = min(1.0, gtc)

    return ObjectStructureMetricsResult(
        goc=float(goc),
        guc=float(guc),
        gtc=float(gtc),
        normalized_gtc=float(normalized_gtc),
        gt_object_count=int(gt_count),
        pred_object_count=int(pred_count),
        matched_gt_count=matched_gt_count,
        matched_pred_count=matched_pred_count,
        split_gt_count=split_gt_count,
        merged_gt_count=merged_gt_count,
        unmatched_gt_count=unmatched_gt_count,
        spurious_pred_count=spurious_pred_count,
        gt_excluded_zero_valid_count=int(gt_excluded),
        pred_excluded_zero_valid_count=int(pred_excluded),
        threshold_provenance=policy.threshold_provenance,
        min_iou_threshold=float(policy.min_iou_threshold),
        min_overlap_gt_threshold=float(policy.min_overlap_gt_threshold),
        min_overlap_pred_threshold=float(policy.min_overlap_pred_threshold),
        match_rule=policy.match_rule,
        stage_scope="stage_d_object_structure_metrics_only",
    )


def build_object_structure_metrics_summary(
    result: ObjectStructureMetricsResult,
) -> dict[str, Any]:
    """Build a compact machine-readable summary for Stage D outputs."""
    if not isinstance(result, ObjectStructureMetricsResult):
        raise ContractError("result must be ObjectStructureMetricsResult.")
    return {
        "stage_scope": result.stage_scope,
        "metrics": {
            "goc": result.goc,
            "guc": result.guc,
            "gtc": result.gtc,
            "normalized_gtc": result.normalized_gtc,
        },
        "counts": {
            "gt_object_count": result.gt_object_count,
            "pred_object_count": result.pred_object_count,
            "matched_gt_count": result.matched_gt_count,
            "matched_pred_count": result.matched_pred_count,
            "split_gt_count": result.split_gt_count,
            "merged_gt_count": result.merged_gt_count,
            "unmatched_gt_count": result.unmatched_gt_count,
            "spurious_pred_count": result.spurious_pred_count,
            "gt_excluded_zero_valid_count": result.gt_excluded_zero_valid_count,
            "pred_excluded_zero_valid_count": result.pred_excluded_zero_valid_count,
        },
        "policy": {
            "threshold_provenance": result.threshold_provenance,
            "min_iou_threshold": result.min_iou_threshold,
            "min_overlap_gt_threshold": result.min_overlap_gt_threshold,
            "min_overlap_pred_threshold": result.min_overlap_pred_threshold,
            "match_rule": result.match_rule,
        },
    }


__all__ = [
    "ObjectMatchingPolicy",
    "ObjectStructureMetricsResult",
    "build_object_structure_metrics_summary",
    "compute_object_structure_metrics",
]
