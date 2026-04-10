"""Minimal Stage D polygonization + conservative cleanup + polygon_confidence.

This layer consumes:
- Stage A validated postprocess input contract;
- Stage C parcel-instance raster result.

It produces:
- baseline vector output in GPKG format;
- per-polygon polygon_confidence attribute;
- strict, conservative, valid-aware polygonization contract.

Out of scope:
- module_eval coupling,
- advanced merge/split heuristics,
- giant topology framework.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError
from ai_fields.common.progress import iter_progress
from ai_fields.module_postprocess_vectorize.input_contract import PostprocessInputContractResult
from ai_fields.module_postprocess_vectorize.instance_core import ParcelInstanceRasterResult

_EPS = 1e-9


@dataclass(frozen=True)
class PolygonizationPolicy:
    """Explicit Stage D policy (minimal, conservative, config-visible)."""

    threshold_provenance: str

    cleanup_policy_name: str = "conservative_topology_cleanup_v1"
    confidence_policy_name: str = "rule_based_polygon_confidence_v1"

    extent_support_min_prob: float = 0.5
    min_polygon_area_m2: float = 0.0
    simplify_tolerance_m: float = 0.0
    make_valid_enabled: bool = True

    confidence_w_extent_inside: float = 0.50
    confidence_w_boundary_inside_inv: float = 0.20
    confidence_w_distance_support: float = 0.15
    confidence_w_extent_overlap: float = 0.15
    confidence_invalid_penalty_weight: float = 0.25
    num_workers: int = 1
    confidence_details_limit: int = 0


@dataclass(frozen=True)
class PostprocessPolygonizationResult:
    """Result contract for Stage D polygonization output."""

    parcels_gpkg_path: Path
    layer_name: str

    polygon_count: int
    crs: str
    geometry_type: str
    attribute_schema: tuple[str, ...]

    polygon_confidence_present: bool

    policy: dict[str, Any]
    diagnostics: dict[str, Any]
    ready_for_stage_e: bool


def _require_runtime_deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import fiona
        import fiona.errors
        import rasterio.features as rfeatures
        from affine import Affine
        from shapely import make_valid
        from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "Stage D polygonization requires fiona, rasterio.features, affine, and shapely."
        ) from exc

    return (
        fiona,
        fiona.errors,
        rfeatures,
        Affine,
        make_valid,
        (Polygon, MultiPolygon, GeometryCollection),
        (mapping, shape, unary_union),
    )


def _validate_policy(policy: PolygonizationPolicy) -> PolygonizationPolicy:
    if not isinstance(policy, PolygonizationPolicy):
        raise ContractError("policy must be PolygonizationPolicy for Stage D.")

    if not isinstance(policy.threshold_provenance, str) or policy.threshold_provenance.strip() == "":
        raise ContractError("policy.threshold_provenance must be a non-empty string.")

    if not (0.0 <= float(policy.extent_support_min_prob) <= 1.0):
        raise ContractError("policy.extent_support_min_prob must be in [0,1].")
    if float(policy.min_polygon_area_m2) < 0.0:
        raise ContractError("policy.min_polygon_area_m2 must be >= 0.")
    if float(policy.simplify_tolerance_m) < 0.0:
        raise ContractError("policy.simplify_tolerance_m must be >= 0.")

    for name, value in (
        ("confidence_w_extent_inside", policy.confidence_w_extent_inside),
        ("confidence_w_boundary_inside_inv", policy.confidence_w_boundary_inside_inv),
        ("confidence_w_distance_support", policy.confidence_w_distance_support),
        ("confidence_w_extent_overlap", policy.confidence_w_extent_overlap),
        ("confidence_invalid_penalty_weight", policy.confidence_invalid_penalty_weight),
    ):
        if float(value) < 0.0:
            raise ContractError(f"policy.{name} must be >= 0.")

    if (
        float(policy.confidence_w_extent_inside)
        + float(policy.confidence_w_boundary_inside_inv)
        + float(policy.confidence_w_distance_support)
        + float(policy.confidence_w_extent_overlap)
    ) <= 0.0:
        raise ContractError("At least one polygon_confidence positive weight must be > 0.")
    if not isinstance(policy.num_workers, int) or int(policy.num_workers) < 1:
        raise ContractError("policy.num_workers must be an integer >= 1.")
    if (
        not isinstance(policy.confidence_details_limit, int)
        or int(policy.confidence_details_limit) < 0
    ):
        raise ContractError("policy.confidence_details_limit must be an integer >= 0.")

    return policy


def _validate_stage_inputs(
    *,
    input_contract: PostprocessInputContractResult,
    instance_result: ParcelInstanceRasterResult,
) -> tuple[np.ndarray, tuple[int, int]]:
    if not isinstance(input_contract, PostprocessInputContractResult):
        raise ContractError("input_contract must be PostprocessInputContractResult from Stage A.")
    if not isinstance(instance_result, ParcelInstanceRasterResult):
        raise ContractError("instance_result must be ParcelInstanceRasterResult from Stage C.")

    if instance_result.ready_for_stage_d is False:
        raise ContractError("instance_result.ready_for_stage_d is False; cannot run Stage D.")

    h = int(input_contract.common_height)
    w = int(input_contract.common_width)
    expected_shape = (h, w)

    parcel = instance_result.parcel_instance
    if not isinstance(parcel, np.ndarray) or parcel.ndim != 2:
        raise ContractError("instance_result.parcel_instance must be a 2-D numpy array.")
    if parcel.shape != expected_shape:
        raise ContractError(
            f"parcel_instance shape mismatch: expected {expected_shape}, got {parcel.shape}."
        )
    if not np.issubdtype(parcel.dtype, np.integer):
        raise ContractError("parcel_instance must have integer dtype.")

    invalid_label = int(instance_result.invalid_label)
    negative_labels = sorted(int(v) for v in np.unique(parcel) if int(v) < 0)
    if negative_labels:
        disallowed = [v for v in negative_labels if v != invalid_label]
        if disallowed:
            raise ContractError(
                "parcel_instance contains negative labels that do not match "
                f"invalid_label={invalid_label}: {disallowed}."
            )

    if not np.any(parcel > 0):
        raise ContractError("parcel_instance contains zero positive instance labels.")

    return parcel.astype(np.int32), expected_shape


def _read_single_band(path: Path, *, role: str, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for Stage D polygonization reads.") from exc

    try:
        with rasterio.open(path) as ds:
            arr = ds.read()
    except rasterio.errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read {role} raster: {path} ({exc})") from exc

    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ContractError(f"{role} must be single-band for Stage D, got shape={arr.shape}.")

    band = arr[0]
    if band.shape != expected_shape:
        raise ContractError(
            f"{role} shape mismatch: expected {expected_shape}, got {band.shape}."
        )
    if not np.isfinite(band).all():
        raise ContractError(f"{role} contains non-finite values.")

    return band.astype(np.float32)


def _read_boundary_presence(path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for Stage D polygonization reads.") from exc

    try:
        with rasterio.open(path) as ds:
            arr = ds.read()
    except rasterio.errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read boundary_prob raster: {path} ({exc})") from exc

    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ContractError(f"boundary_prob must be 3-band, got shape={arr.shape}.")
    if arr.shape[1:] != expected_shape:
        raise ContractError(
            "boundary_prob shape mismatch: "
            f"expected (3, {expected_shape[0]}, {expected_shape[1]}), got {arr.shape}."
        )
    if not np.isfinite(arr).all():
        raise ContractError("boundary_prob contains non-finite values.")

    return np.clip(arr[1] + arr[2], 0.0, 1.0).astype(np.float32)


def _read_valid_binary(path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    raw = _read_single_band(path, role="valid", expected_shape=expected_shape)
    unique = np.unique(raw)
    if not np.all(np.isin(unique, [0, 1])):
        raise ContractError(
            "valid raster must be binary {0,1} for Stage D, "
            f"got unique values {unique.tolist()}."
        )
    return (raw > 0).astype(np.uint8)


def _iter_polygonal_parts(geom: Any, polygon_types: tuple[Any, Any, Any]) -> list[Any]:
    polygon_cls, multipolygon_cls, geometry_collection_cls = polygon_types

    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, polygon_cls):
        return [geom]
    if isinstance(geom, multipolygon_cls):
        return [g for g in geom.geoms if not g.is_empty]
    if isinstance(geom, geometry_collection_cls):
        out: list[Any] = []
        for sub in geom.geoms:
            out.extend(_iter_polygonal_parts(sub, polygon_types))
        return out
    return []


def _cleanup_geometry(
    *,
    geom: Any,
    policy: PolygonizationPolicy,
    make_valid_fn: Any,
    polygon_types: tuple[Any, Any, Any],
    unary_union_fn: Any,
) -> tuple[Any | None, str | None]:
    if geom is None or geom.is_empty:
        return None, "empty"

    cleaned = geom
    if not cleaned.is_valid and bool(policy.make_valid_enabled):
        try:
            cleaned = make_valid_fn(cleaned)
        except Exception as exc:  # pragma: no cover
            raise ContractError(f"Failed to make geometry valid: {exc}") from exc

    parts = _iter_polygonal_parts(cleaned, polygon_types)
    if not parts:
        return None, "non_polygonal"

    min_area = float(policy.min_polygon_area_m2)
    parts = [p for p in parts if p.area > max(min_area, 0.0) + _EPS]
    if not parts:
        return None, "small_area"

    tol = float(policy.simplify_tolerance_m)
    if tol > 0.0:
        parts = [p.simplify(tol, preserve_topology=True) for p in parts]
        parts = [p for p in parts if not p.is_empty and p.area > max(min_area, 0.0) + _EPS]
        if not parts:
            return None, "simplified_away"

    merged = unary_union_fn(parts)
    parts2 = _iter_polygonal_parts(merged, polygon_types)
    if not parts2:
        return None, "non_polygonal"

    merged2 = unary_union_fn(parts2)
    if merged2.is_empty:
        return None, "empty"

    if not merged2.is_valid and bool(policy.make_valid_enabled):
        merged2 = make_valid_fn(merged2)
        parts3 = _iter_polygonal_parts(merged2, polygon_types)
        if not parts3:
            return None, "invalid_after_make_valid"
        merged2 = unary_union_fn(parts3)

    return merged2, None


def _build_label_geometries(
    *,
    parcel_instance: np.ndarray,
    transform: Any,
    valid01: np.ndarray,
    rfeatures: Any,
    shape_fn: Any,
) -> dict[int, list[Any]]:
    parcel_effective = parcel_instance.copy()
    parcel_effective[valid01 == 0] = 0

    label_geoms: dict[int, list[Any]] = {}
    mask = parcel_effective > 0
    if not np.any(mask):
        raise ContractError(
            "No positive parcel labels remain after valid-mask suppression in Stage D."
        )

    for geom_map, value in rfeatures.shapes(parcel_effective.astype(np.int32), mask=mask, transform=transform):
        label = int(value)
        if label <= 0:
            continue
        geom = shape_fn(geom_map)
        if geom.is_empty:
            continue
        label_geoms.setdefault(label, []).append(geom)

    if not label_geoms:
        raise ContractError("Polygonization produced zero candidate geometries from parcel_instance.")

    return label_geoms


def _compute_confidence(
    *,
    polygon_geom: Any,
    extent_prob: np.ndarray,
    boundary_presence: np.ndarray,
    distance_pred: np.ndarray,
    valid01: np.ndarray,
    extent_support_mask: np.ndarray,
    distance_norm: np.ndarray,
    transform: Any,
    rfeatures: Any,
    mapping_fn: Any,
    policy: PolygonizationPolicy,
) -> tuple[float, dict[str, float]]:
    # Use a tight raster window around polygon bounds to avoid full-scene
    # geometry_mask computation for every instance.
    minx, miny, maxx, maxy = polygon_geom.bounds
    inv_transform = ~transform
    cols_rows = [inv_transform * (x, y) for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))]
    cols = [float(c) for c, _ in cols_rows]
    rows = [float(r) for _, r in cols_rows]

    row_min = max(0, int(np.floor(min(rows))))
    row_max = min(extent_prob.shape[0], int(np.ceil(max(rows))))
    col_min = max(0, int(np.floor(min(cols))))
    col_max = min(extent_prob.shape[1], int(np.ceil(max(cols))))

    if row_max <= row_min or col_max <= col_min:
        return 0.0, {
            "mean_extent_inside": 0.0,
            "mean_boundary_inside": 1.0,
            "mean_distance_support": 0.0,
            "extent_overlap_ratio": 0.0,
            "valid_coverage": 0.0,
        }

    local_transform = transform * transform.__class__.translation(col_min, row_min)
    inside = rfeatures.geometry_mask(
        [mapping_fn(polygon_geom)],
        out_shape=(row_max - row_min, col_max - col_min),
        transform=local_transform,
        invert=True,
    )

    valid_local = valid01[row_min:row_max, col_min:col_max]
    extent_local = extent_prob[row_min:row_max, col_min:col_max]
    boundary_local = boundary_presence[row_min:row_max, col_min:col_max]
    distance_norm_local = distance_norm[row_min:row_max, col_min:col_max]
    extent_support_local = extent_support_mask[row_min:row_max, col_min:col_max]

    inside_total = int(inside.sum())
    inside_valid_mask = inside & (valid_local == 1)
    inside_valid = int(inside_valid_mask.sum())

    if inside_total == 0 or inside_valid == 0:
        return 0.0, {
            "mean_extent_inside": 0.0,
            "mean_boundary_inside": 1.0,
            "mean_distance_support": 0.0,
            "extent_overlap_ratio": 0.0,
            "valid_coverage": 0.0,
        }

    mean_extent_inside = float(extent_local[inside_valid_mask].mean())
    mean_boundary_inside = float(boundary_local[inside_valid_mask].mean())
    mean_distance_support = float(distance_norm_local[inside_valid_mask].mean())
    extent_overlap_ratio = float(extent_support_local[inside_valid_mask].mean())
    valid_coverage = float(inside_valid / inside_total)

    score = (
        float(policy.confidence_w_extent_inside) * mean_extent_inside
        + float(policy.confidence_w_boundary_inside_inv) * (1.0 - mean_boundary_inside)
        + float(policy.confidence_w_distance_support) * mean_distance_support
        + float(policy.confidence_w_extent_overlap) * extent_overlap_ratio
        - float(policy.confidence_invalid_penalty_weight) * (1.0 - valid_coverage)
    )
    score = float(np.clip(score, 0.0, 1.0))

    return score, {
        "mean_extent_inside": mean_extent_inside,
        "mean_boundary_inside": mean_boundary_inside,
        "mean_distance_support": mean_distance_support,
        "extent_overlap_ratio": extent_overlap_ratio,
        "valid_coverage": valid_coverage,
    }


def build_postprocess_polygons(
    *,
    input_contract: PostprocessInputContractResult,
    instance_result: ParcelInstanceRasterResult,
    output_gpkg_path: str | Path,
    policy: PolygonizationPolicy,
    layer_name: str = "parcels",
    progress_enabled: bool | None = None,
) -> PostprocessPolygonizationResult:
    """Run minimal Stage D polygonization and write baseline GPKG output."""
    parcel_instance, expected_shape = _validate_stage_inputs(
        input_contract=input_contract,
        instance_result=instance_result,
    )
    _validate_policy(policy)

    if not isinstance(layer_name, str) or layer_name.strip() == "":
        raise ContractError("layer_name must be a non-empty string.")

    output_path = Path(output_gpkg_path)
    if str(output_path).strip() == "":
        raise ContractError("output_gpkg_path must be a non-empty path-like value.")

    (
        fiona,
        _fiona_errors,
        rfeatures,
        Affine,
        make_valid_fn,
        polygon_types,
        shapely_ops,
    ) = _require_runtime_deps()
    mapping_fn, shape_fn, unary_union_fn = shapely_ops

    transform = Affine.from_gdal(*input_contract.common_transform_gdal)

    extent_prob = _read_single_band(
        input_contract.extent_prob.path,
        role="extent_prob",
        expected_shape=expected_shape,
    )
    boundary_presence = _read_boundary_presence(
        input_contract.boundary_prob.path,
        expected_shape=expected_shape,
    )
    distance_pred = _read_single_band(
        input_contract.distance_pred.path,
        role="distance_pred",
        expected_shape=expected_shape,
    )
    valid01 = _read_valid_binary(
        input_contract.valid.path,
        expected_shape=expected_shape,
    )

    extent_support_mask = (extent_prob >= float(policy.extent_support_min_prob)) & (valid01 == 1)

    valid_dist = distance_pred[valid01 == 1]
    if valid_dist.size == 0:
        raise ContractError("valid raster has zero valid pixels for Stage D polygon confidence.")
    d_min = float(valid_dist.min())
    d_max = float(valid_dist.max())
    if d_max - d_min <= _EPS:
        distance_norm = np.zeros_like(distance_pred, dtype=np.float32)
    else:
        distance_norm = ((distance_pred - d_min) / (d_max - d_min)).astype(np.float32)
        distance_norm = np.clip(distance_norm, 0.0, 1.0)

    labels_start = perf_counter()
    label_geoms = _build_label_geometries(
        parcel_instance=parcel_instance,
        transform=transform,
        valid01=valid01,
        rfeatures=rfeatures,
        shape_fn=shape_fn,
    )
    labels_build_seconds = float(perf_counter() - labels_start)

    polygons_written = 0
    dropped: dict[str, int] = {
        "empty": 0,
        "non_polygonal": 0,
        "small_area": 0,
        "simplified_away": 0,
        "invalid_after_make_valid": 0,
    }
    confidence_count = 0
    confidence_sum = 0.0
    confidence_min = float("inf")
    confidence_max = float("-inf")
    confidence_details_sample: dict[int, dict[str, float]] = {}
    confidence_details_limit = int(policy.confidence_details_limit)
    sorted_instance_ids = sorted(label_geoms)
    num_workers = int(policy.num_workers)
    batch_size = max(32, num_workers * 4)

    def _process_instance(
        instance_id: int,
    ) -> tuple[int, dict[str, Any] | None, str | None, float | None, dict[str, float] | None]:
        merged = unary_union_fn(label_geoms[instance_id])
        cleaned, dropped_reason = _cleanup_geometry(
            geom=merged,
            policy=policy,
            make_valid_fn=make_valid_fn,
            polygon_types=polygon_types,
            unary_union_fn=unary_union_fn,
        )
        if cleaned is None:
            return int(instance_id), None, dropped_reason, None, None

        confidence, details = _compute_confidence(
            polygon_geom=cleaned,
            extent_prob=extent_prob,
            boundary_presence=boundary_presence,
            distance_pred=distance_pred,
            valid01=valid01,
            extent_support_mask=extent_support_mask,
            distance_norm=distance_norm,
            transform=transform,
            rfeatures=rfeatures,
            mapping_fn=mapping_fn,
            policy=policy,
        )
        feature = {
            "geometry": mapping_fn(cleaned),
            "properties": {
                "instance_id": int(instance_id),
                "polygon_confidence": float(confidence),
            },
        }
        return int(instance_id), feature, None, float(confidence), details

    def _consume(results_iter: Any, *, writer: Any, total: int) -> None:
        nonlocal polygons_written, confidence_count, confidence_sum, confidence_min, confidence_max
        for instance_id, feature, dropped_reason, confidence, details in iter_progress(
            results_iter,
            total=total,
            desc="postprocess: polygonize instances",
            unit="parcel",
            progress_enabled=progress_enabled,
            leave=False,
        ):
            if feature is None:
                if dropped_reason is not None:
                    dropped[dropped_reason] = dropped.get(dropped_reason, 0) + 1
                continue
            writer.write(feature)
            polygons_written += 1
            if confidence is not None:
                confidence_count += 1
                confidence_sum += float(confidence)
                confidence_min = min(confidence_min, float(confidence))
                confidence_max = max(confidence_max, float(confidence))
            if details is not None and confidence_details_limit > 0:
                if len(confidence_details_sample) < confidence_details_limit:
                    confidence_details_sample[int(instance_id)] = details

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    schema = {
        "geometry": "Unknown",
        "properties": {
            "instance_id": "int",
            "polygon_confidence": "float",
        },
    }

    processing_and_write_start = perf_counter()
    with fiona.open(
        output_path,
        mode="w",
        driver="GPKG",
        schema=schema,
        crs=input_contract.common_crs,
        layer=layer_name,
    ) as dst:
        if num_workers == 1:
            _consume(
                (_process_instance(instance_id) for instance_id in sorted_instance_ids),
                writer=dst,
                total=len(sorted_instance_ids),
            )
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for chunk_start in range(0, len(sorted_instance_ids), batch_size):
                    chunk_ids = sorted_instance_ids[chunk_start : chunk_start + batch_size]
                    _consume(
                        pool.map(_process_instance, chunk_ids),
                        writer=dst,
                        total=len(chunk_ids),
                    )
    processing_and_write_seconds = float(perf_counter() - processing_and_write_start)
    stage_total_seconds = float(perf_counter() - labels_start)

    if polygons_written == 0:
        try:
            output_path.unlink()
        except OSError:
            pass
        raise ContractError(
            "Stage D polygonization produced zero polygons after conservative cleanup."
        )

    confidence_summary: dict[str, float | int] = {
        "count": int(confidence_count),
        "min": float(confidence_min) if confidence_count > 0 else 0.0,
        "max": float(confidence_max) if confidence_count > 0 else 0.0,
        "mean": float(confidence_sum / confidence_count) if confidence_count > 0 else 0.0,
    }

    diagnostics = {
        "input_shape": {
            "height": int(expected_shape[0]),
            "width": int(expected_shape[1]),
        },
        "labels_before_cleanup": int(len(label_geoms)),
        "polygons_written": int(polygons_written),
        "cleanup_dropped": dropped,
        "confidence_summary": confidence_summary,
        "confidence_details_sample": confidence_details_sample,
        "runtime_seconds": {
            "label_geometry_build": labels_build_seconds,
            "processing_and_write": processing_and_write_seconds,
            "stage_total": stage_total_seconds,
        },
    }

    policy_info = {
        "threshold_provenance": policy.threshold_provenance,
        "cleanup_policy_name": policy.cleanup_policy_name,
        "confidence_policy_name": policy.confidence_policy_name,
        "extent_support_min_prob": float(policy.extent_support_min_prob),
        "min_polygon_area_m2": float(policy.min_polygon_area_m2),
        "simplify_tolerance_m": float(policy.simplify_tolerance_m),
        "make_valid_enabled": bool(policy.make_valid_enabled),
        "num_workers": int(policy.num_workers),
        "confidence_details_limit": int(policy.confidence_details_limit),
        "confidence_weights": {
            "extent_inside": float(policy.confidence_w_extent_inside),
            "boundary_inside_inv": float(policy.confidence_w_boundary_inside_inv),
            "distance_support": float(policy.confidence_w_distance_support),
            "extent_overlap": float(policy.confidence_w_extent_overlap),
            "invalid_penalty": float(policy.confidence_invalid_penalty_weight),
        },
    }

    return PostprocessPolygonizationResult(
        parcels_gpkg_path=output_path,
        layer_name=layer_name,
        polygon_count=int(polygons_written),
        crs=input_contract.common_crs,
        geometry_type="Unknown",
        attribute_schema=("instance_id", "polygon_confidence"),
        polygon_confidence_present=True,
        policy=policy_info,
        diagnostics=diagnostics,
        ready_for_stage_e=True,
    )


__all__ = [
    "PolygonizationPolicy",
    "PostprocessPolygonizationResult",
    "build_postprocess_polygons",
]
