"""Evaluation input contract resolver for module_eval (Stage A).

This layer is intentionally narrow and contract-first:
- reads GT / prediction / postprocess artifacts;
- validates strict spatial and semantic compatibility;
- resolves readiness for eval tracks (pixel, boundary, object/structure);
- exposes explicit output contract skeleton for later metric stages.

Out of scope:
- metric computation,
- report generation,
- multi-run comparison runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    SpatialContractError,
    ValidPolicyError,
)
from ai_fields.common.manifests import read_manifest

_EXPECTED_PREDICT_MANIFEST_SCHEMA = "target_predict.predict_manifest"
_EXPECTED_POSTPROCESS_MANIFEST_SCHEMA = "postprocess_vectorize.postprocess_manifest"

_GT_EXTENT_LABELS = {0, 1, 255}
_GT_BOUNDARY_LABELS = {0, 1, 2}
_BINARY_LABELS = {0, 1}
_EPS = 1e-6
_BOUNDARY_SUM_ATOL = 5e-3
_STREAM_WINDOW_SIZE = 1024


@dataclass(frozen=True)
class EvaluationRasterMetadata:
    """Minimal raster metadata required for eval input checks."""

    role: str
    path: Path
    width: int
    height: int
    band_count: int
    crs: str
    transform_gdal: tuple[float, float, float, float, float, float]
    dtype: str
    nodata: float | int | None


@dataclass(frozen=True)
class EvaluationVectorMetadata:
    """Minimal vector metadata required for eval input checks."""

    role: str
    path: Path
    crs: str
    layer_name: str
    geometry_type: str
    feature_count: int
    polygon_confidence_present: bool


@dataclass(frozen=True)
class EvaluationTrackReadiness:
    """Readiness flags for staged eval metric groups."""

    pixel_ready: bool
    boundary_ready: bool
    object_structure_ready: bool

    pixel_reason: str | None
    boundary_reason: str | None
    object_structure_reason: str | None


@dataclass(frozen=True)
class EvaluationOutputContractSkeleton:
    """Explicit skeleton of future metric groups."""

    pixel_metrics: tuple[str, ...]
    boundary_metrics: tuple[str, ...]
    object_structure_metrics: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationInputContractResult:
    """Resolved and validated eval input contract (Stage A output)."""

    gt_extent: EvaluationRasterMetadata
    gt_boundary: EvaluationRasterMetadata
    gt_valid: EvaluationRasterMetadata
    gt_distance: EvaluationRasterMetadata | None
    gt_parcels: EvaluationVectorMetadata | None

    pred_extent_prob: EvaluationRasterMetadata
    pred_boundary_prob: EvaluationRasterMetadata
    pred_distance_pred: EvaluationRasterMetadata
    pred_valid: EvaluationRasterMetadata

    post_parcel_instance: EvaluationRasterMetadata | None
    post_parcels_vector: EvaluationVectorMetadata | None

    common_width: int
    common_height: int
    common_crs: str
    common_transform_gdal: tuple[float, float, float, float, float, float]

    semantic_compatibility_summary: dict[str, Any]
    track_readiness: EvaluationTrackReadiness
    output_contract: EvaluationOutputContractSkeleton

    predict_manifest_path: Path | None
    postprocess_manifest_path: Path | None
    source_run_ids: tuple[str, ...]
    compatible: bool


def _normalize_existing_path(path: Any, *, name: str) -> Path:
    if isinstance(path, (str, PathLike)):
        normalized = Path(path)
    else:
        raise ContractError(f"{name} must be path-like, got {type(path).__name__}.")
    if str(normalized).strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    if not normalized.exists():
        raise ContractError(f"{name} does not exist: {normalized}")
    if not normalized.is_file():
        raise ContractError(f"{name} must point to a regular file: {normalized}")
    return normalized


def _normalize_optional_existing_path(path: Any, *, name: str) -> Path | None:
    if path is None:
        return None
    return _normalize_existing_path(path, name=name)


def _require_rasterio() -> tuple[Any, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for module_eval Stage A.") from exc
    return rasterio, rasterio.errors


def _require_fiona() -> Any:
    try:
        import fiona
    except ImportError as exc:  # pragma: no cover
        raise ContractError("fiona is required for vector contract checks in module_eval Stage A.") from exc
    return fiona


def _to_nodata(raw: Any) -> float | int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, (float, np.floating)):
        return float(raw)
    raise ContractError(f"Unsupported nodata type: {type(raw).__name__}.")


def _crs_to_string(crs: Any, *, role: str) -> str:
    if crs is None:
        raise SpatialContractError(f"{role}: CRS is missing.")
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else crs.to_string()


def _read_raster_metadata(path: Path, *, role: str) -> EvaluationRasterMetadata:
    rasterio, rasterio_errors = _require_rasterio()
    try:
        with rasterio.open(path) as ds:
            if ds.width < 1 or ds.height < 1:
                raise SpatialContractError(
                    f"{role}: invalid raster dimensions ({ds.width}x{ds.height})."
                )
            if ds.count < 1:
                raise ChannelCountError(f"{role}: raster must contain at least one band.")
            crs = _crs_to_string(ds.crs, role=role)
            dtype = str(ds.dtypes[0])
            transform = tuple(float(v) for v in ds.transform.to_gdal())
            if len(transform) != 6:
                raise SpatialContractError(f"{role}: invalid GDAL transform.")

            return EvaluationRasterMetadata(
                role=role,
                path=path,
                width=int(ds.width),
                height=int(ds.height),
                band_count=int(ds.count),
                crs=crs,
                transform_gdal=(
                    transform[0],
                    transform[1],
                    transform[2],
                    transform[3],
                    transform[4],
                    transform[5],
                ),
                dtype=dtype,
                nodata=_to_nodata(ds.nodata),
            )
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to open {role} raster: {path} ({exc})") from exc


def _transforms_match(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), atol=1e-9, rtol=0.0))


def _validate_spatial_alignment(
    reference: EvaluationRasterMetadata,
    candidate: EvaluationRasterMetadata,
) -> None:
    if (candidate.width, candidate.height) != (reference.width, reference.height):
        raise SpatialContractError(
            f"Spatial mismatch: {candidate.role} has {candidate.width}x{candidate.height}, "
            f"expected {reference.width}x{reference.height} from {reference.role}."
        )
    if candidate.crs != reference.crs:
        raise SpatialContractError(
            f"CRS mismatch: {candidate.role} has {candidate.crs!r}, "
            f"expected {reference.crs!r} from {reference.role}."
        )
    if not _transforms_match(candidate.transform_gdal, reference.transform_gdal):
        raise SpatialContractError(
            f"Transform mismatch between {candidate.role} and {reference.role}; "
            "hidden resampling is not allowed."
        )


def _read_array(path: Path, *, role: str) -> np.ndarray:
    rasterio, rasterio_errors = _require_rasterio()
    try:
        with rasterio.open(path) as ds:
            arr = ds.read()
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read {role} raster values: {path} ({exc})") from exc
    if not np.isfinite(arr).all():
        raise ContractError(f"{role} contains non-finite values.")
    return arr


def _require_band_count(meta: EvaluationRasterMetadata, *, expected: int) -> None:
    if meta.band_count != expected:
        raise ChannelCountError(
            f"{meta.role} must have exactly {expected} band(s), got {meta.band_count}."
        )


def _require_floating_dtype(meta: EvaluationRasterMetadata) -> None:
    if not np.issubdtype(np.dtype(meta.dtype), np.floating):
        raise ContractError(f"{meta.role} must have floating dtype, got {meta.dtype!r}.")


def _require_integer_or_bool_dtype(meta: EvaluationRasterMetadata) -> None:
    dtype = np.dtype(meta.dtype)
    if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)):
        raise ContractError(f"{meta.role} must have integer/bool dtype, got {meta.dtype!r}.")


def _extract_single_band(
    arr: np.ndarray,
    *,
    role: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ContractError(f"{role} must be single-band, got shape={arr.shape}.")
    band = arr[0]
    if band.shape != expected_shape:
        raise SpatialContractError(
            f"{role} shape mismatch: expected {expected_shape}, got {band.shape}."
        )
    return band


def _extract_three_band(
    arr: np.ndarray,
    *,
    role: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ContractError(f"{role} must be 3-band, got shape={arr.shape}.")
    if arr.shape[1:] != expected_shape:
        raise SpatialContractError(
            f"{role} shape mismatch: expected (3, {expected_shape[0]}, {expected_shape[1]}), "
            f"got {arr.shape}."
        )
    return arr


def _validate_allowed_integer_labels(
    arr: np.ndarray,
    *,
    role: str,
    allowed: set[int],
) -> tuple[int, ...]:
    unique = np.unique(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ContractError(f"{role} must be integer-encoded labels.")
    if not np.all(np.isin(unique, list(allowed))):
        raise ContractError(
            f"{role} has unsupported labels: {unique.tolist()}, allowed={sorted(allowed)}."
        )
    return tuple(int(v) for v in unique.tolist())


def _validate_binary_mask(
    arr: np.ndarray,
    *,
    role: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    unique = np.unique(arr)
    if not np.all(np.isin(unique, list(_BINARY_LABELS))):
        raise ValidPolicyError(
            f"{role} must be binary {{0,1}}, got unique values {unique.tolist()}."
        )
    mask01 = (arr > 0).astype(np.uint8)
    return mask01, tuple(int(v) for v in unique.tolist())


def _validate_probability_range(
    arr: np.ndarray,
    *,
    role: str,
) -> tuple[float, float]:
    min_v = float(arr.min())
    max_v = float(arr.max())
    if min_v < -_EPS or max_v > 1.0 + _EPS:
        raise ContractError(
            f"{role} must be probability-like in [0,1], got min={min_v}, max={max_v}."
        )
    return min_v, max_v


def _validate_non_negative(
    arr: np.ndarray,
    *,
    role: str,
) -> tuple[float, float]:
    min_v = float(arr.min())
    max_v = float(arr.max())
    if min_v < -_EPS:
        raise ContractError(
            f"{role} must be non-negative for unsigned distance contract, got min={min_v}."
        )
    return min_v, max_v


def _validate_boundary_simplex_on_valid(
    boundary_prob: np.ndarray,
    *,
    valid01: np.ndarray,
) -> tuple[float, float]:
    valid_pixels = valid01 == 1
    if not np.any(valid_pixels):
        raise ValidPolicyError("pred_valid contains zero valid pixels for eval.")
    sums = boundary_prob.sum(axis=0)
    sums_valid = sums[valid_pixels]
    min_sum = float(sums_valid.min())
    max_sum = float(sums_valid.max())
    if not np.all(np.abs(sums_valid - 1.0) <= _BOUNDARY_SUM_ATOL):
        raise ContractError(
            "pred_boundary_prob on pred_valid pixels must represent a 3-class probability "
            f"simplex (sum≈1). Observed sum range [{min_sum}, {max_sum}]."
        )
    return min_sum, max_sum


def _resolve_manifest_path(raw_path: str, *, manifest_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    # Support manifests that store repository-root relative paths
    # (e.g. "runs/module_target_predict/...") while preserving legacy
    # manifest-relative behavior for plain relative filenames.
    cwd_relative = candidate
    manifest_relative = manifest_path.parent / candidate

    if cwd_relative.exists() and manifest_relative.exists():
        if cwd_relative.resolve() != manifest_relative.resolve():
            raise ContractError(
                "manifest path is ambiguous: "
                f"{raw_path!r} resolves to both {cwd_relative} and {manifest_relative}."
            )
        return cwd_relative

    if cwd_relative.exists():
        return cwd_relative
    if manifest_relative.exists():
        return manifest_relative

    # Deterministic fallback for paths that are expected to exist but are missing.
    return manifest_relative


def _paths_equal(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _validate_predict_manifest_consistency(
    *,
    predict_manifest_path: Path,
    pred_extent_prob_path: Path,
    pred_boundary_prob_path: Path,
    pred_distance_pred_path: Path,
    pred_valid_path: Path,
) -> dict[str, Any]:
    manifest = read_manifest(predict_manifest_path)
    schema_name = manifest.get("schema_name")
    if schema_name != _EXPECTED_PREDICT_MANIFEST_SCHEMA:
        raise ContractError(
            "predict_manifest schema_name must be "
            f"{_EXPECTED_PREDICT_MANIFEST_SCHEMA!r}, got {schema_name!r}."
        )

    output_paths = manifest.get("output_paths")
    if not isinstance(output_paths, Mapping):
        raise ContractError("predict_manifest.output_paths must be a mapping/object.")

    expected: dict[str, Path] = {
        "extent_prob": pred_extent_prob_path,
        "boundary_prob": pred_boundary_prob_path,
        "distance_pred": pred_distance_pred_path,
        "valid": pred_valid_path,
    }
    for key, expected_path in expected.items():
        raw = output_paths.get(key)
        if not isinstance(raw, str) or raw.strip() == "":
            raise ContractError(
                "predict_manifest.output_paths is missing required non-empty path for "
                f"{key!r}."
            )
        actual = _resolve_manifest_path(raw, manifest_path=predict_manifest_path)
        if not _paths_equal(actual, expected_path):
            raise ContractError(
                "predict_manifest output path mismatch for "
                f"{key!r}: manifest={actual}, provided={expected_path}."
            )
    return manifest


def _validate_postprocess_manifest_consistency(
    *,
    postprocess_manifest_path: Path,
    pred_extent_prob_path: Path,
    pred_boundary_prob_path: Path,
    pred_distance_pred_path: Path,
    pred_valid_path: Path,
    post_parcel_instance_path: Path | None,
    post_parcels_gpkg_path: Path | None,
) -> dict[str, Any]:
    manifest = read_manifest(postprocess_manifest_path)
    schema_name = manifest.get("schema_name")
    if schema_name != _EXPECTED_POSTPROCESS_MANIFEST_SCHEMA:
        raise ContractError(
            "postprocess_manifest schema_name must be "
            f"{_EXPECTED_POSTPROCESS_MANIFEST_SCHEMA!r}, got {schema_name!r}."
        )

    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ContractError("postprocess_manifest.inputs must be a mapping/object.")

    expected_inputs: dict[str, Path] = {
        "extent_prob_path": pred_extent_prob_path,
        "boundary_prob_path": pred_boundary_prob_path,
        "distance_pred_path": pred_distance_pred_path,
        "valid_path": pred_valid_path,
    }
    for key, expected_path in expected_inputs.items():
        raw = inputs.get(key)
        if not isinstance(raw, str) or raw.strip() == "":
            raise ContractError(
                "postprocess_manifest.inputs is missing required non-empty path for "
                f"{key!r}."
            )
        actual = _resolve_manifest_path(raw, manifest_path=postprocess_manifest_path)
        if not _paths_equal(actual, expected_path):
            raise ContractError(
                "postprocess_manifest input path mismatch for "
                f"{key!r}: manifest={actual}, provided={expected_path}."
            )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ContractError("postprocess_manifest.outputs must be a mapping/object.")

    if post_parcel_instance_path is not None:
        raw_instance = outputs.get("parcel_instance_path")
        if not isinstance(raw_instance, str) or raw_instance.strip() == "":
            raise ContractError(
                "postprocess_manifest.outputs.parcel_instance_path must be a non-empty string "
                "when post_parcel_instance_path is provided."
            )
        actual_instance = _resolve_manifest_path(raw_instance, manifest_path=postprocess_manifest_path)
        if not _paths_equal(actual_instance, post_parcel_instance_path):
            raise ContractError(
                "postprocess_manifest parcel_instance_path mismatch: "
                f"manifest={actual_instance}, provided={post_parcel_instance_path}."
            )

    if post_parcels_gpkg_path is not None:
        raw_polygons = outputs.get("parcels_gpkg_path")
        if not isinstance(raw_polygons, str) or raw_polygons.strip() == "":
            raise ContractError(
                "postprocess_manifest.outputs.parcels_gpkg_path must be a non-empty string "
                "when post_parcels_gpkg_path is provided."
            )
        actual_polygons = _resolve_manifest_path(raw_polygons, manifest_path=postprocess_manifest_path)
        if not _paths_equal(actual_polygons, post_parcels_gpkg_path):
            raise ContractError(
                "postprocess_manifest parcels_gpkg_path mismatch: "
                f"manifest={actual_polygons}, provided={post_parcels_gpkg_path}."
            )
    return manifest


def _resolve_vector_crs_string(src: Any, *, role: str) -> str:
    rasterio, _ = _require_rasterio()
    if getattr(src, "crs_wkt", None):
        try:
            crs = rasterio.crs.CRS.from_wkt(src.crs_wkt)
        except Exception as exc:
            raise SpatialContractError(f"{role}: failed to parse vector CRS WKT: {exc}") from exc
        return _crs_to_string(crs, role=role)

    if getattr(src, "crs", None):
        try:
            crs = rasterio.crs.CRS.from_user_input(src.crs)
        except Exception as exc:
            raise SpatialContractError(f"{role}: failed to parse vector CRS: {exc}") from exc
        return _crs_to_string(crs, role=role)

    raise SpatialContractError(f"{role}: vector CRS is missing.")


def _read_vector_metadata(
    path: Path,
    *,
    role: str,
    expected_crs: str,
    require_polygon_confidence: bool,
) -> EvaluationVectorMetadata:
    fiona = _require_fiona()
    with fiona.open(path) as src:
        crs = _resolve_vector_crs_string(src, role=role)
        if crs != expected_crs:
            raise SpatialContractError(
                f"CRS mismatch: {role} has {crs!r}, expected raster CRS {expected_crs!r}."
            )

        layer_name = str(getattr(src, "name", "layer"))
        geometry_type = str(src.schema.get("geometry", ""))
        geometry_type_normalized = geometry_type.strip()
        if geometry_type_normalized == "":
            raise ContractError(f"{role}: vector schema.geometry is missing.")
        if "Polygon" not in geometry_type_normalized:
            if geometry_type_normalized != "Unknown":
                raise ContractError(
                    f"{role}: expected polygonal geometry type, got {geometry_type!r}."
                )
            # Some GPKG writers emit schema.geometry='Unknown' even when all
            # features are polygonal (Polygon/MultiPolygon). Accept this only
            # after explicit per-feature geometry-type verification.
            for feat in src:
                geom = feat.get("geometry")
                if geom is None:
                    continue
                gtype = str(geom.get("type", ""))
                if "Polygon" not in gtype:
                    raise ContractError(
                        f"{role}: non-polygon geometry {gtype!r} found while "
                        "schema.geometry='Unknown'."
                    )

        props = src.schema.get("properties", {})
        if not isinstance(props, Mapping):
            raise ContractError(f"{role}: vector schema.properties must be a mapping/object.")
        has_polygon_conf = "polygon_confidence" in props
        if require_polygon_confidence and not has_polygon_conf:
            raise ContractError(
                f"{role}: required field 'polygon_confidence' is missing from vector schema."
            )

        feature_count = len(src)
        if feature_count < 0:
            raise ContractError(f"{role}: invalid feature count {feature_count}.")

    return EvaluationVectorMetadata(
        role=role,
        path=path,
        crs=crs,
        layer_name=layer_name,
        geometry_type=geometry_type,
        feature_count=int(feature_count),
        polygon_confidence_present=bool(has_polygon_conf),
    )


def _build_object_track_readiness(
    *,
    gt_parcels: EvaluationVectorMetadata | None,
    post_parcels_vector: EvaluationVectorMetadata | None,
) -> tuple[bool, str | None]:
    missing: list[str] = []
    if gt_parcels is None:
        missing.append("gt_parcels_path")
    if post_parcels_vector is None:
        missing.append("post_parcels_gpkg_path")
    if missing:
        return False, "missing required vector sources: " + ", ".join(missing)

    assert gt_parcels is not None
    assert post_parcels_vector is not None

    if gt_parcels.feature_count == 0 or post_parcels_vector.feature_count == 0:
        return (
            False,
            "vector sources contain zero features: "
            f"gt={gt_parcels.feature_count}, pred={post_parcels_vector.feature_count}.",
        )
    return True, None


def _iter_windows(*, height: int, width: int, window_size: int = _STREAM_WINDOW_SIZE):
    rasterio, _ = _require_rasterio()
    Window = rasterio.windows.Window
    for row_off in range(0, height, window_size):
        win_h = min(window_size, height - row_off)
        for col_off in range(0, width, window_size):
            win_w = min(window_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def _read_window(
    ds: Any,
    indexes: int | tuple[int, ...],
    *,
    window: Any,
    role: str,
) -> np.ndarray:
    arr = ds.read(indexes, window=window)
    if not np.isfinite(arr).all():
        raise ContractError(f"{role} contains non-finite values.")
    return arr


def _validate_streamed_semantics(
    *,
    gt_extent_path: Path,
    gt_boundary_path: Path,
    gt_valid_path: Path,
    pred_extent_prob_path: Path,
    pred_boundary_prob_path: Path,
    pred_distance_pred_path: Path,
    pred_valid_path: Path,
    gt_distance_path: Path | None,
    post_parcel_instance_path: Path | None,
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    rasterio, rasterio_errors = _require_rasterio()
    expected_h, expected_w = expected_shape

    gt_extent_unique: set[int] = set()
    gt_boundary_unique: set[int] = set()
    gt_valid_unique: set[int] = set()
    pred_valid_unique: set[int] = set()
    post_instance_unique: set[int] | None = set() if post_parcel_instance_path is not None else None

    pred_extent_min = float("inf")
    pred_extent_max = float("-inf")
    pred_boundary_min = float("inf")
    pred_boundary_max = float("-inf")
    pred_boundary_sum_min = float("inf")
    pred_boundary_sum_max = float("-inf")
    pred_distance_min = float("inf")
    pred_distance_max = float("-inf")
    gt_distance_min = float("inf")
    gt_distance_max = float("-inf")

    total_valid_pixels = 0

    try:
        with (
            rasterio.open(gt_extent_path) as gt_extent_ds,
            rasterio.open(gt_boundary_path) as gt_boundary_ds,
            rasterio.open(gt_valid_path) as gt_valid_ds,
            rasterio.open(pred_extent_prob_path) as pred_extent_ds,
            rasterio.open(pred_boundary_prob_path) as pred_boundary_ds,
            rasterio.open(pred_distance_pred_path) as pred_distance_ds,
            rasterio.open(pred_valid_path) as pred_valid_ds,
        ):
            gt_distance_ds = rasterio.open(gt_distance_path) if gt_distance_path is not None else None
            post_instance_ds = (
                rasterio.open(post_parcel_instance_path)
                if post_parcel_instance_path is not None
                else None
            )
            try:
                for ds, role in (
                    (gt_extent_ds, "gt_extent"),
                    (gt_boundary_ds, "gt_boundary"),
                    (gt_valid_ds, "gt_valid"),
                    (pred_extent_ds, "pred_extent_prob"),
                    (pred_boundary_ds, "pred_boundary_prob"),
                    (pred_distance_ds, "pred_distance_pred"),
                    (pred_valid_ds, "pred_valid"),
                ):
                    if (ds.height, ds.width) != (expected_h, expected_w):
                        raise SpatialContractError(
                            f"{role} shape mismatch: expected {expected_shape}, "
                            f"got {(ds.height, ds.width)}."
                        )
                if not np.issubdtype(np.dtype(gt_extent_ds.dtypes[0]), np.integer):
                    raise ContractError("gt_extent must be integer-encoded labels.")
                if not np.issubdtype(np.dtype(gt_boundary_ds.dtypes[0]), np.integer):
                    raise ContractError("gt_boundary must be integer-encoded labels.")
                if gt_distance_ds is not None and (gt_distance_ds.height, gt_distance_ds.width) != (
                    expected_h,
                    expected_w,
                ):
                    raise SpatialContractError(
                        f"gt_distance shape mismatch: expected {expected_shape}, "
                        f"got {(gt_distance_ds.height, gt_distance_ds.width)}."
                    )
                if post_instance_ds is not None and (
                    post_instance_ds.height,
                    post_instance_ds.width,
                ) != (expected_h, expected_w):
                    raise SpatialContractError(
                        "post_parcel_instance shape mismatch: "
                        f"expected {expected_shape}, got "
                        f"{(post_instance_ds.height, post_instance_ds.width)}."
                    )
                if post_instance_ds is not None and not np.issubdtype(
                    np.dtype(post_instance_ds.dtypes[0]),
                    np.integer,
                ):
                    raise ContractError("post_parcel_instance must be integer-encoded labels.")

                for window in _iter_windows(height=expected_h, width=expected_w):
                    gt_extent = _read_window(
                        gt_extent_ds, 1, window=window, role="gt_extent"
                    )
                    gt_boundary = _read_window(
                        gt_boundary_ds, 1, window=window, role="gt_boundary"
                    )
                    gt_valid = _read_window(
                        gt_valid_ds, 1, window=window, role="gt_valid"
                    )
                    pred_extent = _read_window(
                        pred_extent_ds, 1, window=window, role="pred_extent_prob"
                    ).astype(np.float32)
                    pred_boundary = _read_window(
                        pred_boundary_ds, (1, 2, 3), window=window, role="pred_boundary_prob"
                    ).astype(np.float32)
                    pred_distance = _read_window(
                        pred_distance_ds, 1, window=window, role="pred_distance_pred"
                    ).astype(np.float32)
                    pred_valid = _read_window(
                        pred_valid_ds, 1, window=window, role="pred_valid"
                    )

                    if pred_boundary.ndim != 3 or pred_boundary.shape[0] != 3:
                        raise ContractError(
                            "pred_boundary_prob must be 3-band while streaming semantic checks."
                        )

                    gt_extent_unique.update(int(v) for v in np.unique(gt_extent).tolist())
                    gt_boundary_unique.update(int(v) for v in np.unique(gt_boundary).tolist())

                    gt_valid_chunk_unique = np.unique(gt_valid)
                    pred_valid_chunk_unique = np.unique(pred_valid)
                    gt_valid_unique.update(int(v) for v in gt_valid_chunk_unique.tolist())
                    pred_valid_unique.update(int(v) for v in pred_valid_chunk_unique.tolist())

                    if not np.all(np.isin(gt_valid_chunk_unique, list(_BINARY_LABELS))):
                        raise ValidPolicyError(
                            "gt_valid must be binary {0,1}, "
                            f"got unique values {sorted(gt_valid_unique)}."
                        )
                    if not np.all(np.isin(pred_valid_chunk_unique, list(_BINARY_LABELS))):
                        raise ValidPolicyError(
                            "pred_valid must be binary {0,1}, "
                            f"got unique values {sorted(pred_valid_unique)}."
                        )

                    gt_valid01 = gt_valid > 0
                    pred_valid01 = pred_valid > 0
                    if not np.array_equal(gt_valid01, pred_valid01):
                        raise ValidPolicyError(
                            "gt_valid and pred_valid masks differ; valid-policy mismatch is not allowed."
                        )
                    total_valid_pixels += int(pred_valid01.sum())

                    pred_extent_min = min(pred_extent_min, float(pred_extent.min()))
                    pred_extent_max = max(pred_extent_max, float(pred_extent.max()))
                    if pred_extent_min < -_EPS or pred_extent_max > 1.0 + _EPS:
                        raise ContractError(
                            "pred_extent_prob must be probability-like in [0,1], "
                            f"got min={pred_extent_min}, max={pred_extent_max}."
                        )

                    pred_boundary_min = min(pred_boundary_min, float(pred_boundary.min()))
                    pred_boundary_max = max(pred_boundary_max, float(pred_boundary.max()))
                    if pred_boundary_min < -_EPS or pred_boundary_max > 1.0 + _EPS:
                        raise ContractError(
                            "pred_boundary_prob must be probability-like in [0,1], "
                            f"got min={pred_boundary_min}, max={pred_boundary_max}."
                        )

                    valid_pixels = pred_valid01
                    if np.any(valid_pixels):
                        sums_valid = pred_boundary.sum(axis=0)[valid_pixels]
                        pred_boundary_sum_min = min(pred_boundary_sum_min, float(sums_valid.min()))
                        pred_boundary_sum_max = max(pred_boundary_sum_max, float(sums_valid.max()))
                        if not np.all(np.abs(sums_valid - 1.0) <= _BOUNDARY_SUM_ATOL):
                            raise ContractError(
                                "pred_boundary_prob on pred_valid pixels must represent a 3-class "
                                "probability simplex (sum≈1). "
                                f"Observed sum range [{pred_boundary_sum_min}, {pred_boundary_sum_max}]."
                            )

                    pred_distance_min = min(pred_distance_min, float(pred_distance.min()))
                    pred_distance_max = max(pred_distance_max, float(pred_distance.max()))
                    if pred_distance_min < -_EPS:
                        raise ContractError(
                            "pred_distance_pred must be non-negative for unsigned distance contract, "
                            f"got min={pred_distance_min}."
                        )

                    if gt_distance_ds is not None:
                        gt_distance = _read_window(
                            gt_distance_ds, 1, window=window, role="gt_distance"
                        ).astype(np.float32)
                        gt_distance_min = min(gt_distance_min, float(gt_distance.min()))
                        gt_distance_max = max(gt_distance_max, float(gt_distance.max()))
                        if gt_distance_min < -_EPS:
                            raise ContractError(
                                "gt_distance must be non-negative for unsigned distance contract, "
                                f"got min={gt_distance_min}."
                            )

                    if post_instance_ds is not None:
                        post_instance = _read_window(
                            post_instance_ds, 1, window=window, role="post_parcel_instance"
                        )
                        min_label = int(post_instance.min())
                        if min_label < -1:
                            raise ContractError(
                                "post_parcel_instance labels below -1 are not supported. "
                                f"Observed min label={min_label}."
                            )
                        assert post_instance_unique is not None
                        post_instance_unique.update(
                            int(v) for v in np.unique(post_instance).tolist()
                        )
            finally:
                if gt_distance_ds is not None:
                    gt_distance_ds.close()
                if post_instance_ds is not None:
                    post_instance_ds.close()
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to stream raster values for semantic checks: {exc}") from exc

    if not gt_extent_unique.issubset(_GT_EXTENT_LABELS):
        raise ContractError(
            f"gt_extent has unsupported labels: {sorted(gt_extent_unique)}, "
            f"allowed={sorted(_GT_EXTENT_LABELS)}."
        )
    if not gt_boundary_unique.issubset(_GT_BOUNDARY_LABELS):
        raise ContractError(
            f"gt_boundary has unsupported labels: {sorted(gt_boundary_unique)}, "
            f"allowed={sorted(_GT_BOUNDARY_LABELS)}."
        )
    if total_valid_pixels <= 0:
        raise ValidPolicyError("pred_valid contains zero valid pixels for eval.")

    return {
        "gt_extent_unique_values": tuple(sorted(gt_extent_unique)),
        "gt_boundary_unique_values": tuple(sorted(gt_boundary_unique)),
        "gt_valid_unique_values": tuple(sorted(gt_valid_unique)),
        "pred_valid_unique_values": tuple(sorted(pred_valid_unique)),
        "pred_extent_prob_range": (pred_extent_min, pred_extent_max),
        "pred_boundary_prob_range": (pred_boundary_min, pred_boundary_max),
        "pred_boundary_prob_sum_range_on_valid": (pred_boundary_sum_min, pred_boundary_sum_max),
        "pred_distance_pred_range": (pred_distance_min, pred_distance_max),
        "gt_distance_range": (
            (gt_distance_min, gt_distance_max) if gt_distance_path is not None else None
        ),
        "post_parcel_instance_unique_values": (
            tuple(sorted(post_instance_unique))
            if post_instance_unique is not None
            else None
        ),
        "valid_mask_consistent_between_gt_and_pred": True,
    }


def resolve_evaluation_input_contract(
    *,
    gt_extent_path: str | Path,
    gt_boundary_path: str | Path,
    gt_valid_path: str | Path,
    pred_extent_prob_path: str | Path,
    pred_boundary_prob_path: str | Path,
    pred_distance_pred_path: str | Path,
    pred_valid_path: str | Path,
    gt_distance_path: str | Path | None = None,
    gt_parcels_path: str | Path | None = None,
    post_parcel_instance_path: str | Path | None = None,
    post_parcels_gpkg_path: str | Path | None = None,
    predict_manifest_path: str | Path | None = None,
    postprocess_manifest_path: str | Path | None = None,
) -> EvaluationInputContractResult:
    """Resolve and validate eval input contract (Stage A)."""
    gt_extent_path = _normalize_existing_path(gt_extent_path, name="gt_extent_path")
    gt_boundary_path = _normalize_existing_path(gt_boundary_path, name="gt_boundary_path")
    gt_valid_path = _normalize_existing_path(gt_valid_path, name="gt_valid_path")

    pred_extent_prob_path = _normalize_existing_path(
        pred_extent_prob_path, name="pred_extent_prob_path"
    )
    pred_boundary_prob_path = _normalize_existing_path(
        pred_boundary_prob_path, name="pred_boundary_prob_path"
    )
    pred_distance_pred_path = _normalize_existing_path(
        pred_distance_pred_path, name="pred_distance_pred_path"
    )
    pred_valid_path = _normalize_existing_path(pred_valid_path, name="pred_valid_path")

    gt_distance_path = _normalize_optional_existing_path(gt_distance_path, name="gt_distance_path")
    gt_parcels_path = _normalize_optional_existing_path(gt_parcels_path, name="gt_parcels_path")
    post_parcel_instance_path = _normalize_optional_existing_path(
        post_parcel_instance_path, name="post_parcel_instance_path"
    )
    post_parcels_gpkg_path = _normalize_optional_existing_path(
        post_parcels_gpkg_path, name="post_parcels_gpkg_path"
    )
    predict_manifest_path = _normalize_optional_existing_path(
        predict_manifest_path, name="predict_manifest_path"
    )
    postprocess_manifest_path = _normalize_optional_existing_path(
        postprocess_manifest_path, name="postprocess_manifest_path"
    )

    gt_extent_meta = _read_raster_metadata(gt_extent_path, role="gt_extent")
    gt_boundary_meta = _read_raster_metadata(gt_boundary_path, role="gt_boundary")
    gt_valid_meta = _read_raster_metadata(gt_valid_path, role="gt_valid")

    pred_extent_meta = _read_raster_metadata(pred_extent_prob_path, role="pred_extent_prob")
    pred_boundary_meta = _read_raster_metadata(pred_boundary_prob_path, role="pred_boundary_prob")
    pred_distance_meta = _read_raster_metadata(pred_distance_pred_path, role="pred_distance_pred")
    pred_valid_meta = _read_raster_metadata(pred_valid_path, role="pred_valid")

    gt_distance_meta: EvaluationRasterMetadata | None = None
    if gt_distance_path is not None:
        gt_distance_meta = _read_raster_metadata(gt_distance_path, role="gt_distance")

    post_parcel_instance_meta: EvaluationRasterMetadata | None = None
    if post_parcel_instance_path is not None:
        post_parcel_instance_meta = _read_raster_metadata(
            post_parcel_instance_path, role="post_parcel_instance"
        )

    for candidate in (
        gt_boundary_meta,
        gt_valid_meta,
        pred_extent_meta,
        pred_boundary_meta,
        pred_distance_meta,
        pred_valid_meta,
    ):
        _validate_spatial_alignment(gt_extent_meta, candidate)
    if gt_distance_meta is not None:
        _validate_spatial_alignment(gt_extent_meta, gt_distance_meta)
    if post_parcel_instance_meta is not None:
        _validate_spatial_alignment(gt_extent_meta, post_parcel_instance_meta)

    _require_band_count(gt_extent_meta, expected=1)
    _require_band_count(gt_boundary_meta, expected=1)
    _require_band_count(gt_valid_meta, expected=1)
    _require_band_count(pred_extent_meta, expected=1)
    _require_band_count(pred_boundary_meta, expected=3)
    _require_band_count(pred_distance_meta, expected=1)
    _require_band_count(pred_valid_meta, expected=1)
    if gt_distance_meta is not None:
        _require_band_count(gt_distance_meta, expected=1)
    if post_parcel_instance_meta is not None:
        _require_band_count(post_parcel_instance_meta, expected=1)

    _require_integer_or_bool_dtype(gt_extent_meta)
    _require_integer_or_bool_dtype(gt_boundary_meta)
    _require_integer_or_bool_dtype(gt_valid_meta)
    _require_floating_dtype(pred_extent_meta)
    _require_floating_dtype(pred_boundary_meta)
    _require_floating_dtype(pred_distance_meta)
    _require_integer_or_bool_dtype(pred_valid_meta)
    if gt_distance_meta is not None:
        _require_floating_dtype(gt_distance_meta)
    if post_parcel_instance_meta is not None:
        _require_integer_or_bool_dtype(post_parcel_instance_meta)

    expected_shape = (int(gt_extent_meta.height), int(gt_extent_meta.width))

    semantic_summary = _validate_streamed_semantics(
        gt_extent_path=gt_extent_path,
        gt_boundary_path=gt_boundary_path,
        gt_valid_path=gt_valid_path,
        pred_extent_prob_path=pred_extent_prob_path,
        pred_boundary_prob_path=pred_boundary_prob_path,
        pred_distance_pred_path=pred_distance_pred_path,
        pred_valid_path=pred_valid_path,
        gt_distance_path=gt_distance_path,
        post_parcel_instance_path=post_parcel_instance_path,
        expected_shape=expected_shape,
    )

    predict_manifest: dict[str, Any] | None = None
    if predict_manifest_path is not None:
        predict_manifest = _validate_predict_manifest_consistency(
            predict_manifest_path=predict_manifest_path,
            pred_extent_prob_path=pred_extent_prob_path,
            pred_boundary_prob_path=pred_boundary_prob_path,
            pred_distance_pred_path=pred_distance_pred_path,
            pred_valid_path=pred_valid_path,
        )

    postprocess_manifest: dict[str, Any] | None = None
    if postprocess_manifest_path is not None:
        postprocess_manifest = _validate_postprocess_manifest_consistency(
            postprocess_manifest_path=postprocess_manifest_path,
            pred_extent_prob_path=pred_extent_prob_path,
            pred_boundary_prob_path=pred_boundary_prob_path,
            pred_distance_pred_path=pred_distance_pred_path,
            pred_valid_path=pred_valid_path,
            post_parcel_instance_path=post_parcel_instance_path,
            post_parcels_gpkg_path=post_parcels_gpkg_path,
        )

    gt_parcels_meta: EvaluationVectorMetadata | None = None
    if gt_parcels_path is not None:
        gt_parcels_meta = _read_vector_metadata(
            gt_parcels_path,
            role="gt_parcels",
            expected_crs=gt_extent_meta.crs,
            require_polygon_confidence=False,
        )

    post_parcels_meta: EvaluationVectorMetadata | None = None
    if post_parcels_gpkg_path is not None:
        post_parcels_meta = _read_vector_metadata(
            post_parcels_gpkg_path,
            role="post_parcels",
            expected_crs=gt_extent_meta.crs,
            require_polygon_confidence=True,
        )

    object_ready, object_reason = _build_object_track_readiness(
        gt_parcels=gt_parcels_meta,
        post_parcels_vector=post_parcels_meta,
    )

    track_readiness = EvaluationTrackReadiness(
        pixel_ready=True,
        boundary_ready=True,
        object_structure_ready=object_ready,
        pixel_reason=None,
        boundary_reason=None,
        object_structure_reason=object_reason,
    )

    output_contract = EvaluationOutputContractSkeleton(
        pixel_metrics=("extent_iou", "extent_f1", "extent_precision", "extent_recall"),
        boundary_metrics=("boundary_f1", "boundary_precision", "boundary_recall", "bde"),
        object_structure_metrics=("goc", "guc", "gtc"),
    )

    source_run_ids: list[str] = []
    if isinstance(predict_manifest, Mapping):
        run_id = predict_manifest.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            source_run_ids.append(run_id)
    if isinstance(postprocess_manifest, Mapping):
        run_id = postprocess_manifest.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            source_run_ids.append(run_id)

    return EvaluationInputContractResult(
        gt_extent=gt_extent_meta,
        gt_boundary=gt_boundary_meta,
        gt_valid=gt_valid_meta,
        gt_distance=gt_distance_meta,
        gt_parcels=gt_parcels_meta,
        pred_extent_prob=pred_extent_meta,
        pred_boundary_prob=pred_boundary_meta,
        pred_distance_pred=pred_distance_meta,
        pred_valid=pred_valid_meta,
        post_parcel_instance=post_parcel_instance_meta,
        post_parcels_vector=post_parcels_meta,
        common_width=int(gt_extent_meta.width),
        common_height=int(gt_extent_meta.height),
        common_crs=gt_extent_meta.crs,
        common_transform_gdal=gt_extent_meta.transform_gdal,
        semantic_compatibility_summary=semantic_summary,
        track_readiness=track_readiness,
        output_contract=output_contract,
        predict_manifest_path=predict_manifest_path,
        postprocess_manifest_path=postprocess_manifest_path,
        source_run_ids=tuple(source_run_ids),
        compatible=True,
    )


__all__ = [
    "EvaluationInputContractResult",
    "EvaluationOutputContractSkeleton",
    "EvaluationRasterMetadata",
    "EvaluationTrackReadiness",
    "EvaluationVectorMetadata",
    "resolve_evaluation_input_contract",
]
