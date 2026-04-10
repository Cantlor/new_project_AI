"""Stage E artifact export layer for module_postprocess_vectorize.

This module writes the postprocess artifact set:
  - postprocess_manifest.json
  - summary.json
  - config_used.yaml

It is intentionally narrow and contract-first:
  - consumes resolved outputs from Stages A-D;
  - preserves provenance/policy transparency;
  - avoids module_eval coupling or orchestration frameworks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import write_manifest, write_summary
from ai_fields.module_postprocess_vectorize.input_contract import (
    PostprocessInputContractResult,
)
from ai_fields.module_postprocess_vectorize.instance_core import (
    ParcelInstanceRasterResult,
    WatershedCorePolicy,
)
from ai_fields.module_postprocess_vectorize.marker_generation import (
    MarkerGenerationResult,
    MarkerThresholdPolicy,
)
from ai_fields.module_postprocess_vectorize.polygonization import (
    PolygonizationPolicy,
    PostprocessPolygonizationResult,
)

_POSTPROCESS_MANIFEST_SCHEMA = "postprocess_vectorize.postprocess_manifest"
_POSTPROCESS_SUMMARY_SCHEMA = "postprocess_vectorize.summary"


@dataclass(frozen=True)
class PostprocessExportArtifacts:
    """Paths of Stage E artifacts for module_postprocess_vectorize."""

    run_dir: Path
    postprocess_manifest_path: Path
    summary_path: Path
    config_used_path: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_output_dir(path: Any) -> Path:
    if not isinstance(path, (str, PathLike)):
        raise ContractError(
            f"output_dir must be path-like (str/Path), got {type(path).__name__}."
        )
    resolved = Path(path)
    if str(resolved).strip() == "":
        raise ContractError("output_dir must be a non-empty path-like value.")
    return resolved


def _normalize_optional_existing_path(path: Any, *, name: str) -> Path | None:
    if path is None:
        return None
    if not isinstance(path, (str, PathLike)):
        raise ContractError(f"{name} must be path-like when provided.")
    resolved = Path(path)
    if str(resolved).strip() == "":
        raise ContractError(f"{name} must be non-empty when provided.")
    if not resolved.exists() or not resolved.is_file():
        raise ContractError(f"{name} must point to an existing file: {resolved}")
    return resolved


def _require_non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{name} must be a non-empty string.")
    return value


def _policy_as_dict(policy: Any, *, name: str) -> dict[str, Any]:
    if is_dataclass(policy):
        return asdict(policy)
    if isinstance(policy, Mapping):
        return dict(policy)
    raise ContractError(f"{name} must be a dataclass instance or mapping/object.")


def _read_valid_coverage(valid_path: Path) -> dict[str, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required to compute valid coverage summary for Stage E."
        ) from exc

    try:
        with rasterio.open(valid_path) as ds:
            arr = ds.read()
    except rasterio.errors.RasterioIOError as exc:
        raise ContractError(f"Failed to read valid raster for summary: {valid_path} ({exc})") from exc

    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ContractError(f"valid raster must be single-band, got shape={arr.shape}.")
    valid_raw = arr[0]
    unique = np.unique(valid_raw)
    if not np.all(np.isin(unique, [0, 1])):
        raise ContractError(
            "valid raster must be binary {0,1} to compute valid coverage; "
            f"got {unique.tolist()}."
        )
    valid01 = (valid_raw > 0).astype(np.uint8)
    valid_pixels = int(valid01.sum())
    total_pixels = int(valid01.size)
    invalid_pixels = int(total_pixels - valid_pixels)
    return {
        "valid_pixels": valid_pixels,
        "invalid_pixels": invalid_pixels,
        "total_pixels": total_pixels,
        "valid_fraction": float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0,
    }


def _read_polygon_confidence_summary(gpkg_path: Path, *, layer_name: str) -> dict[str, Any]:
    try:
        import fiona
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "fiona is required to read polygon_confidence summary from GPKG."
        ) from exc

    with fiona.open(gpkg_path, layer=layer_name) as src:
        props = src.schema.get("properties", {})
        if "polygon_confidence" not in props:
            raise ContractError(
                "parcels output layer must contain polygon_confidence field for Stage E."
            )
        vals: list[float] = []
        for feat in src:
            props_row = feat.get("properties", {})
            if "polygon_confidence" not in props_row:
                raise ContractError("Encountered feature without polygon_confidence field.")
            vals.append(float(props_row["polygon_confidence"]))

    if not vals:
        raise ContractError("parcels.gpkg contains zero polygons; cannot build confidence summary.")

    arr = np.asarray(vals, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ContractError("polygon_confidence contains non-finite values.")
    if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        raise ContractError(
            "polygon_confidence values must be in [0,1], got "
            f"[{float(arr.min())}, {float(arr.max())}]."
        )

    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _write_config_used(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ContractError("PyYAML is required to write config_used.yaml.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            yaml.safe_dump(dict(payload), sort_keys=True, allow_unicode=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ContractError(f"Failed to write config_used.yaml at {path}: {exc}") from exc


def export_postprocess_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    input_contract: PostprocessInputContractResult,
    marker_result: MarkerGenerationResult,
    instance_result: ParcelInstanceRasterResult,
    polygon_result: PostprocessPolygonizationResult,
    marker_policy: MarkerThresholdPolicy,
    watershed_policy: WatershedCorePolicy,
    polygonization_policy: PolygonizationPolicy,
    parcel_instance_path: str | Path | None = None,
    source_predict_run_id: str | None = None,
    source_predict_manifest_path: str | Path | None = None,
    aoi_manifest_path: str | Path | None = None,
    aoi_suppression_applied: bool = False,
    boundary_repair_applied: bool = False,
    extra_effective_config: Mapping[str, Any] | None = None,
) -> PostprocessExportArtifacts:
    """Write Stage E postprocess artifact metadata for Stages A-D outputs."""
    run_id = _require_non_empty_string(run_id, name="run_id")
    run_dir = _normalize_output_dir(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(input_contract, PostprocessInputContractResult):
        raise ContractError("input_contract must be PostprocessInputContractResult.")
    if not isinstance(marker_result, MarkerGenerationResult):
        raise ContractError("marker_result must be MarkerGenerationResult.")
    if not isinstance(instance_result, ParcelInstanceRasterResult):
        raise ContractError("instance_result must be ParcelInstanceRasterResult.")
    if not isinstance(polygon_result, PostprocessPolygonizationResult):
        raise ContractError("polygon_result must be PostprocessPolygonizationResult.")
    if not isinstance(marker_policy, MarkerThresholdPolicy):
        raise ContractError("marker_policy must be MarkerThresholdPolicy.")
    if not isinstance(watershed_policy, WatershedCorePolicy):
        raise ContractError("watershed_policy must be WatershedCorePolicy.")
    if not isinstance(polygonization_policy, PolygonizationPolicy):
        raise ContractError("polygonization_policy must be PolygonizationPolicy.")

    if marker_result.ready_for_stage_c is not True:
        raise ContractError("marker_result.ready_for_stage_c must be True for Stage E export.")
    if instance_result.ready_for_stage_d is not True:
        raise ContractError("instance_result.ready_for_stage_d must be True for Stage E export.")
    if polygon_result.ready_for_stage_e is not True:
        raise ContractError("polygon_result.ready_for_stage_e must be True for Stage E export.")

    parcels_gpkg_path = Path(polygon_result.parcels_gpkg_path)
    if not parcels_gpkg_path.exists() or not parcels_gpkg_path.is_file():
        raise ContractError(f"parcels_gpkg_path does not exist: {parcels_gpkg_path}")

    if polygon_result.polygon_confidence_present is not True:
        raise ContractError("polygon_result must report polygon_confidence_present=True.")

    if polygon_result.crs != input_contract.common_crs:
        raise ContractError(
            "CRS mismatch between Stage A input contract and Stage D polygon output: "
            f"{input_contract.common_crs!r} vs {polygon_result.crs!r}."
        )

    resolved_parcel_instance_path = _normalize_optional_existing_path(
        parcel_instance_path,
        name="parcel_instance_path",
    )
    resolved_source_predict_manifest_path = _normalize_optional_existing_path(
        source_predict_manifest_path,
        name="source_predict_manifest_path",
    )

    if source_predict_run_id is not None:
        source_predict_run_id = _require_non_empty_string(
            source_predict_run_id, name="source_predict_run_id"
        )

    valid_coverage = _read_valid_coverage(input_contract.valid.path)
    polygon_conf_summary = _read_polygon_confidence_summary(
        parcels_gpkg_path,
        layer_name=polygon_result.layer_name,
    )
    if polygon_conf_summary["count"] != int(polygon_result.polygon_count):
        raise ContractError(
            "Polygon count mismatch between Stage D result and written GPKG: "
            f"{polygon_result.polygon_count} vs {polygon_conf_summary['count']}."
        )

    created_at_utc = _utc_now_iso()
    config_used_path = run_dir / "config_used.yaml"
    manifest_path = run_dir / "postprocess_manifest.json"
    summary_path = run_dir / "summary.json"

    effective_config: dict[str, Any] = {
        "module_name": "module_postprocess_vectorize",
        "run_id": run_id,
        "stage_coverage": {
            "stage_a_input_contract": True,
            "stage_b_marker_generation": True,
            "stage_c_instance_core": True,
            "stage_d_polygonization_cleanup_confidence": True,
            "stage_e_artifact_export": True,
            "stage_not_implemented": {
                "module_eval_runtime": True,
                "advanced_topology_framework": True,
            },
        },
        "source_predict": {
            "source_predict_run_id": source_predict_run_id,
            "source_predict_manifest_path": (
                str(resolved_source_predict_manifest_path)
                if resolved_source_predict_manifest_path is not None
                else None
            ),
        },
        "inputs": {
            "extent_prob_path": str(input_contract.extent_prob.path),
            "boundary_prob_path": str(input_contract.boundary_prob.path),
            "distance_pred_path": str(input_contract.distance_pred.path),
            "valid_path": str(input_contract.valid.path),
            "aoi_path": str(input_contract.aoi_path) if input_contract.aoi_path is not None else None,
        },
        "effective_policy_contract": {
            "marker_policy": _policy_as_dict(marker_policy, name="marker_policy"),
            "watershed_policy": _policy_as_dict(watershed_policy, name="watershed_policy"),
            "polygonization_policy": _policy_as_dict(
                polygonization_policy, name="polygonization_policy"
            ),
        },
        "outputs": {
            "parcel_instance_path": (
                str(resolved_parcel_instance_path)
                if resolved_parcel_instance_path is not None
                else None
            ),
            "parcels_gpkg_path": str(parcels_gpkg_path),
        },
    }
    if extra_effective_config is not None:
        if not isinstance(extra_effective_config, Mapping):
            raise ContractError("extra_effective_config must be a mapping/object when provided.")
        effective_config["extra_effective_config"] = dict(extra_effective_config)

    _write_config_used(config_used_path, effective_config)

    manifest_payload: dict[str, Any] = {
        "schema_name": _POSTPROCESS_MANIFEST_SCHEMA,
        "schema_version": "v1",
        "module_name": "module_postprocess_vectorize",
        "module_version": None,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "run_id": run_id,
        "stage_name": "export_postprocess_artifacts",
        "created_at_utc": created_at_utc,
        "status": "success",
        "source_predict_run_id": source_predict_run_id,
        "source_predict_manifest_path": (
            str(resolved_source_predict_manifest_path)
            if resolved_source_predict_manifest_path is not None
            else None
        ),
        "inputs": {
            "extent_prob_path": str(input_contract.extent_prob.path),
            "boundary_prob_path": str(input_contract.boundary_prob.path),
            "distance_pred_path": str(input_contract.distance_pred.path),
            "valid_path": str(input_contract.valid.path),
            "aoi_path": str(input_contract.aoi_path) if input_contract.aoi_path is not None else None,
            "aoi_manifest_path": str(aoi_manifest_path) if aoi_manifest_path is not None else None,
        },
        "resolved_input_contract": {
            "common_width": int(input_contract.common_width),
            "common_height": int(input_contract.common_height),
            "common_crs": input_contract.common_crs,
            "common_transform_gdal": list(input_contract.common_transform_gdal),
            "valid_mask_semantics": dict(input_contract.valid_mask_semantics),
            "probability_semantics": dict(input_contract.probability_semantics),
        },
        "resolved_policy": {
            "valid_suppression": "binary_valid_mask_required",
            "aoi_policy": (
                {
                    "mode": (
                        "aoi_suppression_applied"
                        if aoi_suppression_applied
                        else "aoi_path_provided_suppression_not_applied"
                    ),
                    "aoi_path": str(input_contract.aoi_path),
                    "aoi_manifest_path": (
                        str(aoi_manifest_path) if aoi_manifest_path is not None else None
                    ),
                    "suppression_applied": aoi_suppression_applied,
                }
                if input_contract.aoi_path is not None
                else None
            ),
            "threshold_policy": _policy_as_dict(marker_policy, name="marker_policy"),
            "boundary_repair_policy": (
                "morphological_binary_closing_applied"
                if boundary_repair_applied
                else "not_applied"
            ),
            "marker_generation_policy": marker_result.policy.get(
                "formula", "extent_core_intersection_policy"
            ),
            "watershed_policy": watershed_policy.threshold_provenance,
            "filtering_policy": (
                f"min_region_pixels={int(watershed_policy.min_region_pixels)}"
                if int(watershed_policy.min_region_pixels) > 0
                else "no_filtering"
            ),
            "topology_cleanup_policy": polygonization_policy.cleanup_policy_name,
            "export_format": "GPKG",
        },
        "policy_details": {
            "marker_policy": _policy_as_dict(marker_policy, name="marker_policy"),
            "watershed_policy": _policy_as_dict(watershed_policy, name="watershed_policy"),
            "polygonization_policy": _policy_as_dict(
                polygonization_policy, name="polygonization_policy"
            ),
        },
        "outputs": {
            "parcel_instance_path": (
                str(resolved_parcel_instance_path)
                if resolved_parcel_instance_path is not None
                else None
            ),
            "parcel_instance_encoding": {
                "background_label": int(instance_result.background_label),
                "invalid_label": int(instance_result.invalid_label),
                "instance_labels": "1..N positive integers",
                "nodata_value": int(instance_result.invalid_label),
                "note": (
                    "background (valid pixels with no instance assignment) = background_label (0); "
                    "invalid (NoData/valid==0 pixels) = invalid_label (-1); "
                    "parcel instances = 1,2,...,N"
                ),
            },
            "parcels_gpkg_path": str(parcels_gpkg_path),
            "optional_exports": [],
        },
        "artifacts": {
            "config_used_path": str(config_used_path),
            "summary_path": str(summary_path),
            "postprocess_manifest_path": str(manifest_path),
        },
        "stage_coverage": {
            "stage_a": True,
            "stage_b": True,
            "stage_c": True,
            "stage_d": True,
            "stage_e": True,
        },
        "counts": {
            "instance_count": int(instance_result.instance_count),
            "polygon_count": int(polygon_result.polygon_count),
            "polygon_confidence_count": int(polygon_conf_summary["count"]),
        },
    }
    write_manifest(manifest_path, manifest_payload)

    summary_payload: dict[str, Any] = {
        "schema_name": _POSTPROCESS_SUMMARY_SCHEMA,
        "status": "success",
        "run_id": run_id,
        "module_name": "module_postprocess_vectorize",
        "stage_coverage": {
            "implemented": ["A", "B", "C", "D", "E"],
            "not_implemented": ["module_eval_runtime", "giant_topology_framework"],
        },
        "parcel_count": int(polygon_result.polygon_count),
        "instance_count": int(instance_result.instance_count),
        "export_format": "GPKG",
        "output_paths": {
            "parcel_instance_path": (
                str(resolved_parcel_instance_path)
                if resolved_parcel_instance_path is not None
                else None
            ),
            "parcels_gpkg_path": str(parcels_gpkg_path),
        },
        "valid_coverage": valid_coverage,
        "polygon_confidence_summary": polygon_conf_summary,
        "policy_names": {
            "marker_threshold_policy": marker_policy.threshold_provenance,
            "watershed_policy": watershed_policy.threshold_provenance,
            "topology_cleanup_policy": polygonization_policy.cleanup_policy_name,
            "polygon_confidence_policy": polygonization_policy.confidence_policy_name,
        },
        "warnings": [],
        "key_notes": [
            "Stage E records provenance/contracts for Stages A-D.",
            "module_eval runtime is intentionally out of scope at this stage.",
        ],
    }
    write_summary(summary_path, summary_payload)

    return PostprocessExportArtifacts(
        run_dir=run_dir,
        postprocess_manifest_path=manifest_path,
        summary_path=summary_path,
        config_used_path=config_used_path,
    )


__all__ = [
    "PostprocessExportArtifacts",
    "export_postprocess_artifacts",
]

