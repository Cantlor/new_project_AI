"""Lightweight stage skeleton for `02_prepare_spatial_context` in module_prep_data.

This stage is intentionally minimal: it resolves contract-level spatial context
from metadata-like inputs, writes manifest/summary artifacts, and returns a
small test-friendly result object.

When `runtime_compute_enabled=True`, it delegates to `spatial_context_compute`
for runtime CRS reprojection artifacts and effective AOI bounds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError, SpatialContractError
from ai_fields.common.manifests import write_manifest, write_summary
from ai_fields.module_prep_data import config as prep_data_config
from ai_fields.module_prep_data import validators as prep_data_validators
from ai_fields.module_prep_data.schemas import PrepDataConfig

try:
    from ai_fields.module_prep_data import spatial_context_compute
except ImportError:  # pragma: no cover
    spatial_context_compute = None  # type: ignore[assignment]

_STAGE_NAME = "02_prepare_spatial_context"
_MANIFEST_SCHEMA_NAME = "prep_data.aoi_manifest"
_SUMMARY_SCHEMA_NAME = "prep_data.summary"
_SCHEMA_VERSION = "v1"
_MODULE_NAME = "module_prep_data"
_MANIFEST_FILENAME = "aoi_manifest.json"
_SUMMARY_FILENAME = "summary.json"
_INPUT_REFS_SOURCE = "stage_args_transitional"


@dataclass(frozen=True)
class PrepareSpatialContextStageResult:
    """Minimal stage outcome for `02_prepare_spatial_context`."""

    status: Literal["success", "failed"]
    manifest_path: Path
    summary_path: Path
    spatial_context_mode: Literal["full_raster", "aoi_limited"] | None
    aoi_present: bool | None
    aoi_policy_enabled: bool | None
    resolved_buffer_m: float | None
    blocking_issues: tuple[str, ...]
    checks: dict[str, Any]
    error_type: str | None = None
    error_message: str | None = None
    # Expose resolved spatial artifacts so the orchestrator can reference them
    # without re-reading the manifest JSON.
    aoi_output_path: Path | None = None
    effective_extent_bounds: tuple[float, float, float, float] | None = None
    aoi_source_type: str | None = None  # "user_provided" | "derived_from_labels" | None
    aoi_derivation_method: str | None = None  # "labels_bbox_buffered" | None

    @property
    def success(self) -> bool:
        return self.status == "success"


def _require_non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{name} must be a non-empty string, got {value!r}.")
    return value


def _normalize_path(name: str, value: Any) -> Path:
    if isinstance(value, PathLike):
        as_str = str(value)
    elif isinstance(value, str):
        as_str = value
    else:
        raise ContractError(
            f"{name} must be path-like (str or Path), got {value!r} ({type(value).__name__})."
        )
    if as_str.strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    return Path(value)


def _safe_path_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, PathLike)):
        as_text = str(value)
        return as_text if as_text.strip() != "" else None
    return None


def _now_utc_iso() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return ts.replace("+00:00", "Z")


def _resolve_config(
    *,
    config: PrepDataConfig | Mapping[str, Any] | None,
    config_path: str | Path | None,
) -> tuple[PrepDataConfig, str]:
    if (config is None) == (config_path is None):
        raise ContractError("Provide exactly one of: config or config_path.")

    if config_path is not None:
        cfg_path = _normalize_path("config_path", config_path)
        cfg = prep_data_config.load_config(cfg_path)
        return cfg, str(cfg_path)

    assert config is not None
    if isinstance(config, PrepDataConfig):
        config.validate()
        return config, "<in-memory:PrepDataConfig>"
    if isinstance(config, Mapping):
        cfg = prep_data_config.build_config(dict(config))
        return cfg, "<in-memory:mapping>"
    raise ContractError(
        f"config must be PrepDataConfig or mapping/object, got {type(config).__name__}."
    )


def _extract_aoi_bounds_metadata_hint(aoi_metadata: Any) -> list[float] | None:
    if not isinstance(aoi_metadata, Mapping):
        return None
    if "bounds" not in aoi_metadata:
        return None
    raw = aoi_metadata["bounds"]
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 4:
        raise ContractError(
            "aoi_metadata.bounds must be a 4-element numeric sequence or null "
            "(metadata hint only; not a computed effective extent)."
        )

    bounds: list[float] = []
    for idx, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ContractError(
                f"aoi_metadata.bounds[{idx}] must be a number, got {value!r} ({type(value).__name__})."
            )
        bounds.append(float(value))
    return bounds


def _resolve_spatial_contract(
    *,
    resolved_config: PrepDataConfig,
    raster_path: Any,
    vector_path: Any,
    aoi_path: Any | None,
    raster_metadata: Any,
    vector_metadata: Any,
    aoi_metadata: Any | None,
) -> dict[str, Any]:
    paths = prep_data_validators.validate_input_paths_contract(
        raster_path=raster_path,
        vector_path=vector_path,
        aoi_path=aoi_path,
    )
    crs_summary = prep_data_validators.validate_crs_contract(
        raster_metadata=raster_metadata,
        vector_metadata=vector_metadata,
        aoi_metadata=aoi_metadata,
    )

    aoi_present = paths["aoi_path"] is not None
    aoi_policy_enabled = resolved_config.aoi.enabled

    if aoi_present and not aoi_policy_enabled:
        raise ContractError(
            "AOI contract violation: aoi_path was provided but config.aoi.enabled is False."
        )
    if aoi_policy_enabled and not aoi_present:
        raise ContractError(
            "AOI contract violation: config.aoi.enabled is True but aoi_path is not provided."
        )
    if aoi_present and aoi_metadata is None:
        raise ContractError("aoi_metadata is required when aoi_path is provided.")
    if not aoi_present and aoi_metadata is not None:
        raise ContractError("aoi_metadata must be null when aoi_path is not provided.")

    if aoi_metadata is not None:
        prep_data_validators.validate_vector_geometry_contract(
            vector_metadata=aoi_metadata,
            metadata_name="aoi_metadata",
        )

    derive_aoi_from_labels_if_missing = bool(resolved_config.aoi.derive_from_labels_if_missing)
    buffer_m = (
        float(resolved_config.aoi.buffer_m)
        if aoi_present or derive_aoi_from_labels_if_missing
        else None
    )
    # `bounds` from metadata is treated as a hint only in this lightweight stage.
    aoi_bounds_metadata_hint = _extract_aoi_bounds_metadata_hint(aoi_metadata)
    # No real spatial clipping/reprojection is executed here, so no computed
    # effective extent is produced in stage 02 skeleton.
    effective_bounds = None
    spatial_context_mode: Literal["full_raster", "aoi_limited"] = (
        "aoi_limited" if aoi_present else "full_raster"
    )

    return {
        "paths": paths,
        "crs_summary": crs_summary,
        "aoi_present": aoi_present,
        "aoi_policy_enabled": aoi_policy_enabled,
        "derive_aoi_from_labels_if_missing": derive_aoi_from_labels_if_missing,
        "buffer_m": buffer_m,
        "aoi_bounds_metadata_hint": aoi_bounds_metadata_hint,
        "effective_extent_bounds": effective_bounds,
        "spatial_context_mode": spatial_context_mode,
    }


def _build_success_checks() -> dict[str, Any]:
    return {
        "contract_checks_passed": True,
        "crs_compatible": True,
        "aoi_contract_consistent": True,
        "buffer_policy_resolved": True,
        "spatial_context_resolved": True,
        "blocking_issues": [],
    }


def _build_failure_checks(error: ContractError) -> dict[str, Any]:
    checks = {
        "contract_checks_passed": False,
        "crs_compatible": None,
        "aoi_contract_consistent": None,
        "buffer_policy_resolved": None,
        "spatial_context_resolved": False,
        "blocking_issues": [str(error)],
    }

    message = str(error)
    if isinstance(error, SpatialContractError):
        checks["crs_compatible"] = False
    if "AOI contract violation" in message or "aoi_metadata" in message:
        checks["aoi_contract_consistent"] = False
    if "buffer" in message:
        checks["buffer_policy_resolved"] = False
    return checks


def run_prepare_spatial_context_stage(
    *,
    output_dir: str | Path,
    run_id: str,
    raster_path: Any,
    vector_path: Any,
    raster_metadata: Any,
    vector_metadata: Any,
    config: PrepDataConfig | Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    aoi_path: Any | None = None,
    aoi_metadata: Any | None = None,
    source_manifest_path: str | Path | None = None,
    module_version: str | None = None,
    runtime_compute_enabled: bool = True,
) -> PrepareSpatialContextStageResult:
    """Run lightweight `02_prepare_spatial_context` stage and write artifacts.

    Transitional baseline policy:
      - input references (raster/vector/AOI paths) are provided as stage args;
      - config is validated via config/schema layer, but input refs are still
        stage-owned in this skeleton.
      - `effective_extent_bounds` is not spatially computed in this stage; any
        AOI bounds from metadata are stored as `aoi_bounds_metadata_hint`.
    """
    run_id = _require_non_empty_str("run_id", run_id)
    output_root = _normalize_path("output_dir", output_dir)
    created_at_utc = _now_utc_iso()
    manifest_path = output_root / _MANIFEST_FILENAME
    summary_path = output_root / _SUMMARY_FILENAME

    resolved_config: PrepDataConfig | None = None
    config_used_path: str | None = None
    status: Literal["success", "failed"]
    checks: dict[str, Any]
    blocking_issues: list[str]
    error_type: str | None = None
    error_message: str | None = None

    spatial_context_mode: Literal["full_raster", "aoi_limited"] | None = None
    aoi_present: bool | None = None
    aoi_policy_enabled: bool | None = None
    resolved_buffer_m: float | None = None
    aoi_source_crs: str | None = None
    aoi_target_crs: str | None = None
    vector_source_crs: str | None = None
    vector_target_crs: str | None = None
    vector_reprojection_required: bool | None = None
    aoi_reprojection_required: bool | None = None
    vector_reprojected: bool | None = None
    vector_output_path: str | None = None
    aoi_output_path: str | None = None
    effective_extent_bounds: list[float] | None = None
    aoi_bounds_metadata_hint: list[float] | None = None
    normalized_source_manifest_path: str | None = None
    spatial_compute_mode: str | None = None
    aoi_reprojected: bool | None = None
    aoi_source_type: str | None = None
    aoi_derivation_method: str | None = None
    derive_aoi_from_labels_if_missing = False

    try:
        if source_manifest_path is not None:
            normalized_source_manifest_path = str(
                _normalize_path("source_manifest_path", source_manifest_path)
            )
        resolved_config, config_used_path = _resolve_config(config=config, config_path=config_path)
        spatial = _resolve_spatial_contract(
            resolved_config=resolved_config,
            raster_path=raster_path,
            vector_path=vector_path,
            aoi_path=aoi_path,
            raster_metadata=raster_metadata,
            vector_metadata=vector_metadata,
            aoi_metadata=aoi_metadata,
        )
        spatial_context_mode = spatial["spatial_context_mode"]
        aoi_present = spatial["aoi_present"]
        aoi_policy_enabled = spatial["aoi_policy_enabled"]
        derive_aoi_from_labels_if_missing = bool(spatial["derive_aoi_from_labels_if_missing"])
        resolved_buffer_m = spatial["buffer_m"]
        if aoi_present:
            aoi_source_type = "user_provided"
        vector_source_crs = spatial["crs_summary"]["vector_crs"]
        vector_target_crs = spatial["crs_summary"]["raster_crs"]
        vector_reprojection_required = bool(
            spatial["crs_summary"].get("vector_reprojection_required", False)
        )
        aoi_source_crs = spatial["crs_summary"]["aoi_crs"]
        aoi_target_crs = spatial["crs_summary"]["raster_crs"]
        aoi_reprojection_required = bool(
            spatial["crs_summary"].get("aoi_reprojection_required", False)
        )
        aoi_bounds_metadata_hint = spatial["aoi_bounds_metadata_hint"]
        effective_extent_bounds = spatial["effective_extent_bounds"]

        if runtime_compute_enabled:
            if spatial_context_compute is None:
                raise ContractError(
                    "runtime_compute_enabled=True but spatial_context_compute module "
                    "could not be imported (rasterio/geopandas missing?)."
                )
            compute_result = spatial_context_compute.compute_spatial_context(
                raster_path=raster_path,
                vector_path=vector_path,
                aoi_path=aoi_path,
                buffer_m=float(resolved_buffer_m) if resolved_buffer_m is not None else 0.0,
                output_dir=output_root,
                derive_aoi_from_labels=derive_aoi_from_labels_if_missing,
            )
            effective_extent_bounds = compute_result["effective_extent_bounds"]
            vector_reprojected = compute_result["vector_reprojected"]
            vector_output_path = compute_result.get("vector_reprojected_path")
            aoi_reprojected = compute_result["aoi_reprojected"]
            aoi_output_path = compute_result.get("aoi_reprojected_path")
            aoi_derivation_method = compute_result.get("aoi_derivation_method")
            spatial_compute_mode = compute_result["spatial_compute_mode"]
            vector_source_crs = compute_result.get("vector_crs_source") or vector_source_crs
            vector_target_crs = compute_result["raster_crs"]
            aoi_source_crs = compute_result.get("aoi_crs_source") or aoi_source_crs
            aoi_target_crs = compute_result["raster_crs"]
            resolved_aoi_source_type = compute_result.get("aoi_source_type")
            if resolved_aoi_source_type in {"user_provided", "derived_from_labels"}:
                aoi_source_type = resolved_aoi_source_type
            if aoi_source_type == "derived_from_labels":
                aoi_present = True
                spatial_context_mode = "aoi_limited"

        if aoi_source_type is None and aoi_present:
            aoi_source_type = "user_provided"

        status = "success"
        checks = _build_success_checks()
        blocking_issues = []
    except ContractError as exc:
        status = "failed"
        error_type = type(exc).__name__
        error_message = str(exc)
        checks = _build_failure_checks(exc)
        blocking_issues = [str(exc)]

    manifest_payload = {
        "schema_name": _MANIFEST_SCHEMA_NAME,
        "schema_version": _SCHEMA_VERSION,
        "module_name": _MODULE_NAME,
        "module_version": module_version,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "run_id": run_id,
        "stage_name": _STAGE_NAME,
        "created_at_utc": created_at_utc,
        "status": status,
        "config": {
            "config_used_path": config_used_path,
            "config_hash": None,
            "config_overrides": None,
            "input_refs_source": _INPUT_REFS_SOURCE,
            "aoi_policy_enabled": None if resolved_config is None else resolved_config.aoi.enabled,
            "aoi_policy_buffer_m": None if resolved_config is None else float(resolved_config.aoi.buffer_m),
            "aoi_derive_from_labels_if_missing": (
                None
                if resolved_config is None
                else bool(resolved_config.aoi.derive_from_labels_if_missing)
            ),
            "feature_mode": None if resolved_config is None else resolved_config.feature_mode,
            "runtime_compute_enabled": runtime_compute_enabled,
            "spatial_compute_mode": spatial_compute_mode,
        },
        "provenance": {
            "source_run_ids": [],
            "source_manifest_paths": (
                [] if normalized_source_manifest_path is None else [normalized_source_manifest_path]
            ),
            "source_config_paths": [],
            "code_version": None,
            "git_commit": None,
        },
        "inputs": {
            "artifacts": (
                [
                    {
                        "path": str(aoi_path),
                        "role": "aoi_source",
                        "format": "vector",
                        "is_required": False,
                        "exists": True,
                        "crs": aoi_source_crs,
                    }
                ]
                if (
                    runtime_compute_enabled
                    and aoi_path is not None
                    and aoi_source_type == "user_provided"
                )
                else []
            )
        },
        "outputs": {
            "artifacts": (
                [
                    *(
                        [
                            {
                                "path": vector_output_path,
                                "role": "vector_in_raster_crs",
                                "format": "GPKG",
                                "is_required": False,
                                "exists": True,
                            }
                        ]
                        if vector_output_path is not None
                        else []
                    ),
                    *(
                        [
                            {
                                "path": aoi_output_path,
                                "role": (
                                    "aoi_derived_from_labels_in_raster_crs"
                                    if aoi_source_type == "derived_from_labels"
                                    else "aoi_resolved_in_raster_crs"
                                ),
                                "format": "GPKG",
                                "is_required": False,
                                "exists": True,
                            }
                        ]
                        if aoi_output_path is not None
                        else []
                    ),
                ]
                if runtime_compute_enabled
                else []
            )
        },
        "resolved_contract": {
            "spatial": {
                "spatial_context_mode": spatial_context_mode,
                "raster_crs": vector_target_crs,
                "vector_source_crs": vector_source_crs,
                "vector_target_crs": vector_target_crs,
                "vector_reprojection_required": vector_reprojection_required,
                "vector_reprojection_applied": vector_reprojected,
                "vector_output_path": vector_output_path,
                "aoi_present": aoi_present,
                "aoi_policy_enabled": aoi_policy_enabled,
                "aoi_buffer_m": resolved_buffer_m,
                "aoi_source_crs": aoi_source_crs,
                "aoi_target_crs": aoi_target_crs,
                "aoi_reprojection_required": aoi_reprojection_required,
                "aoi_reprojection_applied": aoi_reprojected,
                "aoi_source_type": aoi_source_type,
                "aoi_derivation_method": aoi_derivation_method,
                "aoi_output_path": aoi_output_path,
                "aoi_bounds_metadata_hint": aoi_bounds_metadata_hint,
                "effective_extent_bounds": effective_extent_bounds,
            },
            "features": {},
            "valid_policy": {},
            "normalization": None,
            "aoi_policy": {
                "enabled": aoi_policy_enabled,
                "buffer_m": resolved_buffer_m,
                "derive_from_labels_if_missing": derive_aoi_from_labels_if_missing,
            },
        },
        "runtime": {
            "device_requested": None,
            "device_resolved": None,
            "amp_requested": None,
            "amp_used": None,
            "oom_fallbacks_applied": [],
            "notes": [],
        },
        "aoi_present": aoi_present if aoi_present is not None else (aoi_path is not None),
        "aoi_source_path": _safe_path_str(aoi_path),
        "aoi_source_type": aoi_source_type,
        "aoi_derivation_method": aoi_derivation_method,
        "aoi_source_crs": aoi_source_crs,
        "aoi_target_crs": aoi_target_crs,
        "aoi_reprojected": aoi_reprojected,
        "vector_source_crs": vector_source_crs,
        "vector_target_crs": vector_target_crs,
        "vector_reprojected": vector_reprojected,
        "vector_output_path": vector_output_path,
        "aoi_output_path": aoi_output_path,
        "buffer_m": resolved_buffer_m,
        "aoi_bounds_metadata_hint": aoi_bounds_metadata_hint,
        "effective_extent_bounds": effective_extent_bounds,
        "spatial_context_mode": spatial_context_mode,
        "checks": checks,
        "diagnostics": {
            "warnings": (
                [
                    (
                        "CRS mismatch is transformable; reprojection remains pending "
                        "because runtime_compute_enabled=false."
                    )
                ]
                if status == "success"
                and not runtime_compute_enabled
                and bool(vector_reprojection_required or aoi_reprojection_required)
                else []
            ),
            "errors": blocking_issues,
        },
    }
    write_manifest(manifest_path, manifest_payload)

    summary_payload = {
        "schema_name": _SUMMARY_SCHEMA_NAME,
        "stage_name": _STAGE_NAME,
        "run_id": run_id,
        "status": status,
        "input_refs_source": _INPUT_REFS_SOURCE,
        "contract_checks_passed": status == "success",
        "spatial_context_mode": spatial_context_mode,
        "aoi_present": aoi_present,
        "aoi_policy_enabled": aoi_policy_enabled,
        "buffer_m": resolved_buffer_m,
        "aoi_source_type": aoi_source_type,
        "aoi_derivation_method": aoi_derivation_method,
        "source_manifest_path": normalized_source_manifest_path,
        "blocking_issues": blocking_issues,
        "manifest_path": str(manifest_path),
        "error_type": error_type,
    }
    write_summary(summary_path, summary_payload)

    return PrepareSpatialContextStageResult(
        status=status,
        manifest_path=manifest_path,
        summary_path=summary_path,
        spatial_context_mode=spatial_context_mode,
        aoi_present=aoi_present,
        aoi_policy_enabled=aoi_policy_enabled,
        resolved_buffer_m=resolved_buffer_m,
        blocking_issues=tuple(blocking_issues),
        checks=checks,
        error_type=error_type,
        error_message=error_message,
        aoi_output_path=Path(aoi_output_path) if aoi_output_path else None,
        effective_extent_bounds=(
            tuple(effective_extent_bounds)  # type: ignore[arg-type]
            if effective_extent_bounds
            else None
        ),
        aoi_source_type=aoi_source_type,
        aoi_derivation_method=aoi_derivation_method,
    )
