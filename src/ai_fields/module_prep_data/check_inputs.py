"""Lightweight stage skeleton for `01_check_inputs` in module_prep_data.

This stage is intentionally minimal: it orchestrates existing config and
validator layers, writes manifest/summary artifacts, and returns a small
test-friendly result object. It does not perform heavy geospatial I/O.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError, SpatialContractError, ValidPolicyError
from ai_fields.common.manifests import write_manifest, write_summary
from ai_fields.module_prep_data import config as prep_data_config
from ai_fields.module_prep_data import input_probe
from ai_fields.module_prep_data import validators as prep_data_validators
from ai_fields.module_prep_data.schemas import PrepDataConfig

_STAGE_NAME = "01_check_inputs"
_MANIFEST_SCHEMA_NAME = "prep_data.check_inputs_manifest"
_SUMMARY_SCHEMA_NAME = "prep_data.summary"
_SCHEMA_VERSION = "v1"
_MODULE_NAME = "module_prep_data"
_MANIFEST_FILENAME = "check_inputs_manifest.json"
_SUMMARY_FILENAME = "summary.json"
_INPUT_REFS_SOURCE = "stage_args_transitional"
_METADATA_SIDECAR_SUFFIX = ".meta.json"
_RUNTIME_PROBE_MODE = input_probe.PROBE_MODE  # "rasterio_fiona_probe_v1"


@dataclass(frozen=True)
class CheckInputsStageResult:
    """Minimal stage outcome for `01_check_inputs`."""

    status: Literal["success", "failed"]
    manifest_path: Path
    summary_path: Path
    blocking_issues: tuple[str, ...]
    checks: dict[str, Any]
    error_type: str | None = None
    error_message: str | None = None

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


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ContractError(
            f"{name} must be a boolean (true/false), got {value!r} ({type(value).__name__})."
        )
    return value


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


def _probe_existing_readable_file(name: str, path: Path) -> None:
    if not path.exists():
        raise ContractError(f"{name} file does not exist: {path}")
    if not path.is_file():
        raise ContractError(f"{name} must point to a regular file, got: {path}")
    try:
        with path.open("rb") as fh:
            fh.read(1)
    except OSError as exc:
        raise ContractError(f"{name} is not readable: {path} ({exc})") from exc


def _load_json_sidecar(path: Path, *, metadata_name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"{metadata_name} sidecar '{path}' is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise ContractError(f"Failed to read {metadata_name} sidecar '{path}': {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ContractError(
            f"{metadata_name} sidecar '{path}' must contain a JSON object at top level."
        )
    return payload


def _sidecar_path_for_input(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}{_METADATA_SIDECAR_SUFFIX}")


def _resolve_runtime_metadata(
    *,
    metadata_name: str,
    metadata_value: Any,
    input_path: Path,
    sidecar_fallback_enabled: bool,
) -> Any:
    if metadata_value is not None:
        return metadata_value
    if not sidecar_fallback_enabled:
        raise ContractError(
            f"{metadata_name} is required for runtime checks when sidecar fallback is disabled."
        )

    sidecar_path = _sidecar_path_for_input(input_path)
    if not sidecar_path.exists():
        raise ContractError(
            f"{metadata_name} is missing and no sidecar metadata file was found at '{sidecar_path}'."
        )
    return _load_json_sidecar(sidecar_path, metadata_name=metadata_name)


def _metadata_snapshot_raster(md: Any, *, path: Any) -> dict[str, Any]:
    if isinstance(md, Mapping):
        return {
            "path": None if path is None else str(path),
            "crs": md.get("crs"),
            "width": md.get("width"),
            "height": md.get("height"),
            "count": md.get("band_count"),
            "dtype": md.get("dtype"),
            "nodata": md.get("nodata"),
        }
    return {"path": None if path is None else str(path), "metadata_type": type(md).__name__}


def _metadata_snapshot_vector(md: Any, *, path: Any) -> dict[str, Any]:
    if isinstance(md, Mapping):
        return {
            "path": None if path is None else str(path),
            "crs": md.get("crs"),
            "feature_count": md.get("feature_count"),
            "geometry_types": md.get("geometry_types"),
        }
    return {"path": None if path is None else str(path), "metadata_type": type(md).__name__}


def _metadata_snapshot_aoi(md: Any, *, path: Any) -> dict[str, Any]:
    if isinstance(md, Mapping):
        return {
            "path": None if path is None else str(path),
            "crs": md.get("crs"),
            "feature_count": md.get("feature_count"),
        }
    return {"path": None if path is None else str(path), "metadata_type": type(md).__name__}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _extract_readability_check_value(metadata: Any) -> bool | None:
    if not isinstance(metadata, Mapping):
        return None
    if "readable" not in metadata:
        return None
    raw = metadata.get("readable")
    return raw if isinstance(raw, bool) else None


def _build_checks_from_success_result(
    result: Mapping[str, Any],
    *,
    raster_metadata: Any,
    vector_metadata: Any,
    aoi_metadata: Any | None,
    aoi_present: bool,
    runtime_readability: Mapping[str, bool | None],
) -> dict[str, Any]:
    crs_summary = result.get("crs") if isinstance(result.get("crs"), Mapping) else {}
    vector_reprojection_required = bool(crs_summary.get("vector_reprojection_required", False))
    aoi_reprojection_required = bool(crs_summary.get("aoi_reprojection_required", False))
    reprojection_required = bool(crs_summary.get("reprojection_required", False))
    return {
        "contract_checks_passed": True,
        "raster_readable": (
            runtime_readability["raster_readable"]
            if runtime_readability["raster_readable"] is not None
            else _extract_readability_check_value(raster_metadata)
        ),
        "vector_readable": (
            runtime_readability["vector_readable"]
            if runtime_readability["vector_readable"] is not None
            else _extract_readability_check_value(vector_metadata)
        ),
        "aoi_readable": (
            runtime_readability["aoi_readable"]
            if runtime_readability["aoi_readable"] is not None
            else (_extract_readability_check_value(aoi_metadata) if aoi_present else None)
        ),
        "crs_compatible": True,
        "crs_match": not reprojection_required,
        "vector_reprojection_required": vector_reprojection_required,
        "aoi_reprojection_required": aoi_reprojection_required if aoi_present else False,
        "reprojection_required": reprojection_required,
        "reprojection_pending_stage": "02_prepare_spatial_context" if reprojection_required else None,
        "band_count_ok": result.get("band_count") == 8,
        "geometry_validity_ok": True,
        "nodata_interpretation_resolved": True,
        "blocking_issues": [],
    }


def _build_failure_checks(*, error: ContractError, aoi_present: bool) -> dict[str, Any]:
    checks = {
        "contract_checks_passed": False,
        "raster_readable": None,
        "vector_readable": None,
        "aoi_readable": None if aoi_present else None,
        "crs_compatible": None,
        "crs_match": None,
        "vector_reprojection_required": None,
        "aoi_reprojection_required": None,
        "reprojection_required": None,
        "reprojection_pending_stage": None,
        "band_count_ok": None,
        "geometry_validity_ok": None,
        "nodata_interpretation_resolved": None,
        "blocking_issues": [str(error)],
    }
    message = str(error)

    if isinstance(error, SpatialContractError):
        checks["crs_compatible"] = False
    if isinstance(error, ValidPolicyError):
        checks["nodata_interpretation_resolved"] = False
    if "band-count" in message or "band_count" in message:
        checks["band_count_ok"] = False
    if "geometry_types" in message or "feature_count" in message or "polygon/multipolygon" in message:
        checks["geometry_validity_ok"] = False
    if "raster_metadata is marked as unreadable" in message:
        checks["raster_readable"] = False
    if "vector_metadata is marked as unreadable" in message:
        checks["vector_readable"] = False
    if "aoi_metadata is marked as unreadable" in message and aoi_present:
        checks["aoi_readable"] = False
    if "raster_path" in message and (
        "does not exist" in message or "regular file" in message or "not readable" in message
    ):
        checks["raster_readable"] = False
    if "vector_path" in message and (
        "does not exist" in message or "regular file" in message or "not readable" in message
    ):
        checks["vector_readable"] = False
    if aoi_present and "aoi_path" in message and (
        "does not exist" in message or "regular file" in message or "not readable" in message
    ):
        checks["aoi_readable"] = False
    return checks


def run_check_inputs_stage(
    *,
    output_dir: str | Path,
    run_id: str,
    raster_path: Any,
    vector_path: Any,
    raster_metadata: Any = None,
    vector_metadata: Any = None,
    config: PrepDataConfig | Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    aoi_path: Any | None = None,
    aoi_metadata: Any | None = None,
    config_override_present: bool = False,
    runtime_probe_enabled: bool = True,
    metadata_sidecar_fallback_enabled: bool = True,
    module_version: str | None = None,
) -> CheckInputsStageResult:
    """Run lightweight `01_check_inputs` stage and write manifest/summary.

    Transitional baseline policy:
      - input references (raster/vector/AOI paths) are provided as stage args;
      - config is still validated via config/schema layer, but is not yet the
        canonical source for input paths in this skeleton stage.
    """
    run_id = _require_non_empty_str("run_id", run_id)
    output_root = _normalize_path("output_dir", output_dir)
    created_at_utc = _now_utc_iso()
    manifest_path = output_root / _MANIFEST_FILENAME
    summary_path = output_root / _SUMMARY_FILENAME

    resolved_config: PrepDataConfig | None = None
    config_used_path: str | None = None
    checks: dict[str, Any]
    blocking_issues: list[str]
    error_type: str | None = None
    error_message: str | None = None
    status: Literal["success", "failed"]
    validator_result: Mapping[str, Any] | None = None
    runtime_readability = {
        "raster_readable": None,
        "vector_readable": None,
        "aoi_readable": None,
    }
    runtime_probe_effective: bool | None = None
    sidecar_fallback_effective: bool | None = None
    resolved_raster_path: Path | None = None
    resolved_vector_path: Path | None = None
    resolved_aoi_path: Path | None = None
    resolved_raster_metadata: Any = raster_metadata
    resolved_vector_metadata: Any = vector_metadata
    resolved_aoi_metadata: Any = aoi_metadata

    try:
        runtime_probe_flag = _require_bool("runtime_probe_enabled", runtime_probe_enabled)
        sidecar_fallback_flag = _require_bool(
            "metadata_sidecar_fallback_enabled",
            metadata_sidecar_fallback_enabled,
        )
        runtime_probe_effective = runtime_probe_flag
        sidecar_fallback_effective = sidecar_fallback_flag
        resolved_config, config_used_path = _resolve_config(config=config, config_path=config_path)
        resolved_raster_path = _normalize_path("raster_path", raster_path)
        resolved_vector_path = _normalize_path("vector_path", vector_path)
        if aoi_path is not None:
            resolved_aoi_path = _normalize_path("aoi_path", aoi_path)

        if runtime_probe_flag:
            if resolved_raster_metadata is None:
                resolved_raster_metadata = input_probe.probe_raster(resolved_raster_path)
            else:
                _probe_existing_readable_file("raster_path", resolved_raster_path)
            runtime_readability["raster_readable"] = True

            if resolved_vector_metadata is None:
                resolved_vector_metadata = input_probe.probe_vector(resolved_vector_path)
            else:
                _probe_existing_readable_file("vector_path", resolved_vector_path)
            runtime_readability["vector_readable"] = True

            if resolved_aoi_path is not None:
                _probe_existing_readable_file("aoi_path", resolved_aoi_path)
                runtime_readability["aoi_readable"] = True

        resolved_raster_metadata = _resolve_runtime_metadata(
            metadata_name="raster_metadata",
            metadata_value=resolved_raster_metadata,
            input_path=resolved_raster_path,
            sidecar_fallback_enabled=sidecar_fallback_flag,
        )
        resolved_vector_metadata = _resolve_runtime_metadata(
            metadata_name="vector_metadata",
            metadata_value=resolved_vector_metadata,
            input_path=resolved_vector_path,
            sidecar_fallback_enabled=sidecar_fallback_flag,
        )
        if resolved_aoi_path is not None:
            resolved_aoi_metadata = _resolve_runtime_metadata(
                metadata_name="aoi_metadata",
                metadata_value=aoi_metadata,
                input_path=resolved_aoi_path,
                sidecar_fallback_enabled=sidecar_fallback_flag,
            )

        validator_result = prep_data_validators.validate_check_inputs_contract(
            raster_path=resolved_raster_path,
            vector_path=resolved_vector_path,
            raster_metadata=resolved_raster_metadata,
            vector_metadata=resolved_vector_metadata,
            aoi_path=resolved_aoi_path,
            aoi_metadata=resolved_aoi_metadata,
            config_override_present=config_override_present,
        )
        status = "success"
        checks = _build_checks_from_success_result(
            validator_result,
            raster_metadata=resolved_raster_metadata,
            vector_metadata=resolved_vector_metadata,
            aoi_metadata=resolved_aoi_metadata,
            aoi_present=resolved_aoi_path is not None,
            runtime_readability=runtime_readability,
        )
        blocking_issues = []
    except ContractError as exc:
        status = "failed"
        error_type = type(exc).__name__
        error_message = str(exc)
        blocking_issues = [str(exc)]
        checks = _build_failure_checks(error=exc, aoi_present=resolved_aoi_path is not None)

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
            "runtime_probe_mode": _RUNTIME_PROBE_MODE,
            "runtime_probe_enabled": runtime_probe_effective,
            "metadata_sidecar_fallback_enabled": sidecar_fallback_effective,
            "feature_mode": None if resolved_config is None else resolved_config.feature_mode,
            "valid_policy_nodata_source": (
                None if resolved_config is None else resolved_config.valid_policy.nodata_source
            ),
        },
        "provenance": {
            "source_run_ids": [],
            "source_manifest_paths": [],
            "source_config_paths": [],
            "code_version": None,
            "git_commit": None,
        },
        "inputs": {"artifacts": []},
        "outputs": {"artifacts": []},
        "resolved_contract": {
            "spatial": {},
            "features": {},
            "valid_policy": {},
            "normalization": None,
            "aoi_policy": None,
        },
        "runtime": {
            "device_requested": None,
            "device_resolved": None,
            "amp_requested": None,
            "amp_used": None,
            "oom_fallbacks_applied": [],
            "notes": [],
        },
        "input_raster": _metadata_snapshot_raster(
            resolved_raster_metadata,
            path=resolved_raster_path if resolved_raster_path is not None else raster_path,
        ),
        "input_vectors": _metadata_snapshot_vector(
            resolved_vector_metadata,
            path=resolved_vector_path if resolved_vector_path is not None else vector_path,
        ),
        "input_aoi": _metadata_snapshot_aoi(
            resolved_aoi_metadata,
            path=resolved_aoi_path if resolved_aoi_path is not None else aoi_path,
        ),
        "checks": checks,
        "validator_result": _to_jsonable(dict(validator_result)) if validator_result is not None else None,
        "diagnostics": {
            "warnings": (
                [
                    (
                        "CRS mismatch detected in check_inputs; reprojection is required "
                        "and deferred to stage 02_prepare_spatial_context."
                    )
                ]
                if validator_result is not None
                and isinstance(validator_result.get("crs"), Mapping)
                and bool(validator_result["crs"].get("reprojection_required", False))
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
        "runtime_probe_mode": _RUNTIME_PROBE_MODE,
        "runtime_probe_enabled": runtime_probe_effective,
        "metadata_sidecar_fallback_enabled": sidecar_fallback_effective,
        "contract_checks_passed": status == "success",
        "blocking_issues": blocking_issues,
        "manifest_path": str(manifest_path),
        "error_type": error_type,
    }
    write_summary(summary_path, summary_payload)

    return CheckInputsStageResult(
        status=status,
        manifest_path=manifest_path,
        summary_path=summary_path,
        blocking_issues=tuple(blocking_issues),
        checks=checks,
        error_type=error_type,
        error_message=error_message,
    )
