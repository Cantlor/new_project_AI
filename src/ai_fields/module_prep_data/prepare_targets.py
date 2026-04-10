"""Lightweight stage skeleton for `04_prepare_targets` in module_prep_data.

This stage intentionally resolves only metadata/contract-level target context.
It writes manifest/summary artifacts and returns a small test-friendly result.
It does not perform real rasterization, boundary generation, distance transform,
or heavy geospatial I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import write_manifest, write_summary
from ai_fields.module_prep_data import config as prep_data_config
from ai_fields.module_prep_data import validators as prep_data_validators
from ai_fields.module_prep_data.schemas import PrepDataConfig

try:
    from ai_fields.module_prep_data import targets_compute
except ImportError:  # pragma: no cover
    targets_compute = None  # type: ignore[assignment]

_STAGE_NAME = "04_prepare_targets"
# Canonical baseline choice for stage-04 manifest naming:
# - module_prep_data.md requires a manifest per stage;
# - MANIFEST_SCHEMAS.md does not yet define stage-04 target schema explicitly.
# We standardize on `prep_data.targets_manifest` and `targets_manifest.json`.
_MANIFEST_SCHEMA_NAME = "prep_data.targets_manifest"
_SUMMARY_SCHEMA_NAME = "prep_data.summary"
_SCHEMA_VERSION = "v1"
_MODULE_NAME = "module_prep_data"
_MANIFEST_FILENAME = "targets_manifest.json"
_SUMMARY_FILENAME = "summary.json"
_INPUT_REFS_SOURCE = "stage_args_transitional"
_TARGET_CONTRACT_MODE = "metadata_snapshot_only"

_TARGET_LAYERS = ("extent", "boundary", "distance", "valid")
_ERROR_CODE_ATTR = "stage_error_code"

_ERR_TARGET_METADATA_TYPE = "target_metadata_type_invalid"
_ERR_TARGET_METADATA_EXPECTED_LAYERS_INVALID = "target_metadata_expected_layers_invalid"
_ERR_TARGET_METADATA_EXPECTED_LAYERS_MISMATCH = "target_metadata_expected_layers_mismatch"
_ERR_TARGET_METADATA_EXPECTED_BOUNDARY_ENCODING_INVALID = (
    "target_metadata_expected_boundary_encoding_invalid"
)
_ERR_TARGET_METADATA_EXPECTED_BOUNDARY_ENCODING_MISMATCH = (
    "target_metadata_expected_boundary_encoding_mismatch"
)
_ERR_TARGET_METADATA_EXPECTED_DISTANCE_TARGET_INVALID = (
    "target_metadata_expected_distance_target_invalid"
)
_ERR_TARGET_METADATA_EXPECTED_DISTANCE_TARGET_MISMATCH = (
    "target_metadata_expected_distance_target_mismatch"
)
_ERR_TARGET_METADATA_EXPECTED_VALID_SAVED_SEPARATELY_INVALID = (
    "target_metadata_expected_valid_saved_separately_invalid"
)
_ERR_TARGET_METADATA_EXPECTED_VALID_SAVED_SEPARATELY_MISMATCH = (
    "target_metadata_expected_valid_saved_separately_mismatch"
)
_ERR_TARGET_METADATA_EXPECTED_BOUNDARY_RAW_ENABLED_INVALID = (
    "target_metadata_expected_boundary_raw_enabled_invalid"
)
_ERR_TARGET_METADATA_EXPECTED_BOUNDARY_RAW_ENABLED_MISMATCH = (
    "target_metadata_expected_boundary_raw_enabled_mismatch"
)

_FAILURE_CHECK_UPDATES_BY_CODE: dict[str, dict[str, Any]] = {
    _ERR_TARGET_METADATA_EXPECTED_LAYERS_INVALID: {
        "target_layers_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_EXPECTED_LAYERS_MISMATCH: {
        "target_layers_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_EXPECTED_BOUNDARY_ENCODING_INVALID: {
        "boundary_policy_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_EXPECTED_BOUNDARY_ENCODING_MISMATCH: {
        "boundary_policy_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_EXPECTED_DISTANCE_TARGET_INVALID: {
        "distance_policy_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_EXPECTED_DISTANCE_TARGET_MISMATCH: {
        "distance_policy_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_EXPECTED_VALID_SAVED_SEPARATELY_INVALID: {
        "valid_semantics_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_EXPECTED_VALID_SAVED_SEPARATELY_MISMATCH: {
        "valid_semantics_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_EXPECTED_BOUNDARY_RAW_ENABLED_INVALID: {
        "boundary_policy_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_EXPECTED_BOUNDARY_RAW_ENABLED_MISMATCH: {
        "boundary_policy_resolved": False,
        "target_metadata_consistent": False,
    },
    _ERR_TARGET_METADATA_TYPE: {
        "target_metadata_consistent": False,
    },
}


@dataclass(frozen=True)
class PrepareTargetsStageResult:
    """Minimal stage outcome for `04_prepare_targets`."""

    status: Literal["success", "failed"]
    manifest_path: Path
    summary_path: Path
    target_layers: tuple[str, ...] | None
    boundary_encoding: str | None
    distance_target: str | None
    valid_saved_separately: bool | None
    boundary_raw_enabled: bool | None
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


def _normalize_optional_path(name: str, value: Any | None) -> str | None:
    if value is None:
        return None
    return str(_normalize_path(name, value))


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


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ContractError(
            f"{name} must be a boolean (true/false), got {value!r} ({type(value).__name__})."
        )
    return value


def _require_str_list(name: str, value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{name} must be a sequence of strings.")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ContractError(
                f"{name}[{idx}] must be a string, got {item!r} ({type(item).__name__})."
            )
        out.append(item)
    return out


def _stage_contract_error(*, code: str, message: str) -> ContractError:
    error = ContractError(message)
    setattr(error, _ERROR_CODE_ATTR, code)
    return error


def _get_error_code(error: ContractError) -> str | None:
    code = getattr(error, _ERROR_CODE_ATTR, None)
    if isinstance(code, str) and code.strip():
        return code
    return None


def _resolve_target_contract(
    *,
    resolved_config: PrepDataConfig,
    target_metadata: Any | None,
) -> dict[str, Any]:
    target_layers = list(_TARGET_LAYERS)
    boundary_encoding = resolved_config.boundary.encoding
    distance_target = resolved_config.distance.target
    valid_saved_separately = True
    boundary_raw_policy = {
        "enabled": True,
        "policy_source": "baseline_v1_default",
        "notes": "boundary_raw is treated as required diagnostic artifact in baseline v1.",
    }
    valid_semantics = {
        "valid_target_layer_required": True,
        "valid_ignore_semantics": "valid=0 is ignore/invalid; valid=1 is usable.",
    }
    target_metadata_checked = target_metadata is not None

    # `target_metadata` is a lightweight consistency hint only.
    if target_metadata is not None:
        if not isinstance(target_metadata, Mapping):
            raise _stage_contract_error(
                code=_ERR_TARGET_METADATA_TYPE,
                message=(
                    "target_metadata must be a mapping/object, got "
                    f"{type(target_metadata).__name__}."
                ),
            )

        if "expected_target_layers" in target_metadata:
            try:
                expected_layers = _require_str_list(
                    "target_metadata.expected_target_layers",
                    target_metadata["expected_target_layers"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_LAYERS_INVALID,
                    message=str(exc),
                ) from exc
            if expected_layers != target_layers:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_LAYERS_MISMATCH,
                    message=(
                        "target_metadata.expected_target_layers is inconsistent with baseline "
                        f"target contract: expected {target_layers}, got {expected_layers}."
                    ),
                )

        if "expected_boundary_encoding" in target_metadata:
            try:
                expected_boundary_encoding = _require_non_empty_str(
                    "target_metadata.expected_boundary_encoding",
                    target_metadata["expected_boundary_encoding"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_BOUNDARY_ENCODING_INVALID,
                    message=str(exc),
                ) from exc
            if expected_boundary_encoding != boundary_encoding:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_BOUNDARY_ENCODING_MISMATCH,
                    message=(
                        "target_metadata.expected_boundary_encoding is inconsistent with config: "
                        f"expected {boundary_encoding!r}, got {expected_boundary_encoding!r}."
                    ),
                )

        if "expected_distance_target" in target_metadata:
            try:
                expected_distance_target = _require_non_empty_str(
                    "target_metadata.expected_distance_target",
                    target_metadata["expected_distance_target"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_DISTANCE_TARGET_INVALID,
                    message=str(exc),
                ) from exc
            if expected_distance_target != distance_target:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_DISTANCE_TARGET_MISMATCH,
                    message=(
                        "target_metadata.expected_distance_target is inconsistent with config: "
                        f"expected {distance_target!r}, got {expected_distance_target!r}."
                    ),
                )

        if "expected_valid_saved_separately" in target_metadata:
            try:
                expected_valid_saved_separately = _require_bool(
                    "target_metadata.expected_valid_saved_separately",
                    target_metadata["expected_valid_saved_separately"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_VALID_SAVED_SEPARATELY_INVALID,
                    message=str(exc),
                ) from exc
            if expected_valid_saved_separately != valid_saved_separately:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_VALID_SAVED_SEPARATELY_MISMATCH,
                    message=(
                        "target_metadata.expected_valid_saved_separately is inconsistent with "
                        f"target contract: expected {valid_saved_separately}, "
                        f"got {expected_valid_saved_separately}."
                    ),
                )

        if "expected_boundary_raw_enabled" in target_metadata:
            try:
                expected_boundary_raw_enabled = _require_bool(
                    "target_metadata.expected_boundary_raw_enabled",
                    target_metadata["expected_boundary_raw_enabled"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_BOUNDARY_RAW_ENABLED_INVALID,
                    message=str(exc),
                ) from exc
            if expected_boundary_raw_enabled != boundary_raw_policy["enabled"]:
                raise _stage_contract_error(
                    code=_ERR_TARGET_METADATA_EXPECTED_BOUNDARY_RAW_ENABLED_MISMATCH,
                    message=(
                        "target_metadata.expected_boundary_raw_enabled is inconsistent with "
                        f"target contract: expected {boundary_raw_policy['enabled']}, "
                        f"got {expected_boundary_raw_enabled}."
                    ),
                )

    return {
        "target_layers": target_layers,
        "boundary_encoding": boundary_encoding,
        "distance_target": distance_target,
        "valid_saved_separately": valid_saved_separately,
        "boundary_raw_policy": boundary_raw_policy,
        "valid_semantics": valid_semantics,
        "target_metadata_checked": target_metadata_checked,
    }


def _build_success_checks(*, target_metadata_checked: bool) -> dict[str, Any]:
    return {
        "contract_checks_passed": True,
        "target_contract_resolved": True,
        "target_layers_resolved": True,
        "boundary_policy_resolved": True,
        "distance_policy_resolved": True,
        "valid_semantics_resolved": True,
        "target_metadata_consistent": True if target_metadata_checked else None,
        "blocking_issues": [],
    }


def _build_failure_checks(error: ContractError) -> dict[str, Any]:
    checks = {
        "contract_checks_passed": False,
        "target_contract_resolved": False,
        "target_layers_resolved": None,
        "boundary_policy_resolved": None,
        "distance_policy_resolved": None,
        "valid_semantics_resolved": None,
        "target_metadata_consistent": None,
        "blocking_issues": [str(error)],
    }
    code = _get_error_code(error)
    if code in _FAILURE_CHECK_UPDATES_BY_CODE:
        checks.update(_FAILURE_CHECK_UPDATES_BY_CODE[code])

    # Legacy fallback for uncoded errors keeps diagnostics usable without
    # introducing a broader error framework in this skeleton stage.
    message = str(error)
    if checks["target_layers_resolved"] is None and "expected_target_layers" in message:
        checks["target_layers_resolved"] = False
    if checks["boundary_policy_resolved"] is None and (
        "expected_boundary_encoding" in message or "expected_boundary_raw_enabled" in message
    ):
        checks["boundary_policy_resolved"] = False
    if checks["distance_policy_resolved"] is None and "expected_distance_target" in message:
        checks["distance_policy_resolved"] = False
    if checks["valid_semantics_resolved"] is None and "expected_valid_saved_separately" in message:
        checks["valid_semantics_resolved"] = False
    if "target_metadata" in message and checks["target_metadata_consistent"] is None:
        checks["target_metadata_consistent"] = False
    return checks


def run_prepare_targets_stage(
    *,
    output_dir: str | Path,
    run_id: str,
    raster_path: Any,
    vector_path: Any,
    config: PrepDataConfig | Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    valid_path: Any | None = None,
    target_metadata: Any | None = None,
    source_manifest_path: str | Path | None = None,
    module_version: str | None = None,
    runtime_compute_enabled: bool = True,
) -> PrepareTargetsStageResult:
    """Run lightweight `04_prepare_targets` stage and write manifest/summary.

    Transitional baseline policy:
      - input references are provided as stage args;
      - this stage resolves target contract metadata only;
      - no real target-generation runtime is executed here.
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

    target_layers: list[str] | None = None
    boundary_encoding: str | None = None
    distance_target: str | None = None
    valid_saved_separately: bool | None = None
    boundary_raw_policy: dict[str, Any] | None = None
    valid_semantics: dict[str, Any] | None = None
    target_metadata_checked = False
    input_raster_path: str | None = None
    input_vector_path: str | None = None
    input_valid_path: str | None = None
    normalized_source_manifest_path: str | None = None
    target_contract_mode = _TARGET_CONTRACT_MODE
    extent_output_path: str | None = None
    boundary_output_path: str | None = None
    boundary_raw_output_path: str | None = None
    distance_output_path: str | None = None
    valid_output_path: str | None = None

    try:
        input_paths = prep_data_validators.validate_input_paths_contract(
            raster_path=raster_path,
            vector_path=vector_path,
            aoi_path=None,
        )
        input_raster_path = str(input_paths["raster_path"])
        input_vector_path = str(input_paths["vector_path"])
        input_valid_path = _normalize_optional_path("valid_path", valid_path)
        if source_manifest_path is not None:
            normalized_source_manifest_path = str(
                _normalize_path("source_manifest_path", source_manifest_path)
            )

        resolved_config, config_used_path = _resolve_config(config=config, config_path=config_path)
        resolved = _resolve_target_contract(
            resolved_config=resolved_config,
            target_metadata=target_metadata,
        )

        target_layers = resolved["target_layers"]
        boundary_encoding = resolved["boundary_encoding"]
        distance_target = resolved["distance_target"]
        valid_saved_separately = resolved["valid_saved_separately"]
        boundary_raw_policy = resolved["boundary_raw_policy"]
        valid_semantics = resolved["valid_semantics"]
        target_metadata_checked = resolved["target_metadata_checked"]

        if runtime_compute_enabled:
            if targets_compute is None:
                raise ContractError(
                    "runtime_compute_enabled=True but targets_compute module "
                    "could not be imported (rasterio/geopandas/scipy missing?)."
                )
            compute_result = targets_compute.compute_and_save_targets(
                raster_path=raster_path,
                vector_path=vector_path,
                output_dir=output_root,
                valid_path=valid_path,
            )
            extent_output_path = str(compute_result["extent_path"])
            boundary_output_path = str(compute_result["boundary_path"])
            boundary_raw_output_path = str(compute_result["boundary_raw_path"])
            distance_output_path = str(compute_result["distance_path"])
            valid_output_path = str(compute_result["valid_path"])
            target_contract_mode = "runtime_compute"

        status = "success"
        checks = _build_success_checks(target_metadata_checked=resolved["target_metadata_checked"])
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
            "feature_mode": None if resolved_config is None else resolved_config.feature_mode,
            "valid_policy_nodata_source": (
                None if resolved_config is None else resolved_config.valid_policy.nodata_source
            ),
            "boundary_encoding": boundary_encoding,
            "distance_target": distance_target,
            "runtime_compute_enabled": runtime_compute_enabled,
            "target_contract_mode": target_contract_mode,
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
        "inputs": {"artifacts": []},
        "outputs": {"artifacts": []},
        "resolved_contract": {
            "spatial": {},
            "features": {
                "feature_mode": None if resolved_config is None else resolved_config.feature_mode,
            },
            "valid_policy": {
                "nodata_source": (
                    None if resolved_config is None else resolved_config.valid_policy.nodata_source
                ),
                "compute_before_fill": (
                    None if resolved_config is None else resolved_config.valid_policy.compute_before_fill
                ),
                "valid_saved_separately": valid_saved_separately,
                "valid_ignore_semantics": (
                    None if valid_semantics is None else valid_semantics["valid_ignore_semantics"]
                ),
            },
            "normalization": None,
            "aoi_policy": (
                None
                if resolved_config is None
                else {
                    "enabled": resolved_config.aoi.enabled,
                    "buffer_m": float(resolved_config.aoi.buffer_m),
                }
            ),
        },
        "runtime": {
            "device_requested": None,
            "device_resolved": None,
            "amp_requested": None,
            "amp_used": None,
            "oom_fallbacks_applied": [],
            "notes": [],
        },
        "input_raster_path": input_raster_path,
        "input_vector_path": input_vector_path,
        "input_valid_path": input_valid_path,
        "extent_output_path": extent_output_path,
        "boundary_output_path": boundary_output_path,
        "boundary_raw_output_path": boundary_raw_output_path,
        "distance_output_path": distance_output_path,
        "valid_output_path": valid_output_path,
        "target_contract_mode": target_contract_mode,
        "target_layers": target_layers if target_layers is not None else [],
        "boundary_encoding": boundary_encoding,
        "distance_target": distance_target,
        "valid_saved_separately": valid_saved_separately,
        "valid_semantics": valid_semantics,
        "boundary_raw_policy": boundary_raw_policy,
        "checks": checks,
        "diagnostics": {
            "warnings": [],
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
        "target_contract_mode": target_contract_mode,
        "target_layers": target_layers,
        "boundary_encoding": boundary_encoding,
        "distance_target": distance_target,
        "target_metadata_consistent": checks["target_metadata_consistent"],
        "source_manifest_path": normalized_source_manifest_path,
        "blocking_issues": blocking_issues,
        "manifest_path": str(manifest_path),
        "error_type": error_type,
    }
    write_summary(summary_path, summary_payload)

    return PrepareTargetsStageResult(
        status=status,
        manifest_path=manifest_path,
        summary_path=summary_path,
        target_layers=None if target_layers is None else tuple(target_layers),
        boundary_encoding=boundary_encoding,
        distance_target=distance_target,
        valid_saved_separately=valid_saved_separately,
        boundary_raw_enabled=(
            None if boundary_raw_policy is None else bool(boundary_raw_policy["enabled"])
        ),
        blocking_issues=tuple(blocking_issues),
        checks=checks,
        error_type=error_type,
        error_message=error_message,
    )
