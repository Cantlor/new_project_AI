"""Stage 07_validate_outputs in module_prep_data.

When runtime_compute_enabled=True (default) and dataset_dir is provided,
performs real dataset scanning via validate_outputs_compute.
Otherwise falls back to metadata-snapshot mode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from ai_fields.common.constants import CHANNEL_COUNTS, DATA_CONTRACT_VERSION, REQUIRED_SAMPLE_LAYERS
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import write_manifest, write_summary
from ai_fields.module_prep_data import config as prep_data_config
from ai_fields.module_prep_data import validators as prep_data_validators
from ai_fields.module_prep_data.schemas import PrepDataConfig

try:
    from ai_fields.module_prep_data import validate_outputs_compute  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    validate_outputs_compute = None  # type: ignore[assignment]

_STAGE_NAME = "07_validate_outputs"
# Canonical baseline choice for stage-07 manifest naming:
# - module_prep_data.md requires a dedicated manifest per stage;
# - MANIFEST_SCHEMAS.md does not yet define stage-07 schema explicitly.
# We standardize on `prep_data.validate_outputs_manifest` and
# `validate_outputs_manifest.json`.
_MANIFEST_SCHEMA_NAME = "prep_data.validate_outputs_manifest"
_SUMMARY_SCHEMA_NAME = "prep_data.summary"
_SCHEMA_VERSION = "v1"
_MODULE_NAME = "module_prep_data"
_MANIFEST_FILENAME = "validate_outputs_manifest.json"
_SUMMARY_FILENAME = "summary.json"
_INPUT_REFS_SOURCE = "stage_args_transitional"
_VALIDATION_CONTRACT_MODE = "metadata_snapshot_only"

_TARGET_LAYERS = ("extent", "boundary", "distance", "valid")
_FEATURE_MODE_TO_ASSEMBLED_VARIANT = {
    "raw8": "raw8_valid",
    "raw8_idx3": "raw8_idx3_valid",
}

_ERROR_CODE_ATTR = "stage_error_code"
_ERR_VALIDATION_METADATA_TYPE = "validation_metadata_type_invalid"
_ERR_EXPECTED_VALIDATION_CONTRACT_MODE_INVALID = "expected_validation_contract_mode_invalid"
_ERR_EXPECTED_VALIDATION_CONTRACT_MODE_MISMATCH = "expected_validation_contract_mode_mismatch"
_ERR_EXPECTED_REQUIRED_DIRS_INVALID = "expected_required_dirs_invalid"
_ERR_EXPECTED_REQUIRED_DIRS_MISMATCH = "expected_required_dirs_mismatch"
_ERR_EXPECTED_TARGET_LAYERS_INVALID = "expected_target_layers_invalid"
_ERR_EXPECTED_TARGET_LAYERS_MISMATCH = "expected_target_layers_mismatch"
_ERR_EXPECTED_FEATURE_CHANNEL_COUNT_INVALID = "expected_feature_channel_count_invalid"
_ERR_EXPECTED_FEATURE_CHANNEL_COUNT_MISMATCH = "expected_feature_channel_count_mismatch"
_ERR_EXPECTED_MODEL_INPUT_CHANNEL_COUNT_INVALID = "expected_model_input_channel_count_invalid"
_ERR_EXPECTED_MODEL_INPUT_CHANNEL_COUNT_MISMATCH = "expected_model_input_channel_count_mismatch"
_ERR_EXPECTED_VALID_SAVED_SEPARATELY_INVALID = "expected_valid_saved_separately_invalid"
_ERR_EXPECTED_VALID_SAVED_SEPARATELY_MISMATCH = "expected_valid_saved_separately_mismatch"

_FAILURE_CHECK_UPDATES_BY_CODE: dict[str, dict[str, Any]] = {
    _ERR_EXPECTED_VALIDATION_CONTRACT_MODE_INVALID: {
        "validation_contract_mode_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_VALIDATION_CONTRACT_MODE_MISMATCH: {
        "validation_contract_mode_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_REQUIRED_DIRS_INVALID: {
        "dataset_structure_contract_resolved": False,
        "split_export_contract_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_REQUIRED_DIRS_MISMATCH: {
        "dataset_structure_contract_resolved": False,
        "split_export_contract_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_TARGET_LAYERS_INVALID: {
        "target_layers_contract_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_TARGET_LAYERS_MISMATCH: {
        "target_layers_contract_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_FEATURE_CHANNEL_COUNT_INVALID: {
        "feature_contract_snapshot_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_FEATURE_CHANNEL_COUNT_MISMATCH: {
        "feature_contract_snapshot_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_MODEL_INPUT_CHANNEL_COUNT_INVALID: {
        "feature_contract_snapshot_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_MODEL_INPUT_CHANNEL_COUNT_MISMATCH: {
        "feature_contract_snapshot_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_VALID_SAVED_SEPARATELY_INVALID: {
        "feature_contract_snapshot_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_EXPECTED_VALID_SAVED_SEPARATELY_MISMATCH: {
        "feature_contract_snapshot_resolved": False,
        "validation_metadata_consistent": False,
    },
    _ERR_VALIDATION_METADATA_TYPE: {
        "validation_metadata_consistent": False,
    },
}


@dataclass(frozen=True)
class ValidateOutputsStageResult:
    """Minimal stage outcome for `07_validate_outputs`."""

    status: Literal["success", "failed"]
    manifest_path: Path
    summary_path: Path
    feature_mode: str | None
    feature_channel_count: int | None
    model_input_channel_count_after_valid: int | None
    validation_runtime_executed: bool | None
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


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(
            f"{name} must be a positive integer, got {value!r} ({type(value).__name__})."
        )
    if value <= 0:
        raise ContractError(f"{name} must be > 0, got {value}.")
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


def _resolve_validation_contract(
    *,
    resolved_config: PrepDataConfig,
    validation_metadata: Any | None,
) -> dict[str, Any]:
    feature_mode = resolved_config.feature_mode
    if feature_mode not in _FEATURE_MODE_TO_ASSEMBLED_VARIANT:
        raise ContractError(
            f"Unsupported feature_mode for stage 07: {feature_mode!r}. "
            f"Supported: {sorted(_FEATURE_MODE_TO_ASSEMBLED_VARIANT)}."
        )
    assembled_variant = _FEATURE_MODE_TO_ASSEMBLED_VARIANT[feature_mode]
    feature_channel_count = CHANNEL_COUNTS[feature_mode]
    model_input_channel_count_after_valid = CHANNEL_COUNTS[assembled_variant]
    required_dirs = list(REQUIRED_SAMPLE_LAYERS)
    target_layers = list(_TARGET_LAYERS)
    valid_saved_separately = True

    # These checks are intentionally unresolved in skeleton mode because no
    # output scanning or raster/vector reads are executed here.
    runtime_checks = {
        "shapes_consistency_checked": None,
        "target_value_domains_checked": None,
        "broken_files_checked": None,
        "valid_nodata_consistency_checked": None,
        "boundary_coverage_checked": None,
        "distance_boundary_consistency_checked": None,
    }
    validation_runtime_executed = False
    validation_metadata_checked = validation_metadata is not None

    if validation_metadata is not None:
        if not isinstance(validation_metadata, Mapping):
            raise _stage_contract_error(
                code=_ERR_VALIDATION_METADATA_TYPE,
                message=(
                    "validation_metadata must be a mapping/object, got "
                    f"{type(validation_metadata).__name__}."
                ),
            )

        if "expected_validation_contract_mode" in validation_metadata:
            try:
                expected_contract_mode = _require_non_empty_str(
                    "validation_metadata.expected_validation_contract_mode",
                    validation_metadata["expected_validation_contract_mode"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_VALIDATION_CONTRACT_MODE_INVALID,
                    message=str(exc),
                ) from exc
            if expected_contract_mode != _VALIDATION_CONTRACT_MODE:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_VALIDATION_CONTRACT_MODE_MISMATCH,
                    message=(
                        "validation_metadata.expected_validation_contract_mode is inconsistent with "
                        f"stage baseline: expected {_VALIDATION_CONTRACT_MODE!r}, "
                        f"got {expected_contract_mode!r}."
                    ),
                )

        if "expected_required_dirs" in validation_metadata:
            try:
                expected_required_dirs = _require_str_list(
                    "validation_metadata.expected_required_dirs",
                    validation_metadata["expected_required_dirs"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_REQUIRED_DIRS_INVALID,
                    message=str(exc),
                ) from exc
            if expected_required_dirs != required_dirs:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_REQUIRED_DIRS_MISMATCH,
                    message=(
                        "validation_metadata.expected_required_dirs is inconsistent with "
                        f"baseline export structure: expected {required_dirs}, "
                        f"got {expected_required_dirs}."
                    ),
                )

        if "expected_target_layers" in validation_metadata:
            try:
                expected_target_layers = _require_str_list(
                    "validation_metadata.expected_target_layers",
                    validation_metadata["expected_target_layers"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_TARGET_LAYERS_INVALID,
                    message=str(exc),
                ) from exc
            if expected_target_layers != target_layers:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_TARGET_LAYERS_MISMATCH,
                    message=(
                        "validation_metadata.expected_target_layers is inconsistent with "
                        f"baseline target layers: expected {target_layers}, "
                        f"got {expected_target_layers}."
                    ),
                )

        if "expected_feature_channel_count" in validation_metadata:
            try:
                expected_feature_channel_count = _require_positive_int(
                    "validation_metadata.expected_feature_channel_count",
                    validation_metadata["expected_feature_channel_count"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_FEATURE_CHANNEL_COUNT_INVALID,
                    message=str(exc),
                ) from exc
            if expected_feature_channel_count != feature_channel_count:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_FEATURE_CHANNEL_COUNT_MISMATCH,
                    message=(
                        "validation_metadata.expected_feature_channel_count is inconsistent with "
                        f"feature_mode={feature_mode!r}: expected {feature_channel_count}, "
                        f"got {expected_feature_channel_count}."
                    ),
                )

        if "expected_model_input_channel_count_after_valid" in validation_metadata:
            try:
                expected_model_input_channel_count = _require_positive_int(
                    "validation_metadata.expected_model_input_channel_count_after_valid",
                    validation_metadata["expected_model_input_channel_count_after_valid"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_MODEL_INPUT_CHANNEL_COUNT_INVALID,
                    message=str(exc),
                ) from exc
            if expected_model_input_channel_count != model_input_channel_count_after_valid:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_MODEL_INPUT_CHANNEL_COUNT_MISMATCH,
                    message=(
                        "validation_metadata.expected_model_input_channel_count_after_valid is "
                        "inconsistent with assembled contract: expected "
                        f"{model_input_channel_count_after_valid}, "
                        f"got {expected_model_input_channel_count}."
                    ),
                )

        if "expected_valid_saved_separately" in validation_metadata:
            try:
                expected_valid_saved_separately = _require_bool(
                    "validation_metadata.expected_valid_saved_separately",
                    validation_metadata["expected_valid_saved_separately"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_VALID_SAVED_SEPARATELY_INVALID,
                    message=str(exc),
                ) from exc
            if expected_valid_saved_separately != valid_saved_separately:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_VALID_SAVED_SEPARATELY_MISMATCH,
                    message=(
                        "validation_metadata.expected_valid_saved_separately is inconsistent with "
                        f"baseline contract: expected {valid_saved_separately}, "
                        f"got {expected_valid_saved_separately}."
                    ),
                )

    return {
        "feature_mode": feature_mode,
        "assembled_model_input": assembled_variant,
        "feature_channel_count": feature_channel_count,
        "model_input_channel_count_after_valid": model_input_channel_count_after_valid,
        "required_dirs": required_dirs,
        "target_layers": target_layers,
        "valid_saved_separately": valid_saved_separately,
        "runtime_checks": runtime_checks,
        "validation_runtime_executed": validation_runtime_executed,
        "validation_metadata_checked": validation_metadata_checked,
    }


def _build_success_checks(*, validation_metadata_checked: bool) -> dict[str, Any]:
    return {
        "contract_checks_passed": True,
        "validation_contract_resolved": True,
        "validation_contract_mode_resolved": True,
        "dataset_structure_contract_resolved": True,
        "target_layers_contract_resolved": True,
        "feature_contract_snapshot_resolved": True,
        "split_export_contract_resolved": True,
        "runtime_output_scan_executed": False,
        "shapes_consistency_checked": None,
        "target_value_domains_checked": None,
        "broken_files_checked": None,
        "valid_nodata_consistency_checked": None,
        "boundary_coverage_checked": None,
        "distance_boundary_consistency_checked": None,
        "validation_metadata_consistent": True if validation_metadata_checked else None,
        "blocking_issues": [],
    }


def _build_failure_checks(error: ContractError) -> dict[str, Any]:
    checks = {
        "contract_checks_passed": False,
        "validation_contract_resolved": False,
        "validation_contract_mode_resolved": None,
        "dataset_structure_contract_resolved": None,
        "target_layers_contract_resolved": None,
        "feature_contract_snapshot_resolved": None,
        "split_export_contract_resolved": None,
        "runtime_output_scan_executed": None,
        "shapes_consistency_checked": None,
        "target_value_domains_checked": None,
        "broken_files_checked": None,
        "valid_nodata_consistency_checked": None,
        "boundary_coverage_checked": None,
        "distance_boundary_consistency_checked": None,
        "validation_metadata_consistent": None,
        "blocking_issues": [str(error)],
    }
    code = _get_error_code(error)
    if code in _FAILURE_CHECK_UPDATES_BY_CODE:
        checks.update(_FAILURE_CHECK_UPDATES_BY_CODE[code])

    # Fallback keeps diagnostics readable for uncoded contract errors.
    message = str(error)
    if checks["validation_contract_mode_resolved"] is None and "expected_validation_contract_mode" in message:
        checks["validation_contract_mode_resolved"] = False
    if checks["dataset_structure_contract_resolved"] is None and "expected_required_dirs" in message:
        checks["dataset_structure_contract_resolved"] = False
        checks["split_export_contract_resolved"] = False
    if checks["target_layers_contract_resolved"] is None and "expected_target_layers" in message:
        checks["target_layers_contract_resolved"] = False
    if checks["feature_contract_snapshot_resolved"] is None and (
        "expected_feature_channel_count" in message
        or "expected_model_input_channel_count_after_valid" in message
        or "expected_valid_saved_separately" in message
    ):
        checks["feature_contract_snapshot_resolved"] = False
    if "validation_metadata" in message and checks["validation_metadata_consistent"] is None:
        checks["validation_metadata_consistent"] = False
    return checks


def run_validate_outputs_stage(
    *,
    output_dir: str | Path,
    run_id: str,
    raster_path: Any,
    vector_path: Any,
    config: PrepDataConfig | Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    validation_metadata: Any | None = None,
    source_manifest_path: str | Path | None = None,
    module_version: str | None = None,
    runtime_compute_enabled: bool = True,
    dataset_dir: Any | None = None,
) -> ValidateOutputsStageResult:
    """Run lightweight `07_validate_outputs` stage and write manifest/summary.

    Transitional baseline policy:
      - input references are provided as stage args;
      - this stage resolves output-validation contract metadata only;
      - no real output scanning or raster/vector runtime validation is executed.
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

    feature_mode: str | None = None
    patch_size: int | None = None
    assembled_model_input: str | None = None
    feature_channel_count: int | None = None
    model_input_channel_count_after_valid: int | None = None
    required_dirs: list[str] | None = None
    target_layers: list[str] | None = None
    valid_saved_separately: bool | None = None
    runtime_checks: dict[str, Any] | None = None
    validation_runtime_executed: bool | None = None
    validation_metadata_checked = False
    input_raster_path: str | None = None
    input_vector_path: str | None = None
    normalized_source_manifest_path: str | None = None

    try:
        input_paths = prep_data_validators.validate_input_paths_contract(
            raster_path=raster_path,
            vector_path=vector_path,
            aoi_path=None,
        )
        input_raster_path = str(input_paths["raster_path"])
        input_vector_path = str(input_paths["vector_path"])
        normalized_source_manifest_path = _normalize_optional_path(
            "source_manifest_path",
            source_manifest_path,
        )

        resolved_config, config_used_path = _resolve_config(config=config, config_path=config_path)
        resolved = _resolve_validation_contract(
            resolved_config=resolved_config,
            validation_metadata=validation_metadata,
        )
        status = "success"
        checks = _build_success_checks(
            validation_metadata_checked=resolved["validation_metadata_checked"]
        )
        blocking_issues = []

        feature_mode = resolved["feature_mode"]
        patch_size = int(resolved_config.patches.patch_size)
        assembled_model_input = resolved["assembled_model_input"]
        feature_channel_count = resolved["feature_channel_count"]
        model_input_channel_count_after_valid = resolved["model_input_channel_count_after_valid"]
        required_dirs = resolved["required_dirs"]
        target_layers = resolved["target_layers"]
        valid_saved_separately = resolved["valid_saved_separately"]
        runtime_checks = resolved["runtime_checks"]
        validation_runtime_executed = resolved["validation_runtime_executed"]
        validation_metadata_checked = resolved["validation_metadata_checked"]

        if runtime_compute_enabled and dataset_dir is not None:
            if validate_outputs_compute is None:
                raise ContractError(
                    "runtime_compute_enabled=True but validate_outputs_compute module "
                    "could not be imported (rasterio missing?)."
                )
            compute_result = validate_outputs_compute.validate_dataset(
                dataset_dir=dataset_dir,
                config=resolved_config,
            )
            runtime_checks = {
                "shapes_consistency_checked": compute_result["shapes_ok"],
                "target_value_domains_checked": compute_result["domains_ok"],
                "broken_files_checked": len(compute_result["issues"]) == 0,
                "valid_nodata_consistency_checked": None,
                "boundary_coverage_checked": None,
                "distance_boundary_consistency_checked": None,
            }
            validation_runtime_executed = True
            if compute_result["issues"]:
                raise ContractError(
                    f"Dataset validation found {len(compute_result['issues'])} issue(s): "
                    + "; ".join(compute_result["issues"][:3])
                )
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
            "feature_mode": feature_mode,
            "validation_contract_mode": _VALIDATION_CONTRACT_MODE,
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
            "spatial": {
                "validation_runtime_executed": validation_runtime_executed,
            },
            "features": {
                "feature_mode": feature_mode,
                "assembled_model_input": assembled_model_input,
                "feature_channel_count": feature_channel_count,
                "final_input_channel_count": model_input_channel_count_after_valid,
                "valid_as_input_channel": True if feature_mode is not None else None,
            },
            "valid_policy": {
                "nodata_source": (
                    None if resolved_config is None else resolved_config.valid_policy.nodata_source
                ),
                "compute_before_fill": (
                    None if resolved_config is None else resolved_config.valid_policy.compute_before_fill
                ),
                "valid_saved_separately": valid_saved_separately,
            },
            "normalization": (
                None
                if resolved_config is None
                else {
                    "normalization_name": resolved_config.normalization.name,
                    "clip_percentiles": [
                        float(resolved_config.normalization.clip_percentiles[0]),
                        float(resolved_config.normalization.clip_percentiles[1]),
                    ],
                    "scaling_range": [
                        float(resolved_config.normalization.scale_range[0]),
                        float(resolved_config.normalization.scale_range[1]),
                    ],
                }
            ),
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
        "validation_contract_mode": (
            "runtime_compute" if validation_runtime_executed else _VALIDATION_CONTRACT_MODE
        ),
        "runtime_compute_enabled": runtime_compute_enabled,
        "required_dirs_contract": required_dirs if required_dirs is not None else list(REQUIRED_SAMPLE_LAYERS),
        "target_layers_contract": target_layers if target_layers is not None else list(_TARGET_LAYERS),
        "feature_mode": feature_mode,
        "patch_size": patch_size,
        "feature_channel_count_before_valid": feature_channel_count,
        "assembled_model_input": assembled_model_input,
        "model_input_channel_count_after_valid": model_input_channel_count_after_valid,
        "valid_saved_separately": valid_saved_separately,
        "validation_runtime_executed": validation_runtime_executed,
        "runtime_checks": runtime_checks
        if runtime_checks is not None
        else {
            "shapes_consistency_checked": None,
            "target_value_domains_checked": None,
            "broken_files_checked": None,
            "valid_nodata_consistency_checked": None,
            "boundary_coverage_checked": None,
            "distance_boundary_consistency_checked": None,
        },
        "validation_metadata_checked": validation_metadata_checked,
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
        "validation_contract_mode": (
            "runtime_compute" if validation_runtime_executed else _VALIDATION_CONTRACT_MODE
        ),
        "feature_mode": feature_mode,
        "patch_size": patch_size,
        "feature_channel_count_before_valid": feature_channel_count,
        "model_input_channel_count_after_valid": model_input_channel_count_after_valid,
        "validation_runtime_executed": validation_runtime_executed,
        "validation_metadata_consistent": checks["validation_metadata_consistent"],
        "source_manifest_path": normalized_source_manifest_path,
        "blocking_issues": blocking_issues,
        "manifest_path": str(manifest_path),
        "error_type": error_type,
    }
    write_summary(summary_path, summary_payload)

    return ValidateOutputsStageResult(
        status=status,
        manifest_path=manifest_path,
        summary_path=summary_path,
        feature_mode=feature_mode,
        feature_channel_count=feature_channel_count,
        model_input_channel_count_after_valid=model_input_channel_count_after_valid,
        validation_runtime_executed=validation_runtime_executed,
        blocking_issues=tuple(blocking_issues),
        checks=checks,
        error_type=error_type,
        error_message=error_message,
    )
