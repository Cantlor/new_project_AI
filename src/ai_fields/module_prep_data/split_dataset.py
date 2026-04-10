"""Stage 06_split_dataset in module_prep_data.

When runtime_compute_enabled=True (default) and patches_dir is provided,
performs real split assignment, export layout creation and normalization stats
computation via split_compute.  Otherwise falls back to metadata-snapshot mode.
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
    from ai_fields.module_prep_data import split_compute  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    split_compute = None  # type: ignore[assignment]

_STAGE_NAME = "06_split_dataset"
_MANIFEST_SCHEMA_NAME = "prep_data.split_manifest"
_SUMMARY_SCHEMA_NAME = "prep_data.summary"
_SCHEMA_VERSION = "v1"
_MODULE_NAME = "module_prep_data"
_MANIFEST_FILENAME = "split_manifest.json"
_SUMMARY_FILENAME = "summary.json"
_INPUT_REFS_SOURCE = "stage_args_transitional"
_SPLIT_CONTRACT_MODE = "metadata_snapshot_only"

_ERROR_CODE_ATTR = "stage_error_code"
_ERR_SPLIT_METADATA_TYPE = "split_metadata_type_invalid"
_ERR_EXPECTED_SPLIT_POLICY_INVALID = "expected_split_policy_invalid"
_ERR_EXPECTED_SPLIT_POLICY_MISMATCH = "expected_split_policy_mismatch"
_ERR_EXPECTED_RANDOM_SEED_INVALID = "expected_random_seed_invalid"
_ERR_EXPECTED_RANDOM_SEED_MISMATCH = "expected_random_seed_mismatch"
_ERR_EXPECTED_FEATURE_CHANNEL_COUNT_INVALID = "expected_feature_channel_count_invalid"
_ERR_EXPECTED_FEATURE_CHANNEL_COUNT_MISMATCH = "expected_feature_channel_count_mismatch"
_ERR_EXPECTED_EXPORT_REQUIRED_DIRS_INVALID = "expected_export_required_dirs_invalid"
_ERR_EXPECTED_EXPORT_REQUIRED_DIRS_MISMATCH = "expected_export_required_dirs_mismatch"

_FAILURE_CHECK_UPDATES_BY_CODE: dict[str, dict[str, Any]] = {
    _ERR_EXPECTED_SPLIT_POLICY_INVALID: {
        "split_policy_resolved": False,
        "split_metadata_consistent": False,
    },
    _ERR_EXPECTED_SPLIT_POLICY_MISMATCH: {
        "split_policy_resolved": False,
        "split_metadata_consistent": False,
    },
    _ERR_EXPECTED_RANDOM_SEED_INVALID: {
        "random_seed_resolved": False,
        "split_metadata_consistent": False,
    },
    _ERR_EXPECTED_RANDOM_SEED_MISMATCH: {
        "random_seed_resolved": False,
        "split_metadata_consistent": False,
    },
    _ERR_EXPECTED_FEATURE_CHANNEL_COUNT_INVALID: {
        "feature_contract_resolved": False,
        "split_metadata_consistent": False,
    },
    _ERR_EXPECTED_FEATURE_CHANNEL_COUNT_MISMATCH: {
        "feature_contract_resolved": False,
        "split_metadata_consistent": False,
    },
    _ERR_EXPECTED_EXPORT_REQUIRED_DIRS_INVALID: {
        "export_structure_resolved": False,
        "split_metadata_consistent": False,
    },
    _ERR_EXPECTED_EXPORT_REQUIRED_DIRS_MISMATCH: {
        "export_structure_resolved": False,
        "split_metadata_consistent": False,
    },
    _ERR_SPLIT_METADATA_TYPE: {
        "split_metadata_consistent": False,
    },
}


@dataclass(frozen=True)
class SplitDatasetStageResult:
    """Minimal stage outcome for `06_split_dataset`."""

    status: Literal["success", "failed"]
    manifest_path: Path
    summary_path: Path
    split_policy: str | None
    random_seed: int | None
    feature_mode: str | None
    feature_channel_count: int | None
    split_assignment_executed: bool | None
    export_layout_materialized: bool | None
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


def _require_int_or_null(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(
            f"{name} must be an integer or null, got {value!r} ({type(value).__name__})."
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


def _resolve_split_contract(
    *,
    resolved_config: PrepDataConfig,
    split_metadata: Any | None,
) -> dict[str, Any]:
    split_policy = resolved_config.split.policy
    random_seed = resolved_config.split.random_seed
    feature_mode = resolved_config.feature_mode
    if feature_mode not in CHANNEL_COUNTS:
        raise ContractError(
            f"feature_mode {feature_mode!r} is not supported by channel-count contract."
        )
    feature_channel_count = CHANNEL_COUNTS[feature_mode]
    required_dirs = list(REQUIRED_SAMPLE_LAYERS)
    split_ratio_plan = None
    channel_semantics: list[str] | None = None

    # Skeleton-level placeholders only; no real split/export runtime is executed.
    splits = {
        "train": {"sample_count": 0},
        "val": {"sample_count": 0},
        "test": {"sample_count": 0},
    }
    split_assignment_executed = False
    export_layout_materialized = False
    split_metadata_checked = split_metadata is not None

    if split_metadata is not None:
        if not isinstance(split_metadata, Mapping):
            raise _stage_contract_error(
                code=_ERR_SPLIT_METADATA_TYPE,
                message=(
                    "split_metadata must be a mapping/object, got "
                    f"{type(split_metadata).__name__}."
                ),
            )

        if "expected_split_policy" in split_metadata:
            try:
                expected_split_policy = _require_non_empty_str(
                    "split_metadata.expected_split_policy",
                    split_metadata["expected_split_policy"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_SPLIT_POLICY_INVALID,
                    message=str(exc),
                ) from exc
            if expected_split_policy != split_policy:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_SPLIT_POLICY_MISMATCH,
                    message=(
                        "split_metadata.expected_split_policy is inconsistent with config: "
                        f"expected {split_policy!r}, got {expected_split_policy!r}."
                    ),
                )

        if "expected_random_seed" in split_metadata:
            try:
                expected_random_seed = _require_int_or_null(
                    "split_metadata.expected_random_seed",
                    split_metadata["expected_random_seed"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_RANDOM_SEED_INVALID,
                    message=str(exc),
                ) from exc
            if expected_random_seed != random_seed:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_RANDOM_SEED_MISMATCH,
                    message=(
                        "split_metadata.expected_random_seed is inconsistent with config: "
                        f"expected {random_seed!r}, got {expected_random_seed!r}."
                    ),
                )

        if "expected_feature_channel_count" in split_metadata:
            try:
                expected_feature_channel_count = _require_positive_int(
                    "split_metadata.expected_feature_channel_count",
                    split_metadata["expected_feature_channel_count"],
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
                        "split_metadata.expected_feature_channel_count is inconsistent with "
                        f"feature_mode={feature_mode!r}: expected {feature_channel_count}, "
                        f"got {expected_feature_channel_count}."
                    ),
                )

        if "expected_export_required_dirs" in split_metadata:
            try:
                expected_dirs = _require_str_list(
                    "split_metadata.expected_export_required_dirs",
                    split_metadata["expected_export_required_dirs"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_EXPORT_REQUIRED_DIRS_INVALID,
                    message=str(exc),
                ) from exc
            if expected_dirs != required_dirs:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_EXPORT_REQUIRED_DIRS_MISMATCH,
                    message=(
                        "split_metadata.expected_export_required_dirs is inconsistent with "
                        f"baseline export structure: expected {required_dirs}, got {expected_dirs}."
                    ),
                )

    return {
        "split_policy": split_policy,
        "random_seed": random_seed,
        "feature_mode": feature_mode,
        "feature_channel_count": feature_channel_count,
        "split_ratio_plan": split_ratio_plan,
        "channel_semantics": channel_semantics,
        "required_dirs": required_dirs,
        "splits": splits,
        "split_assignment_executed": split_assignment_executed,
        "export_layout_materialized": export_layout_materialized,
        "split_metadata_checked": split_metadata_checked,
    }


def _build_success_checks(*, split_metadata_checked: bool) -> dict[str, Any]:
    return {
        "contract_checks_passed": True,
        "split_contract_resolved": True,
        "split_policy_resolved": True,
        "random_seed_resolved": True,
        "feature_contract_resolved": True,
        "split_ratios_resolved": None,
        "channel_semantics_resolved": None,
        "export_structure_resolved": True,
        "split_metadata_consistent": True if split_metadata_checked else None,
        "split_assignment_executed": False,
        "export_layout_materialized": False,
        "blocking_issues": [],
    }


def _build_failure_checks(error: ContractError) -> dict[str, Any]:
    checks = {
        "contract_checks_passed": False,
        "split_contract_resolved": False,
        "split_policy_resolved": None,
        "random_seed_resolved": None,
        "feature_contract_resolved": None,
        "split_ratios_resolved": None,
        "channel_semantics_resolved": None,
        "export_structure_resolved": None,
        "split_metadata_consistent": None,
        "split_assignment_executed": None,
        "export_layout_materialized": None,
        "blocking_issues": [str(error)],
    }
    code = _get_error_code(error)
    if code in _FAILURE_CHECK_UPDATES_BY_CODE:
        checks.update(_FAILURE_CHECK_UPDATES_BY_CODE[code])

    message = str(error)
    if checks["split_policy_resolved"] is None and "expected_split_policy" in message:
        checks["split_policy_resolved"] = False
    if checks["random_seed_resolved"] is None and "expected_random_seed" in message:
        checks["random_seed_resolved"] = False
    if (
        checks["feature_contract_resolved"] is None
        and "expected_feature_channel_count" in message
    ):
        checks["feature_contract_resolved"] = False
    if checks["export_structure_resolved"] is None and "expected_export_required_dirs" in message:
        checks["export_structure_resolved"] = False
    if "split_metadata" in message and checks["split_metadata_consistent"] is None:
        checks["split_metadata_consistent"] = False
    return checks


def run_split_dataset_stage(
    *,
    output_dir: str | Path,
    run_id: str,
    raster_path: Any,
    vector_path: Any,
    config: PrepDataConfig | Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    split_metadata: Any | None = None,
    source_manifest_path: str | Path | None = None,
    module_version: str | None = None,
    runtime_compute_enabled: bool = True,
    patches_dir: Any | None = None,
) -> SplitDatasetStageResult:
    """Run lightweight `06_split_dataset` stage and write manifest/summary.

    Transitional baseline policy:
      - input references are provided as stage args;
      - this stage resolves split/export contract metadata only;
      - no real split assignment or export runtime is executed here.
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

    split_policy: str | None = None
    patch_size: int | None = None
    random_seed: int | None = None
    feature_mode: str | None = None
    feature_channel_count: int | None = None
    split_ratio_plan: Any = None
    channel_semantics: list[str] | None = None
    required_dirs: list[str] | None = None
    splits: dict[str, dict[str, int]] | None = None
    split_assignment_executed: bool | None = None
    export_layout_materialized: bool | None = None
    split_metadata_checked = False
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
        resolved = _resolve_split_contract(
            resolved_config=resolved_config,
            split_metadata=split_metadata,
        )
        status = "success"
        checks = _build_success_checks(split_metadata_checked=resolved["split_metadata_checked"])
        blocking_issues = []

        split_policy = resolved["split_policy"]
        patch_size = int(resolved_config.patches.patch_size)
        random_seed = resolved["random_seed"]
        feature_mode = resolved["feature_mode"]
        feature_channel_count = resolved["feature_channel_count"]
        split_ratio_plan = resolved["split_ratio_plan"]
        channel_semantics = resolved["channel_semantics"]
        required_dirs = resolved["required_dirs"]
        splits = resolved["splits"]
        split_assignment_executed = resolved["split_assignment_executed"]
        export_layout_materialized = resolved["export_layout_materialized"]
        split_metadata_checked = resolved["split_metadata_checked"]

        if runtime_compute_enabled and patches_dir is not None:
            if split_compute is None:
                raise ContractError(
                    "runtime_compute_enabled=True but split_compute module "
                    "could not be imported (numpy missing?)."
                )
            compute_result = split_compute.compute_and_save_split(
                patches_dir=patches_dir,
                output_dir=output_root,
                config=resolved_config,
            )
            splits = {
                "train": {"sample_count": compute_result["train_count"]},
                "val": {"sample_count": compute_result["val_count"]},
                "test": {"sample_count": compute_result["test_count"]},
            }
            split_assignment_executed = compute_result["split_assignment_executed"]
            export_layout_materialized = compute_result["export_layout_materialized"]
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
            "split_policy": split_policy,
            "random_seed": random_seed,
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
                "feature_mode": feature_mode,
                "feature_channel_count": feature_channel_count,
                "channel_semantics_resolved": None,
            },
            "valid_policy": {
                "nodata_source": (
                    None if resolved_config is None else resolved_config.valid_policy.nodata_source
                ),
                "compute_before_fill": (
                    None if resolved_config is None else resolved_config.valid_policy.compute_before_fill
                ),
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
        "split_contract_mode": (
            "runtime_compute" if export_layout_materialized else _SPLIT_CONTRACT_MODE
        ),
        "runtime_compute_enabled": runtime_compute_enabled,
        "split_policy": split_policy,
        "patch_size": patch_size,
        "random_seed": random_seed,
        "feature_mode": feature_mode,
        "feature_channel_count": feature_channel_count,
        "channel_semantics": channel_semantics if channel_semantics is not None else [],
        "channel_semantics_status": "unresolved",
        "split_ratio_plan": split_ratio_plan,
        "split_ratio_plan_status": "unresolved",
        "splits": (
            splits
            if splits is not None
            else {
                "train": {"sample_count": 0},
                "val": {"sample_count": 0},
                "test": {"sample_count": 0},
            }
        ),
        "split_assignment_executed": split_assignment_executed,
        "export_structure": {
            "required_dirs": required_dirs if required_dirs is not None else list(REQUIRED_SAMPLE_LAYERS),
            "materialized": export_layout_materialized,
        },
        "export_layout_materialized": export_layout_materialized,
        "split_metadata_checked": split_metadata_checked,
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
        "split_contract_mode": (
            "runtime_compute" if export_layout_materialized else _SPLIT_CONTRACT_MODE
        ),
        "split_policy": split_policy,
        "patch_size": patch_size,
        "random_seed": random_seed,
        "feature_mode": feature_mode,
        "feature_channel_count": feature_channel_count,
        "split_assignment_executed": split_assignment_executed,
        "export_layout_materialized": export_layout_materialized,
        "split_metadata_consistent": checks["split_metadata_consistent"],
        "source_manifest_path": normalized_source_manifest_path,
        "blocking_issues": blocking_issues,
        "manifest_path": str(manifest_path),
        "error_type": error_type,
    }
    write_summary(summary_path, summary_payload)

    return SplitDatasetStageResult(
        status=status,
        manifest_path=manifest_path,
        summary_path=summary_path,
        split_policy=split_policy,
        random_seed=random_seed,
        feature_mode=feature_mode,
        feature_channel_count=feature_channel_count,
        split_assignment_executed=split_assignment_executed,
        export_layout_materialized=export_layout_materialized,
        blocking_issues=tuple(blocking_issues),
        checks=checks,
        error_type=error_type,
        error_message=error_message,
    )
