"""Lightweight stage skeleton for `03_prepare_features` in module_prep_data.

This stage intentionally resolves only metadata/contract-level feature context.
It writes manifest/summary artifacts and returns a small test-friendly result.
It does not perform real feature computation (no NDVI/SAVI/NDWI runtime),
tensor assembly, or normalization execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from ai_fields.common.constants import CHANNEL_COUNTS, DATA_CONTRACT_VERSION, DERIVED_INDICES
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import write_manifest, write_summary
from ai_fields.module_prep_data import config as prep_data_config
from ai_fields.module_prep_data.schemas import PrepDataConfig

try:
    from ai_fields.module_prep_data import features_compute
except ImportError:  # pragma: no cover
    features_compute = None  # type: ignore[assignment]

_STAGE_NAME = "03_prepare_features"
_MANIFEST_SCHEMA_NAME = "prep_data.features_manifest"
_SUMMARY_SCHEMA_NAME = "prep_data.summary"
_SCHEMA_VERSION = "v1"
_MODULE_NAME = "module_prep_data"
_MANIFEST_FILENAME = "features_manifest.json"
_SUMMARY_FILENAME = "summary.json"
_INPUT_REFS_SOURCE = "stage_args_transitional"

_FEATURE_MODE_TO_ASSEMBLED_VARIANT = {
    "raw8": "raw8_valid",
    "raw8_idx3": "raw8_idx3_valid",
}


@dataclass(frozen=True)
class PrepareFeaturesStageResult:
    """Minimal stage outcome for `03_prepare_features`."""

    status: Literal["success", "failed"]
    manifest_path: Path
    summary_path: Path
    feature_mode: str | None
    feature_channel_count: int | None
    derived_indices: tuple[str, ...] | None
    assembled_model_input_variants: tuple[str, ...] | None
    valid_saved_separately: bool | None
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


def _resolve_feature_contract(
    *,
    resolved_config: PrepDataConfig,
    feature_metadata: Any | None,
) -> dict[str, Any]:
    mode = resolved_config.feature_mode
    if mode not in _FEATURE_MODE_TO_ASSEMBLED_VARIANT:
        raise ContractError(
            f"Unsupported feature_mode in stage 03: {mode!r}. "
            f"Supported: {sorted(_FEATURE_MODE_TO_ASSEMBLED_VARIANT)}."
        )

    feature_channel_count = CHANNEL_COUNTS[mode]
    derived_indices = list(DERIVED_INDICES) if mode == "raw8_idx3" else []
    assembled_variant = _FEATURE_MODE_TO_ASSEMBLED_VARIANT[mode]
    assembled_variants = [assembled_variant]
    channel_semantics: list[str] | None = None
    feature_metadata_checked = feature_metadata is not None

    # `feature_metadata` is a lightweight consistency hint only.
    if feature_metadata is not None:
        if not isinstance(feature_metadata, Mapping):
            raise ContractError(
                f"feature_metadata must be a mapping/object, got {type(feature_metadata).__name__}."
            )
        if "expected_feature_channel_count" in feature_metadata:
            raw_count = feature_metadata["expected_feature_channel_count"]
            if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                raise ContractError(
                    "feature_metadata.expected_feature_channel_count must be an integer."
                )
            if raw_count != feature_channel_count:
                raise ContractError(
                    "feature_metadata.expected_feature_channel_count is inconsistent with "
                    f"feature_mode={mode!r}: expected {feature_channel_count}, got {raw_count}."
                )
        if "expected_derived_indices" in feature_metadata:
            expected_derived = _require_str_list(
                "feature_metadata.expected_derived_indices",
                feature_metadata["expected_derived_indices"],
            )
            if expected_derived != derived_indices:
                raise ContractError(
                    "feature_metadata.expected_derived_indices is inconsistent with "
                    f"feature_mode={mode!r}: expected {derived_indices}, got {expected_derived}."
                )
        if "expected_assembled_model_input_variants" in feature_metadata:
            expected_variants = _require_str_list(
                "feature_metadata.expected_assembled_model_input_variants",
                feature_metadata["expected_assembled_model_input_variants"],
            )
            if expected_variants != assembled_variants:
                raise ContractError(
                    "feature_metadata.expected_assembled_model_input_variants is inconsistent with "
                    f"feature_mode={mode!r}: expected {assembled_variants}, got {expected_variants}."
                )
        if "channel_semantics" in feature_metadata:
            channel_semantics = _require_str_list(
                "feature_metadata.channel_semantics",
                feature_metadata["channel_semantics"],
            )
            if len(channel_semantics) != feature_channel_count:
                raise ContractError(
                    "feature_metadata.channel_semantics length is inconsistent with "
                    f"feature_mode={mode!r}: expected {feature_channel_count}, "
                    f"got {len(channel_semantics)}."
                )

    return {
        "feature_mode": mode,
        "feature_channel_count": feature_channel_count,
        "derived_indices": derived_indices,
        "assembled_model_input_variants": assembled_variants,
        "channel_semantics": channel_semantics,
        "channel_semantics_resolved": True if channel_semantics is not None else None,
        "feature_metadata_checked": feature_metadata_checked,
        "valid_saved_separately": True,
        "normalization_plan": {
            "normalization_name": resolved_config.normalization.name,
            "dtype_before_model": "float32",
            "clip_percentiles": [
                float(resolved_config.normalization.clip_percentiles[0]),
                float(resolved_config.normalization.clip_percentiles[1]),
            ],
            "scaling_range": [
                float(resolved_config.normalization.scale_range[0]),
                float(resolved_config.normalization.scale_range[1]),
            ],
        },
    }


def _build_success_checks(
    *,
    feature_metadata_checked: bool,
    channel_semantics_resolved: bool | None,
) -> dict[str, Any]:
    return {
        "contract_checks_passed": True,
        "feature_mode_resolved": True,
        "derived_indices_resolved": True,
        "assembled_variants_resolved": True,
        "normalization_plan_resolved": True,
        "feature_metadata_consistent": True if feature_metadata_checked else None,
        "channel_semantics_resolved": channel_semantics_resolved,
        "blocking_issues": [],
    }


def _build_failure_checks(error: ContractError) -> dict[str, Any]:
    checks = {
        "contract_checks_passed": False,
        "feature_mode_resolved": None,
        "derived_indices_resolved": None,
        "assembled_variants_resolved": None,
        "normalization_plan_resolved": None,
        "feature_metadata_consistent": None,
        "channel_semantics_resolved": None,
        "blocking_issues": [str(error)],
    }
    message = str(error)
    if "feature_mode" in message:
        checks["feature_mode_resolved"] = False
    if "derived_indices" in message:
        checks["derived_indices_resolved"] = False
        checks["feature_metadata_consistent"] = False
    if "assembled_model_input_variants" in message:
        checks["assembled_variants_resolved"] = False
        checks["feature_metadata_consistent"] = False
    if "normalization" in message:
        checks["normalization_plan_resolved"] = False
    if "feature_metadata" in message and checks["feature_metadata_consistent"] is None:
        checks["feature_metadata_consistent"] = False
    if "channel_semantics" in message:
        checks["channel_semantics_resolved"] = False
        checks["feature_metadata_consistent"] = False
    return checks


def run_prepare_features_stage(
    *,
    output_dir: str | Path,
    run_id: str,
    config: PrepDataConfig | Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    raster_path: Any | None = None,
    valid_path: Any | None = None,
    feature_metadata: Any | None = None,
    source_manifest_path: str | Path | None = None,
    module_version: str | None = None,
    runtime_compute_enabled: bool = True,
) -> PrepareFeaturesStageResult:
    """Run lightweight `03_prepare_features` stage and write manifest/summary.

    Transitional baseline policy:
      - input references are provided as stage args;
      - this stage resolves feature contract metadata only;
      - no real feature computation is executed here.
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
    feature_channel_count: int | None = None
    derived_indices: list[str] | None = None
    assembled_variants: list[str] | None = None
    valid_saved_separately: bool | None = None
    channel_semantics: list[str] | None = None
    channel_semantics_resolved: bool | None = None
    feature_metadata_checked = False
    normalization_plan: dict[str, Any] | None = None
    input_raster_path: str | None = None
    input_valid_path: str | None = None
    normalized_source_manifest_path: str | None = None
    img_output_path: str | None = None
    valid_output_path: str | None = None
    features_compute_mode: str | None = None

    try:
        input_raster_path = _normalize_optional_path("raster_path", raster_path)
        input_valid_path = _normalize_optional_path("valid_path", valid_path)
        if source_manifest_path is not None:
            normalized_source_manifest_path = str(
                _normalize_path("source_manifest_path", source_manifest_path)
            )

        resolved_config, config_used_path = _resolve_config(config=config, config_path=config_path)
        resolved = _resolve_feature_contract(
            resolved_config=resolved_config,
            feature_metadata=feature_metadata,
        )

        feature_mode = resolved["feature_mode"]
        feature_channel_count = resolved["feature_channel_count"]
        derived_indices = resolved["derived_indices"]
        assembled_variants = resolved["assembled_model_input_variants"]
        valid_saved_separately = resolved["valid_saved_separately"]
        channel_semantics = resolved["channel_semantics"]
        channel_semantics_resolved = resolved["channel_semantics_resolved"]
        feature_metadata_checked = resolved["feature_metadata_checked"]
        normalization_plan = resolved["normalization_plan"]

        if runtime_compute_enabled:
            if features_compute is None:
                raise ContractError(
                    "runtime_compute_enabled=True but features_compute module "
                    "could not be imported (rasterio missing?)."
                )
            if raster_path is None:
                raise ContractError(
                    "runtime_compute_enabled=True requires raster_path to be provided."
                )
            compute_result = features_compute.compute_and_save_features(
                raster_path=raster_path,
                output_dir=output_root,
                feature_mode=feature_mode,
            )
            img_output_path = str(compute_result["img_path"])
            valid_output_path = str(compute_result["valid_path"])
            feature_channel_count = compute_result["feature_channel_count"]
            channel_semantics = list(compute_result["channel_semantics"])
            features_compute_mode = compute_result["features_compute_mode"]

        status = "success"
        checks = _build_success_checks(
            feature_metadata_checked=resolved["feature_metadata_checked"],
            channel_semantics_resolved=resolved["channel_semantics_resolved"],
        )
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
            "feature_mode": feature_mode,
            "valid_policy_nodata_source": (
                None if resolved_config is None else resolved_config.valid_policy.nodata_source
            ),
            "runtime_compute_enabled": runtime_compute_enabled,
            "features_compute_mode": features_compute_mode,
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
                "derived_indices": derived_indices,
                "assembled_model_input_variants": assembled_variants,
                "valid_saved_separately": valid_saved_separately,
                "channel_semantics_resolved": channel_semantics_resolved,
                "feature_metadata_checked": feature_metadata_checked,
            },
            "valid_policy": {},
            "normalization": normalization_plan,
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
        "input_raster_path": input_raster_path,
        "input_valid_path": input_valid_path,
        "img_output_path": img_output_path,
        "valid_output_path": valid_output_path,
        "feature_mode": feature_mode,
        "feature_channel_count": feature_channel_count,
        "channel_semantics": channel_semantics if channel_semantics is not None else [],
        "channel_semantics_status": (
            "resolved" if channel_semantics_resolved is True else "unresolved"
        ),
        "derived_indices": derived_indices if derived_indices is not None else [],
        "valid_saved_separately": valid_saved_separately,
        "assembled_model_input_variants": assembled_variants if assembled_variants is not None else [],
        "normalization_plan": normalization_plan,
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
        "feature_mode": feature_mode,
        "feature_channel_count": feature_channel_count,
        "assembled_model_input_variants": assembled_variants,
        "channel_semantics_resolved": channel_semantics_resolved,
        "feature_metadata_consistent": checks["feature_metadata_consistent"],
        "source_manifest_path": normalized_source_manifest_path,
        "blocking_issues": blocking_issues,
        "manifest_path": str(manifest_path),
        "error_type": error_type,
    }
    write_summary(summary_path, summary_payload)

    return PrepareFeaturesStageResult(
        status=status,
        manifest_path=manifest_path,
        summary_path=summary_path,
        feature_mode=feature_mode,
        feature_channel_count=feature_channel_count,
        derived_indices=None if derived_indices is None else tuple(derived_indices),
        assembled_model_input_variants=(
            None if assembled_variants is None else tuple(assembled_variants)
        ),
        valid_saved_separately=valid_saved_separately,
        blocking_issues=tuple(blocking_issues),
        checks=checks,
        error_type=error_type,
        error_message=error_message,
    )
