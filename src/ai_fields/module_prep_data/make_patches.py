"""Stage 05_make_patches in module_prep_data.

When runtime_compute_enabled=True (default) and all five layer paths
(img, extent, boundary, distance, valid) are provided, performs real patch
extraction via patches_compute.  Otherwise falls back to metadata-snapshot mode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from ai_fields.common.constants import DATA_CONTRACT_VERSION, REQUIRED_SAMPLE_LAYERS
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import read_manifest, write_manifest, write_summary
from ai_fields.module_prep_data import config as prep_data_config
from ai_fields.module_prep_data import validators as prep_data_validators
from ai_fields.module_prep_data.schemas import PrepDataConfig

try:
    from ai_fields.module_prep_data import patches_compute  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    patches_compute = None  # type: ignore[assignment]

_STAGE_NAME = "05_make_patches"
_MANIFEST_SCHEMA_NAME = "prep_data.patches_manifest"
_SUMMARY_SCHEMA_NAME = "prep_data.summary"
_SCHEMA_VERSION = "v1"
_MODULE_NAME = "module_prep_data"
_MANIFEST_FILENAME = "patches_manifest.json"
_SUMMARY_FILENAME = "summary.json"
_INPUT_REFS_SOURCE = "stage_args_transitional"
_PATCH_CONTRACT_MODE = "metadata_snapshot_only"
_SPATIAL_CONTEXT_SCHEMA_NAME = "prep_data.aoi_manifest"
_PATCH_RUNTIME_MODE_PATCH_FIRST = "patch_first_from_source"
_PATCH_RUNTIME_MODE_FULL_SCENE = "legacy_full_scene_diagnostic"

_ERROR_CODE_ATTR = "stage_error_code"
_ERR_PATCH_METADATA_TYPE = "patch_metadata_type_invalid"
_ERR_EXPECTED_PATCH_SIZE_INVALID = "expected_patch_size_invalid"
_ERR_EXPECTED_PATCH_SIZE_MISMATCH = "expected_patch_size_mismatch"
_ERR_EXPECTED_SAMPLING_POLICY_INVALID = "expected_sampling_policy_invalid"
_ERR_EXPECTED_SAMPLING_POLICY_MISMATCH = "expected_sampling_policy_mismatch"
_ERR_EXPECTED_PATCH_LAYERS_INVALID = "expected_patch_layers_invalid"
_ERR_EXPECTED_PATCH_LAYERS_MISMATCH = "expected_patch_layers_mismatch"
_ERR_EXPECTED_PATCH_EXPORTS_INVALID = "expected_patch_exports_invalid"
_ERR_EXPECTED_PATCH_EXPORTS_MISMATCH = "expected_patch_exports_mismatch"

_FAILURE_CHECK_UPDATES_BY_CODE: dict[str, dict[str, Any]] = {
    _ERR_EXPECTED_PATCH_SIZE_INVALID: {
        "patch_size_resolved": False,
        "patch_metadata_consistent": False,
    },
    _ERR_EXPECTED_PATCH_SIZE_MISMATCH: {
        "patch_size_resolved": False,
        "patch_metadata_consistent": False,
    },
    _ERR_EXPECTED_SAMPLING_POLICY_INVALID: {
        "sampling_policy_resolved": False,
        "patch_metadata_consistent": False,
    },
    _ERR_EXPECTED_SAMPLING_POLICY_MISMATCH: {
        "sampling_policy_resolved": False,
        "patch_metadata_consistent": False,
    },
    _ERR_EXPECTED_PATCH_LAYERS_INVALID: {
        "patch_layers_resolved": False,
        "patch_metadata_consistent": False,
    },
    _ERR_EXPECTED_PATCH_LAYERS_MISMATCH: {
        "patch_layers_resolved": False,
        "patch_metadata_consistent": False,
    },
    _ERR_EXPECTED_PATCH_EXPORTS_INVALID: {
        "patch_exports_snapshot_resolved": False,
        "patch_metadata_consistent": False,
    },
    _ERR_EXPECTED_PATCH_EXPORTS_MISMATCH: {
        "patch_exports_snapshot_resolved": False,
        "patch_metadata_consistent": False,
    },
    _ERR_PATCH_METADATA_TYPE: {
        "patch_metadata_consistent": False,
    },
}


@dataclass(frozen=True)
class MakePatchesStageResult:
    """Minimal stage outcome for `05_make_patches`."""

    status: Literal["success", "failed"]
    manifest_path: Path
    summary_path: Path
    patch_size: int | None
    sampling_policy: str | None
    patch_layers: tuple[str, ...] | None
    written_total: int | None
    patch_runtime_executed: bool | None
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


def _extract_stage02_runtime_context(
    source_manifest_path: str | None,
) -> dict[str, Any] | None:
    if source_manifest_path is None:
        return None
    manifest = read_manifest(source_manifest_path)
    if manifest.get("schema_name") != _SPATIAL_CONTEXT_SCHEMA_NAME:
        return None

    spatial = manifest.get("resolved_contract", {}).get("spatial", {})
    source_kind = manifest.get("source_kind", spatial.get("source_kind"))
    source_path = manifest.get("source_path", spatial.get("source_path"))
    vector_runtime_path = manifest.get(
        "vector_runtime_path", spatial.get("vector_runtime_path")
    )
    spatial_context_mode = manifest.get(
        "spatial_context_mode", spatial.get("spatial_context_mode", "full_raster")
    )
    effective_extent_bounds = manifest.get(
        "effective_extent_bounds", spatial.get("effective_extent_bounds")
    )

    if not isinstance(source_path, str) or source_path.strip() == "":
        raise ContractError(
            "Stage-02 manifest does not provide canonical source_path for patch-first stage 05."
        )
    if not isinstance(vector_runtime_path, str) or vector_runtime_path.strip() == "":
        raise ContractError(
            "Stage-02 manifest does not provide vector_runtime_path for patch-first stage 05."
        )
    if source_kind not in {"direct_raster", "vrt", "block_cache"}:
        raise ContractError(
            "Stage-02 manifest has invalid source_kind for patch-first stage 05: "
            f"{source_kind!r}."
        )
    if spatial_context_mode not in {"full_raster", "aoi_limited"}:
        raise ContractError(
            "Stage-02 manifest has invalid spatial_context_mode for patch-first stage 05: "
            f"{spatial_context_mode!r}."
        )

    return {
        "source_kind": source_kind,
        "source_path": source_path,
        "vector_runtime_path": vector_runtime_path,
        "spatial_context_mode": spatial_context_mode,
        "effective_extent_bounds": effective_extent_bounds,
        "memory_budget_mb": manifest.get("memory_budget_mb", spatial.get("memory_budget_mb")),
        "large_scene_mode": manifest.get("large_scene_mode", spatial.get("large_scene_mode")),
        "auto_mode_reason": manifest.get("auto_mode_reason", spatial.get("auto_mode_reason")),
        "full_scene_materialization_allowed": manifest.get(
            "full_scene_materialization_allowed",
            spatial.get("full_scene_materialization_allowed"),
        ),
    }


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


def _require_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(
            f"{name} must be a non-negative integer, got {value!r} ({type(value).__name__})."
        )
    if value < 0:
        raise ContractError(f"{name} must be >= 0, got {value}.")
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


def _resolve_patch_contract(
    *,
    resolved_config: PrepDataConfig,
    patch_metadata: Any | None,
) -> dict[str, Any]:
    patch_size = int(resolved_config.patches.patch_size)
    sampling_policy = resolved_config.patches.sampling_policy
    patch_layers = list(REQUIRED_SAMPLE_LAYERS)
    patch_exports = {f"{layer}_count": 0 for layer in patch_layers}
    written_total = 0
    written_center: int | None = None
    written_boundary: int | None = None
    written_negative: int | None = None
    shortfall_negative: int | None = None
    rejection_stats = {
        "invalid_ratio_rejects": None,
        "mask_ratio_rejects": None,
        "boundary_quality_rejects": None,
        "duplicate_or_overlap_rejects": None,
    }
    patch_runtime_executed = False
    patch_artifacts_materialized = False
    patch_metadata_checked = patch_metadata is not None

    if patch_metadata is not None:
        if not isinstance(patch_metadata, Mapping):
            raise _stage_contract_error(
                code=_ERR_PATCH_METADATA_TYPE,
                message=(
                    "patch_metadata must be a mapping/object, got "
                    f"{type(patch_metadata).__name__}."
                ),
            )

        if "expected_patch_size" in patch_metadata:
            try:
                expected_patch_size = _require_non_negative_int(
                    "patch_metadata.expected_patch_size",
                    patch_metadata["expected_patch_size"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_PATCH_SIZE_INVALID,
                    message=str(exc),
                ) from exc
            if expected_patch_size != patch_size:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_PATCH_SIZE_MISMATCH,
                    message=(
                        "patch_metadata.expected_patch_size is inconsistent with config: "
                        f"expected {patch_size}, got {expected_patch_size}."
                    ),
                )

        if "expected_sampling_policy" in patch_metadata:
            try:
                expected_sampling_policy = _require_non_empty_str(
                    "patch_metadata.expected_sampling_policy",
                    patch_metadata["expected_sampling_policy"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_SAMPLING_POLICY_INVALID,
                    message=str(exc),
                ) from exc
            if expected_sampling_policy != sampling_policy:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_SAMPLING_POLICY_MISMATCH,
                    message=(
                        "patch_metadata.expected_sampling_policy is inconsistent with config: "
                        f"expected {sampling_policy!r}, got {expected_sampling_policy!r}."
                    ),
                )

        if "expected_patch_layers" in patch_metadata:
            try:
                expected_patch_layers = _require_str_list(
                    "patch_metadata.expected_patch_layers",
                    patch_metadata["expected_patch_layers"],
                )
            except ContractError as exc:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_PATCH_LAYERS_INVALID,
                    message=str(exc),
                ) from exc
            if expected_patch_layers != patch_layers:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_PATCH_LAYERS_MISMATCH,
                    message=(
                        "patch_metadata.expected_patch_layers is inconsistent with baseline "
                        f"patch contract: expected {patch_layers}, got {expected_patch_layers}."
                    ),
                )

        if "expected_patch_exports" in patch_metadata:
            raw_exports = patch_metadata["expected_patch_exports"]
            if not isinstance(raw_exports, Mapping):
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_PATCH_EXPORTS_INVALID,
                    message=(
                        "patch_metadata.expected_patch_exports must be a mapping/object, got "
                        f"{type(raw_exports).__name__}."
                    ),
                )
            unknown = [k for k in raw_exports if k not in patch_exports]
            missing = [k for k in patch_exports if k not in raw_exports]
            if unknown or missing:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_PATCH_EXPORTS_INVALID,
                    message=(
                        "patch_metadata.expected_patch_exports keys must match baseline patch "
                        f"export keys exactly. missing={sorted(missing)}, unknown={sorted(unknown)}."
                    ),
                )

            normalized_expected_exports: dict[str, int] = {}
            for key, value in raw_exports.items():
                try:
                    normalized_expected_exports[key] = _require_non_negative_int(
                        f"patch_metadata.expected_patch_exports.{key}",
                        value,
                    )
                except ContractError as exc:
                    raise _stage_contract_error(
                        code=_ERR_EXPECTED_PATCH_EXPORTS_INVALID,
                        message=str(exc),
                    ) from exc

            if normalized_expected_exports != patch_exports:
                raise _stage_contract_error(
                    code=_ERR_EXPECTED_PATCH_EXPORTS_MISMATCH,
                    message=(
                        "patch_metadata.expected_patch_exports is inconsistent with skeleton "
                        f"snapshot: expected {patch_exports}, got {normalized_expected_exports}."
                    ),
                )

    return {
        "patch_size": patch_size,
        "sampling_policy": sampling_policy,
        "patch_layers": patch_layers,
        "patch_exports": patch_exports,
        "written_total": written_total,
        "written_center": written_center,
        "written_boundary": written_boundary,
        "written_negative": written_negative,
        "shortfall_negative": shortfall_negative,
        "rejection_stats": rejection_stats,
        "patch_runtime_executed": patch_runtime_executed,
        "patch_artifacts_materialized": patch_artifacts_materialized,
        "patch_metadata_checked": patch_metadata_checked,
    }


def _build_success_checks(*, patch_metadata_checked: bool) -> dict[str, Any]:
    return {
        "contract_checks_passed": True,
        "patch_contract_resolved": True,
        "patch_size_resolved": True,
        "sampling_policy_resolved": True,
        "patch_layers_resolved": True,
        "patch_exports_snapshot_resolved": True,
        "patch_metadata_consistent": True if patch_metadata_checked else None,
        "patch_runtime_executed": False,
        "blocking_issues": [],
    }


def _build_failure_checks(error: ContractError) -> dict[str, Any]:
    checks = {
        "contract_checks_passed": False,
        "patch_contract_resolved": False,
        "patch_size_resolved": None,
        "sampling_policy_resolved": None,
        "patch_layers_resolved": None,
        "patch_exports_snapshot_resolved": None,
        "patch_metadata_consistent": None,
        "patch_runtime_executed": None,
        "blocking_issues": [str(error)],
    }
    code = _get_error_code(error)
    if code in _FAILURE_CHECK_UPDATES_BY_CODE:
        checks.update(_FAILURE_CHECK_UPDATES_BY_CODE[code])

    message = str(error)
    if checks["patch_size_resolved"] is None and "expected_patch_size" in message:
        checks["patch_size_resolved"] = False
    if checks["sampling_policy_resolved"] is None and "expected_sampling_policy" in message:
        checks["sampling_policy_resolved"] = False
    if checks["patch_layers_resolved"] is None and "expected_patch_layers" in message:
        checks["patch_layers_resolved"] = False
    if checks["patch_exports_snapshot_resolved"] is None and "expected_patch_exports" in message:
        checks["patch_exports_snapshot_resolved"] = False
    if "patch_metadata" in message and checks["patch_metadata_consistent"] is None:
        checks["patch_metadata_consistent"] = False
    return checks


def run_make_patches_stage(
    *,
    output_dir: str | Path,
    run_id: str,
    raster_path: Any,
    vector_path: Any,
    config: PrepDataConfig | Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    img_path: Any | None = None,
    extent_path: Any | None = None,
    boundary_path: Any | None = None,
    distance_path: Any | None = None,
    valid_path: Any | None = None,
    patch_metadata: Any | None = None,
    source_manifest_path: str | Path | None = None,
    spatial_manifest_path: str | Path | None = None,
    module_version: str | None = None,
    runtime_compute_enabled: bool = True,
) -> MakePatchesStageResult:
    """Run lightweight `05_make_patches` stage and write manifest/summary.

    Transitional baseline policy:
      - input references are provided as stage args;
      - this stage resolves patching/sampling contract metadata only;
      - no real patch extraction/sampling runtime is executed here.
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

    patch_size: int | None = None
    sampling_policy: str | None = None
    patch_layers: list[str] | None = None
    patch_exports: dict[str, int] | None = None
    written_total: int | None = None
    written_center: int | None = None
    written_boundary: int | None = None
    written_negative: int | None = None
    shortfall_negative: int | None = None
    rejection_stats: dict[str, int | None] | None = None
    patch_runtime_executed: bool | None = None
    patch_artifacts_materialized: bool | None = None
    patch_metadata_checked = False

    input_raster_path: str | None = None
    input_vector_path: str | None = None
    input_img_path: str | None = None
    input_extent_path: str | None = None
    input_boundary_path: str | None = None
    input_distance_path: str | None = None
    input_valid_path: str | None = None
    normalized_source_manifest_path: str | None = None
    normalized_spatial_manifest_path: str | None = None
    stage02_runtime_context: dict[str, Any] | None = None
    patch_runtime_mode = _PATCH_CONTRACT_MODE
    canonical_source_path: str | None = None
    canonical_source_kind: str | None = None
    vector_runtime_path: str | None = None
    spatial_context_mode: str | None = None
    effective_extent_bounds_used: list[float] | None = None
    memory_budget_mb: int | None = None
    large_scene_mode: bool | None = None
    auto_mode_reason: str | None = None
    full_scene_materialization_allowed: bool | None = None
    vector_load_policy: dict[str, Any] | None = None
    block_aware_reading: dict[str, Any] | None = None
    double_read_avoided: bool | None = None
    candidate_windows_total: int | None = None

    try:
        input_paths = prep_data_validators.validate_input_paths_contract(
            raster_path=raster_path,
            vector_path=vector_path,
            aoi_path=None,
        )
        input_raster_path = str(input_paths["raster_path"])
        input_vector_path = str(input_paths["vector_path"])
        input_img_path = _normalize_optional_path("img_path", img_path)
        input_extent_path = _normalize_optional_path("extent_path", extent_path)
        input_boundary_path = _normalize_optional_path("boundary_path", boundary_path)
        input_distance_path = _normalize_optional_path("distance_path", distance_path)
        input_valid_path = _normalize_optional_path("valid_path", valid_path)
        if source_manifest_path is not None:
            normalized_source_manifest_path = str(
                _normalize_path("source_manifest_path", source_manifest_path)
            )
        if spatial_manifest_path is not None:
            normalized_spatial_manifest_path = str(
                _normalize_path("spatial_manifest_path", spatial_manifest_path)
            )

        resolved_config, config_used_path = _resolve_config(config=config, config_path=config_path)
        resolved = _resolve_patch_contract(
            resolved_config=resolved_config,
            patch_metadata=patch_metadata,
        )
        status = "success"
        checks = _build_success_checks(patch_metadata_checked=resolved["patch_metadata_checked"])
        blocking_issues = []

        patch_size = resolved["patch_size"]
        sampling_policy = resolved["sampling_policy"]
        patch_layers = resolved["patch_layers"]
        patch_exports = resolved["patch_exports"]
        written_total = resolved["written_total"]
        written_center = resolved["written_center"]
        written_boundary = resolved["written_boundary"]
        written_negative = resolved["written_negative"]
        shortfall_negative = resolved["shortfall_negative"]
        rejection_stats = resolved["rejection_stats"]
        patch_runtime_executed = resolved["patch_runtime_executed"]
        patch_artifacts_materialized = resolved["patch_artifacts_materialized"]
        patch_metadata_checked = resolved["patch_metadata_checked"]

        if runtime_compute_enabled:
            stage02_manifest_for_patch_first = normalized_spatial_manifest_path
            if (
                stage02_manifest_for_patch_first is None
                and normalized_source_manifest_path is not None
            ):
                maybe_ctx = _extract_stage02_runtime_context(normalized_source_manifest_path)
                if maybe_ctx is not None:
                    stage02_manifest_for_patch_first = normalized_source_manifest_path
                    stage02_runtime_context = maybe_ctx
            if stage02_manifest_for_patch_first is not None and stage02_runtime_context is None:
                stage02_runtime_context = _extract_stage02_runtime_context(
                    stage02_manifest_for_patch_first
                )
            if normalized_spatial_manifest_path is not None and stage02_runtime_context is None:
                raise ContractError(
                    "spatial_manifest_path must reference a valid stage-02 aoi_manifest."
                )

        _all_layer_paths_provided = all(
            p is not None
            for p in [img_path, extent_path, boundary_path, distance_path, valid_path]
        )
        if runtime_compute_enabled and stage02_runtime_context is not None:
            if patches_compute is None:
                raise ContractError(
                    "runtime_compute_enabled=True but patches_compute module "
                    "could not be imported (rasterio/numpy missing?)."
                )
            canonical_source_path = stage02_runtime_context["source_path"]
            canonical_source_kind = stage02_runtime_context["source_kind"]
            vector_runtime_path = stage02_runtime_context["vector_runtime_path"]
            spatial_context_mode = stage02_runtime_context["spatial_context_mode"]
            if isinstance(stage02_runtime_context.get("effective_extent_bounds"), list):
                effective_extent_bounds_used = [
                    float(v) for v in stage02_runtime_context["effective_extent_bounds"]
                ]
            else:
                effective_extent_bounds_used = None
            memory_budget_mb = stage02_runtime_context.get("memory_budget_mb")
            large_scene_mode = stage02_runtime_context.get("large_scene_mode")
            auto_mode_reason = stage02_runtime_context.get("auto_mode_reason")
            full_scene_materialization_allowed = stage02_runtime_context.get(
                "full_scene_materialization_allowed"
            )

            compute_result = patches_compute.compute_and_save_patches_from_source(
                raster_source_path=canonical_source_path,
                vector_runtime_path=vector_runtime_path,
                output_dir=output_root,
                config=resolved_config,
                feature_mode=resolved_config.feature_mode,
                source_kind=canonical_source_kind,
                spatial_context_mode=spatial_context_mode,
                effective_extent_bounds=effective_extent_bounds_used,
            )
            written_total = compute_result["written_total"]
            written_center = compute_result["written_center"]
            written_boundary = compute_result["written_boundary"]
            written_negative = compute_result["written_negative"]
            rejection_stats = compute_result["rejection_stats"]
            vector_load_policy = compute_result.get("vector_load_policy")
            block_aware_reading = compute_result.get("block_aware_reading")
            double_read_avoided = compute_result.get("double_read_avoided")
            candidate_windows_total = compute_result.get("candidate_windows_total")
            patch_runtime_executed = True
            patch_artifacts_materialized = True
            patch_runtime_mode = _PATCH_RUNTIME_MODE_PATCH_FIRST
        elif runtime_compute_enabled and _all_layer_paths_provided:
            if patches_compute is None:
                raise ContractError(
                    "runtime_compute_enabled=True but patches_compute module "
                    "could not be imported (rasterio/numpy missing?)."
                )
            compute_result = patches_compute.compute_and_save_patches(
                img_path=img_path,
                extent_path=extent_path,
                boundary_path=boundary_path,
                distance_path=distance_path,
                valid_path=valid_path,
                output_dir=output_root,
                config=resolved_config,
                feature_mode=resolved_config.feature_mode,
            )
            written_total = compute_result["written_total"]
            written_center = compute_result["written_center"]
            written_boundary = compute_result["written_boundary"]
            written_negative = compute_result["written_negative"]
            rejection_stats = compute_result["rejection_stats"]
            patch_runtime_executed = True
            patch_artifacts_materialized = True
            patch_runtime_mode = _PATCH_RUNTIME_MODE_FULL_SCENE
        elif runtime_compute_enabled:
            any_layer_path_provided = any(
                p is not None
                for p in [img_path, extent_path, boundary_path, distance_path, valid_path]
            )
            if any_layer_path_provided and not _all_layer_paths_provided:
                raise ContractError(
                    "Stage 05 diagnostic full-scene mode requires all layer paths: "
                    "img_path, extent_path, boundary_path, distance_path, valid_path."
                )
            # Metadata-only snapshot mode is still supported for unit-level
            # contract checks when no runtime inputs are provided.
    except ContractError as exc:
        status = "failed"
        error_type = type(exc).__name__
        error_message = str(exc)
        checks = _build_failure_checks(exc)
        blocking_issues = [str(exc)]

    if status == "success":
        checks["patch_runtime_executed"] = patch_runtime_executed

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
            "patch_size": patch_size,
            "sampling_policy": sampling_policy,
            "halo_px": None if resolved_config is None else int(resolved_config.patches.halo_px),
            "distance_clip_px": (
                None if resolved_config is None else int(resolved_config.distance.distance_clip_px)
            ),
        },
        "provenance": {
            "source_run_ids": [],
            "source_manifest_paths": [
                *(
                    [normalized_source_manifest_path]
                    if normalized_source_manifest_path is not None
                    else []
                ),
                *(
                    [normalized_spatial_manifest_path]
                    if normalized_spatial_manifest_path is not None
                    else []
                ),
            ],
            "source_config_paths": [],
            "code_version": None,
            "git_commit": None,
        },
        "inputs": {"artifacts": []},
        "outputs": {"artifacts": []},
        "resolved_contract": {
            "spatial": {
                "source_kind": canonical_source_kind,
                "source_path": canonical_source_path,
                "vector_runtime_path": vector_runtime_path,
                "spatial_context_mode": spatial_context_mode,
                "effective_extent_bounds_used": effective_extent_bounds_used,
                "memory_budget_mb": memory_budget_mb,
                "large_scene_mode": large_scene_mode,
                "auto_mode_reason": auto_mode_reason,
                "full_scene_materialization_allowed": full_scene_materialization_allowed,
            },
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
        "input_img_path": input_img_path,
        "input_extent_path": input_extent_path,
        "input_boundary_path": input_boundary_path,
        "input_distance_path": input_distance_path,
        "input_valid_path": input_valid_path,
        "spatial_manifest_path": normalized_spatial_manifest_path,
        "patch_contract_mode": patch_runtime_mode if patch_artifacts_materialized else _PATCH_CONTRACT_MODE,
        "patch_runtime_mode": patch_runtime_mode,
        "runtime_compute_enabled": runtime_compute_enabled,
        "patch_size": patch_size,
        "halo_px": None if resolved_config is None else int(resolved_config.patches.halo_px),
        "distance_clip_px": (
            None if resolved_config is None else int(resolved_config.distance.distance_clip_px)
        ),
        "sampling_policy": sampling_policy,
        "patch_layers": patch_layers if patch_layers is not None else [],
        "written_total": written_total,
        "written_center": written_center,
        "written_boundary": written_boundary,
        "written_negative": written_negative,
        "shortfall_negative": shortfall_negative,
        "rejection_stats": rejection_stats,
        "patch_exports": patch_exports if patch_exports is not None else {},
        "patch_runtime_executed": patch_runtime_executed,
        "patch_artifacts_materialized": patch_artifacts_materialized,
        "canonical_source_path": canonical_source_path,
        "canonical_source_kind": canonical_source_kind,
        "vector_runtime_path": vector_runtime_path,
        "spatial_context_mode": spatial_context_mode,
        "effective_extent_bounds_used": effective_extent_bounds_used,
        "memory_budget_mb": memory_budget_mb,
        "large_scene_mode": large_scene_mode,
        "auto_mode_reason": auto_mode_reason,
        "full_scene_materialization_allowed": full_scene_materialization_allowed,
        "vector_load_policy": vector_load_policy,
        "block_aware_reading": block_aware_reading,
        "double_read_avoided": double_read_avoided,
        "candidate_windows_total": candidate_windows_total,
        "patch_metadata_checked": patch_metadata_checked,
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
        "patch_contract_mode": patch_runtime_mode if patch_artifacts_materialized else _PATCH_CONTRACT_MODE,
        "patch_runtime_mode": patch_runtime_mode,
        "patch_size": patch_size,
        "sampling_policy": sampling_policy,
        "written_total": written_total,
        "patch_runtime_executed": patch_runtime_executed,
        "patch_metadata_consistent": checks["patch_metadata_consistent"],
        "source_manifest_path": normalized_source_manifest_path,
        "blocking_issues": blocking_issues,
        "manifest_path": str(manifest_path),
        "error_type": error_type,
    }
    write_summary(summary_path, summary_payload)

    return MakePatchesStageResult(
        status=status,
        manifest_path=manifest_path,
        summary_path=summary_path,
        patch_size=patch_size,
        sampling_policy=sampling_policy,
        patch_layers=None if patch_layers is None else tuple(patch_layers),
        written_total=written_total,
        patch_runtime_executed=patch_runtime_executed,
        blocking_issues=tuple(blocking_issues),
        checks=checks,
        error_type=error_type,
        error_message=error_message,
    )
