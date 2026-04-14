"""Thin orchestration script for module_prep_data stage chain (01..07).

This module intentionally adds only a minimal runner over existing stage entry
functions. It does not alter stage logic and does not introduce a new framework.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import write_manifest, write_summary
from ai_fields.module_prep_data import config as prep_data_config
from ai_fields.module_prep_data.check_inputs import run_check_inputs_stage
from ai_fields.module_prep_data.make_patches import run_make_patches_stage
from ai_fields.module_prep_data.prepare_features import run_prepare_features_stage
from ai_fields.module_prep_data.prepare_spatial_context import run_prepare_spatial_context_stage
from ai_fields.module_prep_data.prepare_targets import run_prepare_targets_stage
from ai_fields.module_prep_data.split_dataset import run_split_dataset_stage
from ai_fields.module_prep_data.validate_outputs import run_validate_outputs_stage


@dataclass(frozen=True)
class _StageSpec:
    name: str
    manifest_filename: str


STAGES: tuple[_StageSpec, ...] = (
    _StageSpec("01_check_inputs", "check_inputs_manifest.json"),
    _StageSpec("02_prepare_spatial_context", "aoi_manifest.json"),
    _StageSpec("03_prepare_features", "features_manifest.json"),
    _StageSpec("04_prepare_targets", "targets_manifest.json"),
    _StageSpec("05_make_patches", "patches_manifest.json"),
    _StageSpec("06_split_dataset", "split_manifest.json"),
    _StageSpec("07_validate_outputs", "validate_outputs_manifest.json"),
)
STAGE_NAME_TO_INDEX = {stage.name: idx for idx, stage in enumerate(STAGES)}
_METADATA_SIDECAR_SUFFIX = ".meta.json"
_MULTI_SIZE_MANIFEST_NAME = "multi_size_manifest.json"
_MULTI_SIZE_SUMMARY_NAME = "summary.json"
_SUPPORTED_PATCH_SIZES = (256, 384, 512)


def _default_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"prep-data-{ts}"


def _now_utc_iso() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return ts.replace("+00:00", "Z")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"Failed to read {label} '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} '{path}' must contain a top-level JSON object.")
    return payload


def _load_sidecar_if_exists(input_path: str | None) -> dict[str, Any] | None:
    if input_path is None:
        return None
    sidecar = Path(input_path).with_name(f"{Path(input_path).name}{_METADATA_SIDECAR_SUFFIX}")
    if not sidecar.exists():
        return None
    return _load_json_object(sidecar, label="metadata sidecar")


def _extract_stage01_metadata(stage01_manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    manifest = _load_json_object(stage01_manifest_path, label="stage-01 manifest")

    raster_block = manifest.get("input_raster")
    vector_block = manifest.get("input_vectors")
    aoi_block = manifest.get("input_aoi")

    if not isinstance(raster_block, dict):
        raise ContractError("stage-01 manifest is missing input_raster object.")
    if not isinstance(vector_block, dict):
        raise ContractError("stage-01 manifest is missing input_vectors object.")

    raster_md: dict[str, Any] = {
        "crs": raster_block.get("crs"),
        "band_count": raster_block.get("count"),
        "width": raster_block.get("width"),
        "height": raster_block.get("height"),
        "dtype": raster_block.get("dtype"),
        "nodata": raster_block.get("nodata"),
    }
    vector_md: dict[str, Any] = {
        "crs": vector_block.get("crs"),
        "feature_count": vector_block.get("feature_count"),
        "geometry_types": vector_block.get("geometry_types"),
    }

    aoi_md: dict[str, Any] | None = None
    if isinstance(aoi_block, dict) and aoi_block.get("path") is not None:
        aoi_md = {
            "crs": aoi_block.get("crs"),
            "feature_count": aoi_block.get("feature_count"),
            "geometry_types": aoi_block.get("geometry_types"),
            "bounds": aoi_block.get("bounds"),
        }

    return raster_md, vector_md, aoi_md


def _previous_stage_manifest_for_start(run_dir: Path, start_idx: int) -> Path | None:
    if start_idx == 0:
        return None
    prev = STAGES[start_idx - 1]
    path = run_dir / prev.name / prev.manifest_filename
    if not path.exists():
        raise ContractError(
            f"start_from_stage={STAGES[start_idx].name!r} requires previous manifest at '{path}'."
        )
    return path


def _stage_dir(run_dir: Path, stage_name: str) -> Path:
    return run_dir / stage_name


def _print_stage_start(stage_name: str, stage_output_dir: Path, source_manifest: Path | None) -> None:
    source_text = "<none>" if source_manifest is None else str(source_manifest)
    print(f"[RUN] {stage_name}  output_dir={stage_output_dir}")
    print(f"      source_manifest={source_text}")


def _print_stage_result(stage_name: str, result: Any) -> None:
    print(f"[DONE] {stage_name}  status={result.status}")
    print(f"       manifest={result.manifest_path}")
    print(f"       summary={result.summary_path}")


def _prepare_run_directory(run_dir: Path, *, start_idx: int, overwrite: bool) -> None:
    if start_idx == 0:
        if run_dir.exists() and overwrite:
            shutil.rmtree(run_dir)
        elif run_dir.exists() and not overwrite:
            raise ContractError(
                f"Run directory already exists: '{run_dir}'. Use --overwrite or a different --run-id."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        return

    if overwrite:
        raise ContractError("--overwrite is only supported when starting from 01_check_inputs.")
    if not run_dir.exists():
        raise ContractError(
            f"Run directory '{run_dir}' does not exist for start_from_stage={STAGES[start_idx].name!r}."
        )


def _parse_patch_sizes(value: str) -> list[int]:
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if not tokens:
        raise ContractError(
            "--patch-sizes must contain at least one size, e.g. '256,384,512'."
        )
    parsed: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        try:
            size = int(token)
        except ValueError as exc:
            raise ContractError(f"Invalid patch size '{token}' in --patch-sizes.") from exc
        if size not in _SUPPORTED_PATCH_SIZES:
            raise ContractError(
                f"Unsupported patch size '{size}' in --patch-sizes. "
                f"Supported values: {list(_SUPPORTED_PATCH_SIZES)}."
            )
        if size not in seen:
            seen.add(size)
            parsed.append(size)
    return parsed


def _materialize_dataset_export(
    *,
    source_dataset_dir: Path,
    export_dataset_root: Path,
    overwrite: bool,
) -> None:
    if not source_dataset_dir.exists():
        raise ContractError(
            f"Dataset source directory for export does not exist: '{source_dataset_dir}'."
        )
    if export_dataset_root.exists():
        if not overwrite:
            raise ContractError(
                f"Export dataset root already exists: '{export_dataset_root}'. "
                "Use --overwrite to allow replacing it in multi-size mode."
            )
        shutil.rmtree(export_dataset_root)
    export_dataset_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dataset_dir, export_dataset_root)


def _build_temp_config_with_patch_size(
    *,
    base_config_raw: dict[str, Any],
    patch_size: int,
    work_dir: Path,
) -> Path:
    config_raw = copy.deepcopy(base_config_raw)
    patches_section = config_raw.get("patches")
    if patches_section is None:
        patches_section = {}
        config_raw["patches"] = patches_section
    if not isinstance(patches_section, dict):
        raise ContractError(
            f"Config key 'patches' must be a mapping/object, got {type(patches_section).__name__}."
        )
    patches_section["patch_size"] = patch_size

    cfg_path = work_dir / f"config.patch_size_{patch_size}.yaml"
    cfg_path.write_text(yaml.safe_dump(config_raw, sort_keys=False), encoding="utf-8")
    return cfg_path


def _run_pipeline(args: argparse.Namespace) -> int:
    start_idx = STAGE_NAME_TO_INDEX[args.start_from_stage]
    stop_idx = STAGE_NAME_TO_INDEX[args.stop_after_stage]
    if start_idx > stop_idx:
        raise ContractError(
            f"start_from_stage ({args.start_from_stage}) must be <= stop_after_stage ({args.stop_after_stage})."
        )

    run_id = args.run_id or _default_run_id()
    output_root = Path(args.output_dir)
    run_dir = output_root / run_id

    _prepare_run_directory(run_dir, start_idx=start_idx, overwrite=args.overwrite)
    current_source_manifest = _previous_stage_manifest_for_start(run_dir, start_idx)

    raster_path = str(Path(args.raster))
    vector_path = str(Path(args.vector))
    aoi_path = None if args.aoi is None else str(Path(args.aoi))

    # Seed from stage-01 manifest (resume mode) if available.
    raster_metadata: dict[str, Any] | None = None
    vector_metadata: dict[str, Any] | None = None
    aoi_metadata: dict[str, Any] | None = None

    stage01_manifest_existing = run_dir / STAGES[0].name / STAGES[0].manifest_filename
    if stage01_manifest_existing.exists():
        raster_metadata, vector_metadata, aoi_metadata = _extract_stage01_metadata(stage01_manifest_existing)

    # Sidecars have priority for richer metadata fields (e.g. AOI geometry types).
    raster_sidecar = _load_sidecar_if_exists(raster_path)
    vector_sidecar = _load_sidecar_if_exists(vector_path)
    aoi_sidecar = _load_sidecar_if_exists(aoi_path)
    if raster_sidecar is not None:
        raster_metadata = raster_sidecar
    if vector_sidecar is not None:
        vector_metadata = vector_sidecar
    if aoi_sidecar is not None:
        aoi_metadata = aoi_sidecar

    # Optional runtime artifact paths that can be produced by runtime-enabled stages.
    img_output_path: str | None = None
    valid_output_path: str | None = args.valid_path
    extent_output_path: str | None = None
    boundary_output_path: str | None = None
    distance_output_path: str | None = None

    final_manifest: Path | None = current_source_manifest
    final_summary: Path | None = None
    stage02_manifest_path: Path | None = None
    stage02_manifest_existing = run_dir / "02_prepare_spatial_context" / "aoi_manifest.json"
    if stage02_manifest_existing.exists():
        stage02_manifest_path = stage02_manifest_existing
    feature_target_materialization_mode = (
        "full_scene" if args.diagnostic_full_scene_materialization else "compute_spec_only"
    )

    for idx in range(start_idx, stop_idx + 1):
        stage = STAGES[idx]
        stage_output_dir = _stage_dir(run_dir, stage.name)
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        _print_stage_start(stage.name, stage_output_dir, current_source_manifest)

        if stage.name == "01_check_inputs":
            result = run_check_inputs_stage(
                output_dir=stage_output_dir,
                run_id=run_id,
                config_path=args.config,
                raster_path=raster_path,
                vector_path=vector_path,
                aoi_path=aoi_path,
                raster_metadata=raster_metadata,
                vector_metadata=vector_metadata,
                aoi_metadata=aoi_metadata,
                runtime_probe_enabled=args.runtime_probe_enabled,
                metadata_sidecar_fallback_enabled=args.metadata_sidecar_fallback_enabled,
            )
            if result.success:
                raster_metadata, vector_metadata, aoi_metadata = _extract_stage01_metadata(
                    result.manifest_path
                )
                # Re-apply sidecar priority because stage-01 manifest is intentionally compact.
                if raster_sidecar is not None:
                    raster_metadata = raster_sidecar
                if vector_sidecar is not None:
                    vector_metadata = vector_sidecar
                if aoi_sidecar is not None:
                    aoi_metadata = aoi_sidecar

        elif stage.name == "02_prepare_spatial_context":
            if raster_metadata is None or vector_metadata is None:
                raise ContractError(
                    "Stage 02 requires raster/vector metadata. Run stage 01 first or provide sidecars."
                )
            if aoi_path is not None and aoi_metadata is None:
                raise ContractError(
                    "Stage 02 requires AOI metadata when --aoi is provided. "
                    "Provide '<aoi>.meta.json' or run stage 01 with explicit aoi metadata."
                )
            result = run_prepare_spatial_context_stage(
                output_dir=stage_output_dir,
                run_id=run_id,
                config_path=args.config,
                raster_path=raster_path,
                vector_path=vector_path,
                raster_metadata=raster_metadata,
                vector_metadata=vector_metadata,
                aoi_path=aoi_path,
                aoi_metadata=aoi_metadata,
                source_manifest_path=current_source_manifest,
                runtime_compute_enabled=args.runtime_compute_enabled,
                memory_budget_mb=args.memory_budget_mb,
            )

        elif stage.name == "03_prepare_features":
            result = run_prepare_features_stage(
                output_dir=stage_output_dir,
                run_id=run_id,
                config_path=args.config,
                raster_path=raster_path,
                valid_path=valid_output_path,
                source_manifest_path=current_source_manifest,
                runtime_compute_enabled=args.runtime_compute_enabled,
                materialization_mode=feature_target_materialization_mode,
            )
            if result.success:
                m03 = _load_json_object(result.manifest_path, label="stage-03 manifest")
                img_output_path = m03.get("img_output_path") if isinstance(m03.get("img_output_path"), str) else None
                valid_from_s03 = m03.get("valid_output_path")
                if isinstance(valid_from_s03, str) and valid_from_s03.strip() != "":
                    valid_output_path = valid_from_s03

        elif stage.name == "04_prepare_targets":
            result = run_prepare_targets_stage(
                output_dir=stage_output_dir,
                run_id=run_id,
                config_path=args.config,
                raster_path=raster_path,
                vector_path=vector_path,
                valid_path=valid_output_path,
                source_manifest_path=current_source_manifest,
                runtime_compute_enabled=args.runtime_compute_enabled,
                materialization_mode=feature_target_materialization_mode,
            )
            if result.success:
                m04 = _load_json_object(result.manifest_path, label="stage-04 manifest")
                extent_output_path = (
                    m04.get("extent_output_path") if isinstance(m04.get("extent_output_path"), str) else None
                )
                boundary_output_path = (
                    m04.get("boundary_output_path") if isinstance(m04.get("boundary_output_path"), str) else None
                )
                distance_output_path = (
                    m04.get("distance_output_path") if isinstance(m04.get("distance_output_path"), str) else None
                )
                valid_from_s04 = m04.get("valid_output_path")
                if isinstance(valid_from_s04, str) and valid_from_s04.strip() != "":
                    valid_output_path = valid_from_s04

        elif stage.name == "05_make_patches":
            result = run_make_patches_stage(
                output_dir=stage_output_dir,
                run_id=run_id,
                config_path=args.config,
                raster_path=raster_path,
                vector_path=vector_path,
                img_path=img_output_path,
                extent_path=extent_output_path,
                boundary_path=boundary_output_path,
                distance_path=distance_output_path,
                valid_path=valid_output_path,
                source_manifest_path=current_source_manifest,
                spatial_manifest_path=stage02_manifest_path,
                runtime_compute_enabled=args.runtime_compute_enabled,
            )

        elif stage.name == "06_split_dataset":
            patches_dir_for_stage06 = args.patches_dir
            if patches_dir_for_stage06 is None:
                patches_dir_for_stage06 = str(run_dir / "05_make_patches" / "patches")
            result = run_split_dataset_stage(
                output_dir=stage_output_dir,
                run_id=run_id,
                config_path=args.config,
                raster_path=raster_path,
                vector_path=vector_path,
                source_manifest_path=current_source_manifest,
                runtime_compute_enabled=args.runtime_compute_enabled,
                patches_dir=patches_dir_for_stage06,
            )

        elif stage.name == "07_validate_outputs":
            dataset_dir_for_stage07 = args.dataset_dir
            if dataset_dir_for_stage07 is None:
                dataset_dir_for_stage07 = str(run_dir / "06_split_dataset" / "dataset")
            result = run_validate_outputs_stage(
                output_dir=stage_output_dir,
                run_id=run_id,
                config_path=args.config,
                raster_path=raster_path,
                vector_path=vector_path,
                source_manifest_path=current_source_manifest,
                runtime_compute_enabled=args.runtime_compute_enabled,
                dataset_dir=dataset_dir_for_stage07,
            )

        else:  # pragma: no cover
            raise ContractError(f"Unsupported stage: {stage.name}")

        _print_stage_result(stage.name, result)

        final_manifest = result.manifest_path
        final_summary = result.summary_path
        if stage.name == "02_prepare_spatial_context" and result.success:
            stage02_manifest_path = result.manifest_path

        if not result.success:
            print(f"[FAIL] Pipeline stopped at {stage.name}.")
            return 1

        current_source_manifest = result.manifest_path

    print("[OK] module_prep_data pipeline finished.")
    print(f"     run_dir={run_dir}")
    if final_manifest is not None:
        print(f"     final_manifest={final_manifest}")
    if final_summary is not None:
        print(f"     final_summary={final_summary}")
    return 0


def _run_pipeline_multi_size(args: argparse.Namespace) -> int:
    patch_sizes = _parse_patch_sizes(args.patch_sizes)
    if args.start_from_stage != STAGES[0].name or args.stop_after_stage != STAGES[-1].name:
        raise ContractError(
            "Multi-size mode currently supports only a full stage chain run "
            "(01_check_inputs -> 07_validate_outputs)."
        )
    if not args.runtime_compute_enabled:
        raise ContractError(
            "Multi-size mode requires --runtime-compute-enabled "
            "to produce train-ready datasets."
        )

    base_run_id = args.run_id or _default_run_id()
    output_root = Path(args.output_dir)

    base_raw_config = prep_data_config.load_yaml(Path(args.config))
    base_cfg = prep_data_config.build_config(base_raw_config)
    feature_mode = base_cfg.feature_mode

    export_root_base = (
        Path(args.multi_size_export_root)
        if args.multi_size_export_root is not None
        else output_root / "prep_data_for_train"
    )

    print("[RUN] module_prep_data multi-size mode")
    print(f"      patch_sizes={patch_sizes}")
    print(f"      feature_mode={feature_mode}")
    print(f"      export_root_base={export_root_base}")

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="prep_data_multi_size_cfg_") as tmp_cfg_dir_str:
        tmp_cfg_dir = Path(tmp_cfg_dir_str)

        for patch_size in patch_sizes:
            cfg_path = _build_temp_config_with_patch_size(
                base_config_raw=base_raw_config,
                patch_size=patch_size,
                work_dir=tmp_cfg_dir,
            )
            per_size_run_id = f"{base_run_id}-ps{patch_size}"

            size_args = argparse.Namespace(**vars(args))
            size_args.config = str(cfg_path)
            size_args.run_id = per_size_run_id
            # Always isolate stage06/07 runtime paths per run.
            size_args.patches_dir = None
            size_args.dataset_dir = None

            print(f"[RUN] patch_size={patch_size}  run_id={per_size_run_id}")
            exit_code = _run_pipeline(size_args)
            if exit_code != 0:
                raise ContractError(
                    f"Multi-size run failed for patch_size={patch_size} with exit_code={exit_code}."
                )

            run_dir = output_root / per_size_run_id
            stage05_manifest = run_dir / "05_make_patches" / "patches_manifest.json"
            stage06_manifest = run_dir / "06_split_dataset" / "split_manifest.json"
            stage07_manifest = run_dir / "07_validate_outputs" / "validate_outputs_manifest.json"
            dataset_source_dir = run_dir / "06_split_dataset" / "dataset"
            export_dataset_root = export_root_base / feature_mode / str(patch_size)

            _materialize_dataset_export(
                source_dataset_dir=dataset_source_dir,
                export_dataset_root=export_dataset_root,
                overwrite=args.overwrite,
            )

            m05 = _load_json_object(stage05_manifest, label="stage-05 manifest")
            m06 = _load_json_object(stage06_manifest, label="stage-06 manifest")
            m07 = _load_json_object(stage07_manifest, label="stage-07 manifest")

            records.append(
                {
                    "patch_size": patch_size,
                    "run_id": per_size_run_id,
                    "run_dir": str(run_dir),
                    "dataset_source_dir": str(dataset_source_dir),
                    "export_dataset_root": str(export_dataset_root),
                    "stage05_manifest_path": str(stage05_manifest),
                    "stage06_manifest_path": str(stage06_manifest),
                    "stage07_manifest_path": str(stage07_manifest),
                    "written_total": m05.get("written_total"),
                    "rejection_stats": m05.get("rejection_stats"),
                    "split_counts": m06.get("splits"),
                    "validation_runtime_executed": m07.get("validation_runtime_executed"),
                }
            )

    multi_dir = output_root / f"{base_run_id}__multi_size"
    multi_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = multi_dir / _MULTI_SIZE_MANIFEST_NAME
    summary_path = multi_dir / _MULTI_SIZE_SUMMARY_NAME

    write_manifest(
        manifest_path,
        {
            "schema_name": "prep_data.multi_size_manifest",
            "schema_version": "v1",
            "module_name": "module_prep_data",
            "module_version": None,
            "data_contract_version": DATA_CONTRACT_VERSION,
            "run_id": base_run_id,
            "stage_name": "multi_size_export",
            "created_at_utc": _now_utc_iso(),
            "status": "success",
            "mode": "multi_size",
            "feature_mode": feature_mode,
            "patch_sizes": patch_sizes,
            "export_root_base": str(export_root_base),
            "runs": records,
        },
    )
    write_summary(
        summary_path,
        {
            "schema_name": "prep_data.summary",
            "stage_name": "multi_size_export",
            "run_id": base_run_id,
            "status": "success",
            "mode": "multi_size",
            "feature_mode": feature_mode,
            "patch_sizes": patch_sizes,
            "export_root_base": str(export_root_base),
            "generated_dataset_roots": [r["export_dataset_root"] for r in records],
            "manifest_path": str(manifest_path),
        },
    )

    print("[OK] module_prep_data multi-size mode finished.")
    print(f"     multi_size_manifest={manifest_path}")
    print(f"     multi_size_summary={summary_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run module_prep_data stages 01..07 in fixed order as a thin orchestration layer."
        )
    )
    parser.add_argument("--config", required=True, help="Path to module_prep_data YAML config.")
    parser.add_argument("--raster", required=True, help="Input raster path.")
    parser.add_argument("--vector", required=True, help="Input vector labels path.")
    parser.add_argument("--aoi", default=None, help="Optional AOI vector path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root directory; run artifacts are stored under <output-dir>/<run-id>/.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id (default: UTC timestamp-based).")
    parser.add_argument(
        "--start-from-stage",
        choices=[stage.name for stage in STAGES],
        default=STAGES[0].name,
        help="Optional stage to start from (resume-style).",
    )
    parser.add_argument(
        "--stop-after-stage",
        choices=[stage.name for stage in STAGES],
        default=STAGES[-1].name,
        help="Optional stage to stop after.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing run directory before execution (only allowed from stage 01).",
    )
    parser.add_argument(
        "--valid-path",
        default=None,
        help="Optional precomputed valid raster path used by downstream stages when available.",
    )
    parser.add_argument(
        "--patches-dir",
        default=None,
        help="Optional patches directory for stage 06 runtime split compute.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Optional dataset directory for stage 07 runtime output validation.",
    )
    parser.add_argument(
        "--runtime-probe-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable real input probing in stage 01 (default: true).",
    )
    parser.add_argument(
        "--metadata-sidecar-fallback-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow '<input>.meta.json' fallback in stage 01 (default: true).",
    )
    parser.add_argument(
        "--runtime-compute-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable runtime compute branches in stages 02..07 (default: true). "
            "When false, stages stay metadata/contract-driven."
        ),
    )
    parser.add_argument(
        "--diagnostic-full-scene-materialization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic-only flag. When true, stages 03/04 materialize full-scene "
            "img/target rasters. Default false keeps baseline compute_spec_only."
        ),
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=int,
        default=None,
        help=(
            "Optional stage-02 memory budget override in MB. "
            "By default stage-02 auto-resolves a conservative budget."
        ),
    )
    parser.add_argument(
        "--patch-sizes",
        default=None,
        help=(
            "Optional comma-separated patch-size list (e.g. '256,384,512'). "
            "When set, enables multi-size mode and runs a full 01..07 pipeline "
            "for each size in isolated outputs."
        ),
    )
    parser.add_argument(
        "--multi-size-export-root",
        default=None,
        help=(
            "Optional dataset export root for multi-size mode. "
            "Default: <output-dir>/prep_data_for_train."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.patch_sizes is not None:
            return _run_pipeline_multi_size(args)
        return _run_pipeline(args)
    except ContractError as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
