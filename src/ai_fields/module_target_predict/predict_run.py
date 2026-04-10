"""Output writers and scene-level predict orchestrator for module_target_predict.

This module implements:
- write_predict_raster_outputs(): writes the 4 mandatory predict rasters;
- run_predict_for_scene(): orchestrates full predict flow for one input raster,
  writes raster outputs, predict_manifest.json, summary.json, config_used.yaml.

It does NOT implement tiled inference logic (see tiled_predict.py).
No thresholding, watershed, or polygonization is performed here.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ai_fields.common.constants import DATA_CONTRACT_VERSION
from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import read_manifest, write_manifest, write_summary
from ai_fields.module_target_predict.checkpoint_contract import (
    CheckpointDrivenPredictContract,
    resolve_checkpoint_predict_contract,
)
from ai_fields.module_target_predict.inference_core import (
    LoadedPredictModel,
    load_predict_model,
)
from ai_fields.module_target_predict.raster_contract import (
    PredictRasterMetadata,
    read_predict_raster_metadata,
    resolve_predict_valid_mask,
)
from ai_fields.module_target_predict.tiled_predict import (
    TiledPredictResult,
    cleanup_tiled_predict_result,
    run_tiled_predict,
)

try:
    import rasterio
    import rasterio.transform
    from rasterio.transform import Affine as _RasterioAffine

    _RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None  # type: ignore[assignment]
    _RasterioAffine = None  # type: ignore[assignment, misc]
    _RASTERIO_AVAILABLE = False


_PREDICT_MODULE_NAME = "module_target_predict"
_PREDICT_MANIFEST_SCHEMA_NAME = "target_predict.predict_manifest"
_PREDICT_SUMMARY_SCHEMA_NAME = "target_predict.summary"
_PREDICT_SCHEMA_VERSION = "v1"


def _require_rasterio() -> Any:
    if not _RASTERIO_AVAILABLE:
        raise ContractError(
            "rasterio is required for module_target_predict output writers. "
            "Install the 'geo' optional dependencies to use this layer."
        )
    return rasterio


def write_predict_raster_outputs(
    *,
    output_dir: Path,
    result: TiledPredictResult,
    metadata: PredictRasterMetadata,
) -> dict[str, Path]:
    """Write the 4 mandatory predict raster outputs to output_dir.

    Writes:
        - ``extent_prob.tif``   (1 band, float32)
        - ``boundary_prob.tif`` (3 bands, float32)
        - ``distance_pred.tif`` (1 band, float32)
        - ``valid.tif``         (1 band, uint8)

    Returns a mapping from canonical output name to the written Path.
    """
    _rio = _require_rasterio()
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write_band_windowed(
        ds: Any,
        *,
        array2d: np.ndarray,
        band_index: int,
        dtype: str,
        window_rows: int = 512,
    ) -> None:
        height, width = array2d.shape
        for row_off in range(0, height, window_rows):
            row_end = min(height, row_off + window_rows)
            window = _rio.windows.Window(0, row_off, width, row_end - row_off)
            chunk = np.asarray(
                array2d[row_off:row_end, :],
                dtype=np.float32 if dtype == "float32" else np.uint8,
            )
            ds.write(chunk, band_index, window=window)

    transform = _RasterioAffine.from_gdal(*metadata.transform_gdal)

    base_kwargs: dict[str, Any] = {
        "driver": "GTiff",
        "crs": metadata.crs,
        "transform": transform,
        "height": result.scene_height,
        "width": result.scene_width,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    written: dict[str, Path] = {}

    # 1. extent_prob.tif — single band float32
    extent_path = output_dir / "extent_prob.tif"
    with _rio.open(extent_path, "w", count=1, dtype="float32", **base_kwargs) as ds:
        _write_band_windowed(
            ds,
            array2d=result.extent_prob,
            band_index=1,
            dtype="float32",
        )
    written["extent_prob"] = extent_path

    # 2. boundary_prob.tif — 3 bands float32
    boundary_path = output_dir / "boundary_prob.tif"
    with _rio.open(boundary_path, "w", count=3, dtype="float32", **base_kwargs) as ds:
        for band_idx in range(3):
            _write_band_windowed(
                ds,
                array2d=result.boundary_prob[band_idx],
                band_index=band_idx + 1,
                dtype="float32",
            )
    written["boundary_prob"] = boundary_path

    # 3. distance_pred.tif — single band float32
    distance_path = output_dir / "distance_pred.tif"
    with _rio.open(distance_path, "w", count=1, dtype="float32", **base_kwargs) as ds:
        _write_band_windowed(
            ds,
            array2d=result.distance_pred,
            band_index=1,
            dtype="float32",
        )
    written["distance_pred"] = distance_path

    # 4. valid.tif — single band uint8
    valid_path = output_dir / "valid.tif"
    with _rio.open(valid_path, "w", count=1, dtype="uint8", **base_kwargs) as ds:
        _write_band_windowed(
            ds,
            array2d=result.valid_mask,
            band_index=1,
            dtype="uint8",
        )
    written["valid"] = valid_path

    return written


def _write_config_used(output_dir: Path, config: dict[str, Any]) -> Path:
    """Write config_used.yaml to output_dir."""
    config_path = output_dir / "config_used.yaml"
    config_path.write_text(
        yaml.dump(config, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


def _resolve_predict_aoi_policy(
    *,
    aoi_path: "str | Path | None",
    aoi_manifest_path: "str | Path | None",
) -> dict[str, Any]:
    """Build aoi_policy dict for predict_manifest.

    When no AOI is provided, returns the baseline (aoi_present=False).
    When aoi_manifest_path is provided (upstream prep_data aoi_manifest.json),
    reads buffer_m from it and classifies source as upstream_prep_data_resolved.
    When only aoi_path is provided, records it as user_provided.
    Predict does not currently clip outputs to AOI; AOI is recorded for
    downstream provenance only.
    """
    if aoi_path is None and aoi_manifest_path is None:
        return {
            "aoi_present": False,
            "buffer_m": None,
            "output_extent_mode": "full_raster",
        }

    buffer_m: float | None = None
    aoi_source_type = "user_provided"
    resolved_aoi_path_str: str | None = str(aoi_path) if aoi_path is not None else None

    if aoi_manifest_path is not None:
        # Explicit caller-provided path: failure is an explicit ContractError.
        try:
            m = read_manifest(Path(aoi_manifest_path))
        except Exception as exc:
            raise ContractError(
                f"Failed to read aoi_manifest_path '{aoi_manifest_path}': {exc}"
            ) from exc
        expected_schema = "prep_data.aoi_manifest"
        if m.get("schema_name") != expected_schema:
            raise ContractError(
                f"aoi_manifest_path does not point to a {expected_schema!r} artifact "
                f"(got schema_name={m.get('schema_name')!r}): {aoi_manifest_path}"
            )
        buffer_m_raw = m.get("buffer_m")
        buffer_m = float(buffer_m_raw) if buffer_m_raw is not None else None
        aoi_source_type = "upstream_prep_data_resolved"
        if resolved_aoi_path_str is None:
            upstream_aoi = m.get("aoi_output_path")
            if isinstance(upstream_aoi, str) and upstream_aoi.strip():
                resolved_aoi_path_str = upstream_aoi

    return {
        "aoi_present": True,
        "aoi_path": resolved_aoi_path_str,
        "aoi_manifest_path": str(aoi_manifest_path) if aoi_manifest_path is not None else None,
        "aoi_source_type": aoi_source_type,
        "buffer_m": buffer_m,
        "output_extent_mode": "full_raster",
        "note": (
            "predict does not clip outputs to AOI; "
            "AOI recorded for downstream provenance only"
        ),
    }


def _resolve_train_run_id(train_manifest_path: Path | None) -> str | None:
    """Resolve train run_id from optional train_manifest.json for provenance."""
    if train_manifest_path is None:
        return None
    manifest = read_manifest(train_manifest_path)
    run_id_raw = manifest.get("run_id")
    if run_id_raw is None:
        return None
    if not isinstance(run_id_raw, str) or run_id_raw.strip() == "":
        raise ContractError(
            "train_manifest.run_id must be a non-empty string when provided."
        )
    return run_id_raw.strip()


def run_predict_for_scene(
    *,
    raster_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_metadata_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    tile_size: int = 512,
    overlap: float = 0.25,
    normalization_stats: Mapping[str, Any] | None = None,
    normalization_stats_path: str | Path | None = None,
    explicit_valid_mask_override: np.ndarray | None = None,
    run_id: str | None = None,
    train_manifest_path: str | Path | None = None,
    config_used_path: str | Path | None = None,
    progress_enabled: bool | None = None,
    aoi_path: str | Path | None = None,
    aoi_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Orchestrate full predict for a single scene.

    Steps:
      1. Resolve checkpoint contract.
      2. Read raster metadata.
      3. Resolve valid mask.
      4. Load model.
      5. Run tiled inference with Gaussian blending.
      6. Write raster outputs.
      7. Write config_used.yaml.
      8. Write predict_manifest.json.
      9. Write summary.json.

    Returns a summary dict with all artifact paths and key stats.
    """
    raster_path = Path(raster_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    created_at_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Resolve checkpoint contract.
    checkpoint_contract = resolve_checkpoint_predict_contract(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        train_manifest_path=train_manifest_path,
        config_used_path=config_used_path,
    )
    train_run_id = _resolve_train_run_id(checkpoint_contract.train_manifest_path)

    # 2. Read raster metadata.
    metadata = read_predict_raster_metadata(raster_path)

    # 3. Resolve valid mask.
    valid_mask_resolution = resolve_predict_valid_mask(
        raster_path=raster_path,
        explicit_valid_mask_override=explicit_valid_mask_override,
    )

    # 4. Load model.
    loaded_model = load_predict_model(
        checkpoint_contract=checkpoint_contract,
        device=device,
    )

    # 5. Run tiled inference.
    result = run_tiled_predict(
        raster_path=raster_path,
        checkpoint_contract=checkpoint_contract,
        loaded_model=loaded_model,
        valid_mask=valid_mask_resolution.valid_mask,
        tile_size=tile_size,
        overlap=overlap,
        normalization_stats=normalization_stats,
        normalization_stats_path=normalization_stats_path,
        progress_enabled=progress_enabled,
    )

    # 6. Write raster outputs.
    try:
        output_paths = write_predict_raster_outputs(
            output_dir=output_dir_path,
            result=result,
            metadata=metadata,
        )
    finally:
        cleanup_tiled_predict_result(result)

    # 7. Write config_used.yaml.
    config_dict: dict[str, Any] = {
        "raster_path": str(raster_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata_path": str(checkpoint_metadata_path),
        "output_dir": str(output_dir_path),
        "device": device,
        "tile_size": tile_size,
        "overlap": overlap,
        "train_manifest_path": (
            str(train_manifest_path) if train_manifest_path is not None else None
        ),
        "config_used_path": (
            str(config_used_path) if config_used_path is not None else None
        ),
        "aoi_path": str(aoi_path) if aoi_path is not None else None,
        "aoi_manifest_path": str(aoi_manifest_path) if aoi_manifest_path is not None else None,
    }
    config_path = _write_config_used(output_dir_path, config_dict)

    # 8. Write predict_manifest.json.
    manifest_payload: dict[str, Any] = {
        "schema_name": _PREDICT_MANIFEST_SCHEMA_NAME,
        "schema_version": _PREDICT_SCHEMA_VERSION,
        "module_name": _PREDICT_MODULE_NAME,
        "module_version": None,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "run_id": run_id,
        "stage_name": "predict_scene",
        "created_at_utc": created_at_utc,
        "status": "success",
        "checkpoint": {
            "checkpoint_path": str(checkpoint_contract.checkpoint_path),
            "checkpoint_metadata_path": str(checkpoint_contract.checkpoint_metadata_path),
            "train_run_id": train_run_id,
            "train_manifest_path": (
                str(checkpoint_contract.train_manifest_path)
                if checkpoint_contract.train_manifest_path is not None
                else None
            ),
        },
        "input_raster": {
            "path": str(metadata.raster_path),
            "crs": metadata.crs,
            "width": metadata.width,
            "height": metadata.height,
            "count": metadata.band_count,
            "dtype": metadata.dtype,
            "nodata": metadata.nodata,
        },
        "resolved_contract": {
            "features": {
                "dataset_feature_mode": checkpoint_contract.feature_mode,
                "assembled_model_input": checkpoint_contract.assembled_model_input,
                "feature_channel_count": checkpoint_contract.feature_channel_count,
                "final_input_channel_count": checkpoint_contract.in_channels,
                "channel_semantics": list(checkpoint_contract.channel_semantics),
                "valid_as_input_channel": checkpoint_contract.valid_as_input_channel,
            },
            "normalization": {
                "normalization_name": checkpoint_contract.normalization.get(
                    "normalization_name"
                ),
                "stats_source": checkpoint_contract.normalization.get("stats_source"),
                "clip_percentiles": checkpoint_contract.normalization.get("clip_percentiles"),
                "scaling_range": checkpoint_contract.normalization.get("scaling_range"),
            },
            "valid_policy": {
                "valid_source": valid_mask_resolution.source,
                "invalid_handling": "suppress_and_skip_invalid_only_tiles",
            },
            "aoi_policy": _resolve_predict_aoi_policy(
                aoi_path=aoi_path,
                aoi_manifest_path=aoi_manifest_path,
            ),
        },
        "tiling": {
            "tile_size": tile_size,
            "overlap_fraction": overlap,
            "blending": result.blending,
            "invalid_only_tiles_skipped": result.tiles_skipped_invalid,
            "processed_tiles": result.tiles_processed,
            "total_tiles": result.tiles_total,
        },
        "outputs_expected": list(checkpoint_contract.required_outputs),
        "output_paths": {k: str(v) for k, v in output_paths.items()},
        "runtime": {
            "device_requested": device,
            "device_resolved": loaded_model.device,
            "amp_requested": None,
            "amp_used": None,
            "oom_fallbacks_applied": [],
            "notes": [],
        },
        "valid_coverage": {
            "valid_pixels": valid_mask_resolution.valid_pixels,
            "invalid_pixels": valid_mask_resolution.invalid_pixels,
            "valid_ratio": valid_mask_resolution.valid_ratio,
        },
    }

    manifest_path = output_dir_path / "predict_manifest.json"
    write_manifest(manifest_path, manifest_payload)

    # 9. Write summary.json.
    summary_payload: dict[str, Any] = {
        "schema_name": _PREDICT_SUMMARY_SCHEMA_NAME,
        "status": "success",
        "input_raster_path": str(raster_path),
        "feature_mode": checkpoint_contract.feature_mode,
        "assembled_model_input": checkpoint_contract.assembled_model_input,
        "output_paths": {k: str(v) for k, v in output_paths.items()},
        "tiles_total": result.tiles_total,
        "tiles_processed": result.tiles_processed,
        "tiles_skipped_invalid": result.tiles_skipped_invalid,
        "valid_ratio": valid_mask_resolution.valid_ratio,
        "run_id": run_id,
        "warnings": [],
        "key_notes": [],
    }
    summary_path = output_dir_path / "summary.json"
    write_summary(summary_path, summary_payload)

    return {
        "run_id": run_id,
        "output_dir": str(output_dir_path),
        "output_paths": {k: str(v) for k, v in output_paths.items()},
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "config_path": str(config_path),
        "tiles_total": result.tiles_total,
        "tiles_processed": result.tiles_processed,
        "tiles_skipped_invalid": result.tiles_skipped_invalid,
        "valid_ratio": valid_mask_resolution.valid_ratio,
        "feature_mode": checkpoint_contract.feature_mode,
        "assembled_model_input": checkpoint_contract.assembled_model_input,
        "device_resolved": loaded_model.device,
    }


__all__ = [
    "write_predict_raster_outputs",
    "run_predict_for_scene",
]
