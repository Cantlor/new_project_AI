"""Run-level orchestration for module_postprocess_vectorize (single-scene path).

This layer stitches Stage A -> B -> C -> D -> E for one scene:
  - input contract resolution;
  - marker generation;
  - marker-controlled watershed / parcel_instance;
  - polygonization + conservative cleanup + polygon_confidence;
  - artifact export (manifest / summary / config_used).

Scope is intentionally narrow:
  - no module_eval runtime;
  - no batch manager;
  - no retry/resume framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

import dataclasses

from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import read_manifest
from ai_fields.module_postprocess_vectorize.boundary_repair import (
    BoundaryRepairPolicy,
    apply_boundary_repair,
)
from ai_fields.module_postprocess_vectorize.export import (
    PostprocessExportArtifacts,
    export_postprocess_artifacts,
)
from ai_fields.module_postprocess_vectorize.input_contract import (
    PostprocessInputContractResult,
    PostprocessRasterMetadata,
    resolve_postprocess_input_contract,
)
from ai_fields.module_postprocess_vectorize.instance_core import (
    ParcelInstanceRasterResult,
    WatershedCorePolicy,
    build_parcel_instance_raster,
)
from ai_fields.module_postprocess_vectorize.marker_generation import (
    MarkerGenerationResult,
    MarkerThresholdPolicy,
    build_marker_candidates,
)
from ai_fields.module_postprocess_vectorize.polygonization import (
    PolygonizationPolicy,
    PostprocessPolygonizationResult,
    build_postprocess_polygons,
)


@dataclass(frozen=True)
class PostprocessRunPolicies:
    """Policy contract passed into run-level postprocess orchestration."""

    marker_policy: MarkerThresholdPolicy
    watershed_policy: WatershedCorePolicy
    polygonization_policy: PolygonizationPolicy
    aoi_suppression_enabled: bool = False
    boundary_repair_policy: BoundaryRepairPolicy | None = None


@dataclass(frozen=True)
class PostprocessRunResult:
    """Result contract for run_postprocess_for_scene."""

    run_id: str
    run_dir: Path

    parcel_instance_path: Path
    parcels_gpkg_path: Path

    postprocess_manifest_path: Path
    summary_path: Path
    config_used_path: Path

    instance_count: int
    polygon_count: int
    success: bool


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _normalize_pathlike(path: Any, *, name: str) -> Path:
    if not isinstance(path, (str, PathLike)):
        raise ContractError(f"{name} must be path-like (str/Path).")
    resolved = Path(path)
    if str(resolved).strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    return resolved


def _normalize_optional_pathlike(path: Any, *, name: str) -> Path | None:
    if path is None:
        return None
    return _normalize_pathlike(path, name=name)


def _require_non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{name} must be a non-empty string.")
    return value


def _apply_aoi_suppression(
    *,
    valid_path: Path,
    aoi_path: Path,
    output_path: Path,
) -> Path:
    """Rasterize AOI vector onto valid grid and write effective_valid.tif (valid & aoi_mask).

    Returns output_path.  Raises ContractError on spatial issues or missing deps.
    """
    try:
        import rasterio
        import rasterio.features
        import rasterio.transform as rio_transform
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for AOI suppression.") from exc
    try:
        import geopandas as gpd
        from shapely.geometry import mapping as shape_mapping
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "geopandas/shapely is required for AOI vector rasterization in AOI suppression."
        ) from exc

    with rasterio.open(valid_path) as src:
        valid_arr = src.read(1).astype(bool)
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        height, width = src.height, src.width

    try:
        aoi_gdf = gpd.read_file(aoi_path)
    except Exception as exc:
        raise ContractError(f"AOI vector could not be read: {aoi_path}: {exc}") from exc

    if aoi_gdf.empty:
        raise ContractError(f"AOI vector file is empty (no features): {aoi_path}")

    # Reproject AOI to match valid raster CRS
    if crs is not None and aoi_gdf.crs is not None and aoi_gdf.crs != crs:
        try:
            aoi_gdf = aoi_gdf.to_crs(crs)
        except Exception as exc:
            raise ContractError(
                f"AOI CRS reprojection failed ({aoi_gdf.crs} -> {crs}): {exc}"
            ) from exc

    shapes = [(shape_mapping(geom), 1) for geom in aoi_gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        raise ContractError("AOI vector contains no valid geometries after reprojection.")

    aoi_mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    effective_valid = (valid_arr & aoi_mask).astype("uint8")

    profile.update(dtype="uint8", count=1, nodata=None, compress="deflate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(effective_valid, 1)

    return output_path


def _apply_boundary_repair_to_raster(
    *,
    boundary_prob_path: Path,
    repair_policy: BoundaryRepairPolicy,
    output_path: Path,
    input_contract: PostprocessInputContractResult,
) -> PostprocessRasterMetadata:
    """Apply morphological boundary repair and write repaired_boundary_prob.tif.

    Returns updated PostprocessRasterMetadata pointing to the repaired file.
    """
    try:
        import rasterio
        import rasterio.transform as rio_transform
    except ImportError as exc:  # pragma: no cover
        raise ContractError("rasterio is required for boundary repair raster step.") from exc
    import numpy as np

    with rasterio.open(boundary_prob_path) as src:
        boundary_arr = src.read().astype(np.float32)  # shape (3, H, W)
        profile = src.profile.copy()

    if boundary_arr.ndim != 3 or boundary_arr.shape[0] != 3:
        raise ContractError(
            f"boundary_prob must be a 3-band raster for repair, got shape {boundary_arr.shape}."
        )

    # Presence = skeleton (ch1) + buffer (ch2)
    presence = boundary_arr[1] + boundary_arr[2]
    presence_binary = (presence > 0.0).astype(np.uint8)

    repair_result = apply_boundary_repair(presence_binary, policy=repair_policy)

    if repair_result.repair_applied:
        repaired_presence = repair_result.repaired_boundary_mask.astype(np.float32)
        # Rebuild simplex: background = 1 - presence, skeleton = presence, buffer = 0
        repaired_arr = np.stack([
            np.clip(1.0 - repaired_presence, 0.0, 1.0),
            repaired_presence,
            np.zeros_like(repaired_presence),
        ], axis=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(repaired_arr)
    else:
        # No repair applied — just copy the path reference
        output_path = boundary_prob_path

    return dataclasses.replace(
        input_contract.boundary_prob,
        path=output_path,
    )


def _write_parcel_instance_raster(
    *,
    output_path: Path,
    input_contract: PostprocessInputContractResult,
    instance_result: ParcelInstanceRasterResult,
) -> Path:
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required to write parcel_instance.tif in postprocess run orchestration."
        ) from exc

    if instance_result.parcel_instance.shape != (
        int(input_contract.common_height),
        int(input_contract.common_width),
    ):
        raise ContractError("parcel_instance shape mismatch against Stage A input contract.")

    transform = Affine.from_gdal(*input_contract.common_transform_gdal)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=int(input_contract.common_width),
        height=int(input_contract.common_height),
        count=1,
        dtype="int32",
        crs=input_contract.common_crs,
        transform=transform,
        nodata=int(instance_result.invalid_label),
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as ds:
        ds.write(instance_result.parcel_instance.astype("int32"), 1)

    return output_path


def _infer_source_predict_run_id(
    *,
    source_predict_manifest_path: Path | None,
    explicit_source_predict_run_id: str | None,
) -> str | None:
    if explicit_source_predict_run_id is not None:
        return _require_non_empty_string(
            explicit_source_predict_run_id,
            name="source_predict_run_id",
        )
    if source_predict_manifest_path is None:
        return None
    manifest = read_manifest(source_predict_manifest_path)
    run_id = manifest.get("run_id")
    return _require_non_empty_string(run_id, name="source_predict_manifest.run_id")


def run_postprocess_for_scene(
    *,
    extent_prob_path: str | Path,
    boundary_prob_path: str | Path,
    distance_pred_path: str | Path,
    valid_path: str | Path,
    output_dir: str | Path,
    policies: PostprocessRunPolicies,
    run_id: str | None = None,
    aoi_path: str | Path | None = None,
    aoi_manifest_path: str | Path | None = None,
    source_predict_manifest_path: str | Path | None = None,
    source_predict_run_id: str | None = None,
    layer_name: str = "parcels",
    progress_enabled: bool | None = None,
) -> PostprocessRunResult:
    """Execute Stage A-E postprocess flow for one scene with fail-fast behavior."""
    if not isinstance(policies, PostprocessRunPolicies):
        raise ContractError("policies must be PostprocessRunPolicies.")
    if not isinstance(layer_name, str) or layer_name.strip() == "":
        raise ContractError("layer_name must be a non-empty string.")

    resolved_run_id = _default_run_id() if run_id is None else _require_non_empty_string(
        run_id, name="run_id"
    )
    base_output_dir = _normalize_pathlike(output_dir, name="output_dir")
    run_dir = base_output_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_source_manifest = _normalize_optional_pathlike(
        source_predict_manifest_path,
        name="source_predict_manifest_path",
    )
    resolved_source_run_id = _infer_source_predict_run_id(
        source_predict_manifest_path=resolved_source_manifest,
        explicit_source_predict_run_id=source_predict_run_id,
    )

    resolved_aoi_manifest_path = _normalize_optional_pathlike(
        aoi_manifest_path, name="aoi_manifest_path"
    )

    # Stage A.0: AOI suppression (if enabled and AOI path provided)
    effective_valid_path: str | Path = valid_path
    resolved_aoi_path = _normalize_optional_pathlike(aoi_path, name="aoi_path")
    if policies.aoi_suppression_enabled and resolved_aoi_path is not None:
        effective_valid_tif = run_dir / "effective_valid.tif"
        try:
            _apply_aoi_suppression(
                valid_path=Path(valid_path),
                aoi_path=resolved_aoi_path,
                output_path=effective_valid_tif,
            )
            effective_valid_path = effective_valid_tif
        except ContractError as exc:
            raise ContractError(f"Stage A.0 (AOI suppression) failed: {exc}") from exc

    # Stage A: input contract
    try:
        input_contract = resolve_postprocess_input_contract(
            extent_prob_path=extent_prob_path,
            boundary_prob_path=boundary_prob_path,
            distance_pred_path=distance_pred_path,
            valid_path=effective_valid_path,
            aoi_path=aoi_path,
            predict_manifest_path=resolved_source_manifest,
        )
    except ContractError as exc:
        raise ContractError(f"Stage A (input contract) failed: {exc}") from exc

    # Stage B: marker generation
    try:
        marker_result: MarkerGenerationResult = build_marker_candidates(
            input_contract=input_contract,
            policy=policies.marker_policy,
        )
    except ContractError as exc:
        raise ContractError(f"Stage B (marker generation) failed: {exc}") from exc

    # Stage B.5: hybrid boundary repair (optional)
    repair_applied = False
    if policies.boundary_repair_policy is not None and policies.boundary_repair_policy.enabled:
        repaired_boundary_tif = run_dir / "repaired_boundary_prob.tif"
        try:
            repaired_boundary_meta = _apply_boundary_repair_to_raster(
                boundary_prob_path=Path(input_contract.boundary_prob.path),
                repair_policy=policies.boundary_repair_policy,
                output_path=repaired_boundary_tif,
                input_contract=input_contract,
            )
            if repaired_boundary_meta.path != input_contract.boundary_prob.path:
                input_contract = dataclasses.replace(
                    input_contract, boundary_prob=repaired_boundary_meta
                )
                repair_applied = True
        except ContractError as exc:
            raise ContractError(f"Stage B.5 (boundary repair) failed: {exc}") from exc

    # Stage C: instance raster core
    try:
        instance_result: ParcelInstanceRasterResult = build_parcel_instance_raster(
            input_contract=input_contract,
            marker_result=marker_result,
            policy=policies.watershed_policy,
        )
    except ContractError as exc:
        raise ContractError(f"Stage C (instance core) failed: {exc}") from exc

    parcel_instance_path = run_dir / "parcel_instance.tif"
    try:
        _write_parcel_instance_raster(
            output_path=parcel_instance_path,
            input_contract=input_contract,
            instance_result=instance_result,
        )
    except ContractError as exc:
        raise ContractError(f"Stage C writer (parcel_instance.tif) failed: {exc}") from exc

    # Stage D: polygonization + cleanup + confidence
    parcels_gpkg_path = run_dir / "parcels.gpkg"
    try:
        polygon_result: PostprocessPolygonizationResult = build_postprocess_polygons(
            input_contract=input_contract,
            instance_result=instance_result,
            output_gpkg_path=parcels_gpkg_path,
            policy=policies.polygonization_policy,
            layer_name=layer_name,
            progress_enabled=progress_enabled,
        )
    except ContractError as exc:
        raise ContractError(f"Stage D (polygonization) failed: {exc}") from exc

    # Stage E: manifest / summary / config_used
    try:
        artifacts: PostprocessExportArtifacts = export_postprocess_artifacts(
            output_dir=run_dir,
            run_id=resolved_run_id,
            input_contract=input_contract,
            marker_result=marker_result,
            instance_result=instance_result,
            polygon_result=polygon_result,
            marker_policy=policies.marker_policy,
            watershed_policy=policies.watershed_policy,
            polygonization_policy=policies.polygonization_policy,
            parcel_instance_path=parcel_instance_path,
            source_predict_run_id=resolved_source_run_id,
            source_predict_manifest_path=resolved_source_manifest,
            aoi_manifest_path=resolved_aoi_manifest_path,
            aoi_suppression_applied=(
                policies.aoi_suppression_enabled and resolved_aoi_path is not None
            ),
            boundary_repair_applied=repair_applied,
            extra_effective_config={
                "run_level": {
                    "layer_name": layer_name,
                    "run_dir": str(run_dir),
                    "boundary_repair_applied": repair_applied,
                    "boundary_repair_policy": (
                        dataclasses.asdict(policies.boundary_repair_policy)
                        if policies.boundary_repair_policy is not None
                        else None
                    ),
                }
            },
        )
    except ContractError as exc:
        raise ContractError(f"Stage E (artifact export) failed: {exc}") from exc

    return PostprocessRunResult(
        run_id=resolved_run_id,
        run_dir=run_dir,
        parcel_instance_path=parcel_instance_path,
        parcels_gpkg_path=parcels_gpkg_path,
        postprocess_manifest_path=artifacts.postprocess_manifest_path,
        summary_path=artifacts.summary_path,
        config_used_path=artifacts.config_used_path,
        instance_count=int(instance_result.instance_count),
        polygon_count=int(polygon_result.polygon_count),
        success=True,
    )


__all__ = [
    "PostprocessRunPolicies",
    "PostprocessRunResult",
    "run_postprocess_for_scene",
]

