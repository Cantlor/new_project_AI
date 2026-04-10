"""Unit tests for Stage D polygonization in module_postprocess_vectorize."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_postprocess_vectorize.input_contract import (
    resolve_postprocess_input_contract,
)
from ai_fields.module_postprocess_vectorize.instance_core import (
    ParcelInstanceRasterResult,
    WatershedCorePolicy,
    build_parcel_instance_raster,
)
from ai_fields.module_postprocess_vectorize.marker_generation import (
    MarkerThresholdPolicy,
    build_marker_candidates,
)
from ai_fields.module_postprocess_vectorize.polygonization import (
    PolygonizationPolicy,
    build_postprocess_polygons,
)

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
pytest.importorskip("scipy")
shape = pytest.importorskip("shapely.geometry").shape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_raster(
    path: Path,
    *,
    array: np.ndarray,
    crs: str = "EPSG:32637",
    transform: Any | None = None,
) -> Path:
    from rasterio.transform import from_origin

    if transform is None:
        transform = from_origin(100.0, 200.0, 1.0, 1.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(array.dtype),
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(array)
    return path


def _write_postprocess_inputs(tmp_path: Path, arrays: dict[str, np.ndarray]) -> Any:
    extent_path = _write_raster(tmp_path / "extent_prob.tif", array=arrays["extent"])
    boundary_path = _write_raster(tmp_path / "boundary_prob.tif", array=arrays["boundary"])
    distance_path = _write_raster(tmp_path / "distance_pred.tif", array=arrays["distance"])
    valid_path = _write_raster(tmp_path / "valid.tif", array=arrays["valid"])

    return resolve_postprocess_input_contract(
        extent_prob_path=extent_path,
        boundary_prob_path=boundary_path,
        distance_pred_path=distance_path,
        valid_path=valid_path,
    )


def _base_arrays(height: int = 5, width: int = 7) -> dict[str, np.ndarray]:
    extent = np.full((1, height, width), 0.9, dtype=np.float32)

    boundary_presence = np.full((height, width), 0.15, dtype=np.float32)
    boundary_presence[:, width // 2] = 0.8
    boundary = np.zeros((3, height, width), dtype=np.float32)
    boundary[1] = boundary_presence * 0.6
    boundary[2] = boundary_presence * 0.4
    boundary[0] = 1.0 - boundary_presence

    distance = np.ones((1, height, width), dtype=np.float32)
    distance[0, :, : width // 2] = 2.5

    valid = np.ones((1, height, width), dtype=np.uint8)

    return {
        "extent": extent,
        "boundary": boundary,
        "distance": distance,
        "valid": valid,
    }


def _manual_instance_result(parcel_instance: np.ndarray, *, ready_for_stage_d: bool = True) -> ParcelInstanceRasterResult:
    positive = sorted(int(v) for v in np.unique(parcel_instance) if int(v) > 0)
    return ParcelInstanceRasterResult(
        parcel_instance=parcel_instance.astype(np.int32),
        shape=tuple(int(v) for v in parcel_instance.shape),
        dtype="int32",
        instance_count=len(positive),
        background_label=0,
        invalid_label=-1,
        valid_pixels=int(parcel_instance.size),
        invalid_pixels=0,
        domain_pixels=int((parcel_instance > 0).sum()),
        labeled_valid_pixels=int((parcel_instance > 0).sum()),
        unlabeled_valid_pixels=int((parcel_instance == 0).sum()),
        policy={"threshold_provenance": "unit_test"},
        diagnostics={},
        ready_for_stage_d=ready_for_stage_d,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_postprocess_polygons_happy_path(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _base_arrays())

    marker_result = build_marker_candidates(
        input_contract=input_contract,
        policy=MarkerThresholdPolicy(
            extent_core_min_prob=0.7,
            boundary_low_max_prob=0.4,
            distance_high_min_value=1.0,
            threshold_provenance="validation_calibrated_baseline_v1",
        ),
    )
    instance_result = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=WatershedCorePolicy(
            extent_support_min_prob=0.5,
            threshold_provenance="validation_calibrated_baseline_v1",
            boundary_weight=2.0,
            extent_weight=1.0,
            distance_weight=0.5,
            boundary_barrier_max_prob=0.95,
        ),
    )

    out_path = tmp_path / "parcels.gpkg"
    result = build_postprocess_polygons(
        input_contract=input_contract,
        instance_result=instance_result,
        output_gpkg_path=out_path,
        policy=PolygonizationPolicy(
            threshold_provenance="validation_calibrated_baseline_v1",
        ),
    )

    assert result.ready_for_stage_e is True
    assert result.parcels_gpkg_path == out_path
    assert out_path.exists()
    assert result.crs == "EPSG:32637"
    assert result.polygon_count >= 1
    assert result.polygon_confidence_present is True

    with fiona.open(out_path, layer=result.layer_name) as src:
        features = list(src)
        assert len(features) == result.polygon_count
        assert "polygon_confidence" in src.schema["properties"]
        assert "instance_id" in src.schema["properties"]
        for feat in features:
            assert int(feat["properties"]["instance_id"]) > 0
            conf = float(feat["properties"]["polygon_confidence"])
            assert 0.0 <= conf <= 1.0


def test_valid_awareness_invalid_zero_vs_background_zero_handled_via_valid_mask(tmp_path: Path) -> None:
    arrays = _base_arrays(height=2, width=2)
    arrays["valid"][0, 0, 0] = 0
    input_contract = _write_postprocess_inputs(tmp_path, arrays)

    parcel_instance = np.array(
        [
            [-1, 1],
            [1, 1],
        ],
        dtype=np.int32,
    )
    instance_result = _manual_instance_result(parcel_instance)

    out_path = tmp_path / "parcels_valid_clip.gpkg"
    build_postprocess_polygons(
        input_contract=input_contract,
        instance_result=instance_result,
        output_gpkg_path=out_path,
        policy=PolygonizationPolicy(threshold_provenance="validation_calibrated_baseline_v1"),
    )

    with fiona.open(out_path, layer="parcels") as src:
        feats = list(src)
        assert len(feats) == 1
        geom = shape(feats[0]["geometry"])
        assert geom.area == pytest.approx(3.0, abs=1e-6)
        assert int(feats[0]["properties"]["instance_id"]) > 0


def test_negative_labels_other_than_invalid_label_fail_explicitly(tmp_path: Path) -> None:
    arrays = _base_arrays(height=2, width=2)
    input_contract = _write_postprocess_inputs(tmp_path, arrays)

    # -2 is not the configured invalid label (-1), so Stage D must fail explicitly.
    parcel_instance = np.array(
        [
            [-2, 1],
            [1, 1],
        ],
        dtype=np.int32,
    )
    instance_result = _manual_instance_result(parcel_instance)

    with pytest.raises(ContractError, match="do not match invalid_label"):
        build_postprocess_polygons(
            input_contract=input_contract,
            instance_result=instance_result,
            output_gpkg_path=tmp_path / "bad_negative_label.gpkg",
            policy=PolygonizationPolicy(threshold_provenance="validation_calibrated_baseline_v1"),
        )


def test_conservative_cleanup_drops_tiny_polygons_only(tmp_path: Path) -> None:
    arrays = _base_arrays(height=3, width=3)
    input_contract = _write_postprocess_inputs(tmp_path, arrays)

    parcel_instance = np.array(
        [
            [1, 0, 2],
            [0, 0, 2],
            [0, 0, 2],
        ],
        dtype=np.int32,
    )
    instance_result = _manual_instance_result(parcel_instance)

    out_path = tmp_path / "parcels_cleanup.gpkg"
    result = build_postprocess_polygons(
        input_contract=input_contract,
        instance_result=instance_result,
        output_gpkg_path=out_path,
        policy=PolygonizationPolicy(
            threshold_provenance="validation_calibrated_baseline_v1",
            min_polygon_area_m2=1.5,
        ),
    )

    assert result.polygon_count == 1
    with fiona.open(out_path, layer=result.layer_name) as src:
        feats = list(src)
        assert len(feats) == 1
        assert int(feats[0]["properties"]["instance_id"]) == 2


def test_polygon_confidence_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _base_arrays())

    marker_result = build_marker_candidates(
        input_contract=input_contract,
        policy=MarkerThresholdPolicy(
            extent_core_min_prob=0.7,
            boundary_low_max_prob=0.4,
            distance_high_min_value=1.0,
            threshold_provenance="validation_calibrated_baseline_v1",
        ),
    )
    instance_result = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=WatershedCorePolicy(
            extent_support_min_prob=0.5,
            threshold_provenance="validation_calibrated_baseline_v1",
            boundary_weight=2.0,
            extent_weight=1.0,
            distance_weight=0.5,
            boundary_barrier_max_prob=0.95,
        ),
    )

    p1 = tmp_path / "run1.gpkg"
    p2 = tmp_path / "run2.gpkg"

    build_postprocess_polygons(
        input_contract=input_contract,
        instance_result=instance_result,
        output_gpkg_path=p1,
        policy=PolygonizationPolicy(threshold_provenance="validation_calibrated_baseline_v1"),
    )
    build_postprocess_polygons(
        input_contract=input_contract,
        instance_result=instance_result,
        output_gpkg_path=p2,
        policy=PolygonizationPolicy(threshold_provenance="validation_calibrated_baseline_v1"),
    )

    def _collect(path: Path) -> dict[int, float]:
        with fiona.open(path, layer="parcels") as src:
            rows = {
                int(feat["properties"]["instance_id"]): float(feat["properties"]["polygon_confidence"])
                for feat in src
            }
        return rows

    assert _collect(p1) == _collect(p2)


def test_parallel_workers_keep_polygon_output_contract(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _base_arrays())

    marker_result = build_marker_candidates(
        input_contract=input_contract,
        policy=MarkerThresholdPolicy(
            extent_core_min_prob=0.7,
            boundary_low_max_prob=0.4,
            distance_high_min_value=1.0,
            threshold_provenance="validation_calibrated_baseline_v1",
        ),
    )
    instance_result = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=WatershedCorePolicy(
            extent_support_min_prob=0.5,
            threshold_provenance="validation_calibrated_baseline_v1",
            boundary_weight=2.0,
            extent_weight=1.0,
            distance_weight=0.5,
            boundary_barrier_max_prob=0.95,
        ),
    )

    p_seq = tmp_path / "seq.gpkg"
    p_par = tmp_path / "par.gpkg"

    r_seq = build_postprocess_polygons(
        input_contract=input_contract,
        instance_result=instance_result,
        output_gpkg_path=p_seq,
        policy=PolygonizationPolicy(
            threshold_provenance="validation_calibrated_baseline_v1",
            num_workers=1,
        ),
    )
    r_par = build_postprocess_polygons(
        input_contract=input_contract,
        instance_result=instance_result,
        output_gpkg_path=p_par,
        policy=PolygonizationPolicy(
            threshold_provenance="validation_calibrated_baseline_v1",
            num_workers=2,
        ),
    )

    assert r_seq.polygon_count == r_par.polygon_count
    assert r_par.policy["num_workers"] == 2
    assert r_par.diagnostics["confidence_summary"]["count"] == r_par.polygon_count
    assert r_par.diagnostics["confidence_details_sample"] == {}

    def _collect_ids(path: Path) -> set[int]:
        with fiona.open(path, layer="parcels") as src:
            return {int(feat["properties"]["instance_id"]) for feat in src}

    assert _collect_ids(p_seq) == _collect_ids(p_par)


def test_confidence_details_sample_can_be_bounded(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _base_arrays())

    marker_result = build_marker_candidates(
        input_contract=input_contract,
        policy=MarkerThresholdPolicy(
            extent_core_min_prob=0.7,
            boundary_low_max_prob=0.4,
            distance_high_min_value=1.0,
            threshold_provenance="validation_calibrated_baseline_v1",
        ),
    )
    instance_result = build_parcel_instance_raster(
        input_contract=input_contract,
        marker_result=marker_result,
        policy=WatershedCorePolicy(
            extent_support_min_prob=0.5,
            threshold_provenance="validation_calibrated_baseline_v1",
            boundary_weight=2.0,
            extent_weight=1.0,
            distance_weight=0.5,
            boundary_barrier_max_prob=0.95,
        ),
    )

    result = build_postprocess_polygons(
        input_contract=input_contract,
        instance_result=instance_result,
        output_gpkg_path=tmp_path / "bounded_details.gpkg",
        policy=PolygonizationPolicy(
            threshold_provenance="validation_calibrated_baseline_v1",
            confidence_details_limit=1,
        ),
    )

    sample = result.diagnostics["confidence_details_sample"]
    assert isinstance(sample, dict)
    assert len(sample) <= 1
    assert result.diagnostics["confidence_summary"]["count"] == result.polygon_count


def test_zero_positive_instances_fail_explicitly(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _base_arrays(height=2, width=2))
    instance_result = _manual_instance_result(np.zeros((2, 2), dtype=np.int32))

    with pytest.raises(ContractError, match="zero positive instance labels"):
        build_postprocess_polygons(
            input_contract=input_contract,
            instance_result=instance_result,
            output_gpkg_path=tmp_path / "bad.gpkg",
            policy=PolygonizationPolicy(threshold_provenance="validation_calibrated_baseline_v1"),
        )


def test_invalid_num_workers_fails_explicitly(tmp_path: Path) -> None:
    input_contract = _write_postprocess_inputs(tmp_path, _base_arrays(height=2, width=2))
    instance_result = _manual_instance_result(np.array([[1, 1], [1, 1]], dtype=np.int32))

    with pytest.raises(ContractError, match="num_workers"):
        build_postprocess_polygons(
            input_contract=input_contract,
            instance_result=instance_result,
            output_gpkg_path=tmp_path / "bad_workers.gpkg",
            policy=PolygonizationPolicy(
                threshold_provenance="validation_calibrated_baseline_v1",
                num_workers=0,
            ),
        )
