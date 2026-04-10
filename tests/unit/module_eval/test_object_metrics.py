"""Unit tests for module_eval Stage D minimal object/structure metrics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_eval.input_contract import resolve_evaluation_input_contract
from ai_fields.module_eval.object_metrics import (
    ObjectMatchingPolicy,
    build_object_structure_metrics_summary,
    compute_object_structure_metrics,
)

rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")


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


def _write_polygons(
    path: Path,
    *,
    polygons: list[list[tuple[float, float]]],
    include_confidence: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    properties: dict[str, str] = {"instance_id": "int"}
    if include_confidence:
        properties["polygon_confidence"] = "float"

    schema = {
        "geometry": "Polygon",
        "properties": properties,
    }
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs="EPSG:32637") as dst:
        for idx, ring in enumerate(polygons, start=1):
            props: dict[str, Any] = {"instance_id": idx}
            if include_confidence:
                props["polygon_confidence"] = 0.8
            dst.write(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": props,
                }
            )
    return path


def _rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> list[tuple[float, float]]:
    return [
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
        (x0, y0),
    ]


def _build_eval_arrays(*, gt_valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    h, w = 8, 8

    gt_extent = np.zeros((1, h, w), dtype=np.uint8)
    gt_extent[0, 1:7, 1:7] = 1
    gt_extent[0, 0, 0] = 255

    gt_boundary = np.zeros((1, h, w), dtype=np.uint8)
    gt_boundary[0, 1:7, 4] = 1
    gt_boundary[0, 1:7, 3] = 2
    gt_boundary[0, 1:7, 5] = 2

    if gt_valid is None:
        gt_valid = np.ones((1, h, w), dtype=np.uint8)
    pred_valid = gt_valid.copy()

    pred_extent_prob = np.full((1, h, w), 0.2, dtype=np.float32)
    pred_extent_prob[0, 1:7, 1:7] = 0.85

    pred_boundary_prob = np.zeros((3, h, w), dtype=np.float32)
    pred_boundary_prob[0] = 0.8
    pred_boundary_prob[1] = 0.1
    pred_boundary_prob[2] = 0.1
    pred_boundary_prob[0, 1:7, 4] = 0.15
    pred_boundary_prob[1, 1:7, 4] = 0.65
    pred_boundary_prob[2, 1:7, 4] = 0.20

    pred_distance_pred = np.full((1, h, w), 1.0, dtype=np.float32)

    return {
        "gt_extent": gt_extent,
        "gt_boundary": gt_boundary,
        "gt_valid": gt_valid,
        "pred_extent_prob": pred_extent_prob,
        "pred_boundary_prob": pred_boundary_prob,
        "pred_distance_pred": pred_distance_pred,
        "pred_valid": pred_valid,
    }


def _resolve_contract(
    tmp_path: Path,
    *,
    gt_polygons: list[list[tuple[float, float]]],
    pred_polygons: list[list[tuple[float, float]]],
    gt_valid: np.ndarray | None = None,
):
    arrays = _build_eval_arrays(gt_valid=gt_valid)
    paths = {
        "gt_extent_path": _write_raster(tmp_path / "gt_extent.tif", array=arrays["gt_extent"]),
        "gt_boundary_path": _write_raster(tmp_path / "gt_boundary.tif", array=arrays["gt_boundary"]),
        "gt_valid_path": _write_raster(tmp_path / "gt_valid.tif", array=arrays["gt_valid"]),
        "pred_extent_prob_path": _write_raster(
            tmp_path / "extent_prob.tif",
            array=arrays["pred_extent_prob"],
        ),
        "pred_boundary_prob_path": _write_raster(
            tmp_path / "boundary_prob.tif",
            array=arrays["pred_boundary_prob"],
        ),
        "pred_distance_pred_path": _write_raster(
            tmp_path / "distance_pred.tif",
            array=arrays["pred_distance_pred"],
        ),
        "pred_valid_path": _write_raster(tmp_path / "valid.tif", array=arrays["pred_valid"]),
        "gt_parcels_path": _write_polygons(
            tmp_path / "gt_parcels.gpkg",
            polygons=gt_polygons,
            include_confidence=False,
        ),
        "post_parcels_gpkg_path": _write_polygons(
            tmp_path / "parcels.gpkg",
            polygons=pred_polygons,
            include_confidence=True,
        ),
    }
    return resolve_evaluation_input_contract(**paths)


def _default_policy() -> ObjectMatchingPolicy:
    return ObjectMatchingPolicy(
        threshold_provenance="eval_object_matching_v1",
        min_iou_threshold=0.2,
        min_overlap_gt_threshold=0.2,
        min_overlap_pred_threshold=0.2,
        match_rule="iou_or_overlap",
    )


def test_compute_object_structure_metrics_happy_path(tmp_path: Path) -> None:
    gt_polys = [
        _rect(101.0, 199.0, 104.0, 196.0),
        _rect(104.0, 199.0, 107.0, 196.0),
    ]
    pred_polys = list(gt_polys)
    contract = _resolve_contract(
        tmp_path,
        gt_polygons=gt_polys,
        pred_polygons=pred_polys,
    )

    result = compute_object_structure_metrics(
        input_contract=contract,
        policy=_default_policy(),
    )

    assert result.goc == pytest.approx(0.0)
    assert result.guc == pytest.approx(0.0)
    assert result.gtc == pytest.approx(0.0)
    assert result.gt_object_count == 2
    assert result.pred_object_count == 2
    assert result.matched_gt_count == 2
    assert result.matched_pred_count == 2

    summary = build_object_structure_metrics_summary(result)
    assert summary["metrics"]["gtc"] == pytest.approx(0.0)


def test_compute_object_structure_metrics_merge_case(tmp_path: Path) -> None:
    gt_polys = [
        _rect(101.0, 199.0, 104.0, 196.0),
        _rect(104.0, 199.0, 107.0, 196.0),
    ]
    pred_polys = [
        _rect(101.0, 199.0, 107.0, 196.0),
    ]
    contract = _resolve_contract(
        tmp_path,
        gt_polygons=gt_polys,
        pred_polygons=pred_polys,
    )

    result = compute_object_structure_metrics(
        input_contract=contract,
        policy=_default_policy(),
    )
    assert result.split_gt_count == 0
    assert result.merged_gt_count == 2
    assert result.goc == pytest.approx(0.0)
    assert result.guc == pytest.approx(1.0)
    assert result.gtc == pytest.approx(1.0)


def test_compute_object_structure_metrics_policy_semantics_are_explicit(
    tmp_path: Path,
) -> None:
    gt_polys = [_rect(101.0, 199.0, 107.0, 196.0)]
    pred_polys = [
        _rect(101.0, 199.0, 104.0, 196.0),
        _rect(104.0, 199.0, 107.0, 196.0),
    ]
    contract = _resolve_contract(
        tmp_path,
        gt_polygons=gt_polys,
        pred_polygons=pred_polys,
    )

    overlap_policy = ObjectMatchingPolicy(
        threshold_provenance="policy_overlap",
        min_iou_threshold=0.95,
        min_overlap_gt_threshold=0.4,
        min_overlap_pred_threshold=0.4,
        match_rule="overlap_only",
    )
    iou_policy = ObjectMatchingPolicy(
        threshold_provenance="policy_iou",
        min_iou_threshold=0.6,
        min_overlap_gt_threshold=0.95,
        min_overlap_pred_threshold=0.95,
        match_rule="iou_only",
    )

    res_overlap = compute_object_structure_metrics(
        input_contract=contract,
        policy=overlap_policy,
    )
    res_iou = compute_object_structure_metrics(
        input_contract=contract,
        policy=iou_policy,
    )

    assert res_overlap.goc > res_iou.goc
    assert res_overlap.match_rule == "overlap_only"
    assert res_iou.match_rule == "iou_only"


def test_compute_object_structure_metrics_not_ready_failure(tmp_path: Path) -> None:
    gt_polys = [_rect(101.0, 199.0, 107.0, 196.0)]
    pred_polys = [_rect(101.0, 199.0, 107.0, 196.0)]
    contract = _resolve_contract(
        tmp_path,
        gt_polygons=gt_polys,
        pred_polygons=pred_polys,
    )
    not_ready = replace(
        contract,
        track_readiness=replace(
            contract.track_readiness,
            object_structure_ready=False,
            object_structure_reason="synthetic_object_not_ready",
        ),
    )

    with pytest.raises(ContractError, match="object track is not ready"):
        compute_object_structure_metrics(
            input_contract=not_ready,
            policy=_default_policy(),
        )


def test_compute_object_structure_metrics_valid_aware(tmp_path: Path) -> None:
    gt_polys = [_rect(101.0, 199.0, 107.0, 196.0)]
    pred_polys = [
        _rect(101.0, 199.0, 104.0, 196.0),
        _rect(104.0, 199.0, 107.0, 196.0),
    ]

    valid_all = np.ones((1, 8, 8), dtype=np.uint8)
    valid_half = np.ones((1, 8, 8), dtype=np.uint8)
    valid_half[0, :, 4:] = 0

    contract_all = _resolve_contract(
        tmp_path / "all_valid",
        gt_polygons=gt_polys,
        pred_polygons=pred_polys,
        gt_valid=valid_all,
    )
    contract_half = _resolve_contract(
        tmp_path / "half_valid",
        gt_polygons=gt_polys,
        pred_polygons=pred_polys,
        gt_valid=valid_half,
    )

    m_all = compute_object_structure_metrics(
        input_contract=contract_all,
        policy=_default_policy(),
    )
    m_half = compute_object_structure_metrics(
        input_contract=contract_half,
        policy=_default_policy(),
    )

    assert m_all.goc > m_half.goc
    assert m_half.pred_excluded_zero_valid_count >= 1


def test_compute_object_structure_metrics_invalid_policy_failure(tmp_path: Path) -> None:
    gt_polys = [_rect(101.0, 199.0, 107.0, 196.0)]
    pred_polys = [_rect(101.0, 199.0, 107.0, 196.0)]
    contract = _resolve_contract(
        tmp_path,
        gt_polygons=gt_polys,
        pred_polygons=pred_polys,
    )

    bad_policy = ObjectMatchingPolicy(
        threshold_provenance="x",
        min_iou_threshold=0.2,
        min_overlap_gt_threshold=0.2,
        min_overlap_pred_threshold=0.2,
        match_rule="unsupported_rule",
    )
    with pytest.raises(ContractError, match="Unsupported policy.match_rule"):
        compute_object_structure_metrics(
            input_contract=contract,
            policy=bad_policy,
        )


def test_compute_object_structure_metrics_many_objects_smoke(tmp_path: Path) -> None:
    from rasterio.transform import from_origin

    h, w = 256, 256
    transform = from_origin(100.0, 200.0, 1.0, 1.0)

    gt_extent = np.ones((1, h, w), dtype=np.uint8)
    gt_extent[0, 0, 0] = 255
    gt_boundary = np.zeros((1, h, w), dtype=np.uint8)
    gt_valid = np.ones((1, h, w), dtype=np.uint8)
    pred_extent_prob = np.full((1, h, w), 0.9, dtype=np.float32)
    pred_boundary_prob = np.zeros((3, h, w), dtype=np.float32)
    pred_boundary_prob[0] = 1.0
    pred_distance_pred = np.zeros((1, h, w), dtype=np.float32)
    pred_valid = gt_valid.copy()

    gt_extent_path = _write_raster(
        tmp_path / "gt_extent.tif",
        array=gt_extent,
        transform=transform,
    )
    gt_boundary_path = _write_raster(
        tmp_path / "gt_boundary.tif",
        array=gt_boundary,
        transform=transform,
    )
    gt_valid_path = _write_raster(
        tmp_path / "gt_valid.tif",
        array=gt_valid,
        transform=transform,
    )
    pred_extent_path = _write_raster(
        tmp_path / "extent_prob.tif",
        array=pred_extent_prob,
        transform=transform,
    )
    pred_boundary_path = _write_raster(
        tmp_path / "boundary_prob.tif",
        array=pred_boundary_prob,
        transform=transform,
    )
    pred_distance_path = _write_raster(
        tmp_path / "distance_pred.tif",
        array=pred_distance_pred,
        transform=transform,
    )
    pred_valid_path = _write_raster(
        tmp_path / "valid.tif",
        array=pred_valid,
        transform=transform,
    )

    polygons: list[list[tuple[float, float]]] = []
    for rr in range(10):
        for cc in range(10):
            x0 = 105.0 + cc * 20.0
            x1 = x0 + 8.0
            y0 = 195.0 - rr * 20.0
            y1 = y0 - 8.0
            polygons.append(_rect(x0, y0, x1, y1))

    gt_parcels_path = _write_polygons(
        tmp_path / "gt_parcels.gpkg",
        polygons=polygons,
        include_confidence=False,
    )
    pred_parcels_path = _write_polygons(
        tmp_path / "parcels.gpkg",
        polygons=polygons,
        include_confidence=True,
    )

    contract = resolve_evaluation_input_contract(
        gt_extent_path=gt_extent_path,
        gt_boundary_path=gt_boundary_path,
        gt_valid_path=gt_valid_path,
        pred_extent_prob_path=pred_extent_path,
        pred_boundary_prob_path=pred_boundary_path,
        pred_distance_pred_path=pred_distance_path,
        pred_valid_path=pred_valid_path,
        gt_parcels_path=gt_parcels_path,
        post_parcels_gpkg_path=pred_parcels_path,
    )

    result = compute_object_structure_metrics(
        input_contract=contract,
        policy=_default_policy(),
    )
    assert result.gt_object_count == 100
    assert result.pred_object_count == 100
    assert result.gtc == pytest.approx(0.0)
