"""Predict-time feature adapter and minimal normalization apply.

This module is the next narrow contract-first layer for module_target_predict:
- build dataset-side feature stack (raw8 / raw8_idx3) from source raster;
- apply train-derived normalization using exported policy + per-band stats;
- assemble final model input with valid as the last channel.

It intentionally does NOT implement tiled inference, model forward, blending,
or output writing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.constants import CHANNEL_COUNTS
from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    NormalizationContractError,
)
from ai_fields.module_target_predict.checkpoint_contract import (
    CheckpointDrivenPredictContract,
)
from ai_fields.module_target_predict.raster_contract import (
    PredictRasterContractResult,
    resolve_predict_raster_contract,
)


_EPS = 1e-6
_CANONICAL_SEMANTICS_BY_MODE: dict[str, tuple[str, ...]] = {
    "raw8": (
        "coastal",
        "blue",
        "green",
        "yellow",
        "red",
        "rededge",
        "nir1",
        "nir2",
    ),
    "raw8_idx3": (
        "coastal",
        "blue",
        "green",
        "yellow",
        "red",
        "rededge",
        "nir1",
        "nir2",
        "NDVI",
        "SAVI",
        "NDWI",
    ),
}


@dataclass(frozen=True)
class PredictInputAdapterResult:
    """Prepared predict-time model input contract for the next inference step."""

    feature_mode: str
    assembled_model_input: str
    channel_semantics: tuple[str, ...]

    valid_mask_source: str

    raw_feature_shape: tuple[int, int, int]
    normalized_feature_shape: tuple[int, int, int]
    assembled_input_shape: tuple[int, int, int]

    raw_feature_stack: np.ndarray
    normalized_feature_stack: np.ndarray
    assembled_input: np.ndarray
    valid_mask: np.ndarray

    normalization_summary: dict[str, Any]
    input_ready_for_model: bool


@dataclass(frozen=True)
class _PerBandStat:
    band_idx: int
    p_lo: float
    p_hi: float


def _normalize_existing_path(path: Any, *, name: str) -> Path:
    if isinstance(path, (str, PathLike)):
        normalized = Path(path)
    else:
        raise ContractError(f"{name} must be path-like, got {type(path).__name__}.")
    if str(normalized).strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    if not normalized.exists():
        raise ContractError(f"{name} does not exist: {normalized}")
    if not normalized.is_file():
        raise ContractError(f"{name} must point to a regular file: {normalized}")
    return normalized


def _require_rasterio() -> tuple[Any, Any]:
    try:
        import rasterio
        import rasterio.errors
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "rasterio is required for module_target_predict feature adapter layer."
        ) from exc
    return rasterio, rasterio.errors


def _validate_semantics_against_feature_mode(contract: CheckpointDrivenPredictContract) -> None:
    mode = contract.feature_mode
    canonical = _CANONICAL_SEMANTICS_BY_MODE.get(mode)
    if canonical is None:
        raise ContractError(f"Unsupported feature_mode in checkpoint contract: {mode!r}.")

    expected = (*canonical, "valid")
    if tuple(contract.channel_semantics) != expected:
        raise ContractError(
            "checkpoint channel_semantics does not match canonical assembled input order for "
            f"feature_mode={mode!r}. Expected {list(expected)}, got {list(contract.channel_semantics)}."
        )


def _read_source_raw8(raster_path: str | Path) -> np.ndarray:
    resolved_path = _normalize_existing_path(raster_path, name="raster_path")
    rasterio, rasterio_errors = _require_rasterio()
    try:
        with rasterio.open(resolved_path) as ds:
            if ds.count < 8:
                raise ChannelCountError(
                    f"Input raster must provide at least 8 source bands, got {ds.count}."
                )
            raw = ds.read(list(range(1, 9))).astype(np.float32)
    except rasterio_errors.RasterioIOError as exc:
        raise ContractError(f"Failed to open raster_path with rasterio: {resolved_path} ({exc})") from exc
    except OSError as exc:
        raise ContractError(f"Failed to read raster bands from {resolved_path}: {exc}") from exc

    if raw.ndim != 3 or raw.shape[0] != 8:
        raise ContractError(
            f"Source raw8 stack must have shape (8, H, W), got {raw.shape}."
        )
    return raw


def _normalize_numeric_pair(value: Any, *, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise NormalizationContractError(f"{name} must be a 2-item numeric sequence.")
    try:
        first = float(value[0])
        second = float(value[1])
    except (TypeError, ValueError) as exc:
        raise NormalizationContractError(
            f"{name} must contain numeric values, got {value!r}."
        ) from exc
    if not np.isfinite(first) or not np.isfinite(second):
        raise NormalizationContractError(f"{name} values must be finite numbers.")
    return first, second


def build_predict_feature_stack(
    *,
    raster_path: str | Path,
    checkpoint_contract: CheckpointDrivenPredictContract,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build dataset-side feature stack for predict (`raw8` or `raw8_idx3`)."""

    _validate_semantics_against_feature_mode(checkpoint_contract)

    raw8 = _read_source_raw8(raster_path)

    if valid_mask.ndim != 2:
        raise ContractError(f"valid_mask must be 2-D (H, W), got shape={valid_mask.shape}.")
    if raw8.shape[1:] != valid_mask.shape:
        raise ContractError(
            "valid_mask shape must match raster spatial shape: "
            f"{valid_mask.shape} != {raw8.shape[1:]}"
        )

    valid_bool = valid_mask.astype(bool)

    if checkpoint_contract.feature_mode == "raw8":
        feature_stack = raw8.copy()
        channel_semantics = _CANONICAL_SEMANTICS_BY_MODE["raw8"]
    elif checkpoint_contract.feature_mode == "raw8_idx3":
        red = raw8[4]
        nir1 = raw8[6]
        green = raw8[2]

        # Compute derived features only on valid pixels, keeping invalid pixels
        # explicitly non-informative in the assembled contract.
        ndvi = np.zeros_like(red, dtype=np.float32)
        savi = np.zeros_like(red, dtype=np.float32)
        ndwi = np.zeros_like(red, dtype=np.float32)
        if np.any(valid_bool):
            red_valid = red[valid_bool]
            nir1_valid = nir1[valid_bool]
            green_valid = green[valid_bool]
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi_valid = (nir1_valid - red_valid) / (nir1_valid + red_valid + _EPS)
                savi_valid = (
                    ((nir1_valid - red_valid) / (nir1_valid + red_valid + 0.5 + _EPS))
                    * 1.5
                )
                ndwi_valid = (green_valid - nir1_valid) / (green_valid + nir1_valid + _EPS)

            ndvi[valid_bool] = np.nan_to_num(
                ndvi_valid, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            savi[valid_bool] = np.nan_to_num(
                savi_valid, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            ndwi[valid_bool] = np.nan_to_num(
                ndwi_valid, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)

        feature_stack = np.concatenate([raw8, np.stack([ndvi, savi, ndwi], axis=0)], axis=0)
        channel_semantics = _CANONICAL_SEMANTICS_BY_MODE["raw8_idx3"]
    else:
        raise ContractError(
            f"Unsupported checkpoint feature_mode: {checkpoint_contract.feature_mode!r}."
        )

    # Keep invalid area explicit and non-informative before normalization.
    feature_stack[:, ~valid_bool] = 0.0

    expected_channels = CHANNEL_COUNTS[checkpoint_contract.feature_mode]
    if feature_stack.shape[0] != expected_channels:
        raise ChannelCountError(
            "Built feature stack channel count is inconsistent with checkpoint feature_mode: "
            f"expected {expected_channels}, got {feature_stack.shape[0]}."
        )

    return feature_stack.astype(np.float32), channel_semantics


def _parse_per_band_stats(stats_payload: Mapping[str, Any], *, expected_channels: int) -> list[_PerBandStat]:
    band_stats_raw = stats_payload.get("band_stats")
    if not isinstance(band_stats_raw, Sequence) or isinstance(band_stats_raw, (str, bytes)):
        raise NormalizationContractError(
            "normalization stats payload must contain 'band_stats' sequence."
        )

    parsed: list[_PerBandStat] = []
    for idx, item in enumerate(band_stats_raw):
        if not isinstance(item, Mapping):
            raise NormalizationContractError(
                f"band_stats[{idx}] must be a mapping/object."
            )

        band_idx_raw = item.get("band_idx")
        p_lo_raw = item.get("p_lo")
        p_hi_raw = item.get("p_hi")

        if isinstance(band_idx_raw, bool) or not isinstance(band_idx_raw, int):
            raise NormalizationContractError(
                f"band_stats[{idx}].band_idx must be an integer."
            )
        if isinstance(p_lo_raw, bool) or not isinstance(p_lo_raw, (int, float)):
            raise NormalizationContractError(
                f"band_stats[{idx}].p_lo must be numeric."
            )
        if isinstance(p_hi_raw, bool) or not isinstance(p_hi_raw, (int, float)):
            raise NormalizationContractError(
                f"band_stats[{idx}].p_hi must be numeric."
            )

        p_lo = float(p_lo_raw)
        p_hi = float(p_hi_raw)
        if not p_hi > p_lo:
            raise NormalizationContractError(
                f"band_stats[{idx}] must satisfy p_hi > p_lo, got p_lo={p_lo}, p_hi={p_hi}."
            )

        parsed.append(_PerBandStat(band_idx=int(band_idx_raw), p_lo=p_lo, p_hi=p_hi))

    if len(parsed) != expected_channels:
        raise NormalizationContractError(
            "band_stats count must match feature channel count: "
            f"expected {expected_channels}, got {len(parsed)}."
        )

    parsed_sorted = sorted(parsed, key=lambda x: x.band_idx)
    expected_band_indices = list(range(expected_channels))
    actual_indices = [s.band_idx for s in parsed_sorted]
    if actual_indices != expected_band_indices:
        raise NormalizationContractError(
            "band_stats must provide contiguous band_idx values [0..C-1]. "
            f"Expected {expected_band_indices}, got {actual_indices}."
        )

    return parsed_sorted


def _load_stats_payload_from_path(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NormalizationContractError(
            f"Normalization stats file is not valid JSON: {path} ({exc})"
        ) from exc
    except OSError as exc:
        raise NormalizationContractError(
            f"Failed to read normalization stats file: {path} ({exc})"
        ) from exc

    if not isinstance(payload, Mapping):
        raise NormalizationContractError(
            f"Normalization stats payload must be a mapping/object, got {type(payload).__name__}."
        )
    return dict(payload)


def _resolve_stats_payload(
    *,
    checkpoint_contract: CheckpointDrivenPredictContract,
    expected_channels: int,
    normalization_stats: Mapping[str, Any] | None,
    normalization_stats_path: str | Path | None,
) -> tuple[list[_PerBandStat], str]:
    # Priority:
    # 1) explicit normalization_stats mapping,
    # 2) explicit normalization_stats_path,
    # 3) embedded per_band_stats in checkpoint normalization,
    # 4) stats_source-resolved file path,
    # 5) explicit error.

    if normalization_stats is not None:
        if not isinstance(normalization_stats, Mapping):
            raise NormalizationContractError(
                "normalization_stats must be a mapping/object when provided."
            )
        parsed = _parse_per_band_stats(normalization_stats, expected_channels=expected_channels)
        return parsed, "explicit_mapping"

    if normalization_stats_path is not None:
        resolved = _normalize_existing_path(normalization_stats_path, name="normalization_stats_path")
        payload = _load_stats_payload_from_path(resolved)
        parsed = _parse_per_band_stats(payload, expected_channels=expected_channels)
        return parsed, f"explicit_path:{resolved}"

    embedded = checkpoint_contract.normalization.get("per_band_stats")
    if embedded is not None:
        if not isinstance(embedded, Sequence) or isinstance(embedded, (str, bytes)):
            raise NormalizationContractError(
                "checkpoint normalization.per_band_stats must be a sequence when present."
            )
        parsed = _parse_per_band_stats(
            {"band_stats": list(embedded)},
            expected_channels=expected_channels,
        )
        return parsed, "checkpoint_embedded"

    stats_source = checkpoint_contract.normalization.get("stats_source")
    if not isinstance(stats_source, str) or stats_source.strip() == "":
        raise NormalizationContractError(
            "checkpoint normalization.stats_source must be a non-empty string."
        )

    candidate_paths: list[Path] = []
    source_as_path = Path(stats_source)
    candidate_paths.append(source_as_path)
    candidate_paths.append(checkpoint_contract.checkpoint_metadata_path.parent / source_as_path)

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            payload = _load_stats_payload_from_path(candidate)
            parsed = _parse_per_band_stats(payload, expected_channels=expected_channels)
            return parsed, f"stats_source_path:{candidate}"

    raise NormalizationContractError(
        "Train-derived per-band normalization stats are required for predict-time apply, "
        "but no stats payload was provided and stats_source could not be resolved to a file. "
        f"stats_source={stats_source!r}."
    )


def apply_predict_normalization(
    *,
    feature_stack: np.ndarray,
    valid_mask: np.ndarray,
    checkpoint_contract: CheckpointDrivenPredictContract,
    normalization_stats: Mapping[str, Any] | None = None,
    normalization_stats_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply train-derived robust normalization to predict feature stack."""

    if feature_stack.ndim != 3:
        raise ContractError(
            f"feature_stack must be 3-D (C, H, W), got shape={feature_stack.shape}."
        )
    if valid_mask.ndim != 2:
        raise ContractError(f"valid_mask must be 2-D (H, W), got shape={valid_mask.shape}.")
    if feature_stack.shape[1:] != valid_mask.shape:
        raise ContractError(
            "valid_mask shape must match feature stack spatial shape: "
            f"{valid_mask.shape} != {feature_stack.shape[1:]}"
        )

    norm = checkpoint_contract.normalization
    norm_name = norm.get("normalization_name")
    if not isinstance(norm_name, str) or norm_name.strip() == "":
        raise NormalizationContractError("checkpoint normalization.normalization_name must be non-empty.")

    clip_lo, clip_hi = _normalize_numeric_pair(
        norm.get("clip_percentiles"),
        name="checkpoint normalization.clip_percentiles",
    )
    if not (0.0 <= clip_lo < clip_hi <= 100.0):
        raise NormalizationContractError(
            "checkpoint normalization.clip_percentiles must satisfy 0 <= lo < hi <= 100."
        )

    scale_min, scale_max = _normalize_numeric_pair(
        norm.get("scaling_range"),
        name="checkpoint normalization.scaling_range",
    )
    if not scale_max > scale_min:
        raise NormalizationContractError(
            "checkpoint normalization.scaling_range must satisfy max > min: "
            f"got [{scale_min}, {scale_max}]."
        )

    expected_channels = CHANNEL_COUNTS[checkpoint_contract.feature_mode]
    if feature_stack.shape[0] != expected_channels:
        raise ChannelCountError(
            "feature stack channel count is inconsistent with checkpoint feature_mode: "
            f"expected {expected_channels}, got {feature_stack.shape[0]}."
        )

    per_band_stats, stats_origin = _resolve_stats_payload(
        checkpoint_contract=checkpoint_contract,
        expected_channels=expected_channels,
        normalization_stats=normalization_stats,
        normalization_stats_path=normalization_stats_path,
    )

    normalized = feature_stack.astype(np.float32, copy=True)
    valid_bool = valid_mask.astype(bool)

    for st in per_band_stats:
        band = normalized[st.band_idx]
        clipped = np.clip(band, st.p_lo, st.p_hi)
        denom = st.p_hi - st.p_lo
        scaled01 = (clipped - st.p_lo) / denom
        scaled = scale_min + scaled01 * (scale_max - scale_min)
        scaled = np.nan_to_num(scaled, nan=scale_min, posinf=scale_max, neginf=scale_min)
        scaled[~valid_bool] = scale_min
        normalized[st.band_idx] = scaled.astype(np.float32)

    summary = {
        "normalization_name": norm_name,
        "stats_source": norm.get("stats_source"),
        "stats_origin": stats_origin,
        "clip_percentiles": [clip_lo, clip_hi],
        "scaling_range": [scale_min, scale_max],
        "bands_normalized": expected_channels,
    }
    return normalized, summary


def resolve_predict_input(
    *,
    raster_path: str | Path,
    checkpoint_contract: CheckpointDrivenPredictContract,
    explicit_valid_mask_override: np.ndarray | None = None,
    normalization_stats: Mapping[str, Any] | None = None,
    normalization_stats_path: str | Path | None = None,
) -> PredictInputAdapterResult:
    """Resolve full predict-time assembled input (without model forward)."""

    raster_contract: PredictRasterContractResult = resolve_predict_raster_contract(
        raster_path=raster_path,
        checkpoint_contract=checkpoint_contract,
        explicit_valid_mask_override=explicit_valid_mask_override,
    )

    valid_mask = raster_contract.valid_mask_resolution.valid_mask
    raw_feature_stack, dataset_channel_semantics = build_predict_feature_stack(
        raster_path=raster_path,
        checkpoint_contract=checkpoint_contract,
        valid_mask=valid_mask,
    )

    normalized_feature_stack, norm_summary = apply_predict_normalization(
        feature_stack=raw_feature_stack,
        valid_mask=valid_mask,
        checkpoint_contract=checkpoint_contract,
        normalization_stats=normalization_stats,
        normalization_stats_path=normalization_stats_path,
    )

    valid_channel = valid_mask[np.newaxis, ...].astype(np.float32)
    assembled_input = np.concatenate([normalized_feature_stack, valid_channel], axis=0)

    expected_input_channels = checkpoint_contract.in_channels
    if assembled_input.shape[0] != expected_input_channels:
        raise ChannelCountError(
            "Assembled predict input channel count mismatch: "
            f"expected {expected_input_channels}, got {assembled_input.shape[0]}."
        )

    if tuple(checkpoint_contract.channel_semantics) != (*dataset_channel_semantics, "valid"):
        raise ContractError(
            "Assembled input channel semantics are inconsistent with checkpoint contract."
        )

    return PredictInputAdapterResult(
        feature_mode=checkpoint_contract.feature_mode,
        assembled_model_input=checkpoint_contract.assembled_model_input,
        channel_semantics=tuple(checkpoint_contract.channel_semantics),
        valid_mask_source=raster_contract.valid_mask_resolution.source,
        raw_feature_shape=tuple(raw_feature_stack.shape),
        normalized_feature_shape=tuple(normalized_feature_stack.shape),
        assembled_input_shape=tuple(assembled_input.shape),
        raw_feature_stack=raw_feature_stack,
        normalized_feature_stack=normalized_feature_stack,
        assembled_input=assembled_input,
        valid_mask=valid_mask,
        normalization_summary=norm_summary,
        input_ready_for_model=True,
    )


__all__ = [
    "PredictInputAdapterResult",
    "build_predict_feature_stack",
    "apply_predict_normalization",
    "resolve_predict_input",
]
