"""Tiled inference engine for module_target_predict.

This module implements:
- sliding window tile offset generation;
- Gaussian blending kernel;
- per-tile feature assembly and normalization;
- invalid-only tile skip (no model forward);
- Gaussian-weighted output accumulation across the full scene.

It operates at a lower level than resolve_predict_input:
the raster is read per-window without loading the full scene into memory.
"""

from __future__ import annotations

import json
import shutil
import tempfile
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
from ai_fields.common.progress import progress_bar
from ai_fields.module_target_predict.checkpoint_contract import (
    CheckpointDrivenPredictContract,
)
from ai_fields.module_target_predict.inference_core import LoadedPredictModel

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


_EPS = 1e-6
_MEMMAP_ACCUM_THRESHOLD_BYTES = 1_073_741_824  # 1 GiB
_FINALIZE_CHUNK_ROWS = 256


@dataclass(frozen=True)
class _PerBandNormStat:
    band_idx: int
    p_lo: float
    p_hi: float


@dataclass(frozen=True)
class TiledPredictResult:
    """Result of Gaussian-blended tiled inference over a full scene."""

    extent_prob: np.ndarray      # (H, W) float32, range [0, 1]
    boundary_prob: np.ndarray    # (3, H, W) float32, per-class probs summing to 1 per pixel
    distance_pred: np.ndarray    # (H, W) float32
    valid_mask: np.ndarray       # (H, W) uint8, values 0/1

    scene_height: int
    scene_width: int

    tiles_total: int
    tiles_processed: int
    tiles_skipped_invalid: int

    feature_mode: str
    tile_size: int
    overlap: float
    blending: str
    temp_work_dir: str | None = None


def generate_tile_offsets(total_size: int, tile_size: int, stride: int) -> list[int]:
    """Generate 1-D tile start-offsets for sliding window traversal.

    Always includes offset=0. Ensures the last tile ends exactly at total_size
    by adding the ``last_start = total_size - tile_size`` offset when needed.
    When total_size <= tile_size a single offset [0] is returned.
    """
    if not isinstance(total_size, int) or total_size < 1:
        raise ContractError(f"total_size must be a positive integer, got {total_size!r}.")
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ContractError(f"tile_size must be a positive integer, got {tile_size!r}.")
    if not isinstance(stride, int) or stride < 1:
        raise ContractError(f"stride must be a positive integer, got {stride!r}.")

    if total_size <= tile_size:
        return [0]

    offsets: list[int] = []
    current = 0
    while current + tile_size <= total_size:
        offsets.append(current)
        current += stride

    last_start = total_size - tile_size
    if not offsets or offsets[-1] != last_start:
        offsets.append(last_start)

    return sorted(set(offsets))


def build_gaussian_kernel(tile_size: int) -> np.ndarray:
    """Build a 2-D symmetric Gaussian weight kernel for tile blending.

    sigma = tile_size / 4. All values are strictly positive.
    The center value is the maximum; corner values are the minimum.
    The kernel is NOT normalised to sum-to-1; normalisation happens via
    the Gaussian-weighted accumulation in run_tiled_predict.
    """
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ContractError(f"tile_size must be a positive integer, got {tile_size!r}.")

    sigma = tile_size / 4.0
    axis = np.arange(tile_size, dtype=np.float64) - (tile_size - 1) / 2.0
    gauss_1d = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel = np.outer(gauss_1d, gauss_1d).astype(np.float32)
    return kernel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ContractError(
            "torch is required for module_target_predict tiled inference. "
            "Install torch to use this layer."
        )


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


def _require_rasterio() -> Any:
    try:
        import rasterio
        import rasterio.windows
    except ImportError as exc:
        raise ContractError(
            "rasterio is required for module_target_predict tiled inference."
        ) from exc
    return rasterio


def _parse_per_band_norm_stats(
    stats_payload: Mapping[str, Any],
    *,
    expected_channels: int,
) -> list[_PerBandNormStat]:
    """Parse band_stats sequence from a normalization stats mapping."""
    band_stats_raw = stats_payload.get("band_stats")
    if not isinstance(band_stats_raw, Sequence) or isinstance(band_stats_raw, (str, bytes)):
        raise NormalizationContractError(
            "normalization stats payload must contain 'band_stats' sequence."
        )

    parsed: list[_PerBandNormStat] = []
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
                f"band_stats[{idx}] must satisfy p_hi > p_lo, "
                f"got p_lo={p_lo}, p_hi={p_hi}."
            )
        parsed.append(_PerBandNormStat(band_idx=int(band_idx_raw), p_lo=p_lo, p_hi=p_hi))

    if len(parsed) != expected_channels:
        raise NormalizationContractError(
            "band_stats count must match feature channel count: "
            f"expected {expected_channels}, got {len(parsed)}."
        )

    parsed_sorted = sorted(parsed, key=lambda x: x.band_idx)
    expected_indices = list(range(expected_channels))
    actual_indices = [s.band_idx for s in parsed_sorted]
    if actual_indices != expected_indices:
        raise NormalizationContractError(
            "band_stats must provide contiguous band_idx values [0..C-1]. "
            f"Expected {expected_indices}, got {actual_indices}."
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
            "Normalization stats payload must be a mapping/object, "
            f"got {type(payload).__name__}."
        )
    return dict(payload)


def _accumulator_bytes_required(
    *,
    height: int,
    width: int,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> int:
    # extent_acc + boundary_acc(3 bands) + distance_acc + weight_sum = 6 planes
    plane_count = 6
    return int(height * width * plane_count * dtype.itemsize)


def _allocate_zero_array(
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    use_memmap: bool,
    path: Path | None = None,
) -> np.ndarray:
    if use_memmap:
        if path is None:
            raise ContractError("Internal error: memmap path is required when use_memmap=True.")
        arr = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        arr.fill(0)
        return arr
    return np.zeros(shape, dtype=dtype)


def _finalize_weighted_outputs_in_place(
    *,
    extent_acc: np.ndarray,
    boundary_acc: np.ndarray,
    distance_acc: np.ndarray,
    weight_sum: np.ndarray,
    chunk_rows: int = _FINALIZE_CHUNK_ROWS,
) -> None:
    if not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ContractError(f"chunk_rows must be a positive integer, got {chunk_rows!r}.")

    height = int(weight_sum.shape[0])
    for row_off in range(0, height, chunk_rows):
        row_end = min(height, row_off + chunk_rows)
        weights = weight_sum[row_off:row_end, :]
        positive = weights > 0.0
        non_positive = ~positive

        extent_chunk = extent_acc[row_off:row_end, :]
        np.divide(extent_chunk, weights, out=extent_chunk, where=positive)
        extent_chunk[non_positive] = 0.0

        distance_chunk = distance_acc[row_off:row_end, :]
        np.divide(distance_chunk, weights, out=distance_chunk, where=positive)
        distance_chunk[non_positive] = 0.0

        for cls_idx in range(boundary_acc.shape[0]):
            boundary_chunk = boundary_acc[cls_idx, row_off:row_end, :]
            np.divide(boundary_chunk, weights, out=boundary_chunk, where=positive)
            boundary_chunk[non_positive] = 0.0


def cleanup_tiled_predict_result(result: TiledPredictResult) -> None:
    """Clean temporary disk-backed buffers created by run_tiled_predict."""
    work_dir = result.temp_work_dir
    if not work_dir:
        return
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except OSError:
        # Best-effort cleanup: temp artifacts are non-contract outputs.
        return


def _resolve_per_band_norm_stats(
    *,
    checkpoint_contract: CheckpointDrivenPredictContract,
    expected_channels: int,
    normalization_stats: Mapping[str, Any] | None,
    normalization_stats_path: str | Path | None,
) -> tuple[list[_PerBandNormStat], str]:
    """Resolve per-band normalization stats using the same priority as feature_adapter.

    Priority:
      1) explicit normalization_stats mapping;
      2) explicit normalization_stats_path;
      3) embedded per_band_stats in checkpoint normalization;
      4) stats_source-resolved file path;
      5) explicit NormalizationContractError.
    """
    if normalization_stats is not None:
        if not isinstance(normalization_stats, Mapping):
            raise NormalizationContractError(
                "normalization_stats must be a mapping/object when provided."
            )
        parsed = _parse_per_band_norm_stats(
            normalization_stats, expected_channels=expected_channels
        )
        return parsed, "explicit_mapping"

    if normalization_stats_path is not None:
        resolved = _normalize_existing_path(
            normalization_stats_path, name="normalization_stats_path"
        )
        payload = _load_stats_payload_from_path(resolved)
        parsed = _parse_per_band_norm_stats(payload, expected_channels=expected_channels)
        return parsed, f"explicit_path:{resolved}"

    embedded = checkpoint_contract.normalization.get("per_band_stats")
    if embedded is not None:
        if not isinstance(embedded, Sequence) or isinstance(embedded, (str, bytes)):
            raise NormalizationContractError(
                "checkpoint normalization.per_band_stats must be a sequence when present."
            )
        parsed = _parse_per_band_norm_stats(
            {"band_stats": list(embedded)},
            expected_channels=expected_channels,
        )
        return parsed, "checkpoint_embedded"

    stats_source = checkpoint_contract.normalization.get("stats_source")
    if not isinstance(stats_source, str) or stats_source.strip() == "":
        raise NormalizationContractError(
            "checkpoint normalization.stats_source must be a non-empty string."
        )

    candidate_paths: list[Path] = [
        Path(stats_source),
        checkpoint_contract.checkpoint_metadata_path.parent / Path(stats_source),
    ]
    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            payload = _load_stats_payload_from_path(candidate)
            parsed = _parse_per_band_norm_stats(payload, expected_channels=expected_channels)
            return parsed, f"stats_source_path:{candidate}"

    raise NormalizationContractError(
        "Train-derived per-band normalization stats are required for predict-time tiled "
        "inference, but no stats payload was provided and stats_source could not be "
        f"resolved to a file. stats_source={stats_source!r}."
    )


def _build_tile_feature_stack(
    raw8_tile: np.ndarray,
    valid_tile: np.ndarray,
    *,
    feature_mode: str,
) -> np.ndarray:
    """Build dataset-side feature stack for a single tile (raw8 or raw8_idx3)."""
    if raw8_tile.ndim != 3 or raw8_tile.shape[0] != 8:
        raise ContractError(
            f"raw8_tile must have shape (8, H, W), got {raw8_tile.shape}."
        )
    if valid_tile.ndim != 2 or valid_tile.shape != raw8_tile.shape[1:]:
        raise ContractError(
            f"valid_tile must have shape {raw8_tile.shape[1:]}, got {valid_tile.shape}."
        )

    valid_bool = valid_tile.astype(bool)

    if feature_mode == "raw8":
        stack = raw8_tile.copy()
    elif feature_mode == "raw8_idx3":
        red = raw8_tile[4]
        nir1 = raw8_tile[6]
        green = raw8_tile[2]

        ndvi = np.zeros_like(red, dtype=np.float32)
        savi = np.zeros_like(red, dtype=np.float32)
        ndwi = np.zeros_like(red, dtype=np.float32)

        if np.any(valid_bool):
            r = red[valid_bool]
            n = nir1[valid_bool]
            g = green[valid_bool]
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi_v = (n - r) / (n + r + _EPS)
                savi_v = ((n - r) / (n + r + 0.5 + _EPS)) * 1.5
                ndwi_v = (g - n) / (g + n + _EPS)
            ndvi[valid_bool] = np.nan_to_num(
                ndvi_v, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            savi[valid_bool] = np.nan_to_num(
                savi_v, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            ndwi[valid_bool] = np.nan_to_num(
                ndwi_v, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)

        stack = np.concatenate(
            [raw8_tile, np.stack([ndvi, savi, ndwi], axis=0)], axis=0
        )
    else:
        raise ContractError(f"Unsupported feature_mode: {feature_mode!r}.")

    stack[:, ~valid_bool] = 0.0
    return stack.astype(np.float32)


def _normalize_tile_stack(
    feature_stack: np.ndarray,
    valid_tile: np.ndarray,
    *,
    per_band_stats: list[_PerBandNormStat],
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    """Apply per-band clipping and scaling to a tile feature stack."""
    normalized = feature_stack.astype(np.float32, copy=True)
    valid_bool = valid_tile.astype(bool)

    for st in per_band_stats:
        band = normalized[st.band_idx]
        clipped = np.clip(band, st.p_lo, st.p_hi)
        denom = st.p_hi - st.p_lo
        scaled01 = (clipped - st.p_lo) / denom
        scaled = scale_min + scaled01 * (scale_max - scale_min)
        scaled = np.nan_to_num(scaled, nan=scale_min, posinf=scale_max, neginf=scale_min)
        scaled[~valid_bool] = scale_min
        normalized[st.band_idx] = scaled.astype(np.float32)

    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tiled_predict(
    *,
    raster_path: str | Path,
    checkpoint_contract: CheckpointDrivenPredictContract,
    loaded_model: LoadedPredictModel,
    valid_mask: np.ndarray,
    tile_size: int = 512,
    overlap: float = 0.25,
    normalization_stats: Mapping[str, Any] | None = None,
    normalization_stats_path: str | Path | None = None,
    progress_enabled: bool | None = None,
    accumulator_memmap_threshold_bytes: int = _MEMMAP_ACCUM_THRESHOLD_BYTES,
) -> TiledPredictResult:
    """Run Gaussian-blended tiled inference over a full raster scene.

    Parameters
    ----------
    raster_path:
        Path to the input 8-band GeoTIFF.
    checkpoint_contract:
        Resolved checkpoint predict contract.
    loaded_model:
        Loaded predict model from load_predict_model().
    valid_mask:
        Resolved valid mask, shape (H, W), dtype uint8 (0/1).
        Must be pre-computed by the caller (e.g. from resolve_predict_valid_mask).
    tile_size:
        Tile size in pixels (default 512).
    overlap:
        Fractional overlap between adjacent tiles, in [0, 1) (default 0.25).
    normalization_stats:
        Optional explicit per-band stats mapping (same priority as feature_adapter).
    normalization_stats_path:
        Optional path to JSON stats file.
    accumulator_memmap_threshold_bytes:
        Threshold for switching accumulation buffers to disk-backed memmap.
        When estimated accumulator footprint is >= threshold, memmap is used.
        Set to 0 to always use memmap.

    Returns
    -------
    TiledPredictResult
        Scene-level accumulated predictions and tile coverage stats.
    """
    _require_torch()
    _rio = _require_rasterio()

    resolved_path = _normalize_existing_path(raster_path, name="raster_path")

    if not isinstance(valid_mask, np.ndarray) or valid_mask.ndim != 2:
        raise ContractError("valid_mask must be a 2-D numpy ndarray (H, W).")
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ContractError(f"tile_size must be a positive integer, got {tile_size!r}.")
    if not isinstance(overlap, float) or not (0.0 <= overlap < 1.0):
        raise ContractError(f"overlap must be a float in [0, 1), got {overlap!r}.")
    if (
        isinstance(accumulator_memmap_threshold_bytes, bool)
        or not isinstance(accumulator_memmap_threshold_bytes, int)
        or accumulator_memmap_threshold_bytes < 0
    ):
        raise ContractError(
            "accumulator_memmap_threshold_bytes must be an integer >= 0."
        )

    feature_mode = checkpoint_contract.feature_mode
    expected_feature_channels = CHANNEL_COUNTS[feature_mode]
    in_channels = checkpoint_contract.in_channels

    # Validate normalization contract fields up-front.
    norm = checkpoint_contract.normalization
    norm_name = norm.get("normalization_name")
    if not isinstance(norm_name, str) or norm_name.strip() == "":
        raise NormalizationContractError(
            "checkpoint normalization.normalization_name must be non-empty."
        )

    scale_range_raw = norm.get("scaling_range")
    if (
        not isinstance(scale_range_raw, Sequence)
        or isinstance(scale_range_raw, (str, bytes))
        or len(scale_range_raw) != 2
    ):
        raise NormalizationContractError(
            "checkpoint normalization.scaling_range must be a 2-element sequence."
        )
    scale_min = float(scale_range_raw[0])
    scale_max = float(scale_range_raw[1])
    if not scale_max > scale_min:
        raise NormalizationContractError(
            f"scaling_range must satisfy max > min: [{scale_min}, {scale_max}]."
        )

    per_band_stats, _stats_origin = _resolve_per_band_norm_stats(
        checkpoint_contract=checkpoint_contract,
        expected_channels=expected_feature_channels,
        normalization_stats=normalization_stats,
        normalization_stats_path=normalization_stats_path,
    )

    gaussian_kernel = build_gaussian_kernel(tile_size)
    stride = max(1, int(tile_size * (1.0 - overlap)))

    H = int(valid_mask.shape[0])
    W = int(valid_mask.shape[1])

    # Use float32 accumulators and switch to disk-backed memmap when scene size
    # implies a high RAM footprint. This preserves contract semantics while
    # avoiding OOM peaks on full-scene inference.
    accum_dtype = np.dtype(np.float32)
    required_bytes = _accumulator_bytes_required(height=H, width=W, dtype=accum_dtype)
    use_memmap = required_bytes >= accumulator_memmap_threshold_bytes
    temp_work_dir: str | None = None
    if use_memmap:
        temp_work_dir = tempfile.mkdtemp(prefix="target_predict_accum_")
        temp_dir = Path(temp_work_dir)
        extent_acc = _allocate_zero_array(
            shape=(H, W),
            dtype=accum_dtype,
            use_memmap=True,
            path=temp_dir / "extent_acc.f32.mmap",
        )
        boundary_acc = _allocate_zero_array(
            shape=(3, H, W),
            dtype=accum_dtype,
            use_memmap=True,
            path=temp_dir / "boundary_acc.f32.mmap",
        )
        distance_acc = _allocate_zero_array(
            shape=(H, W),
            dtype=accum_dtype,
            use_memmap=True,
            path=temp_dir / "distance_acc.f32.mmap",
        )
        weight_sum = _allocate_zero_array(
            shape=(H, W),
            dtype=accum_dtype,
            use_memmap=True,
            path=temp_dir / "weight_sum.f32.mmap",
        )
    else:
        extent_acc = _allocate_zero_array(
            shape=(H, W),
            dtype=accum_dtype,
            use_memmap=False,
        )
        boundary_acc = _allocate_zero_array(
            shape=(3, H, W),
            dtype=accum_dtype,
            use_memmap=False,
        )
        distance_acc = _allocate_zero_array(
            shape=(H, W),
            dtype=accum_dtype,
            use_memmap=False,
        )
        weight_sum = _allocate_zero_array(
            shape=(H, W),
            dtype=accum_dtype,
            use_memmap=False,
        )

    row_offsets = generate_tile_offsets(H, tile_size, stride)
    col_offsets = generate_tile_offsets(W, tile_size, stride)

    tiles_total = len(row_offsets) * len(col_offsets)
    tiles_processed = 0
    tiles_skipped_invalid = 0

    try:
        with progress_bar(
            total=tiles_total,
            desc="predict: tiles",
            unit="tile",
            progress_enabled=progress_enabled,
            leave=True,
        ) as bar:
            with _rio.open(resolved_path) as ds:
                for row_off in row_offsets:
                    for col_off in col_offsets:
                        actual_h = min(tile_size, H - row_off)
                        actual_w = min(tile_size, W - col_off)

                        valid_tile = valid_mask[
                            row_off: row_off + actual_h,
                            col_off: col_off + actual_w,
                        ]

                        # Skip tiles where every pixel is invalid (§11.2).
                        if not np.any(valid_tile):
                            tiles_skipped_invalid += 1
                            bar.update(1)
                            bar.set_postfix(
                                processed=tiles_processed,
                                skipped=tiles_skipped_invalid,
                            )
                            continue

                        # Windowed read of the 8 source bands.
                        window = _rio.windows.Window(col_off, row_off, actual_w, actual_h)
                        raw8_actual = ds.read(list(range(1, 9)), window=window).astype(np.float32)

                        # Pad border tiles to tile_size × tile_size with zeros.
                        if actual_h < tile_size or actual_w < tile_size:
                            raw8_padded = np.zeros((8, tile_size, tile_size), dtype=np.float32)
                            raw8_padded[:, :actual_h, :actual_w] = raw8_actual
                            valid_padded = np.zeros((tile_size, tile_size), dtype=np.uint8)
                            valid_padded[:actual_h, :actual_w] = valid_tile
                        else:
                            raw8_padded = raw8_actual
                            valid_padded = valid_tile

                        # Build dataset-side feature stack.
                        feature_stack = _build_tile_feature_stack(
                            raw8_padded, valid_padded, feature_mode=feature_mode
                        )

                        # Apply per-band normalization.
                        normalized = _normalize_tile_stack(
                            feature_stack,
                            valid_padded,
                            per_band_stats=per_band_stats,
                            scale_min=scale_min,
                            scale_max=scale_max,
                        )

                        # Assemble model input: normalized features + valid channel.
                        valid_channel = valid_padded[np.newaxis, ...].astype(np.float32)
                        assembled = np.concatenate([normalized, valid_channel], axis=0)

                        if assembled.shape[0] != in_channels:
                            raise ChannelCountError(
                                f"Assembled tile input channel count {assembled.shape[0]} does not "
                                f"match checkpoint contract {in_channels}."
                            )

                        # Model forward.
                        input_tensor = torch.from_numpy(assembled).unsqueeze(0).to(
                            device=loaded_model.device, dtype=torch.float32
                        )
                        with torch.no_grad():
                            outputs = loaded_model.model(input_tensor)

                        # Apply activations.
                        extent_t = torch.sigmoid(outputs["extent"]).squeeze(0).squeeze(0)  # (T, T)
                        boundary_t = torch.softmax(outputs["boundary"], dim=1).squeeze(0)  # (3, T, T)
                        distance_t = outputs["distance"].squeeze(0).squeeze(0)             # (T, T)

                        # Crop padded output back to actual tile dimensions.
                        extent_np = extent_t.detach().cpu().numpy().astype(np.float32, copy=False)
                        boundary_np = boundary_t.detach().cpu().numpy().astype(np.float32, copy=False)
                        distance_np = distance_t.detach().cpu().numpy().astype(np.float32, copy=False)

                        extent_np = extent_np[:actual_h, :actual_w]
                        boundary_np = boundary_np[:, :actual_h, :actual_w]
                        distance_np = distance_np[:actual_h, :actual_w]

                        # Gaussian weight × valid_tile for this tile region.
                        kernel_crop = gaussian_kernel[:actual_h, :actual_w].astype(np.float32, copy=False)
                        weights = kernel_crop * valid_tile.astype(np.float32, copy=False)

                        # Weighted accumulation.
                        sl_r = slice(row_off, row_off + actual_h)
                        sl_c = slice(col_off, col_off + actual_w)

                        extent_acc[sl_r, sl_c] += extent_np * weights
                        boundary_acc[:, sl_r, sl_c] += boundary_np * weights[np.newaxis, ...]
                        distance_acc[sl_r, sl_c] += distance_np * weights
                        weight_sum[sl_r, sl_c] += weights

                        tiles_processed += 1
                        bar.update(1)
                        bar.set_postfix(
                            processed=tiles_processed,
                            skipped=tiles_skipped_invalid,
                        )

        # Finalise in-place to avoid allocating a second full-scene output copy.
        _finalize_weighted_outputs_in_place(
            extent_acc=extent_acc,
            boundary_acc=boundary_acc,
            distance_acc=distance_acc,
            weight_sum=weight_sum,
        )
        if isinstance(extent_acc, np.memmap):
            extent_acc.flush()
        if isinstance(boundary_acc, np.memmap):
            boundary_acc.flush()
        if isinstance(distance_acc, np.memmap):
            distance_acc.flush()
    except Exception:
        if temp_work_dir is not None:
            shutil.rmtree(temp_work_dir, ignore_errors=True)
        raise

    return TiledPredictResult(
        extent_prob=extent_acc,
        boundary_prob=boundary_acc,
        distance_pred=distance_acc,
        valid_mask=valid_mask.astype(np.uint8, copy=False),
        scene_height=H,
        scene_width=W,
        tiles_total=tiles_total,
        tiles_processed=tiles_processed,
        tiles_skipped_invalid=tiles_skipped_invalid,
        feature_mode=feature_mode,
        tile_size=tile_size,
        overlap=overlap,
        blending="gaussian",
        temp_work_dir=temp_work_dir,
    )


__all__ = [
    "TiledPredictResult",
    "generate_tile_offsets",
    "build_gaussian_kernel",
    "cleanup_tiled_predict_result",
    "run_tiled_predict",
]
