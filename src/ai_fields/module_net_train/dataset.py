"""Dataset reader and model-input assembly for module_net_train.

Pure numpy functions:
  - list_sample_ids()      — discover samples in a split directory
  - read_sample()          — load one sample (img, targets, valid, meta)
  - assemble_model_input() — append valid as the last input channel

PyTorch class (requires torch):
  - FieldsDataset          — torch.utils.data.Dataset over a split directory

Contract anchors:
  DATA_CONTRACT.md §7.4: assembled model input = feature stack + valid channel.
  DATA_CONTRACT.md §6.4: valid has dual role — mask AND input channel (DEC-002).
  DATA_CONTRACT.md §7.5: raw8_valid=9ch, raw8_idx3_valid=12ch.
  module_net_train.md §10.2: runtime sample dict must contain 'image' (assembled tensor)
    and 'valid' kept separately for ignore/masking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.constants import CHANNEL_COUNTS, FEATURE_MODES
from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    FeatureModeError,
)


# ---------------------------------------------------------------------------
# Pure numpy helpers
# ---------------------------------------------------------------------------


def list_sample_ids(split_dir: Any) -> list[str]:
    """Return a sorted list of sample IDs found in split_dir/img/.

    Sample ID = the stem of *_img.tif files (everything before '_img.tif').

    Parameters
    ----------
    split_dir:
        Path to a split directory, e.g. dataset/train/.
        Must contain an 'img/' sub-directory with *_img.tif files.

    Raises
    ------
    ContractError
        If img/ does not exist or contains no *_img.tif files.
    """
    split_dir = Path(split_dir)
    img_dir = split_dir / "img"
    if not img_dir.exists():
        raise ContractError(
            f"img sub-directory not found in split dir: {img_dir}.  "
            "Expected canonical module_prep_data layout: split_dir/img/*_img.tif"
        )

    sample_ids = sorted(
        p.name.removesuffix("_img.tif") for p in img_dir.glob("*_img.tif")
    )

    if not sample_ids:
        raise ContractError(
            f"No *_img.tif files found in {img_dir}.  "
            "The split directory appears to be empty."
        )

    return sample_ids


def read_sample(split_dir: Any, sample_id: str, feature_mode: str) -> dict:
    """Load one train-ready sample from the canonical split layout.

    Reads GeoTIFF layers and meta.json from the module_prep_data export
    structure:  split_dir/{img,extent,boundary,distance,valid,meta}/.

    Parameters
    ----------
    split_dir:
        Path to a split directory, e.g. dataset/train/.
    sample_id:
        Patch identifier (prefix of all layer files).
    feature_mode:
        Dataset-side feature mode: "raw8" or "raw8_idx3".

    Returns
    -------
    dict with keys:
        img       : (C, H, W) float32 — dataset-side feature stack (no valid yet)
        extent    : (H, W) uint8
        boundary  : (H, W) uint8  (0=background, 1=skeleton, 2=buffer)
        distance  : (H, W) float32
        valid     : (H, W) uint8  (0=invalid, 1=valid)
        meta      : dict — parsed meta.json
        sample_id : str

    Raises
    ------
    FeatureModeError
        If feature_mode is not in FEATURE_MODES.
    ContractError
        If any required file is missing.
    ChannelCountError
        If img band count does not match the expected feature_mode channel count.
    ContractError
        If spatial shapes are inconsistent across layers.
    """
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:
        raise ContractError("rasterio is required for read_sample") from exc

    if feature_mode not in FEATURE_MODES:
        raise FeatureModeError(
            f"Unsupported feature_mode: {feature_mode!r}.  "
            f"Expected one of {list(FEATURE_MODES)} (DATA_CONTRACT.md §7.1)."
        )

    split_dir = Path(split_dir)
    sid = sample_id

    paths = {
        "img":      split_dir / "img"      / f"{sid}_img.tif",
        "extent":   split_dir / "extent"   / f"{sid}_extent.tif",
        "boundary": split_dir / "boundary" / f"{sid}_boundary.tif",
        "distance": split_dir / "distance" / f"{sid}_distance.tif",
        "valid":    split_dir / "valid"    / f"{sid}_valid.tif",
        "meta":     split_dir / "meta"     / f"{sid}_meta.json",
    }

    for layer, path in paths.items():
        if not path.exists():
            raise ContractError(
                f"Required sample file missing: {path}  "
                f"(sample_id={sid!r}, layer={layer!r})."
            )

    def _read(path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            return ds.read()

    img      = _read(paths["img"]).astype(np.float32)       # (C, H, W)
    extent   = _read(paths["extent"])[0].astype(np.uint8)   # (H, W)
    boundary = _read(paths["boundary"])[0].astype(np.uint8)
    distance = _read(paths["distance"])[0].astype(np.float32)
    valid    = _read(paths["valid"])[0].astype(np.uint8)

    # --- channel count check (DATA_CONTRACT.md §7.5) ---
    expected_ch = CHANNEL_COUNTS[feature_mode]
    if img.shape[0] != expected_ch:
        raise ChannelCountError(
            f"img has {img.shape[0]} channels but feature_mode='{feature_mode}' "
            f"expects {expected_ch}.  "
            "Verify that the dataset was prepared with the matching feature_mode "
            "(DATA_CONTRACT.md §7.5)."
        )

    # --- spatial shape consistency ---
    h, w = img.shape[1], img.shape[2]
    for name, arr in (
        ("extent",   extent),
        ("boundary", boundary),
        ("distance", distance),
        ("valid",    valid),
    ):
        if arr.shape != (h, w):
            raise ContractError(
                f"Spatial shape mismatch: img is ({h}, {w}) but '{name}' is "
                f"{arr.shape}.  All layers must share the same (H, W)."
            )

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))

    return {
        "img":       img,
        "extent":    extent,
        "boundary":  boundary,
        "distance":  distance,
        "valid":     valid,
        "meta":      meta,
        "sample_id": sample_id,
    }


def assemble_model_input(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Append valid as the final input channel to produce the assembled model input.

    Implements the assembled model input contract:
      raw8_valid     = raw8 (8ch) + valid (1ch) = 9ch
      raw8_idx3_valid = raw8_idx3 (11ch) + valid (1ch) = 12ch

    Parameters
    ----------
    img:
        (C, H, W) float32 — dataset-side feature stack.
    valid:
        (H, W) uint8 — valid mask (0=invalid, 1=valid).

    Returns
    -------
    (C+1, H, W) float32 — assembled model input with valid as the last channel.

    Contract
    --------
    DATA_CONTRACT.md §7.4, §6.4 (DEC-002):
        valid is both a service mask and an additional input channel.
        This function implements the input-channel role.
    """
    if img.ndim != 3:
        raise ContractError(
            f"img must be a 3-D (C, H, W) array, got {img.ndim}D with shape {img.shape}."
        )
    if valid.ndim != 2:
        raise ContractError(
            f"valid must be a 2-D (H, W) array, got {valid.ndim}D with shape {valid.shape}."
        )
    h, w = img.shape[1], img.shape[2]
    if valid.shape != (h, w):
        raise ContractError(
            f"valid shape {valid.shape} does not match img spatial dims ({h}, {w})."
        )

    valid_channel = valid[np.newaxis, ...].astype(np.float32)  # (1, H, W)
    return np.concatenate([img, valid_channel], axis=0)        # (C+1, H, W)


def _apply_spatial_augmentation(
    *,
    img: np.ndarray,
    extent: np.ndarray,
    boundary: np.ndarray,
    distance: np.ndarray,
    valid: np.ndarray,
    rng: "np.random.Generator | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Apply synchronized spatial augmentations to all sample layers.

    Augmentations applied (module_net_train.md §11.2 baseline):
      - horizontal flip (p=0.5)
      - vertical flip (p=0.5)
      - 90°-multiple rotation (k ∈ {0,1,2,3} uniform)

    All operations are applied identically to img and every target/mask layer
    so that spatial alignment is preserved.

    Parameters
    ----------
    img:
        (C, H, W) float32 — feature stack.
    extent:
        (H, W) int — extent target.
    boundary:
        (H, W) int — boundary target.
    distance:
        (H, W) float32 — distance target.
    valid:
        (H, W) uint8 — valid mask.
    rng:
        Optional numpy Generator; a fresh default_rng() is created if None.

    Returns
    -------
    Augmented (img, extent, boundary, distance, valid) — same dtypes as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    arrays_2d = [extent, boundary, distance, valid]

    if rng.integers(0, 2):  # horizontal flip (left/right)
        img = img[:, :, ::-1].copy()
        arrays_2d = [a[:, ::-1].copy() for a in arrays_2d]

    if rng.integers(0, 2):  # vertical flip (up/down)
        img = img[:, ::-1, :].copy()
        arrays_2d = [a[::-1, :].copy() for a in arrays_2d]

    k = int(rng.integers(0, 4))  # rotation: 0=none, 1=90°, 2=180°, 3=270°
    if k:
        img = np.rot90(img, k=k, axes=(1, 2)).copy()
        arrays_2d = [np.rot90(a, k=k, axes=(0, 1)).copy() for a in arrays_2d]

    return img, arrays_2d[0], arrays_2d[1], arrays_2d[2], arrays_2d[3]


# ---------------------------------------------------------------------------
# FieldsDataset — PyTorch Dataset (requires torch)
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TorchDataset = object  # type: ignore[misc,assignment]
    _TORCH_AVAILABLE = False


class FieldsDataset(_TorchDataset):  # type: ignore[misc]
    """PyTorch Dataset over a single split directory from module_prep_data.

    Reads train-ready samples, assembles the model input tensor
    (feature stack + valid channel), and returns the runtime sample dict
    required by the train loop.

    CONTRACT (module_net_train.md §10.2):
      The returned dict contains:
        'image'     — assembled model input (C+1, H, W) float32 tensor
        'extent'    — (H, W) int64 tensor
        'boundary'  — (H, W) int64 tensor
        'distance'  — (H, W) float32 tensor
        'valid'     — (H, W) bool tensor  ← kept separate for ignore policy
        'sample_id' — str
        'meta'      — dict

    valid is explicitly preserved as a separate key because it is both
    the ignore mask for loss/metrics AND an input channel (DEC-002).
    """

    def __init__(
        self,
        split_dir: Any,
        feature_mode: str,
        augment: bool = False,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ContractError(
                "torch is required for FieldsDataset.  "
                "Install torch or use the pure numpy read_sample() function."
            )
        if feature_mode not in FEATURE_MODES:
            raise FeatureModeError(
                f"Unsupported feature_mode: {feature_mode!r}.  "
                f"Expected one of {list(FEATURE_MODES)} (DATA_CONTRACT.md §7.1)."
            )

        self._split_dir = Path(split_dir)
        self._feature_mode = feature_mode
        self._augment = augment
        self._sample_ids = list_sample_ids(self._split_dir)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_mode(self) -> str:
        """Dataset-side feature mode."""
        return self._feature_mode

    @property
    def assembled_input_channels(self) -> int:
        """Number of channels in the assembled model input (feature channels + 1 for valid).

        raw8  → 9  (8 spectral + valid)
        raw8_idx3 → 12  (11 feature + valid)
        DATA_CONTRACT.md §7.5.
        """
        return CHANNEL_COUNTS[self._feature_mode] + 1

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self._sample_ids[idx]
        raw = read_sample(self._split_dir, sample_id, self._feature_mode)

        img, extent, boundary, distance, valid = (
            raw["img"], raw["extent"], raw["boundary"], raw["distance"], raw["valid"]
        )

        if self._augment:
            img, extent, boundary, distance, valid = _apply_spatial_augmentation(
                img=img, extent=extent, boundary=boundary,
                distance=distance, valid=valid,
            )

        # Assemble model input: feature stack + valid channel  (DEC-002)
        model_input = assemble_model_input(img, valid)  # (C+1, H, W)

        # torch.tensor() (unlike torch.from_numpy) always copies data into
        # PyTorch-owned ("resizable") storage.  This is required for DataLoader
        # workers with num_workers > 0: the IPC mechanism calls
        # storage.share_memory_() which fails on numpy-backed storage
        # ("Trying to resize storage that is not resizable").
        # np.ascontiguousarray() guards against non-contiguous arrays that
        # can arise from augmentation (e.g. rot90 without copy) or from
        # rasterio-backed views returned by astype() when dtype already matches.
        return {
            # assembled model input — what the model's forward() receives
            "image":     _torch.tensor(np.ascontiguousarray(model_input),    dtype=_torch.float32),

            # targets
            "extent":    _torch.tensor(np.ascontiguousarray(extent),         dtype=_torch.int64),
            "boundary":  _torch.tensor(np.ascontiguousarray(boundary),       dtype=_torch.int64),
            "distance":  _torch.tensor(np.ascontiguousarray(distance),       dtype=_torch.float32),

            # valid kept separately — ignore mask for loss/metrics/diagnostics
            "valid":     _torch.tensor(np.ascontiguousarray(valid),          dtype=_torch.bool),

            # provenance
            "sample_id": raw["sample_id"],
            "meta":      raw["meta"],
        }


def fields_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for FieldsDataset batches.

    Uses torch.stack() for the five tensor fields and leaves 'sample_id' and
    'meta' as plain Python lists, preventing default_collate from recursively
    tensor-ifying meta values.

    Why this is needed:
      default_collate recurses into the 'meta' dict and tries to convert every
      value it finds.  This can fail in two ways:
        1. Mixed types across samples (e.g. source_crs is None in some tiles
           and a string in others) → TypeError.
        2. Any numpy-derived scalar that slips in triggers torch.as_tensor()
           which may return numpy-backed (non-resizable) storage, causing
           "Trying to resize storage that is not resizable" in the worker's
           IPC share_memory_() call when num_workers > 0.

    This function must be passed as collate_fn= to every DataLoader that
    iterates a FieldsDataset.

    Returns
    -------
    dict with the same keys as FieldsDataset.__getitem__:
        'image'     — (B, C+1, H, W) float32 tensor
        'extent'    — (B, H, W) int64 tensor
        'boundary'  — (B, H, W) int64 tensor
        'distance'  — (B, H, W) float32 tensor
        'valid'     — (B, H, W) bool tensor
        'sample_id' — list[str]  (not a tensor)
        'meta'      — list[dict]  (not collated into tensors)
    """
    if not _TORCH_AVAILABLE:
        raise ContractError(
            "torch is required for fields_collate_fn.  "
            "Install torch or use the pure numpy read_sample() function."
        )
    return {
        "image":     _torch.stack([b["image"]    for b in batch]),
        "extent":    _torch.stack([b["extent"]   for b in batch]),
        "boundary":  _torch.stack([b["boundary"] for b in batch]),
        "distance":  _torch.stack([b["distance"] for b in batch]),
        "valid":     _torch.stack([b["valid"]    for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
        "meta":      [b["meta"]      for b in batch],
    }
