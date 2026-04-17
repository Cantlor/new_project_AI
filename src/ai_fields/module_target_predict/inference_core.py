"""Minimal predict-side inference core (single forward contract).

This layer intentionally stays narrow:
- load checkpoint payload and strict model weights;
- validate checkpoint/input contracts;
- run a single full-frame forward pass;
- validate output head contract.

It does NOT implement tiled inference, blending, output writing, or postprocess.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import (
    ChannelCountError,
    ContractError,
    FeatureModeError,
)
from ai_fields.module_net_train.model import EdgeAwareMultitaskNet
from ai_fields.module_target_predict.checkpoint_contract import (
    CheckpointDrivenPredictContract,
)
from ai_fields.module_target_predict.feature_adapter import PredictInputAdapterResult

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


_ARCHITECTURE_NAME = "edge_aware_multitask_v1"
_MODEL_SHAPE_SOURCE_METADATA = "checkpoint_metadata_explicit"
_MODEL_SHAPE_SOURCE_LEGACY_FALLBACK = "legacy_state_dict_fallback"
_REQUIRED_CHECKPOINT_FIELDS = (
    "feature_mode",
    "assembled_model_input",
    "in_channels",
    "valid_as_input_channel",
    "channel_semantics",
    "model_state_dict",
)


@dataclass(frozen=True)
class LoadedPredictModel:
    """Loaded model runtime for predict forward contract."""

    model: Any
    device: str
    in_channels: int
    encoder_depth: int
    base_channels: int
    architecture: str
    model_shape_source: str
    aux_head_count: int
    checkpoint_epochs_completed: int | None


@dataclass(frozen=True)
class PredictForwardResult:
    """Result of a single predict forward pass with contract validation."""

    input_shape: tuple[int, int, int, int]
    extent_shape: tuple[int, int, int, int]
    boundary_shape: tuple[int, int, int, int]
    distance_shape: tuple[int, int, int, int]
    aux_count: int
    device: str
    dtype: str

    extent_logits: np.ndarray
    extent_prob: np.ndarray
    boundary_logits: np.ndarray
    boundary_prob: np.ndarray
    distance_pred: np.ndarray
    forward_contract_ok: bool


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ContractError(
            "torch is required for module_target_predict inference core. Install torch to use this layer."
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


def _resolve_device(device: str | None) -> str:
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    if not isinstance(device, str) or device.strip() == "":
        raise ContractError("device must be a non-empty string or None.")
    if device not in {"cpu", "cuda", "mps"}:
        raise ContractError("device must be one of: 'cpu', 'cuda', 'mps'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ContractError("Requested device='cuda' but CUDA is not available.")
    if device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ContractError("Requested device='mps' but MPS is not available.")
    return device


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except OSError as exc:
        raise ContractError(f"Failed to read checkpoint from {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ContractError(
            f"Checkpoint payload must be a mapping/object, got {type(payload).__name__}."
        )
    return dict(payload)


def _infer_model_shape_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    expected_in_channels: int,
) -> tuple[int, int]:
    stem_key = "stem.conv.weight"
    if stem_key not in state_dict:
        raise ContractError(
            f"checkpoint model_state_dict is missing required key: {stem_key!r}."
        )
    stem_weight = state_dict[stem_key]
    if not torch.is_tensor(stem_weight) or stem_weight.ndim != 4:
        raise ContractError(
            "checkpoint model_state_dict['stem.conv.weight'] must be a 4-D tensor."
        )
    base_channels = int(stem_weight.shape[0])
    in_channels = int(stem_weight.shape[1])
    if in_channels != expected_in_channels:
        raise ChannelCountError(
            "checkpoint state_dict in_channels is inconsistent with checkpoint metadata "
            f"contract: expected {expected_in_channels}, got {in_channels}."
        )
    if base_channels < 1:
        raise ContractError(
            f"Inferred base_channels must be >= 1, got {base_channels}."
        )

    depth_indices: list[int] = []
    pattern = re.compile(r"^encoders\.(\d+)\.")
    for key in state_dict:
        m = pattern.match(str(key))
        if m is not None:
            depth_indices.append(int(m.group(1)))
    if not depth_indices:
        raise ContractError(
            "Unable to infer encoder_depth from checkpoint state_dict (no 'encoders.*' keys)."
        )
    encoder_depth = max(depth_indices) + 1
    if encoder_depth < 2:
        raise ContractError(
            f"Inferred encoder_depth must be >= 2, got {encoder_depth}."
        )
    return encoder_depth, base_channels


def _resolve_model_shape_from_checkpoint_metadata(
    checkpoint_contract: CheckpointDrivenPredictContract,
) -> tuple[str, int, int, str] | None:
    architecture = checkpoint_contract.model_architecture
    encoder_depth = checkpoint_contract.model_encoder_depth
    base_channels = checkpoint_contract.model_base_channels

    provided_flags = (
        architecture is not None,
        encoder_depth is not None,
        base_channels is not None,
    )
    if any(provided_flags) and not all(provided_flags):
        raise ContractError(
            "checkpoint metadata model architecture fields must be provided together: "
            "'model_architecture', 'encoder_depth', and 'base_channels'."
        )
    if not any(provided_flags):
        return None

    if architecture != _ARCHITECTURE_NAME:
        raise ContractError(
            "checkpoint metadata model_architecture is not supported by baseline predict core: "
            f"expected {_ARCHITECTURE_NAME!r}, got {architecture!r}."
        )
    if encoder_depth is None or encoder_depth < 2:
        raise ContractError(
            f"checkpoint metadata encoder_depth must be >= 2, got {encoder_depth!r}."
        )
    if base_channels is None or base_channels < 1:
        raise ContractError(
            f"checkpoint metadata base_channels must be >= 1, got {base_channels!r}."
        )

    return (
        architecture,
        int(encoder_depth),
        int(base_channels),
        _MODEL_SHAPE_SOURCE_METADATA,
    )


def _validate_checkpoint_payload_contract(
    payload: Mapping[str, Any],
    *,
    checkpoint_contract: CheckpointDrivenPredictContract,
) -> None:
    for field_name in _REQUIRED_CHECKPOINT_FIELDS:
        if field_name not in payload:
            raise ContractError(
                f"checkpoint payload is missing required field: {field_name!r}."
            )

    feature_mode = payload["feature_mode"]
    if feature_mode != checkpoint_contract.feature_mode:
        raise FeatureModeError(
            "checkpoint payload feature_mode is inconsistent with metadata contract: "
            f"{feature_mode!r} != {checkpoint_contract.feature_mode!r}."
        )

    assembled_input = payload["assembled_model_input"]
    if assembled_input != checkpoint_contract.assembled_model_input:
        raise ChannelCountError(
            "checkpoint payload assembled_model_input is inconsistent with metadata contract: "
            f"{assembled_input!r} != {checkpoint_contract.assembled_model_input!r}."
        )

    in_channels = payload["in_channels"]
    if (
        isinstance(in_channels, bool)
        or not isinstance(in_channels, int)
        or in_channels < 1
    ):
        raise ContractError(
            "checkpoint payload 'in_channels' must be an integer >= 1."
        )
    if in_channels != checkpoint_contract.in_channels:
        raise ChannelCountError(
            "checkpoint payload in_channels is inconsistent with metadata contract: "
            f"{in_channels} != {checkpoint_contract.in_channels}."
        )

    valid_as_input = payload["valid_as_input_channel"]
    if valid_as_input is not True:
        raise ContractError(
            "checkpoint payload valid_as_input_channel must be True for baseline contract."
        )

    channel_semantics = payload["channel_semantics"]
    if not isinstance(channel_semantics, (list, tuple)):
        raise ContractError(
            "checkpoint payload channel_semantics must be a sequence."
        )
    if tuple(channel_semantics) != tuple(checkpoint_contract.channel_semantics):
        raise ContractError(
            "checkpoint payload channel_semantics is inconsistent with metadata contract."
        )

    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ContractError(
            "checkpoint payload model_state_dict must be a mapping/object."
        )


def load_predict_model(
    *,
    checkpoint_contract: CheckpointDrivenPredictContract,
    device: str | None = None,
) -> LoadedPredictModel:
    """Load strict predict model from checkpoint payload."""

    _require_torch()
    if not isinstance(checkpoint_contract, CheckpointDrivenPredictContract):
        raise ContractError(
            "checkpoint_contract must be CheckpointDrivenPredictContract."
        )

    checkpoint_path = _normalize_existing_path(
        checkpoint_contract.checkpoint_path,
        name="checkpoint_contract.checkpoint_path",
    )
    payload = _load_checkpoint_payload(checkpoint_path)
    _validate_checkpoint_payload_contract(payload, checkpoint_contract=checkpoint_contract)

    state_dict_raw = payload["model_state_dict"]
    # torch.nn.Module.load_state_dict expects a mutable mapping-like object.
    state_dict = dict(state_dict_raw)
    shape_from_metadata = _resolve_model_shape_from_checkpoint_metadata(checkpoint_contract)
    if shape_from_metadata is not None:
        architecture_name, encoder_depth, base_channels, model_shape_source = (
            shape_from_metadata
        )
    else:
        encoder_depth, base_channels = _infer_model_shape_from_state_dict(
            state_dict,
            expected_in_channels=checkpoint_contract.in_channels,
        )
        architecture_name = _ARCHITECTURE_NAME
        model_shape_source = _MODEL_SHAPE_SOURCE_LEGACY_FALLBACK

    model = EdgeAwareMultitaskNet(
        in_channels=checkpoint_contract.in_channels,
        encoder_depth=encoder_depth,
        base_channels=base_channels,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        if model_shape_source == _MODEL_SHAPE_SOURCE_METADATA:
            raise ContractError(
                "checkpoint metadata architecture fields are inconsistent with "
                "checkpoint model_state_dict."
            ) from exc
        raise ContractError(
            "checkpoint model_state_dict is incompatible with resolved predict model contract."
        ) from exc

    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    model.eval()

    epochs_completed_raw = payload.get("epochs_completed")
    if epochs_completed_raw is not None:
        if (
            isinstance(epochs_completed_raw, bool)
            or not isinstance(epochs_completed_raw, int)
            or epochs_completed_raw < 0
        ):
            raise ContractError(
                "checkpoint payload epochs_completed must be an integer >= 0 or null."
            )
        epochs_completed = int(epochs_completed_raw)
    else:
        epochs_completed = None

    aux_head_count = len(getattr(model, "aux_heads", []))

    return LoadedPredictModel(
        model=model,
        device=resolved_device,
        in_channels=checkpoint_contract.in_channels,
        encoder_depth=encoder_depth,
        base_channels=base_channels,
        architecture=architecture_name,
        model_shape_source=model_shape_source,
        aux_head_count=aux_head_count,
        checkpoint_epochs_completed=epochs_completed,
    )


def _validate_forward_input_contract(
    *,
    predict_input: PredictInputAdapterResult,
    checkpoint_contract: CheckpointDrivenPredictContract,
    loaded_model: LoadedPredictModel,
) -> tuple[np.ndarray, int, int]:
    if not isinstance(predict_input, PredictInputAdapterResult):
        raise ContractError("predict_input must be PredictInputAdapterResult.")
    if not isinstance(checkpoint_contract, CheckpointDrivenPredictContract):
        raise ContractError(
            "checkpoint_contract must be CheckpointDrivenPredictContract."
        )
    if not isinstance(loaded_model, LoadedPredictModel):
        raise ContractError("loaded_model must be LoadedPredictModel.")

    if predict_input.input_ready_for_model is not True:
        raise ContractError(
            "predict_input.input_ready_for_model must be True before model forward."
        )
    if predict_input.feature_mode != checkpoint_contract.feature_mode:
        raise FeatureModeError(
            "predict_input.feature_mode is inconsistent with checkpoint contract: "
            f"{predict_input.feature_mode!r} != {checkpoint_contract.feature_mode!r}."
        )
    if predict_input.assembled_model_input != checkpoint_contract.assembled_model_input:
        raise ChannelCountError(
            "predict_input.assembled_model_input is inconsistent with checkpoint contract: "
            f"{predict_input.assembled_model_input!r} != {checkpoint_contract.assembled_model_input!r}."
        )
    if tuple(predict_input.channel_semantics) != tuple(checkpoint_contract.channel_semantics):
        raise ContractError(
            "predict_input.channel_semantics is inconsistent with checkpoint contract."
        )

    assembled = predict_input.assembled_input
    if not isinstance(assembled, np.ndarray):
        raise ContractError(
            f"predict_input.assembled_input must be numpy ndarray, got {type(assembled).__name__}."
        )
    if assembled.ndim != 3:
        raise ContractError(
            f"predict_input.assembled_input must be 3-D (C, H, W), got shape={assembled.shape}."
        )

    c, h, w = int(assembled.shape[0]), int(assembled.shape[1]), int(assembled.shape[2])
    if c != checkpoint_contract.in_channels:
        raise ChannelCountError(
            "predict input channel count is inconsistent with checkpoint contract: "
            f"expected {checkpoint_contract.in_channels}, got {c}."
        )
    if c != loaded_model.in_channels:
        raise ChannelCountError(
            "predict input channel count is inconsistent with loaded model: "
            f"expected {loaded_model.in_channels}, got {c}."
        )
    if h < 1 or w < 1:
        raise ContractError(f"predict input spatial shape must be positive, got {(h, w)}.")
    return assembled.astype(np.float32, copy=False), h, w


def _validate_forward_outputs(
    *,
    outputs: Any,
    expected_hw: tuple[int, int],
) -> tuple[Any, Any, Any, int]:
    if not isinstance(outputs, Mapping):
        raise ContractError(
            f"model forward output must be a mapping/object, got {type(outputs).__name__}."
        )

    for head in ("extent", "boundary", "distance"):
        if head not in outputs:
            raise ContractError(f"model forward output is missing required head: {head!r}.")

    extent = outputs["extent"]
    boundary = outputs["boundary"]
    distance = outputs["distance"]
    for head_name, tensor in (
        ("extent", extent),
        ("boundary", boundary),
        ("distance", distance),
    ):
        if not torch.is_tensor(tensor):
            raise ContractError(
                f"model output head {head_name!r} must be torch.Tensor, got {type(tensor).__name__}."
            )
        if tensor.ndim != 4:
            raise ContractError(
                f"model output head {head_name!r} must be 4-D (B, C, H, W), got shape={tuple(tensor.shape)}."
            )
        if int(tensor.shape[0]) != 1:
            raise ContractError(
                f"model output head {head_name!r} batch size must be 1 in single-forward mode, "
                f"got {int(tensor.shape[0])}."
            )
        if tuple(tensor.shape[2:]) != expected_hw:
            raise ContractError(
                f"model output head {head_name!r} spatial shape mismatch: "
                f"expected {expected_hw}, got {tuple(tensor.shape[2:])}."
            )

    if int(extent.shape[1]) != 1:
        raise ChannelCountError(
            f"extent output channel count must be 1, got {int(extent.shape[1])}."
        )
    if int(boundary.shape[1]) != 3:
        raise ChannelCountError(
            f"boundary output channel count must be 3, got {int(boundary.shape[1])}."
        )
    if int(distance.shape[1]) != 1:
        raise ChannelCountError(
            f"distance output channel count must be 1, got {int(distance.shape[1])}."
        )

    aux = outputs.get("aux", [])
    if aux is None:
        aux = []
    if not isinstance(aux, (list, tuple)):
        raise ContractError("model output 'aux' must be a list/tuple when present.")
    return extent, boundary, distance, int(len(aux))


def run_predict_forward(
    *,
    loaded_model: LoadedPredictModel,
    predict_input: PredictInputAdapterResult,
    checkpoint_contract: CheckpointDrivenPredictContract,
) -> PredictForwardResult:
    """Run a single predict forward pass with strict contract checks."""

    _require_torch()
    assembled, h, w = _validate_forward_input_contract(
        predict_input=predict_input,
        checkpoint_contract=checkpoint_contract,
        loaded_model=loaded_model,
    )

    input_tensor = torch.from_numpy(assembled).unsqueeze(0).to(
        device=loaded_model.device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        outputs = loaded_model.model(input_tensor)

    extent, boundary, distance, aux_count = _validate_forward_outputs(
        outputs=outputs,
        expected_hw=(h, w),
    )

    extent_prob = torch.sigmoid(extent)
    # Reduce 3-class boundary softmax to single-channel P(any boundary).
    # P(any_boundary) = 1 − P(background) = P(skeleton) + P(buffer).
    # DATA_CONTRACT.md §16.1: boundary_prob.tif is a single-channel raster.
    boundary_softmax = torch.softmax(boundary, dim=1)
    boundary_prob = 1.0 - boundary_softmax[:, 0:1, ...]  # keep (B,1,H,W) shape

    extent_logits_np = extent.detach().cpu().numpy()
    boundary_logits_np = boundary.detach().cpu().numpy()
    distance_pred_np = distance.detach().cpu().numpy()
    extent_prob_np = extent_prob.detach().cpu().numpy()
    boundary_prob_np = boundary_prob.detach().cpu().numpy()

    return PredictForwardResult(
        input_shape=tuple(int(v) for v in input_tensor.shape),
        extent_shape=tuple(int(v) for v in extent.shape),
        boundary_shape=tuple(int(v) for v in boundary.shape),
        distance_shape=tuple(int(v) for v in distance.shape),
        aux_count=aux_count,
        device=loaded_model.device,
        dtype=str(input_tensor.dtype),
        extent_logits=extent_logits_np,
        extent_prob=extent_prob_np,
        boundary_logits=boundary_logits_np,
        boundary_prob=boundary_prob_np,
        distance_pred=distance_pred_np,
        forward_contract_ok=True,
    )


__all__ = [
    "LoadedPredictModel",
    "PredictForwardResult",
    "load_predict_model",
    "run_predict_forward",
]
