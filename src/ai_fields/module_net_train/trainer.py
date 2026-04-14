"""Minimal train/eval loop layer for module_net_train (Stage D).

This module is intentionally narrow:
  - consumes an already-built model and MultitaskLoss-compatible interface;
  - runs train/eval steps and epoch-level aggregation;
  - keeps contract checks explicit and raises ContractError on violations.

It does not implement checkpoint export, metrics framework, or full run
orchestration. Those belong to later stages.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from ai_fields.common.errors import ContractError
from ai_fields.common.progress import progress_bar
from ai_fields.module_net_train.losses import EXTENT_IGNORE_LABEL

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ContractError(
            "torch is required for module_net_train trainer. Install torch to use this module."
        )


def _resolve_device(requested_device: str | None = None) -> "torch.device":
    """Resolve runtime device with CUDA -> MPS -> CPU fallback order."""
    _require_torch()
    if requested_device is not None:
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise ContractError(
                    "Requested training.device='cuda', but CUDA is not available."
                )
            return torch.device("cuda")
        if requested_device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise ContractError(
                    "Requested training.device='mps', but MPS is not available."
                )
            return torch.device("mps")
        if requested_device == "cpu":
            return torch.device("cpu")
        raise ContractError(
            "requested_device must be one of {'cuda', 'mps', 'cpu'} or null, "
            f"got {requested_device!r}."
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_runtime_execution(
    *,
    requested_device: str | None,
    amp_requested: bool,
) -> dict[str, Any]:
    """Resolve runtime execution policy for device and AMP usage."""
    _require_torch()
    if requested_device is not None and not isinstance(requested_device, str):
        raise ContractError(
            "requested_device must be a string or null, "
            f"got {type(requested_device).__name__}."
        )
    if not isinstance(amp_requested, bool):
        raise ContractError(
            f"amp_requested must be a boolean, got {type(amp_requested).__name__}."
        )

    resolved_device = _resolve_device(requested_device)
    amp_used = bool(amp_requested and resolved_device.type == "cuda")
    return {
        "device_requested": requested_device,
        "device_resolved": str(resolved_device),
        "amp_requested": amp_requested,
        "amp_used": amp_used,
        "resolved_device": resolved_device,
    }


def _autocast_context(*, amp_enabled: bool, device: "torch.device") -> Any:
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=amp_enabled)
    return nullcontext()


def build_optimizer(model: "torch.nn.Module", config: Any) -> "torch.optim.Optimizer":
    """Build baseline optimizer from validated NetTrainConfig (AdamW in v1)."""
    _require_torch()
    name = config.optimizer.name
    if name != "adamw":
        raise ContractError(
            f"Unsupported optimizer.name={name!r}. Stage D supports baseline AdamW only."
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
    )


def build_scheduler(
    optimizer: "torch.optim.Optimizer",
    config: Any,
    *,
    total_epochs: int | None = None,
) -> tuple[Any, str]:
    """Build minimal baseline scheduler and return (scheduler, step_policy).

    step_policy values:
      - "epoch_end": call scheduler.step() once per epoch.
      - "epoch_end_val_total": call scheduler.step(val_total_loss) once per epoch.
    """
    _require_torch()
    scheduler_cfg = getattr(config, "scheduler", None)
    if scheduler_cfg is None:
        raise ContractError("config.scheduler is required to build scheduler.")

    name = getattr(scheduler_cfg, "name", None)
    if not isinstance(name, str):
        raise ContractError("config.scheduler.name must be a string.")
    if total_epochs is None:
        total_epochs = getattr(getattr(config, "training", None), "num_epochs", None)
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs < 1:
        raise ContractError("total_epochs must be an integer >= 1 for scheduler setup.")

    min_lr = getattr(scheduler_cfg, "min_lr", None)
    if isinstance(min_lr, bool) or not isinstance(min_lr, (int, float)) or float(min_lr) < 0:
        raise ContractError("config.scheduler.min_lr must be a number >= 0.")
    min_lr_f = float(min_lr)

    if name == "cosine_with_warmup":
        warmup_epochs = getattr(scheduler_cfg, "warmup_epochs", None)
        if (
            isinstance(warmup_epochs, bool)
            or not isinstance(warmup_epochs, int)
            or warmup_epochs < 0
        ):
            raise ContractError("config.scheduler.warmup_epochs must be an integer >= 0.")
        if total_epochs > 1 and warmup_epochs >= total_epochs:
            # Short smoke-runs (e.g. epochs_override=1..3) should remain valid:
            # clamp warmup to keep at least one cosine epoch.
            warmup_epochs = max(0, total_epochs - 1)

        if warmup_epochs == 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_epochs),
                eta_min=min_lr_f,
            )
            return scheduler, "epoch_end"

        start_factor = max(1e-6, 1.0 / float(warmup_epochs + 1))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=min_lr_f,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return scheduler, "epoch_end"

    if name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=min_lr_f,
        )
        return scheduler, "epoch_end_val_total"

    if name == "one_cycle":
        raise ContractError(
            "scheduler.name='one_cycle' is not supported in the current run-level wiring. "
            "It requires per-batch scheduler stepping, which is outside this minimal stage."
        )

    raise ContractError(f"Unsupported scheduler.name={name!r}.")


def _require_batch_dict(batch: Any) -> Mapping[str, Any]:
    if not isinstance(batch, Mapping):
        raise ContractError(f"batch must be a mapping/object, got {type(batch).__name__}.")
    return batch


def _require_batch_tensor(batch: Mapping[str, Any], key: str) -> "torch.Tensor":
    if key not in batch:
        raise ContractError(f"batch is missing required key: {key!r}.")
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise ContractError(
            f"batch[{key!r}] must be a torch.Tensor, got {type(value).__name__}."
        )
    return value


def _require_finite_tensor(tensor: "torch.Tensor", *, name: str) -> None:
    if torch.isfinite(tensor).all():
        return
    n_nan = int(torch.isnan(tensor).sum().item())
    n_inf = int(torch.isinf(tensor).sum().item())
    raise ContractError(
        f"{name} contains non-finite values (nan={n_nan}, inf={n_inf})."
    )


def _prepare_batch(
    batch: Mapping[str, Any],
    *,
    device: "torch.device",
    input_normalizer: Any | None = None,
) -> tuple[dict[str, "torch.Tensor"], dict[str, "torch.Tensor"], "torch.Tensor", int]:
    image = _require_batch_tensor(batch, "image")
    extent = _require_batch_tensor(batch, "extent")
    boundary = _require_batch_tensor(batch, "boundary")
    distance = _require_batch_tensor(batch, "distance")
    valid = _require_batch_tensor(batch, "valid")

    if image.ndim != 4:
        raise ContractError(
            f"batch['image'] must have shape (B, C, H, W), got {tuple(image.shape)}."
        )
    if extent.ndim != 3 or boundary.ndim != 3 or distance.ndim != 3 or valid.ndim != 3:
        raise ContractError(
            "batch targets/mask must have shape (B, H, W) for keys: "
            "'extent', 'boundary', 'distance', 'valid'."
        )
    b, _, h, w = image.shape
    for name, tensor in (
        ("extent", extent),
        ("boundary", boundary),
        ("distance", distance),
        ("valid", valid),
    ):
        if tensor.shape[0] != b or tensor.shape[1] != h or tensor.shape[2] != w:
            raise ContractError(
                f"batch[{name!r}] shape {tuple(tensor.shape)} is incompatible with "
                f"batch['image'] shape {tuple(image.shape)}."
            )

    if valid.dtype != torch.bool:
        raise ContractError(
            f"batch['valid'] must be bool tensor, got dtype={valid.dtype}."
        )

    image_tensor = image.to(device=device, dtype=torch.float32, non_blocking=True)
    _require_finite_tensor(image_tensor, name="batch['image'] before normalization")
    if input_normalizer is not None:
        image_tensor = input_normalizer(image_tensor)
    _require_finite_tensor(image_tensor, name="batch['image'] after normalization")

    preds_input = {"image": image_tensor}
    targets = {
        "extent": extent.to(device=device, dtype=torch.long, non_blocking=True),
        "boundary": boundary.to(device=device, dtype=torch.long, non_blocking=True),
        "distance": distance.to(device=device, dtype=torch.float32, non_blocking=True),
    }
    valid_mask = valid.to(device=device, dtype=torch.bool, non_blocking=True)
    _require_finite_tensor(targets["distance"], name="batch['distance']")

    return preds_input, targets, valid_mask, int(b)


def _require_pred_heads(preds: Mapping[str, Any], *, where: str) -> None:
    for key in ("extent", "boundary", "distance"):
        if key not in preds:
            raise ContractError(f"{where} is missing required prediction key: {key!r}.")
        if not isinstance(preds[key], torch.Tensor):
            raise ContractError(
                f"{where}[{key!r}] must be a torch.Tensor, got {type(preds[key]).__name__}."
            )


def _binary_counts(
    *,
    pred_positive: "torch.Tensor",
    target_positive: "torch.Tensor",
    mask: "torch.Tensor",
) -> tuple[int, int, int]:
    """Return TP/FP/FN counts for masked binary F1 computation."""
    effective_mask = mask.bool()
    pred_pos = pred_positive.bool()
    target_pos = target_positive.bool()

    tp = int((pred_pos & target_pos & effective_mask).sum().item())
    fp = int((pred_pos & (~target_pos) & effective_mask).sum().item())
    fn = int(((~pred_pos) & target_pos & effective_mask).sum().item())
    return tp, fp, fn


def _precision_from_counts(*, tp: int, fp: int) -> float:
    denom = tp + fp
    if denom <= 0:
        return 0.0
    return float(tp) / float(denom)


def _recall_from_counts(*, tp: int, fn: int) -> float:
    denom = tp + fn
    if denom <= 0:
        return 0.0
    return float(tp) / float(denom)


def _f1_from_counts(*, tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    if denom <= 0:
        return 0.0
    return float((2.0 * tp) / float(denom))


def _iou_from_counts(*, tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    if denom <= 0:
        return 0.0
    return float(tp) / float(denom)


def _macro2(a: float, b: float) -> float:
    return 0.5 * (float(a) + float(b))


def _compute_batch_metric_counts(
    *,
    preds: Mapping[str, "torch.Tensor"],
    targets: Mapping[str, "torch.Tensor"],
    valid_mask: "torch.Tensor",
) -> dict[str, int]:
    """Compute valid-aware segmentation counts for extent and boundary classes."""
    extent_logits = preds["extent"]
    if extent_logits.dim() == 4 and extent_logits.shape[1] == 1:
        extent_logits = extent_logits.squeeze(1)
    extent_pred_positive = extent_logits.detach() >= 0.0
    extent_target_positive = targets["extent"] == 1
    extent_effective_mask = valid_mask & (targets["extent"] != EXTENT_IGNORE_LABEL)
    extent_tp, extent_fp, extent_fn = _binary_counts(
        pred_positive=extent_pred_positive,
        target_positive=extent_target_positive,
        mask=extent_effective_mask,
    )

    boundary_logits = preds["boundary"]
    if boundary_logits.dim() != 4 or boundary_logits.shape[1] < 2:
        raise ContractError(
            "boundary prediction tensor must have shape (B, C, H, W) with C>=2."
        )

    boundary_pred = boundary_logits.detach().argmax(dim=1)
    boundary_target = targets["boundary"]

    boundary_any_tp, boundary_any_fp, boundary_any_fn = _binary_counts(
        pred_positive=boundary_pred != 0,
        target_positive=boundary_target != 0,
        mask=valid_mask,
    )
    boundary_skeleton_tp, boundary_skeleton_fp, boundary_skeleton_fn = _binary_counts(
        pred_positive=boundary_pred == 1,
        target_positive=boundary_target == 1,
        mask=valid_mask,
    )
    boundary_buffer_tp, boundary_buffer_fp, boundary_buffer_fn = _binary_counts(
        pred_positive=boundary_pred == 2,
        target_positive=boundary_target == 2,
        mask=valid_mask,
    )

    return {
        "extent_tp": extent_tp,
        "extent_fp": extent_fp,
        "extent_fn": extent_fn,
        "boundary_any_tp": boundary_any_tp,
        "boundary_any_fp": boundary_any_fp,
        "boundary_any_fn": boundary_any_fn,
        "boundary_skeleton_tp": boundary_skeleton_tp,
        "boundary_skeleton_fp": boundary_skeleton_fp,
        "boundary_skeleton_fn": boundary_skeleton_fn,
        "boundary_buffer_tp": boundary_buffer_tp,
        "boundary_buffer_fp": boundary_buffer_fp,
        "boundary_buffer_fn": boundary_buffer_fn,
    }


def _compute_losses(
    model: "torch.nn.Module",
    loss_fn: Any,
    batch: Mapping[str, Any],
    *,
    device: "torch.device",
    amp_enabled: bool,
    aux_weight: float,
    input_normalizer: Any | None = None,
) -> tuple[dict[str, "torch.Tensor"], int, int, int, dict[str, int]]:
    model_inputs, targets, valid_mask, batch_size = _prepare_batch(
        batch,
        device=device,
        input_normalizer=input_normalizer,
    )

    with _autocast_context(amp_enabled=amp_enabled, device=device):
        model_out = model(model_inputs["image"])
        if not isinstance(model_out, Mapping):
            raise ContractError(
                f"model output must be a mapping/object, got {type(model_out).__name__}."
            )
        _require_pred_heads(model_out, where="model output")
        main_preds = {
            "extent": model_out["extent"],
            "boundary": model_out["boundary"],
            "distance": model_out["distance"],
        }
        _require_finite_tensor(main_preds["extent"], name="model output extent")
        _require_finite_tensor(main_preds["boundary"], name="model output boundary")
        _require_finite_tensor(main_preds["distance"], name="model output distance")
        main_losses = loss_fn(main_preds, targets, valid_mask)
        for key in ("extent", "boundary", "distance", "total", "n_valid"):
            if key not in main_losses:
                raise ContractError(f"loss_fn output is missing required key: {key!r}.")

        extent_loss = main_losses["extent"]
        boundary_loss = main_losses["boundary"]
        distance_loss = main_losses["distance"]
        total_loss = main_losses["total"]
        _require_finite_tensor(extent_loss, name="extent loss")
        _require_finite_tensor(boundary_loss, name="boundary loss")
        _require_finite_tensor(distance_loss, name="distance loss")
        _require_finite_tensor(total_loss, name="total loss")

        aux_losses_total = total_loss.new_zeros(())
        n_aux = 0
        aux_raw = model_out.get("aux", [])
        if aux_raw is None:
            aux_raw = []
        if isinstance(aux_raw, list):
            n_aux = len(aux_raw)
            if n_aux > 0 and aux_weight > 0:
                for idx, aux_preds in enumerate(aux_raw):
                    if not isinstance(aux_preds, Mapping):
                        raise ContractError(
                            f"model output aux[{idx}] must be a mapping/object, got "
                            f"{type(aux_preds).__name__}."
                        )
                    _require_pred_heads(aux_preds, where=f"model output aux[{idx}]")
                    aux_res = loss_fn(
                        {
                            "extent": aux_preds["extent"],
                            "boundary": aux_preds["boundary"],
                            "distance": aux_preds["distance"],
                        },
                        targets,
                        valid_mask,
                    )
                    _require_finite_tensor(aux_res["total"], name=f"aux[{idx}] total loss")
                    w = float(aux_weight)
                    extent_loss = extent_loss + w * aux_res["extent"]
                    boundary_loss = boundary_loss + w * aux_res["boundary"]
                    distance_loss = distance_loss + w * aux_res["distance"]
                    weighted_aux_total = w * aux_res["total"]
                    total_loss = total_loss + weighted_aux_total
                    aux_losses_total = aux_losses_total + weighted_aux_total
        else:
            raise ContractError(
                f"model output 'aux' must be a list (or null), got {type(aux_raw).__name__}."
            )

    metric_counts = _compute_batch_metric_counts(preds=main_preds, targets=targets, valid_mask=valid_mask)
    n_valid = int(main_losses["n_valid"])
    return (
        {
            "extent": extent_loss,
            "boundary": boundary_loss,
            "distance": distance_loss,
            "total": total_loss,
            "aux_total": aux_losses_total,
        },
        n_valid,
        batch_size,
        n_aux,
        metric_counts,
    )


def _loss_tensors_to_result(
    *,
    losses: Mapping[str, "torch.Tensor"],
    n_valid: int,
    n_batches: int,
    n_samples: int,
    n_aux: int,
    metric_counts: Mapping[str, int],
) -> dict[str, Any]:
    extent_tp = int(metric_counts["extent_tp"])
    extent_fp = int(metric_counts["extent_fp"])
    extent_fn = int(metric_counts["extent_fn"])

    boundary_skel_tp = int(metric_counts["boundary_skeleton_tp"])
    boundary_skel_fp = int(metric_counts["boundary_skeleton_fp"])
    boundary_skel_fn = int(metric_counts["boundary_skeleton_fn"])
    boundary_buf_tp = int(metric_counts["boundary_buffer_tp"])
    boundary_buf_fp = int(metric_counts["boundary_buffer_fp"])
    boundary_buf_fn = int(metric_counts["boundary_buffer_fn"])
    boundary_any_tp = int(metric_counts["boundary_any_tp"])
    boundary_any_fp = int(metric_counts["boundary_any_fp"])
    boundary_any_fn = int(metric_counts["boundary_any_fn"])

    extent_precision = _precision_from_counts(tp=extent_tp, fp=extent_fp)
    extent_recall = _recall_from_counts(tp=extent_tp, fn=extent_fn)
    extent_iou = _iou_from_counts(tp=extent_tp, fp=extent_fp, fn=extent_fn)
    extent_f1 = _f1_from_counts(tp=extent_tp, fp=extent_fp, fn=extent_fn)

    boundary_skel_precision = _precision_from_counts(tp=boundary_skel_tp, fp=boundary_skel_fp)
    boundary_skel_recall = _recall_from_counts(tp=boundary_skel_tp, fn=boundary_skel_fn)
    boundary_skel_f1 = _f1_from_counts(tp=boundary_skel_tp, fp=boundary_skel_fp, fn=boundary_skel_fn)
    boundary_buf_precision = _precision_from_counts(tp=boundary_buf_tp, fp=boundary_buf_fp)
    boundary_buf_recall = _recall_from_counts(tp=boundary_buf_tp, fn=boundary_buf_fn)
    boundary_buf_f1 = _f1_from_counts(tp=boundary_buf_tp, fp=boundary_buf_fp, fn=boundary_buf_fn)
    boundary_any_precision = _precision_from_counts(tp=boundary_any_tp, fp=boundary_any_fp)
    boundary_any_recall = _recall_from_counts(tp=boundary_any_tp, fn=boundary_any_fn)
    boundary_any_f1 = _f1_from_counts(tp=boundary_any_tp, fp=boundary_any_fp, fn=boundary_any_fn)

    return {
        "extent": float(losses["extent"].detach().item()),
        "boundary": float(losses["boundary"].detach().item()),
        "distance": float(losses["distance"].detach().item()),
        "total": float(losses["total"].detach().item()),
        "aux_total": float(losses["aux_total"].detach().item()),
        "n_valid": int(n_valid),
        "n_batches": int(n_batches),
        "n_samples": int(n_samples),
        "n_aux": int(n_aux),
        "extent_precision": extent_precision,
        "extent_recall": extent_recall,
        "extent_iou": extent_iou,
        "extent_f1": extent_f1,
        "boundary_precision": _macro2(boundary_skel_precision, boundary_buf_precision),
        "boundary_recall": _macro2(boundary_skel_recall, boundary_buf_recall),
        "boundary_f1": _macro2(boundary_skel_f1, boundary_buf_f1),
        "boundary_skeleton_precision": boundary_skel_precision,
        "boundary_skeleton_recall": boundary_skel_recall,
        "boundary_skeleton_f1": boundary_skel_f1,
        "boundary_buffer_precision": boundary_buf_precision,
        "boundary_buffer_recall": boundary_buf_recall,
        "boundary_buffer_f1": boundary_buf_f1,
        "boundary_any_precision": boundary_any_precision,
        "boundary_any_recall": boundary_any_recall,
        "boundary_any_f1": boundary_any_f1,
    }


def train_step(
    model: "torch.nn.Module",
    batch: Mapping[str, Any],
    loss_fn: Any,
    optimizer: "torch.optim.Optimizer",
    *,
    device: "torch.device | str | None" = None,
    amp_enabled: bool = False,
    aux_weight: float = 0.0,
    gradient_clip_norm: float | None = None,
    input_normalizer: Any | None = None,
) -> dict[str, Any]:
    """Run one optimization step for a single batch."""
    _require_torch()
    resolved_device = _resolve_device(None if device is None else str(device))
    model.to(resolved_device)
    model.train()

    optimizer.zero_grad(set_to_none=True)
    losses, n_valid, batch_size, n_aux, metric_counts = _compute_losses(
        model,
        loss_fn,
        _require_batch_dict(batch),
        device=resolved_device,
        amp_enabled=amp_enabled,
        aux_weight=aux_weight,
        input_normalizer=input_normalizer,
    )
    losses["total"].backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
    optimizer.step()

    return _loss_tensors_to_result(
        losses=losses,
        n_valid=n_valid,
        n_batches=1,
        n_samples=batch_size,
        n_aux=n_aux,
        metric_counts=metric_counts,
    )


def train_one_epoch(
    model: "torch.nn.Module",
    dataloader: Any,
    loss_fn: Any,
    optimizer: "torch.optim.Optimizer",
    *,
    device: "torch.device | str | None" = None,
    amp_enabled: bool = False,
    aux_weight: float = 0.0,
    gradient_clip_norm: float | None = None,
    gradient_accumulation_steps: int = 1,
    progress_enabled: bool | None = None,
    input_normalizer: Any | None = None,
) -> dict[str, Any]:
    """Run one training epoch and return aggregated loss summary."""
    _require_torch()
    if (
        isinstance(gradient_accumulation_steps, bool)
        or not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps < 1
    ):
        raise ContractError(
            "gradient_accumulation_steps must be an integer >= 1."
        )

    resolved_device = _resolve_device(None if device is None else str(device))
    model.to(resolved_device)
    model.train()

    optimizer.zero_grad(set_to_none=True)

    agg = {
        "extent": 0.0,
        "boundary": 0.0,
        "distance": 0.0,
        "total": 0.0,
        "aux_total": 0.0,
    }
    total_valid = 0
    total_samples = 0
    n_batches = 0
    n_aux = 0
    metric_counts = {
        "extent_tp": 0,
        "extent_fp": 0,
        "extent_fn": 0,
        "boundary_any_tp": 0,
        "boundary_any_fp": 0,
        "boundary_any_fn": 0,
        "boundary_skeleton_tp": 0,
        "boundary_skeleton_fp": 0,
        "boundary_skeleton_fn": 0,
        "boundary_buffer_tp": 0,
        "boundary_buffer_fp": 0,
        "boundary_buffer_fn": 0,
    }

    _dl_len = getattr(dataloader, "__len__", None)
    dl_total = int(_dl_len()) if callable(_dl_len) else None

    with progress_bar(
        total=dl_total,
        desc="  train",
        unit="batch",
        progress_enabled=progress_enabled,
        leave=False,
    ) as bar:
        for step_idx, batch in enumerate(dataloader, start=1):
            losses, n_valid, batch_size, batch_aux_count, batch_metric_counts = _compute_losses(
                model,
                loss_fn,
                _require_batch_dict(batch),
                device=resolved_device,
                amp_enabled=amp_enabled,
                aux_weight=aux_weight,
                input_normalizer=input_normalizer,
            )
            if step_idx == 1:
                n_aux = batch_aux_count

            (losses["total"] / float(gradient_accumulation_steps)).backward()

            do_step = (step_idx % gradient_accumulation_steps) == 0
            if do_step:
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float(gradient_clip_norm)
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            n_batches += 1
            total_valid += n_valid
            total_samples += batch_size
            for key in ("extent", "boundary", "distance", "total", "aux_total"):
                agg[key] += float(losses[key].detach().item())
            for key in (
                "extent_tp",
                "extent_fp",
                "extent_fn",
                "boundary_any_tp",
                "boundary_any_fp",
                "boundary_any_fn",
                "boundary_skeleton_tp",
                "boundary_skeleton_fp",
                "boundary_skeleton_fn",
                "boundary_buffer_tp",
                "boundary_buffer_fp",
                "boundary_buffer_fn",
            ):
                metric_counts[key] += int(batch_metric_counts[key])

            bar.update(1)
            bar.set_postfix(loss=f"{agg['total'] / n_batches:.4f}")

    if n_batches == 0:
        raise ContractError("train_one_epoch received an empty dataloader.")

    if (n_batches % gradient_accumulation_steps) != 0:
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(gradient_clip_norm)
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    extent_tp = metric_counts["extent_tp"]
    extent_fp = metric_counts["extent_fp"]
    extent_fn = metric_counts["extent_fn"]
    boundary_skel_tp = metric_counts["boundary_skeleton_tp"]
    boundary_skel_fp = metric_counts["boundary_skeleton_fp"]
    boundary_skel_fn = metric_counts["boundary_skeleton_fn"]
    boundary_buf_tp = metric_counts["boundary_buffer_tp"]
    boundary_buf_fp = metric_counts["boundary_buffer_fp"]
    boundary_buf_fn = metric_counts["boundary_buffer_fn"]
    boundary_any_tp = metric_counts["boundary_any_tp"]
    boundary_any_fp = metric_counts["boundary_any_fp"]
    boundary_any_fn = metric_counts["boundary_any_fn"]

    extent_precision = _precision_from_counts(tp=extent_tp, fp=extent_fp)
    extent_recall = _recall_from_counts(tp=extent_tp, fn=extent_fn)
    extent_iou = _iou_from_counts(tp=extent_tp, fp=extent_fp, fn=extent_fn)
    extent_f1 = _f1_from_counts(tp=extent_tp, fp=extent_fp, fn=extent_fn)
    boundary_skel_precision = _precision_from_counts(tp=boundary_skel_tp, fp=boundary_skel_fp)
    boundary_skel_recall = _recall_from_counts(tp=boundary_skel_tp, fn=boundary_skel_fn)
    boundary_skel_f1 = _f1_from_counts(tp=boundary_skel_tp, fp=boundary_skel_fp, fn=boundary_skel_fn)
    boundary_buf_precision = _precision_from_counts(tp=boundary_buf_tp, fp=boundary_buf_fp)
    boundary_buf_recall = _recall_from_counts(tp=boundary_buf_tp, fn=boundary_buf_fn)
    boundary_buf_f1 = _f1_from_counts(tp=boundary_buf_tp, fp=boundary_buf_fp, fn=boundary_buf_fn)
    boundary_any_precision = _precision_from_counts(tp=boundary_any_tp, fp=boundary_any_fp)
    boundary_any_recall = _recall_from_counts(tp=boundary_any_tp, fn=boundary_any_fn)
    boundary_any_f1 = _f1_from_counts(tp=boundary_any_tp, fp=boundary_any_fp, fn=boundary_any_fn)

    return {
        "extent": agg["extent"] / n_batches,
        "boundary": agg["boundary"] / n_batches,
        "distance": agg["distance"] / n_batches,
        "total": agg["total"] / n_batches,
        "aux_total": agg["aux_total"] / n_batches,
        "n_valid": total_valid,
        "n_batches": n_batches,
        "n_samples": total_samples,
        "n_aux": n_aux,
        "extent_precision": extent_precision,
        "extent_recall": extent_recall,
        "extent_iou": extent_iou,
        "extent_f1": extent_f1,
        "boundary_precision": _macro2(boundary_skel_precision, boundary_buf_precision),
        "boundary_recall": _macro2(boundary_skel_recall, boundary_buf_recall),
        "boundary_f1": _macro2(boundary_skel_f1, boundary_buf_f1),
        "boundary_skeleton_precision": boundary_skel_precision,
        "boundary_skeleton_recall": boundary_skel_recall,
        "boundary_skeleton_f1": boundary_skel_f1,
        "boundary_buffer_precision": boundary_buf_precision,
        "boundary_buffer_recall": boundary_buf_recall,
        "boundary_buffer_f1": boundary_buf_f1,
        "boundary_any_precision": boundary_any_precision,
        "boundary_any_recall": boundary_any_recall,
        "boundary_any_f1": boundary_any_f1,
    }


def evaluate_one_epoch(
    model: "torch.nn.Module",
    dataloader: Any,
    loss_fn: Any,
    *,
    device: "torch.device | str | None" = None,
    amp_enabled: bool = False,
    aux_weight: float = 0.0,
    progress_enabled: bool | None = None,
    input_normalizer: Any | None = None,
) -> dict[str, Any]:
    """Run one eval epoch (no optimizer step) and return aggregated loss summary."""
    _require_torch()
    resolved_device = _resolve_device(None if device is None else str(device))
    model.to(resolved_device)
    model.eval()

    agg = {
        "extent": 0.0,
        "boundary": 0.0,
        "distance": 0.0,
        "total": 0.0,
        "aux_total": 0.0,
    }
    total_valid = 0
    total_samples = 0
    n_batches = 0
    n_aux = 0
    metric_counts = {
        "extent_tp": 0,
        "extent_fp": 0,
        "extent_fn": 0,
        "boundary_any_tp": 0,
        "boundary_any_fp": 0,
        "boundary_any_fn": 0,
        "boundary_skeleton_tp": 0,
        "boundary_skeleton_fp": 0,
        "boundary_skeleton_fn": 0,
        "boundary_buffer_tp": 0,
        "boundary_buffer_fp": 0,
        "boundary_buffer_fn": 0,
    }

    _dl_len = getattr(dataloader, "__len__", None)
    dl_total = int(_dl_len()) if callable(_dl_len) else None

    with torch.no_grad(), progress_bar(
        total=dl_total,
        desc="    val",
        unit="batch",
        progress_enabled=progress_enabled,
        leave=False,
    ) as bar:
        for batch in dataloader:
            losses, n_valid, batch_size, batch_aux_count, batch_metric_counts = _compute_losses(
                model,
                loss_fn,
                _require_batch_dict(batch),
                device=resolved_device,
                amp_enabled=amp_enabled,
                aux_weight=aux_weight,
                input_normalizer=input_normalizer,
            )
            if n_batches == 0:
                n_aux = batch_aux_count

            n_batches += 1
            total_valid += n_valid
            total_samples += batch_size
            for key in ("extent", "boundary", "distance", "total", "aux_total"):
                agg[key] += float(losses[key].detach().item())
            for key in (
                "extent_tp",
                "extent_fp",
                "extent_fn",
                "boundary_any_tp",
                "boundary_any_fp",
                "boundary_any_fn",
                "boundary_skeleton_tp",
                "boundary_skeleton_fp",
                "boundary_skeleton_fn",
                "boundary_buffer_tp",
                "boundary_buffer_fp",
                "boundary_buffer_fn",
            ):
                metric_counts[key] += int(batch_metric_counts[key])

            bar.update(1)
            bar.set_postfix(loss=f"{agg['total'] / n_batches:.4f}")

    if n_batches == 0:
        raise ContractError("evaluate_one_epoch received an empty dataloader.")

    extent_tp = metric_counts["extent_tp"]
    extent_fp = metric_counts["extent_fp"]
    extent_fn = metric_counts["extent_fn"]
    boundary_skel_tp = metric_counts["boundary_skeleton_tp"]
    boundary_skel_fp = metric_counts["boundary_skeleton_fp"]
    boundary_skel_fn = metric_counts["boundary_skeleton_fn"]
    boundary_buf_tp = metric_counts["boundary_buffer_tp"]
    boundary_buf_fp = metric_counts["boundary_buffer_fp"]
    boundary_buf_fn = metric_counts["boundary_buffer_fn"]
    boundary_any_tp = metric_counts["boundary_any_tp"]
    boundary_any_fp = metric_counts["boundary_any_fp"]
    boundary_any_fn = metric_counts["boundary_any_fn"]

    extent_precision = _precision_from_counts(tp=extent_tp, fp=extent_fp)
    extent_recall = _recall_from_counts(tp=extent_tp, fn=extent_fn)
    extent_iou = _iou_from_counts(tp=extent_tp, fp=extent_fp, fn=extent_fn)
    extent_f1 = _f1_from_counts(tp=extent_tp, fp=extent_fp, fn=extent_fn)
    boundary_skel_precision = _precision_from_counts(tp=boundary_skel_tp, fp=boundary_skel_fp)
    boundary_skel_recall = _recall_from_counts(tp=boundary_skel_tp, fn=boundary_skel_fn)
    boundary_skel_f1 = _f1_from_counts(tp=boundary_skel_tp, fp=boundary_skel_fp, fn=boundary_skel_fn)
    boundary_buf_precision = _precision_from_counts(tp=boundary_buf_tp, fp=boundary_buf_fp)
    boundary_buf_recall = _recall_from_counts(tp=boundary_buf_tp, fn=boundary_buf_fn)
    boundary_buf_f1 = _f1_from_counts(tp=boundary_buf_tp, fp=boundary_buf_fp, fn=boundary_buf_fn)
    boundary_any_precision = _precision_from_counts(tp=boundary_any_tp, fp=boundary_any_fp)
    boundary_any_recall = _recall_from_counts(tp=boundary_any_tp, fn=boundary_any_fn)
    boundary_any_f1 = _f1_from_counts(tp=boundary_any_tp, fp=boundary_any_fp, fn=boundary_any_fn)

    return {
        "extent": agg["extent"] / n_batches,
        "boundary": agg["boundary"] / n_batches,
        "distance": agg["distance"] / n_batches,
        "total": agg["total"] / n_batches,
        "aux_total": agg["aux_total"] / n_batches,
        "n_valid": total_valid,
        "n_batches": n_batches,
        "n_samples": total_samples,
        "n_aux": n_aux,
        "extent_precision": extent_precision,
        "extent_recall": extent_recall,
        "extent_iou": extent_iou,
        "extent_f1": extent_f1,
        "boundary_precision": _macro2(boundary_skel_precision, boundary_buf_precision),
        "boundary_recall": _macro2(boundary_skel_recall, boundary_buf_recall),
        "boundary_f1": _macro2(boundary_skel_f1, boundary_buf_f1),
        "boundary_skeleton_precision": boundary_skel_precision,
        "boundary_skeleton_recall": boundary_skel_recall,
        "boundary_skeleton_f1": boundary_skel_f1,
        "boundary_buffer_precision": boundary_buf_precision,
        "boundary_buffer_recall": boundary_buf_recall,
        "boundary_buffer_f1": boundary_buf_f1,
        "boundary_any_precision": boundary_any_precision,
        "boundary_any_recall": boundary_any_recall,
        "boundary_any_f1": boundary_any_f1,
    }
    
