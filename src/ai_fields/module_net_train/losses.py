"""Loss functions for module_net_train.

Implements the three-head loss contract from module_net_train.md §8 and §22:

  - extent_loss:    focal BCE + soft Dice  (§8.2)
  - boundary_loss:  class-weighted focal CE + λ·soft Dice (skeleton class)  (§8.3, §22.1)
  - distance_loss:  SmoothL1  (§8.4, §22.2)
  - MultitaskLoss:  combined weighted sum with configurable weights

Ignore policy  (module_net_train.md §9, §19):
  - valid == 0 pixels excluded from ALL head-level loss computations.
  - Extent ignore label (EXTENT_IGNORE_LABEL = 255) additionally excluded from extent.
  - Boundary target uses 3 classes (0=background, 1=skeleton, 2=buffer) — no separate
    boundary ignore label in baseline v1; the mask is driven by valid only.
  - If the effective valid region is completely empty, head-level losses and
    MultitaskLoss.forward() raise ContractError (§19).

Low-level helper functions (_safe_mean, focal_bce_loss, etc.) may return 0.0 on empty
masks so they remain composable; ContractError enforcement lives at the head level.

Requires torch.  ContractError is raised if torch is unavailable.
"""

from __future__ import annotations

from typing import Any

from ai_fields.common.errors import ContractError

# Extent target value that marks ignore zones.
# Source: module_prep_data/targets_compute.py  _IGNORE_LABEL = 255
# DATA_CONTRACT.md §8.2: ignore-policy is mandatory for extent.
EXTENT_IGNORE_LABEL: int = 255

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ContractError(
            "torch is required for loss computation in module_net_train.  "
            "Install torch to use this module."
        )


# ---------------------------------------------------------------------------
# Shared helpers  (may return 0.0 on empty mask — head level enforces ContractError)
# ---------------------------------------------------------------------------


def _check_nonempty_valid(valid: "torch.Tensor", context: str) -> None:
    """Raise ContractError if no valid pixels (module_net_train.md §19).

    Called at head level — not in low-level helpers.
    """
    if valid.sum() == 0:
        raise ContractError(
            f"Loss aborted: valid region is completely empty in {context}.  "
            "All pixels are masked out.  This indicates a data pipeline issue or "
            "a fully-invalid patch that should have been skipped "
            "(module_net_train.md §19)."
        )


def _safe_mean(values: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """Mean of values[mask].  Returns 0.0 scalar if mask is all False."""
    n = mask.sum()
    if n == 0:
        return values.new_zeros(())
    return values[mask].mean()


# ---------------------------------------------------------------------------
# Extent loss:  focal BCE  +  soft Dice
# ---------------------------------------------------------------------------


def focal_bce_loss(
    pred_logits: "torch.Tensor",
    target_binary: "torch.Tensor",
    mask: "torch.Tensor",
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> "torch.Tensor":
    """Focal Binary Cross-Entropy.

    Parameters
    ----------
    pred_logits: (B, H, W) float32  (squeeze channel dim before calling)
    target_binary: (B, H, W) int64 or float32  —  0 or 1 only (ignore pixels already removed by mask)
    mask: (B, H, W) bool  —  True where loss is computed
    gamma: focal exponent (default 2.0)
    alpha: foreground balance weight (default 0.25)

    Returns
    -------
    Masked mean scalar.  0.0 if mask is empty.
    """
    _require_torch()
    target_f = target_binary.float()
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_f, reduction="none")

    p = torch.sigmoid(pred_logits)
    p_t = p * target_f + (1.0 - p) * (1.0 - target_f)
    alpha_t = alpha * target_f + (1.0 - alpha) * (1.0 - target_f)
    focal = alpha_t * (1.0 - p_t).pow(gamma) * bce

    return _safe_mean(focal, mask)


def soft_dice_loss_binary(
    pred_logits: "torch.Tensor",
    target_binary: "torch.Tensor",
    mask: "torch.Tensor",
    smooth: float = 1.0,
) -> "torch.Tensor":
    """Soft Dice loss for binary segmentation.

    Parameters
    ----------
    pred_logits: (B, H, W) float32
    target_binary: (B, H, W) 0/1
    mask: (B, H, W) bool
    smooth: Laplace smoothing

    Returns
    -------
    Scalar Dice loss in [0, 1].  0.0 if mask is empty.
    """
    _require_torch()
    p = torch.sigmoid(pred_logits)
    p_flat = p[mask].float()
    t_flat = target_binary.float()[mask]

    if p_flat.numel() == 0:
        return pred_logits.new_zeros(())

    intersection = (p_flat * t_flat).sum()
    denom = p_flat.sum() + t_flat.sum() + smooth
    return 1.0 - (2.0 * intersection + smooth) / denom


def extent_loss(
    pred_logits: "torch.Tensor",
    target: "torch.Tensor",
    valid: "torch.Tensor",
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> "torch.Tensor":
    """Focal BCE + soft Dice for extent head.

    Ignore policy (DATA_CONTRACT.md §8.2, module_net_train.md §9):
      - valid == 0 excluded
      - target == EXTENT_IGNORE_LABEL (255) excluded

    Parameters
    ----------
    pred_logits: (B, H, W) or (B, 1, H, W) float32
    target: (B, H, W) int64  —  0=background, 1=foreground, 255=ignore
    valid: (B, H, W) bool

    Raises
    ------
    ContractError if valid region is completely empty (§19).
    """
    _require_torch()
    _check_nonempty_valid(valid, "extent_loss")

    if pred_logits.dim() == 4 and pred_logits.shape[1] == 1:
        pred_logits = pred_logits.squeeze(1)

    # Combined ignore mask: valid AND not extent-ignore-label
    mask = valid & (target != EXTENT_IGNORE_LABEL)

    binary_target = (target == 1).long()
    bce = focal_bce_loss(pred_logits, binary_target, mask, gamma=focal_gamma, alpha=focal_alpha)
    dice = soft_dice_loss_binary(pred_logits, binary_target, mask)
    return bce_weight * bce + dice_weight * dice


# ---------------------------------------------------------------------------
# Boundary loss:  focal CE  +  λ · soft Dice (skeleton class)
# ---------------------------------------------------------------------------

# Default class weights for boundary encoding (0=bg, 1=skeleton, 2=buffer).
# Skeleton (class 1) is geometrically rare → high weight for accurate localization.
_DEFAULT_BOUNDARY_CLASS_WEIGHTS = (0.5, 4.0, 1.5)


def focal_ce_loss(
    pred_logits: "torch.Tensor",
    target: "torch.Tensor",
    mask: "torch.Tensor",
    class_weights: tuple = _DEFAULT_BOUNDARY_CLASS_WEIGHTS,
    gamma: float = 2.0,
) -> "torch.Tensor":
    """Class-weighted focal Cross-Entropy for 3-class boundary segmentation.

    Boundary encoding: 0=background, 1=skeleton, 2=buffer.
    No separate boundary ignore label in baseline v1 — masking is by valid only.

    Parameters
    ----------
    pred_logits: (B, num_classes, H, W) float32
    target: (B, H, W) int64  —  values in {0, 1, 2}
    mask: (B, H, W) bool
    class_weights: per-class weights  (len == num_classes)
    gamma: focal exponent

    Returns
    -------
    Masked mean scalar.  0.0 if mask is empty.
    """
    _require_torch()
    num_classes = pred_logits.shape[1]
    w = pred_logits.new_tensor(class_weights)

    # Standard CE (unreduced)
    ce = F.cross_entropy(
        pred_logits,
        target.long(),
        weight=w,
        reduction="none",
    )  # (B, H, W)

    # Focal weight: (1 - p_t)^gamma
    probs = F.softmax(pred_logits, dim=1)                       # (B, C, H, W)
    t_idx = target.clamp(0, num_classes - 1)
    p_t = probs.gather(1, t_idx.unsqueeze(1)).squeeze(1)        # (B, H, W)
    focal_weight = (1.0 - p_t).pow(gamma)

    return _safe_mean(focal_weight * ce, mask)


def soft_dice_skeleton(
    pred_logits: "torch.Tensor",
    target: "torch.Tensor",
    mask: "torch.Tensor",
    skeleton_class: int = 1,
    smooth: float = 1.0,
) -> "torch.Tensor":
    """Soft Dice for the skeleton class (class 1) only.

    module_net_train.md §22.1: boundary_loss includes a separate soft Dice term
    over the skeleton class to sharpen thin-boundary localization.

    Parameters
    ----------
    pred_logits: (B, num_classes, H, W)
    target: (B, H, W) int64
    mask: (B, H, W) bool
    skeleton_class: class index for skeleton (default 1)

    Returns
    -------
    Scalar in [0, 1].  0.0 if mask is empty.
    """
    _require_torch()
    probs = F.softmax(pred_logits, dim=1)
    p_skel = probs[:, skeleton_class, :]  # (B, H, W)
    t_skel = (target == skeleton_class).float()

    p_flat = p_skel[mask]
    t_flat = t_skel[mask]

    if p_flat.numel() == 0:
        return pred_logits.new_zeros(())

    intersection = (p_flat * t_flat).sum()
    denom = p_flat.sum() + t_flat.sum() + smooth
    return 1.0 - (2.0 * intersection + smooth) / denom


def boundary_loss(
    pred_logits: "torch.Tensor",
    target: "torch.Tensor",
    valid: "torch.Tensor",
    lambda_skel_dice: float = 0.5,
    class_weights: tuple = _DEFAULT_BOUNDARY_CLASS_WEIGHTS,
    focal_gamma: float = 2.0,
) -> "torch.Tensor":
    """Boundary head loss: focal CE + λ·soft Dice (skeleton class).

    module_net_train.md §22.1:
      boundary_loss = focal_CE_boundary + lambda_skel_dice * soft_dice_skeleton

    Boundary encoding: 0=background, 1=skeleton, 2=buffer.
    No additional ignore label for boundary in baseline v1.
    mask = (valid == 1).

    Parameters
    ----------
    pred_logits: (B, 3, H, W)
    target: (B, H, W) int64  —  values in {0, 1, 2}
    valid: (B, H, W) bool
    lambda_skel_dice: weight for soft Dice skeleton term (from LossConfig)
    class_weights: per-class weights for the focal CE component
    focal_gamma: focal exponent

    Raises
    ------
    ContractError if valid region is completely empty (§19).
    """
    _require_torch()
    _check_nonempty_valid(valid, "boundary_loss")

    fce = focal_ce_loss(pred_logits, target, valid, class_weights=class_weights, gamma=focal_gamma)
    skel_dice = soft_dice_skeleton(pred_logits, target, valid)
    return fce + lambda_skel_dice * skel_dice


# ---------------------------------------------------------------------------
# Distance loss:  SmoothL1
# ---------------------------------------------------------------------------


def distance_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    valid: "torch.Tensor",
    beta: float = 1.0,
) -> "torch.Tensor":
    """SmoothL1 (Huber) loss on normalized unsigned distance map.

    module_net_train.md §22.2: SmoothL1 preferred over MSE.
    Invalid pixels excluded per valid mask (§9).

    Parameters
    ----------
    pred: (B, H, W) or (B, 1, H, W) float32
    target: (B, H, W) float32  —  normalized unsigned distance
    valid: (B, H, W) bool
    beta: SmoothL1 transition point (default 1.0)

    Raises
    ------
    ContractError if valid region is completely empty (§19).
    """
    _require_torch()
    _check_nonempty_valid(valid, "distance_loss")

    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)

    sl1 = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return _safe_mean(sl1, valid)


# ---------------------------------------------------------------------------
# MultitaskLoss — combined loss for all three heads
# ---------------------------------------------------------------------------


class MultitaskLoss:
    """Combined multitask loss for extent + boundary + distance heads.

    total = extent_weight * L_extent
          + boundary_weight * L_boundary
          + distance_weight * L_distance

    For deep supervision, call forward() on each auxiliary output separately
    with aux_weight multiplied into the caller — MultitaskLoss itself does not
    apply aux_weight (it's applied externally per auxiliary head).

    Baseline weights (module_net_train.md §8.6, §21.9):
      extent=1.0, boundary=2.5, distance=1.0

    Requires torch.
    """

    def __init__(
        self,
        extent_weight: float = 1.0,
        boundary_weight: float = 2.5,
        distance_weight: float = 1.0,
        lambda_skel_dice: float = 0.5,
    ) -> None:
        _require_torch()
        self.extent_weight = extent_weight
        self.boundary_weight = boundary_weight
        self.distance_weight = distance_weight
        self.lambda_skel_dice = lambda_skel_dice

    @classmethod
    def from_config(cls, config: Any) -> "MultitaskLoss":
        """Construct from a LossConfig dataclass instance."""
        return cls(
            extent_weight=config.extent_weight,
            boundary_weight=config.boundary_weight,
            distance_weight=config.distance_weight,
            lambda_skel_dice=getattr(config, "boundary_lambda_skel_dice", 0.5),
        )

    def forward(
        self,
        preds: dict,
        targets: dict,
        valid_mask: "torch.Tensor",
    ) -> dict:
        """Compute per-head losses and the weighted total.

        Parameters
        ----------
        preds: dict with keys
            "extent"    : (B, H, W) or (B, 1, H, W) float32 — logits
            "boundary"  : (B, 3, H, W) float32 — logits
            "distance"  : (B, H, W) or (B, 1, H, W) float32 — regression output
        targets: dict with keys
            "extent"    : (B, H, W) int64  —  0/1/255
            "boundary"  : (B, H, W) int64  —  0/1/2
            "distance"  : (B, H, W) float32
        valid_mask: (B, H, W) bool

        Returns
        -------
        dict with keys:
            "extent"   : scalar tensor
            "boundary" : scalar tensor
            "distance" : scalar tensor
            "total"    : scalar tensor  (weighted sum)
            "n_valid"  : int  (total valid pixels in batch)

        Raises
        ------
        ContractError if valid_mask is entirely False (§19).
        """
        _require_torch()
        _check_nonempty_valid(valid_mask, "MultitaskLoss.forward")

        l_extent = extent_loss(preds["extent"], targets["extent"], valid_mask)
        l_boundary = boundary_loss(
            preds["boundary"], targets["boundary"], valid_mask,
            lambda_skel_dice=self.lambda_skel_dice,
        )
        l_distance = distance_loss(preds["distance"], targets["distance"], valid_mask)

        total = (
            self.extent_weight * l_extent
            + self.boundary_weight * l_boundary
            + self.distance_weight * l_distance
        )

        return {
            "extent":   l_extent,
            "boundary": l_boundary,
            "distance": l_distance,
            "total":    total,
            "n_valid":  int(valid_mask.sum().item()),
        }

    def __call__(
        self,
        preds: dict,
        targets: dict,
        valid_mask: "torch.Tensor",
    ) -> dict:
        return self.forward(preds, targets, valid_mask)
