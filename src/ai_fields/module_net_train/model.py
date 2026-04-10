"""Baseline model for module_net_train: EdgeAwareMultitaskNet.

Architecture: hybrid edge-aware multitask encoder-decoder.
Implements the architectural contract from module_net_train.md §5–§6.

Components:
  - Stem / input adapter  (in_channels = 9 or 12)                    §5.3
  - CNN-first residual encoder  (encoder_depth stride-2 stages)       §6.1
  - ASPP-style multi-scale context / bottleneck                       §6.2
  - UNet-style decoder with skip connections
  - EdgeEnhanceBlock at the two finest decoder levels                 §6.3
  - AuxHead deep supervision on coarse decoder levels                 §6.4
  - RefineBlock  (two stacked residual blocks)                        §6.5
  - Three output heads                                                §7

Forward output contract:
  "extent"    : (B, 1, H, W) — binary segmentation logits
  "boundary"  : (B, 3, H, W) — 3-class logits (0=bg / 1=skeleton / 2=buffer)
  "distance"  : (B, 1, H, W) — unsigned distance regression output
  "aux"       : list of dicts, same three keys, at input (H×W) resolution.
                len(aux) == max(0, encoder_depth - 2).
                Used for deep supervision; empty when encoder_depth <= 2.

Output shapes are directly consumable by losses.py:
  extent_loss / boundary_loss / distance_loss all accept (B,1,H,W).
  MultitaskLoss.forward() receives the dict keys "extent", "boundary", "distance".

Requires torch.  Raises ContractError if torch is unavailable.
"""

from __future__ import annotations

from typing import Any, List

from ai_fields.common.constants import CHANNEL_COUNTS
from ai_fields.common.errors import ContractError, FeatureModeError

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

    # Minimal stub so `class Foo(nn.Module):` does not fail at import time.
    # Every __init__ calls _require_torch() before accessing any real nn attribute.
    class _FakeModule:  # type: ignore[misc]
        pass

    class nn:  # type: ignore[assignment,no-redef]
        Module = _FakeModule
        ModuleList = list


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ContractError(
            "torch is required for model in module_net_train.  "
            "Install torch to use this module."
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBnRelu(nn.Module):  # type: ignore[misc]
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        _require_torch()
        super().__init__()
        self.conv = nn.Conv2d(  # type: ignore[attr-defined]
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_ch)  # type: ignore[attr-defined]
        self.relu = nn.ReLU(inplace=True)  # type: ignore[attr-defined]

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):  # type: ignore[misc]
    """Single residual block with optional projection shortcut.

    Two 3×3 conv-BN-ReLU layers; shortcut projects when stride > 1 or
    channel count changes.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        _require_torch()
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)  # type: ignore[attr-defined]
        self.bn1 = nn.BatchNorm2d(out_ch)  # type: ignore[attr-defined]
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)  # type: ignore[attr-defined]
        self.bn2 = nn.BatchNorm2d(out_ch)  # type: ignore[attr-defined]
        self.relu = nn.ReLU(inplace=True)  # type: ignore[attr-defined]

        if stride != 1 or in_ch != out_ch:
            self.shortcut: "nn.Module" = nn.Sequential(  # type: ignore[attr-defined]
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),  # type: ignore[attr-defined]
                nn.BatchNorm2d(out_ch),  # type: ignore[attr-defined]
            )
        else:
            self.shortcut = nn.Identity()  # type: ignore[attr-defined]

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ContextModule(nn.Module):  # type: ignore[misc]
    """ASPP-style multi-scale context aggregation (module_net_train.md §6.2).

    Five parallel branches: 1×1, 3×3, dilated-3×3 (r=6), dilated-3×3 (r=12),
    global average pool.  Output fused with 1×1 convolution.
    Same spatial resolution in and out.
    """

    def __init__(self, channels: int) -> None:
        _require_torch()
        super().__init__()
        mid = max(channels // 4, 1)
        self.b1x1 = ConvBnRelu(channels, mid, kernel_size=1, padding=0)
        self.b3x3 = ConvBnRelu(channels, mid, kernel_size=3, padding=1)
        self.bd6 = nn.Sequential(  # type: ignore[attr-defined]
            nn.Conv2d(channels, mid, 3, padding=6, dilation=6, bias=False),  # type: ignore[attr-defined]
            nn.BatchNorm2d(mid),  # type: ignore[attr-defined]
            nn.ReLU(inplace=True),  # type: ignore[attr-defined]
        )
        self.bd12 = nn.Sequential(  # type: ignore[attr-defined]
            nn.Conv2d(channels, mid, 3, padding=12, dilation=12, bias=False),  # type: ignore[attr-defined]
            nn.BatchNorm2d(mid),  # type: ignore[attr-defined]
            nn.ReLU(inplace=True),  # type: ignore[attr-defined]
        )
        self.pool = nn.Sequential(  # type: ignore[attr-defined]
            nn.AdaptiveAvgPool2d(1),  # type: ignore[attr-defined]
            nn.Conv2d(channels, mid, 1, bias=False),  # type: ignore[attr-defined]
            nn.BatchNorm2d(mid),  # type: ignore[attr-defined]
            nn.ReLU(inplace=True),  # type: ignore[attr-defined]
        )
        self.fuse = ConvBnRelu(mid * 5, channels, kernel_size=1, padding=0)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        b1 = self.b1x1(x)
        b3 = self.b3x3(x)
        b6 = self.bd6(x)
        b12 = self.bd12(x)
        bp = F.interpolate(self.pool(x), size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([b1, b3, b6, b12, bp], dim=1))


class EdgeEnhanceBlock(nn.Module):  # type: ignore[misc]
    """Edge-aware channel-attention block (module_net_train.md §6.3).

    A dedicated edge branch (two 3×3 convs) produces an edge residual,
    which is modulated by squeeze-excite channel attention before being
    added back to the input.
    """

    def __init__(self, channels: int) -> None:
        _require_torch()
        super().__init__()
        mid = max(channels // 2, 1)
        self.edge = nn.Sequential(  # type: ignore[attr-defined]
            nn.Conv2d(channels, mid, 3, padding=1, bias=False),  # type: ignore[attr-defined]
            nn.BatchNorm2d(mid),  # type: ignore[attr-defined]
            nn.ReLU(inplace=True),  # type: ignore[attr-defined]
            nn.Conv2d(mid, channels, 3, padding=1, bias=False),  # type: ignore[attr-defined]
            nn.BatchNorm2d(channels),  # type: ignore[attr-defined]
        )
        ca_mid = max(channels // 4, 1)
        self.ca = nn.Sequential(  # type: ignore[attr-defined]
            nn.AdaptiveAvgPool2d(1),  # type: ignore[attr-defined]
            nn.Flatten(),  # type: ignore[attr-defined]
            nn.Linear(channels, ca_mid),  # type: ignore[attr-defined]
            nn.ReLU(inplace=True),  # type: ignore[attr-defined]
            nn.Linear(ca_mid, channels),  # type: ignore[attr-defined]
            nn.Sigmoid(),  # type: ignore[attr-defined]
        )
        self.relu = nn.ReLU(inplace=True)  # type: ignore[attr-defined]

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        edge = self.edge(x)
        ca = self.ca(x).view(x.size(0), -1, 1, 1)
        return self.relu(x + edge * ca)


class UpBlock(nn.Module):  # type: ignore[misc]
    """Decoder block: bilinear upsample + skip-connection concat + ResBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        _require_torch()
        super().__init__()
        self.resblock = ResBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: "torch.Tensor", skip: "torch.Tensor") -> "torch.Tensor":
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.resblock(torch.cat([x, skip], dim=1))


class RefineBlock(nn.Module):  # type: ignore[misc]
    """Lightweight boundary refinement stage (module_net_train.md §6.5).

    Two stacked residual blocks as a compact correction head.
    """

    def __init__(self, channels: int) -> None:
        _require_torch()
        super().__init__()
        self.block = nn.Sequential(  # type: ignore[attr-defined]
            ResBlock(channels, channels),
            ResBlock(channels, channels),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.block(x)


class OutputHead(nn.Module):  # type: ignore[misc]
    """Single-task output head: features → logits / regression."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        _require_torch()
        super().__init__()
        mid = max(in_ch // 2, 1)
        self.head = nn.Sequential(  # type: ignore[attr-defined]
            ConvBnRelu(in_ch, mid, kernel_size=3, padding=1),
            nn.Conv2d(mid, out_ch, kernel_size=1),  # type: ignore[attr-defined]
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.head(x)


class AuxHead(nn.Module):  # type: ignore[misc]
    """Deep supervision head for an intermediate decoder level (§6.4).

    Produces extent / boundary / distance outputs at target (H×W) via
    bilinear upsampling from a coarser intermediate feature map.
    """

    def __init__(self, in_ch: int) -> None:
        _require_torch()
        super().__init__()
        mid = max(in_ch // 2, 1)
        self.extent = nn.Sequential(ConvBnRelu(in_ch, mid), nn.Conv2d(mid, 1, 1))  # type: ignore[attr-defined]
        self.boundary = nn.Sequential(ConvBnRelu(in_ch, mid), nn.Conv2d(mid, 3, 1))  # type: ignore[attr-defined]
        self.distance = nn.Sequential(ConvBnRelu(in_ch, mid), nn.Conv2d(mid, 1, 1))  # type: ignore[attr-defined]

    def forward(
        self,
        x: "torch.Tensor",
        target_size: tuple,
    ) -> dict:
        e = self.extent(x)
        b = self.boundary(x)
        d = self.distance(x)
        if e.shape[-2] != target_size[0] or e.shape[-1] != target_size[1]:
            e = F.interpolate(e, size=target_size, mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=target_size, mode="bilinear", align_corners=False)
            d = F.interpolate(d, size=target_size, mode="bilinear", align_corners=False)
        return {"extent": e, "boundary": b, "distance": d}


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class EdgeAwareMultitaskNet(nn.Module):  # type: ignore[misc]
    """Hybrid edge-aware multitask encoder-decoder.

    Architecture contract (module_net_train.md §5–§6):
      - Input adapter stem for 9ch (raw8_valid) or 12ch (raw8_idx3_valid)
      - CNN-first residual encoder: encoder_depth stride-2 stages
      - ASPP-style context module at bottleneck
      - UNet decoder with skip connections from all encoder levels + stem
      - EdgeEnhanceBlock applied at the two finest decoder levels
      - AuxHead deep supervision at coarser decoder levels (D-2 heads)
      - RefineBlock (two stacked ResBlocks) before output heads
      - Three output heads: extent (B,1,H,W), boundary (B,3,H,W), distance (B,1,H,W)

    Channel progression (base_channels=32, encoder_depth=4):
      stem  : H×W     32ch
      enc[0]: H/2     32ch   (stride-2 ResBlock)
      enc[1]: H/4     64ch
      enc[2]: H/8    128ch
      enc[3]: H/16   256ch
      context                (ASPP, same size)
      dec[0]: H/8    128ch   (UpBlock + skip from enc[2])
      dec[1]: H/4     64ch   (UpBlock + skip from enc[1])
      dec[2]: H/2     32ch   (UpBlock + skip from enc[0] + EdgeEnhance)
      dec[3]: H×W     32ch   (UpBlock + skip from stem  + EdgeEnhance)
      refine: H×W     32ch   (2× ResBlock)
      heads:  extent/boundary/distance

    Parameters
    ----------
    in_channels : int
        9 for raw8_valid or 12 for raw8_idx3_valid.
    encoder_depth : int
        Number of stride-2 encoder stages.  Default 4; must be >= 2.
    base_channels : int
        Base channel width at first encoder stage.  Default 32.
    """

    def __init__(
        self,
        in_channels: int,
        encoder_depth: int = 4,
        base_channels: int = 32,
    ) -> None:
        _require_torch()
        super().__init__()

        self.in_channels = in_channels
        self.encoder_depth = encoder_depth
        self.base_channels = base_channels

        D = encoder_depth
        B = base_channels
        # enc_chs[i] = output channels of encoder stage i
        enc_chs: List[int] = [B * (2 ** i) for i in range(D)]

        # ---- Stem ----
        self.stem = ConvBnRelu(in_channels, B, kernel_size=3, padding=1)

        # ---- Encoder (encoder_depth stride-2 ResBlocks) ----
        self.encoders: "nn.ModuleList" = nn.ModuleList()  # type: ignore[attr-defined]
        prev = B
        for out_ch in enc_chs:
            self.encoders.append(ResBlock(prev, out_ch, stride=2))
            prev = out_ch

        # ---- Context / bottleneck ----
        self.context = ContextModule(enc_chs[-1])

        # ---- Decoder ----
        # decoder[i] fuses: previous_features + skip from enc_feats[D-1-i]
        #   i=0 : skip = enc_feats[D-1] (second deepest raw encoder output)
        #   ...
        #   i=D-1: skip = enc_feats[0]  (stem output)
        self.decoders: "nn.ModuleList" = nn.ModuleList()  # type: ignore[attr-defined]
        self._dec_out_chs: List[int] = []
        prev = enc_chs[-1]
        for i in range(D):
            skip_idx = D - 2 - i
            skip_ch = enc_chs[skip_idx] if skip_idx >= 0 else B
            out_ch = enc_chs[skip_idx] if skip_idx >= 0 else B
            self.decoders.append(UpBlock(prev, skip_ch, out_ch))
            self._dec_out_chs.append(out_ch)
            prev = out_ch

        # ---- Edge-aware enhancement (two finest decoder levels) ----
        self.edge_enhance: "nn.ModuleList" = nn.ModuleList(  # type: ignore[attr-defined]
            [
                EdgeEnhanceBlock(self._dec_out_chs[-2]),
                EdgeEnhanceBlock(self._dec_out_chs[-1]),
            ]
        )

        # ---- Refine module ----
        self.refine = RefineBlock(self._dec_out_chs[-1])

        # ---- Output heads ----
        final_ch = self._dec_out_chs[-1]
        self.extent_head = OutputHead(final_ch, 1)
        self.boundary_head = OutputHead(final_ch, 3)
        self.distance_head = OutputHead(final_ch, 1)

        # ---- Deep supervision aux heads (D-2 coarser levels) ----
        n_aux = max(0, D - 2)
        self.aux_heads: "nn.ModuleList" = nn.ModuleList(  # type: ignore[attr-defined]
            [AuxHead(self._dec_out_chs[i]) for i in range(n_aux)]
        )

    def forward(self, x: "torch.Tensor") -> dict:
        """Forward pass.

        Parameters
        ----------
        x : (B, in_channels, H, W) float32 — assembled model input

        Returns
        -------
        dict with keys:
            "extent"    : (B, 1, H, W) — binary segmentation logits
            "boundary"  : (B, 3, H, W) — 3-class boundary logits
            "distance"  : (B, 1, H, W) — unsigned distance regression
            "aux"       : list of dicts with same three keys, all at (H, W).
                         len == max(0, encoder_depth - 2).
        """
        input_size = (x.shape[2], x.shape[3])
        D = self.encoder_depth

        # Stem
        stem_feat = self.stem(x)

        # Encoder  — collect all intermediate features for skip connections
        enc_feats: List["torch.Tensor"] = [stem_feat]   # [0] = stem (H×W)
        feat = stem_feat
        for enc in self.encoders:
            feat = enc(feat)
            enc_feats.append(feat)
        # enc_feats[k] = output of encoder[k-1]; enc_feats[D] = deepest

        # Context bottleneck (on deepest encoder output)
        feat = self.context(feat)

        # Decoder
        dec_feats: List["torch.Tensor"] = []
        for i, dec_block in enumerate(self.decoders):
            # Skip source: enc_feats[D-i] counts from deepest ENCODER output downward.
            # For i=0 → enc_feats[D-1] (second deepest raw encoder output, H/2^(D-1))
            # For i=D-1 → enc_feats[0] = stem (H×W)
            skip = enc_feats[D - 1 - i]
            feat = dec_block(feat, skip)
            # Apply edge enhancement at the two finest decoder levels
            if i >= D - 2:
                feat = self.edge_enhance[i - (D - 2)](feat)
            dec_feats.append(feat)

        # Refinement
        final_feat = self.refine(dec_feats[-1])

        # Main output heads
        extent = self.extent_head(final_feat)      # (B, 1, H, W)
        boundary = self.boundary_head(final_feat)  # (B, 3, H, W)
        distance = self.distance_head(final_feat)  # (B, 1, H, W)

        # Deep supervision auxiliary heads
        aux = [
            aux_head(dec_feats[i], input_size)
            for i, aux_head in enumerate(self.aux_heads)
        ]

        return {
            "extent": extent,
            "boundary": boundary,
            "distance": distance,
            "aux": aux,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(config: Any) -> "EdgeAwareMultitaskNet":
    """Construct EdgeAwareMultitaskNet from a NetTrainConfig.

    Derives in_channels from feature_mode + valid_as_input_channel contract
    (DATA_CONTRACT.md §7.4–§7.5, module_net_train.md §5.3).

    Parameters
    ----------
    config : NetTrainConfig
        Fully-validated configuration.

    Raises
    ------
    ContractError   if torch is unavailable.
    FeatureModeError if assembled input name is not in CHANNEL_COUNTS.
    """
    _require_torch()
    assembled = f"{config.feature_mode}_valid"   # "raw8_valid" or "raw8_idx3_valid"
    if assembled not in CHANNEL_COUNTS:
        raise FeatureModeError(
            f"No channel count for assembled input '{assembled}'.  "
            f"Expected one of: {[k for k in CHANNEL_COUNTS if '_valid' in k]}.  "
            "(DATA_CONTRACT.md §7.4–§7.5)"
        )
    in_channels = CHANNEL_COUNTS[assembled]
    return EdgeAwareMultitaskNet(
        in_channels=in_channels,
        encoder_depth=config.model.encoder_depth,
        base_channels=config.model.base_channels,
    )
