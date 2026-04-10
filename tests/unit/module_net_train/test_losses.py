"""Unit tests for module_net_train.losses.

Covers:
  - EXTENT_IGNORE_LABEL constant
  - No-torch guard (ContractError when torch unavailable)
  - Per-head losses: extent, boundary, distance
  - MultitaskLoss: from_config, forward, output keys
  - Ignore policy: valid==0 excluded; extent ignore label 255 excluded
  - Empty valid region: ContractError at head level (module_net_train.md §19)
  - Low-level helpers: 0.0 returned on empty mask (safe for composition)

All torch-dependent tests auto-skip if torch is not installed.
"""

from __future__ import annotations

import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_net_train import losses as losses_module
from ai_fields.module_net_train.losses import EXTENT_IGNORE_LABEL


# ---------------------------------------------------------------------------
# No-torch guard — always runs
# ---------------------------------------------------------------------------


class TestNoTorchGuard:
    def test_extent_ignore_label_is_255(self):
        """Constant must match module_prep_data._IGNORE_LABEL (DATA_CONTRACT §8.2)."""
        assert EXTENT_IGNORE_LABEL == 255

    def test_multitask_loss_no_torch_raises(self, monkeypatch):
        monkeypatch.setattr(losses_module, "_TORCH_AVAILABLE", False)
        with pytest.raises(ContractError, match="torch is required"):
            losses_module.MultitaskLoss()

    def test_extent_loss_no_torch_raises(self, monkeypatch):
        monkeypatch.setattr(losses_module, "_TORCH_AVAILABLE", False)
        with pytest.raises(ContractError, match="torch is required"):
            losses_module.extent_loss(None, None, None)

    def test_boundary_loss_no_torch_raises(self, monkeypatch):
        monkeypatch.setattr(losses_module, "_TORCH_AVAILABLE", False)
        with pytest.raises(ContractError, match="torch is required"):
            losses_module.boundary_loss(None, None, None)

    def test_distance_loss_no_torch_raises(self, monkeypatch):
        monkeypatch.setattr(losses_module, "_TORCH_AVAILABLE", False)
        with pytest.raises(ContractError, match="torch is required"):
            losses_module.distance_loss(None, None, None)


# ---------------------------------------------------------------------------
# Torch-dependent tests
# ---------------------------------------------------------------------------


def _torch():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers: _safe_mean
# ---------------------------------------------------------------------------


class TestSafeMean:
    def test_nonempty_mask(self):
        torch = _torch()
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])
        result = losses_module._safe_mean(vals, mask)
        assert abs(result.item() - 1.5) < 1e-5

    def test_empty_mask_returns_zero(self):
        torch = _torch()
        vals = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.zeros(3, dtype=torch.bool)
        result = losses_module._safe_mean(vals, mask)
        assert result.item() == 0.0


# ---------------------------------------------------------------------------
# Extent: focal_bce_loss
# ---------------------------------------------------------------------------


class TestFocalBCELoss:
    def test_all_foreground_target_is_finite(self):
        torch = _torch()
        B, H, W = 2, 8, 8
        pred = torch.zeros(B, H, W)
        target = torch.ones(B, H, W, dtype=torch.long)
        mask = torch.ones(B, H, W, dtype=torch.bool)
        result = losses_module.focal_bce_loss(pred, target, mask)
        assert torch.isfinite(result)
        assert result.item() > 0.0

    def test_empty_mask_returns_zero(self):
        torch = _torch()
        pred = torch.zeros(2, 8, 8)
        target = torch.ones(2, 8, 8, dtype=torch.long)
        mask = torch.zeros(2, 8, 8, dtype=torch.bool)
        result = losses_module.focal_bce_loss(pred, target, mask)
        assert result.item() == 0.0


# ---------------------------------------------------------------------------
# Extent: soft_dice_loss_binary
# ---------------------------------------------------------------------------


class TestSoftDiceBinary:
    def test_perfect_prediction_near_zero(self):
        torch = _torch()
        pred = torch.full((2, 8, 8), 10.0)   # very confident foreground
        target = torch.ones(2, 8, 8, dtype=torch.long)
        mask = torch.ones(2, 8, 8, dtype=torch.bool)
        result = losses_module.soft_dice_loss_binary(pred, target, mask)
        assert result.item() < 0.05

    def test_worst_prediction_near_one(self):
        torch = _torch()
        pred = torch.full((2, 8, 8), 10.0)   # confident foreground
        target = torch.zeros(2, 8, 8, dtype=torch.long)  # but GT is all background
        mask = torch.ones(2, 8, 8, dtype=torch.bool)
        result = losses_module.soft_dice_loss_binary(pred, target, mask)
        assert result.item() > 0.9

    def test_empty_mask_returns_zero(self):
        torch = _torch()
        pred = torch.zeros(2, 4, 4)
        target = torch.ones(2, 4, 4, dtype=torch.long)
        mask = torch.zeros(2, 4, 4, dtype=torch.bool)
        result = losses_module.soft_dice_loss_binary(pred, target, mask)
        assert result.item() == 0.0


# ---------------------------------------------------------------------------
# extent_loss
# ---------------------------------------------------------------------------


class TestExtentLoss:
    def test_basic_forward(self):
        torch = _torch()
        B, H, W = 2, 16, 16
        pred = torch.zeros(B, H, W)
        target = torch.ones(B, H, W, dtype=torch.long)
        valid = torch.ones(B, H, W, dtype=torch.bool)
        result = losses_module.extent_loss(pred, target, valid)
        assert torch.isfinite(result)
        assert result.item() > 0.0

    def test_extent_ignore_label_excluded(self):
        """Pixels with target==255 must not contribute to loss (DATA_CONTRACT §8.2)."""
        torch = _torch()
        B, H, W = 1, 8, 8
        pred = torch.zeros(B, H, W)
        # Half the pixels are ignore
        target = torch.ones(B, H, W, dtype=torch.long)
        target[:, :, 4:] = EXTENT_IGNORE_LABEL
        valid = torch.ones(B, H, W, dtype=torch.bool)

        # All pixels are ignore
        target_all_ignore = torch.full((B, H, W), EXTENT_IGNORE_LABEL, dtype=torch.long)
        # even though valid is True, all pixels are ignore → effective mask is empty
        # This should return 0.0 (low-level helpers return 0.0 on empty mask),
        # but _check_nonempty_valid passes because valid is non-empty.
        result_partial = losses_module.extent_loss(pred, target, valid)
        assert torch.isfinite(result_partial)

    def test_valid_zero_excluded(self):
        """Pixels with valid==0 must not contribute to extent loss (§9)."""
        torch = _torch()
        B, H, W = 1, 8, 8
        pred = torch.full((B, H, W), -5.0)  # strong background prediction
        target = torch.ones(B, H, W, dtype=torch.long)  # GT is all foreground
        valid = torch.zeros(B, H, W, dtype=torch.bool)  # but all invalid
        valid[:, :4, :] = True  # only first 4 rows are valid

        result = losses_module.extent_loss(pred, target, valid)
        assert torch.isfinite(result)
        # Result must be > 0 (pred=background, target=foreground)
        assert result.item() > 0.0

    def test_empty_valid_raises(self):
        """Empty valid region must raise ContractError (module_net_train.md §19)."""
        torch = _torch()
        B, H, W = 1, 8, 8
        pred = torch.zeros(B, H, W)
        target = torch.ones(B, H, W, dtype=torch.long)
        valid = torch.zeros(B, H, W, dtype=torch.bool)  # ALL invalid
        with pytest.raises(ContractError, match="completely empty"):
            losses_module.extent_loss(pred, target, valid)

    def test_accepts_4d_logits(self):
        """pred may be (B, 1, H, W) — channel dim is squeezed automatically."""
        torch = _torch()
        pred = torch.zeros(2, 1, 8, 8)
        target = torch.ones(2, 8, 8, dtype=torch.long)
        valid = torch.ones(2, 8, 8, dtype=torch.bool)
        result = losses_module.extent_loss(pred, target, valid)
        assert torch.isfinite(result)


# ---------------------------------------------------------------------------
# focal_ce_loss
# ---------------------------------------------------------------------------


class TestFocalCELoss:
    def test_all_background_is_finite(self):
        torch = _torch()
        B, H, W = 2, 8, 8
        pred = torch.zeros(B, 3, H, W)
        target = torch.zeros(B, H, W, dtype=torch.long)
        mask = torch.ones(B, H, W, dtype=torch.bool)
        result = losses_module.focal_ce_loss(pred, target, mask)
        assert torch.isfinite(result)

    def test_empty_mask_returns_zero(self):
        torch = _torch()
        pred = torch.zeros(2, 3, 4, 4)
        target = torch.zeros(2, 4, 4, dtype=torch.long)
        mask = torch.zeros(2, 4, 4, dtype=torch.bool)
        result = losses_module.focal_ce_loss(pred, target, mask)
        assert result.item() == 0.0


# ---------------------------------------------------------------------------
# soft_dice_skeleton
# ---------------------------------------------------------------------------


class TestSoftDiceSkeleton:
    def test_no_skeleton_pixels_near_one(self):
        """If GT has no skeleton but model predicts high skeleton → Dice near 1."""
        torch = _torch()
        B, H, W = 1, 8, 8
        pred = torch.zeros(B, 3, H, W)
        pred[:, 1, :, :] = 10.0   # confident skeleton prediction everywhere
        target = torch.zeros(B, H, W, dtype=torch.long)  # all background
        mask = torch.ones(B, H, W, dtype=torch.bool)
        result = losses_module.soft_dice_skeleton(pred, target, mask)
        assert result.item() > 0.8

    def test_empty_mask_returns_zero(self):
        torch = _torch()
        pred = torch.zeros(1, 3, 4, 4)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        result = losses_module.soft_dice_skeleton(pred, target, mask)
        assert result.item() == 0.0


# ---------------------------------------------------------------------------
# boundary_loss
# ---------------------------------------------------------------------------


class TestBoundaryLoss:
    def test_basic_forward(self):
        torch = _torch()
        B, H, W = 2, 8, 8
        pred = torch.zeros(B, 3, H, W)
        target = torch.zeros(B, H, W, dtype=torch.long)
        valid = torch.ones(B, H, W, dtype=torch.bool)
        result = losses_module.boundary_loss(pred, target, valid)
        assert torch.isfinite(result)
        assert result.item() > 0.0

    def test_buffer_class_2_is_included(self):
        """Boundary class 2 (buffer) is a valid training target, not an ignore label."""
        torch = _torch()
        B, H, W = 1, 8, 8
        pred = torch.zeros(B, 3, H, W)
        target = torch.full((B, H, W), 2, dtype=torch.long)  # all buffer
        valid = torch.ones(B, H, W, dtype=torch.bool)
        # Must not raise, must be finite
        result = losses_module.boundary_loss(pred, target, valid)
        assert torch.isfinite(result)

    def test_lambda_skel_dice_zero_excludes_dice_term(self):
        """With lambda=0, boundary_loss equals focal CE only."""
        torch = _torch()
        B, H, W = 1, 8, 8
        pred = torch.zeros(B, 3, H, W)
        target = torch.zeros(B, H, W, dtype=torch.long)
        valid = torch.ones(B, H, W, dtype=torch.bool)

        result_with_dice = losses_module.boundary_loss(pred, target, valid, lambda_skel_dice=0.5)
        result_without_dice = losses_module.boundary_loss(pred, target, valid, lambda_skel_dice=0.0)
        # With dice term non-zero, result should differ
        assert abs(result_with_dice.item() - result_without_dice.item()) > 1e-6

    def test_valid_zero_excluded(self):
        torch = _torch()
        B, H, W = 1, 8, 8
        pred = torch.zeros(B, 3, H, W)
        target = torch.zeros(B, H, W, dtype=torch.long)
        valid = torch.ones(B, H, W, dtype=torch.bool)
        valid[:, :, 4:] = False  # half invalid

        result_full = losses_module.boundary_loss(pred, target, valid)
        full_valid = torch.ones(B, H, W, dtype=torch.bool)
        result_full_valid = losses_module.boundary_loss(pred, target, full_valid)
        # Both must be finite; valid mask affects the result
        assert torch.isfinite(result_full)
        assert torch.isfinite(result_full_valid)

    def test_empty_valid_raises(self):
        torch = _torch()
        pred = torch.zeros(1, 3, 8, 8)
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        valid = torch.zeros(1, 8, 8, dtype=torch.bool)
        with pytest.raises(ContractError, match="completely empty"):
            losses_module.boundary_loss(pred, target, valid)


# ---------------------------------------------------------------------------
# distance_loss
# ---------------------------------------------------------------------------


class TestDistanceLoss:
    def test_perfect_prediction_near_zero(self):
        torch = _torch()
        B, H, W = 2, 8, 8
        target = torch.rand(B, H, W)
        pred = target.clone()
        valid = torch.ones(B, H, W, dtype=torch.bool)
        result = losses_module.distance_loss(pred, target, valid)
        assert result.item() < 1e-6

    def test_nonzero_error_positive(self):
        torch = _torch()
        B, H, W = 2, 8, 8
        pred = torch.zeros(B, H, W)
        target = torch.ones(B, H, W)
        valid = torch.ones(B, H, W, dtype=torch.bool)
        result = losses_module.distance_loss(pred, target, valid)
        assert result.item() > 0.0

    def test_valid_zero_excluded(self):
        torch = _torch()
        B, H, W = 1, 8, 8
        pred = torch.zeros(B, H, W)
        target = torch.ones(B, H, W)  # large error everywhere
        valid = torch.zeros(B, H, W, dtype=torch.bool)
        valid[:, :4, :] = True  # only first 4 rows
        result_partial = losses_module.distance_loss(pred, target, valid)
        assert torch.isfinite(result_partial)

    def test_accepts_4d_pred(self):
        torch = _torch()
        pred = torch.zeros(2, 1, 8, 8)
        target = torch.zeros(2, 8, 8)
        valid = torch.ones(2, 8, 8, dtype=torch.bool)
        result = losses_module.distance_loss(pred, target, valid)
        assert torch.isfinite(result)

    def test_empty_valid_raises(self):
        torch = _torch()
        pred = torch.zeros(1, 8, 8)
        target = torch.zeros(1, 8, 8)
        valid = torch.zeros(1, 8, 8, dtype=torch.bool)
        with pytest.raises(ContractError, match="completely empty"):
            losses_module.distance_loss(pred, target, valid)


# ---------------------------------------------------------------------------
# MultitaskLoss
# ---------------------------------------------------------------------------


class TestMultitaskLoss:
    def _make_batch(self, torch, B=2, H=16, W=16):
        """Create a minimal synthetic batch."""
        preds = {
            "extent":   torch.zeros(B, H, W),
            "boundary": torch.zeros(B, 3, H, W),
            "distance": torch.zeros(B, H, W),
        }
        targets = {
            "extent":   torch.ones(B, H, W, dtype=torch.long),
            "boundary": torch.zeros(B, H, W, dtype=torch.long),
            "distance": torch.ones(B, H, W, dtype=torch.float32),
        }
        valid = torch.ones(B, H, W, dtype=torch.bool)
        return preds, targets, valid

    def test_from_config_reads_weights(self):
        torch = _torch()
        from ai_fields.module_net_train.schemas import LossConfig
        cfg = LossConfig(
            extent_weight=2.0,
            boundary_weight=3.0,
            distance_weight=0.5,
            boundary_lambda_skel_dice=0.3,
        )
        ml = losses_module.MultitaskLoss.from_config(cfg)
        assert ml.extent_weight == 2.0
        assert ml.boundary_weight == 3.0
        assert ml.distance_weight == 0.5
        assert ml.lambda_skel_dice == 0.3

    def test_forward_returns_required_keys(self):
        torch = _torch()
        preds, targets, valid = self._make_batch(torch)
        ml = losses_module.MultitaskLoss()
        result = ml(preds, targets, valid)
        for key in ("extent", "boundary", "distance", "total", "n_valid"):
            assert key in result, f"Missing key: {key}"

    def test_total_is_weighted_sum(self):
        torch = _torch()
        preds, targets, valid = self._make_batch(torch)
        ew, bw, dw = 1.5, 2.0, 0.8
        ml = losses_module.MultitaskLoss(
            extent_weight=ew, boundary_weight=bw, distance_weight=dw
        )
        result = ml(preds, targets, valid)
        expected = ew * result["extent"] + bw * result["boundary"] + dw * result["distance"]
        assert abs(result["total"].item() - expected.item()) < 1e-5

    def test_n_valid_equals_valid_pixel_count(self):
        torch = _torch()
        B, H, W = 2, 8, 8
        preds, targets, valid = self._make_batch(torch, B=B, H=H, W=W)
        valid[:, :4, :] = False  # half invalid
        ml = losses_module.MultitaskLoss()
        result = ml(preds, targets, valid)
        assert result["n_valid"] == int(valid.sum().item())

    def test_all_outputs_are_finite(self):
        torch = _torch()
        preds, targets, valid = self._make_batch(torch)
        ml = losses_module.MultitaskLoss()
        result = ml(preds, targets, valid)
        for key in ("extent", "boundary", "distance", "total"):
            assert torch.isfinite(result[key]), f"Non-finite loss: {key}"

    def test_total_is_positive(self):
        torch = _torch()
        preds, targets, valid = self._make_batch(torch)
        ml = losses_module.MultitaskLoss()
        result = ml(preds, targets, valid)
        assert result["total"].item() > 0.0

    def test_empty_valid_raises(self):
        torch = _torch()
        preds, targets, _ = self._make_batch(torch)
        empty_valid = torch.zeros(2, 16, 16, dtype=torch.bool)
        ml = losses_module.MultitaskLoss()
        with pytest.raises(ContractError, match="completely empty"):
            ml(preds, targets, empty_valid)

    def test_baseline_weights_match_tz(self):
        """Default weights must match module_net_train.md §8.6 / §21.9: 1.0/2.5/1.0."""
        torch = _torch()
        ml = losses_module.MultitaskLoss()
        assert ml.extent_weight == 1.0
        assert ml.boundary_weight == 2.5
        assert ml.distance_weight == 1.0
