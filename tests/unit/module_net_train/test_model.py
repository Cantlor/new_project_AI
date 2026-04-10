"""Unit tests for module_net_train/model.py.

Tests are grouped into:
  - Always-run (no torch required): ContractError on instantiation without torch
  - Torch-dependent (auto-skip if torch unavailable): forward shapes, loss compat,
    build_model factory, deep supervision contract

Torch tests use small spatial sizes (64×64) and batch size 2 to keep them fast.
"""

from __future__ import annotations

import pytest

from ai_fields.common.errors import ContractError
import ai_fields.module_net_train.model as model_module


def _torch():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# No-torch guard — always runs
# ---------------------------------------------------------------------------


class TestNoTorchGuard:
    def test_edge_aware_net_raises_contract_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_module, "_TORCH_AVAILABLE", False)
        with pytest.raises(ContractError, match="torch is required"):
            model_module.EdgeAwareMultitaskNet(in_channels=9)

    def test_build_model_raises_contract_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ai_fields.module_net_train.schemas import NetTrainConfig

        monkeypatch.setattr(model_module, "_TORCH_AVAILABLE", False)
        cfg = NetTrainConfig()
        with pytest.raises(ContractError, match="torch is required"):
            model_module.build_model(cfg)

    def test_conv_bn_relu_raises_contract_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_module, "_TORCH_AVAILABLE", False)
        with pytest.raises(ContractError, match="torch is required"):
            model_module.ConvBnRelu(3, 16)

    def test_res_block_raises_contract_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_module, "_TORCH_AVAILABLE", False)
        with pytest.raises(ContractError, match="torch is required"):
            model_module.ResBlock(16, 32)


# ---------------------------------------------------------------------------
# Torch-dependent helpers
# ---------------------------------------------------------------------------


def _make_net(in_channels: int = 9, encoder_depth: int = 4, base_channels: int = 32):
    net = model_module.EdgeAwareMultitaskNet(in_channels, encoder_depth, base_channels)
    net.eval()
    return net


def _rand_input(batch: int, in_ch: int, h: int = 64, w: int = 64):
    torch = _torch()
    return torch.rand(batch, in_ch, h, w)


# ---------------------------------------------------------------------------
# Forward shape contract
# ---------------------------------------------------------------------------


class TestForwardShapes:
    """Model outputs must match the shapes expected by losses.py."""

    def test_extent_shape_raw8_valid(self) -> None:
        torch = _torch()
        net = _make_net(in_channels=9)
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        assert out["extent"].shape == (2, 1, 64, 64), out["extent"].shape

    def test_boundary_shape_raw8_valid(self) -> None:
        torch = _torch()
        net = _make_net(in_channels=9)
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        assert out["boundary"].shape == (2, 3, 64, 64), out["boundary"].shape

    def test_distance_shape_raw8_valid(self) -> None:
        torch = _torch()
        net = _make_net(in_channels=9)
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        assert out["distance"].shape == (2, 1, 64, 64), out["distance"].shape

    def test_extent_shape_raw8_idx3_valid(self) -> None:
        torch = _torch()
        net = _make_net(in_channels=12)
        with torch.no_grad():
            out = net(_rand_input(2, 12))
        assert out["extent"].shape == (2, 1, 64, 64)

    def test_boundary_shape_raw8_idx3_valid(self) -> None:
        torch = _torch()
        net = _make_net(in_channels=12)
        with torch.no_grad():
            out = net(_rand_input(2, 12))
        assert out["boundary"].shape == (2, 3, 64, 64)

    def test_distance_shape_raw8_idx3_valid(self) -> None:
        torch = _torch()
        net = _make_net(in_channels=12)
        with torch.no_grad():
            out = net(_rand_input(2, 12))
        assert out["distance"].shape == (2, 1, 64, 64)

    def test_output_preserves_spatial_size(self) -> None:
        """Spatial dims of output == spatial dims of input."""
        torch = _torch()
        net = _make_net(in_channels=9)
        for h, w in [(32, 32), (64, 64), (128, 96)]:
            x = torch.rand(1, 9, h, w)
            with torch.no_grad():
                out = net(x)
            assert out["extent"].shape[-2:] == (h, w)
            assert out["boundary"].shape[-2:] == (h, w)
            assert out["distance"].shape[-2:] == (h, w)


# ---------------------------------------------------------------------------
# Output dict keys and dtype
# ---------------------------------------------------------------------------


class TestOutputDictContract:
    def test_required_keys_present(self) -> None:
        torch = _torch()
        net = _make_net()
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        for key in ("extent", "boundary", "distance", "aux"):
            assert key in out, f"missing key: {key}"

    def test_output_float32(self) -> None:
        torch = _torch()
        net = _make_net()
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        assert out["extent"].dtype == torch.float32
        assert out["boundary"].dtype == torch.float32
        assert out["distance"].dtype == torch.float32

    def test_no_nan_in_outputs(self) -> None:
        torch = _torch()
        net = _make_net()
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        assert not torch.isnan(out["extent"]).any()
        assert not torch.isnan(out["boundary"]).any()
        assert not torch.isnan(out["distance"]).any()


# ---------------------------------------------------------------------------
# Deep supervision / aux
# ---------------------------------------------------------------------------


class TestDeepSupervision:
    @pytest.mark.parametrize("depth,expected_n_aux", [(2, 0), (3, 1), (4, 2), (5, 3)])
    def test_aux_length_matches_depth(self, depth: int, expected_n_aux: int) -> None:
        torch = _torch()
        net = _make_net(in_channels=9, encoder_depth=depth)
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        assert len(out["aux"]) == expected_n_aux, (
            f"depth={depth}: expected {expected_n_aux} aux entries, got {len(out['aux'])}"
        )

    def test_aux_entries_have_required_keys(self) -> None:
        torch = _torch()
        net = _make_net(encoder_depth=4)
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        for i, entry in enumerate(out["aux"]):
            for key in ("extent", "boundary", "distance"):
                assert key in entry, f"aux[{i}] missing key '{key}'"

    def test_aux_shapes_at_full_resolution(self) -> None:
        """All aux entries must be upsampled to input H×W."""
        torch = _torch()
        net = _make_net(encoder_depth=4)
        with torch.no_grad():
            out = net(_rand_input(2, 9, 64, 64))
        for i, entry in enumerate(out["aux"]):
            assert entry["extent"].shape == (2, 1, 64, 64), f"aux[{i}].extent shape wrong"
            assert entry["boundary"].shape == (2, 3, 64, 64), f"aux[{i}].boundary shape wrong"
            assert entry["distance"].shape == (2, 1, 64, 64), f"aux[{i}].distance shape wrong"

    def test_aux_empty_for_depth_2(self) -> None:
        torch = _torch()
        net = _make_net(encoder_depth=2)
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        assert out["aux"] == []

    def test_aux_no_nan(self) -> None:
        torch = _torch()
        net = _make_net(encoder_depth=4)
        with torch.no_grad():
            out = net(_rand_input(2, 9))
        for i, entry in enumerate(out["aux"]):
            for key in ("extent", "boundary", "distance"):
                assert not torch.isnan(entry[key]).any(), f"NaN in aux[{i}].{key}"


# ---------------------------------------------------------------------------
# build_model factory
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_raw8_produces_9_in_channels(self) -> None:
        _torch()
        from ai_fields.module_net_train.schemas import NetTrainConfig

        cfg = NetTrainConfig(feature_mode="raw8")
        net = model_module.build_model(cfg)
        assert net.in_channels == 9

    def test_raw8_idx3_produces_12_in_channels(self) -> None:
        _torch()
        from ai_fields.module_net_train.schemas import NetTrainConfig

        cfg = NetTrainConfig(feature_mode="raw8_idx3")
        net = model_module.build_model(cfg)
        assert net.in_channels == 12

    def test_build_model_uses_encoder_depth(self) -> None:
        _torch()
        from ai_fields.module_net_train.schemas import ModelConfig, NetTrainConfig

        cfg = NetTrainConfig(model=ModelConfig(encoder_depth=3, base_channels=16))
        net = model_module.build_model(cfg)
        assert net.encoder_depth == 3

    def test_build_model_uses_base_channels(self) -> None:
        _torch()
        from ai_fields.module_net_train.schemas import ModelConfig, NetTrainConfig

        cfg = NetTrainConfig(model=ModelConfig(encoder_depth=4, base_channels=16))
        net = model_module.build_model(cfg)
        assert net.base_channels == 16

    def test_build_model_forward_runs(self) -> None:
        torch = _torch()
        from ai_fields.module_net_train.schemas import NetTrainConfig

        cfg = NetTrainConfig(feature_mode="raw8_idx3")
        net = model_module.build_model(cfg)
        net.eval()
        x = torch.rand(1, 12, 64, 64)
        with torch.no_grad():
            out = net(x)
        assert out["extent"].shape == (1, 1, 64, 64)
        assert out["boundary"].shape == (1, 3, 64, 64)
        assert out["distance"].shape == (1, 1, 64, 64)


# ---------------------------------------------------------------------------
# Compatibility with losses.py
# ---------------------------------------------------------------------------


class TestModelLossCompatibility:
    """Model outputs must be directly consumable by MultitaskLoss.forward()."""

    def test_model_outputs_accepted_by_multitask_loss(self) -> None:
        torch = _torch()
        from ai_fields.module_net_train.losses import MultitaskLoss

        net = _make_net(in_channels=9)
        loss_fn = MultitaskLoss()

        x = torch.rand(2, 9, 64, 64)
        valid = torch.ones(2, 64, 64, dtype=torch.bool)
        targets = {
            "extent": torch.zeros(2, 64, 64, dtype=torch.long),
            "boundary": torch.zeros(2, 64, 64, dtype=torch.long),
            "distance": torch.zeros(2, 64, 64),
        }

        with torch.no_grad():
            model_out = net(x)

        preds = {k: model_out[k] for k in ("extent", "boundary", "distance")}
        result = loss_fn(preds, targets, valid)
        assert "total" in result
        assert not torch.isnan(result["total"])

    def test_aux_outputs_also_accepted_by_multitask_loss(self) -> None:
        """Each aux entry has the same shape contract as the main output."""
        torch = _torch()
        from ai_fields.module_net_train.losses import MultitaskLoss

        net = _make_net(in_channels=9, encoder_depth=4)
        loss_fn = MultitaskLoss()

        x = torch.rand(2, 9, 64, 64)
        valid = torch.ones(2, 64, 64, dtype=torch.bool)
        targets = {
            "extent": torch.zeros(2, 64, 64, dtype=torch.long),
            "boundary": torch.zeros(2, 64, 64, dtype=torch.long),
            "distance": torch.zeros(2, 64, 64),
        }

        with torch.no_grad():
            model_out = net(x)

        for i, aux_entry in enumerate(model_out["aux"]):
            result = loss_fn(aux_entry, targets, valid)
            assert not torch.isnan(result["total"]), f"NaN in aux[{i}] loss"

    def test_extent_head_compatible_with_extent_loss(self) -> None:
        torch = _torch()
        from ai_fields.module_net_train.losses import extent_loss

        net = _make_net(in_channels=9)
        x = torch.rand(1, 9, 64, 64)
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        valid = torch.ones(1, 64, 64, dtype=torch.bool)

        with torch.no_grad():
            out = net(x)
        loss = extent_loss(out["extent"], target, valid)
        assert not torch.isnan(loss)

    def test_boundary_head_compatible_with_boundary_loss(self) -> None:
        torch = _torch()
        from ai_fields.module_net_train.losses import boundary_loss

        net = _make_net(in_channels=9)
        x = torch.rand(1, 9, 64, 64)
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        valid = torch.ones(1, 64, 64, dtype=torch.bool)

        with torch.no_grad():
            out = net(x)
        loss = boundary_loss(out["boundary"], target, valid)
        assert not torch.isnan(loss)

    def test_distance_head_compatible_with_distance_loss(self) -> None:
        torch = _torch()
        from ai_fields.module_net_train.losses import distance_loss

        net = _make_net(in_channels=9)
        x = torch.rand(1, 9, 64, 64)
        target = torch.zeros(1, 64, 64)
        valid = torch.ones(1, 64, 64, dtype=torch.bool)

        with torch.no_grad():
            out = net(x)
        loss = distance_loss(out["distance"], target, valid)
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Architecture structural properties
# ---------------------------------------------------------------------------


class TestArchitectureProperties:
    def test_in_channels_attribute(self) -> None:
        _torch()
        assert _make_net(in_channels=9).in_channels == 9
        assert _make_net(in_channels=12).in_channels == 12

    def test_encoder_depth_attribute(self) -> None:
        _torch()
        for depth in (2, 3, 4):
            assert _make_net(encoder_depth=depth).encoder_depth == depth

    def test_deeper_model_has_more_encoder_stages(self) -> None:
        _torch()
        net3 = _make_net(encoder_depth=3)
        net4 = _make_net(encoder_depth=4)
        assert len(net4.encoders) > len(net3.encoders)

    def test_larger_base_channels_more_params(self) -> None:
        _torch()

        def param_count(net):
            return sum(p.numel() for p in net.parameters())

        assert param_count(_make_net(base_channels=32)) > param_count(_make_net(base_channels=16))

    def test_two_edge_enhance_blocks(self) -> None:
        _torch()
        assert len(_make_net(encoder_depth=4).edge_enhance) == 2

    def test_model_has_refine_block(self) -> None:
        _torch()
        assert isinstance(_make_net().refine, model_module.RefineBlock)

    def test_model_has_three_output_heads(self) -> None:
        _torch()
        net = _make_net()
        assert isinstance(net.extent_head, model_module.OutputHead)
        assert isinstance(net.boundary_head, model_module.OutputHead)
        assert isinstance(net.distance_head, model_module.OutputHead)
