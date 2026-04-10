"""Unit tests for module_net_train.trainer (Stage D train/eval loop).

These tests verify that the trainer layer:
  - reuses existing model/loss contracts from stages A/B/C;
  - performs train/eval steps with clear, testable summaries;
  - preserves strict contract behavior (raises ContractError on bad inputs).
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_net_train.losses import MultitaskLoss
from ai_fields.module_net_train.model import build_model
from ai_fields.module_net_train.schemas import (
    ModelConfig,
    NetTrainConfig,
    SchedulerConfig,
    TrainingConfig,
)
from ai_fields.module_net_train.trainer import (
    build_scheduler,
    build_optimizer,
    evaluate_one_epoch,
    train_one_epoch,
    train_step,
)

torch = pytest.importorskip("torch")


def _make_config(*, encoder_depth: int = 4, base_channels: int = 8) -> NetTrainConfig:
    cfg = NetTrainConfig(
        feature_mode="raw8",
        model=ModelConfig(encoder_depth=encoder_depth, base_channels=base_channels),
        training=TrainingConfig(batch_size=2, num_epochs=1, num_workers=1, device="cpu"),
    )
    cfg.validate()
    return cfg


def _make_batch(
    *,
    batch_size: int = 2,
    in_channels: int = 9,
    h: int = 32,
    w: int = 32,
    all_valid: bool = True,
) -> dict[str, torch.Tensor]:
    image = torch.rand(batch_size, in_channels, h, w, dtype=torch.float32)
    extent = torch.randint(0, 2, (batch_size, h, w), dtype=torch.long)
    boundary = torch.randint(0, 3, (batch_size, h, w), dtype=torch.long)
    distance = torch.rand(batch_size, h, w, dtype=torch.float32)
    valid = torch.ones(batch_size, h, w, dtype=torch.bool)
    if not all_valid:
        valid.zero_()
    return {
        "image": image,
        "extent": extent,
        "boundary": boundary,
        "distance": distance,
        "valid": valid,
    }


def _clone_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def _params_changed(before: list[torch.Tensor], model: torch.nn.Module) -> bool:
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    return any(not torch.allclose(p0, p1) for p0, p1 in zip(before, after, strict=True))


def _params_unchanged(before: list[torch.Tensor], model: torch.nn.Module) -> bool:
    return not _params_changed(before, model)


def _assert_summary_keys(summary: Mapping[str, object]) -> None:
    for key in (
        "extent",
        "boundary",
        "distance",
        "total",
        "aux_total",
        "extent_f1",
        "boundary_f1",
        "n_valid",
        "n_batches",
        "n_samples",
        "n_aux",
    ):
        assert key in summary, f"missing key in trainer summary: {key}"
    assert 0.0 <= float(summary["extent_f1"]) <= 1.0
    assert 0.0 <= float(summary["boundary_f1"]) <= 1.0


class TestTrainerStageD:
    def test_build_optimizer_returns_adamw(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        optimizer = build_optimizer(model, cfg)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_build_scheduler_returns_scheduler_and_step_policy(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler, step_policy = build_scheduler(
            optimizer,
            cfg,
            total_epochs=cfg.training.num_epochs,
        )
        assert scheduler is not None
        assert step_policy == "epoch_end"

    def test_build_scheduler_one_cycle_raises_explicit_contract_error(self) -> None:
        cfg = NetTrainConfig(
            feature_mode="raw8",
            model=ModelConfig(encoder_depth=4, base_channels=8),
            scheduler=SchedulerConfig(name="one_cycle", warmup_epochs=0, min_lr=1e-6),
            training=TrainingConfig(batch_size=2, num_epochs=2, num_workers=1, device="cpu"),
        )
        cfg.validate()
        model = build_model(cfg)
        optimizer = build_optimizer(model, cfg)

        with pytest.raises(ContractError, match="one_cycle"):
            build_scheduler(optimizer, cfg, total_epochs=cfg.training.num_epochs)

    def test_train_step_returns_expected_keys(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        loss_fn = MultitaskLoss.from_config(cfg.loss)
        optimizer = build_optimizer(model, cfg)
        batch = _make_batch(in_channels=model.in_channels)

        result = train_step(
            model,
            batch,
            loss_fn,
            optimizer,
            device="cpu",
            aux_weight=cfg.loss.aux_weight,
            gradient_clip_norm=cfg.training.gradient_clip,
        )

        _assert_summary_keys(result)
        assert result["n_batches"] == 1
        assert result["n_samples"] == batch["image"].shape[0]
        assert result["n_valid"] > 0

    def test_train_step_total_matches_multitask_weights_when_aux_disabled(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        loss_fn = MultitaskLoss.from_config(cfg.loss)
        optimizer = build_optimizer(model, cfg)
        batch = _make_batch(in_channels=model.in_channels)

        result = train_step(
            model,
            batch,
            loss_fn,
            optimizer,
            device="cpu",
            aux_weight=0.0,
        )
        expected_total = (
            cfg.loss.extent_weight * result["extent"]
            + cfg.loss.boundary_weight * result["boundary"]
            + cfg.loss.distance_weight * result["distance"]
        )
        assert result["total"] == pytest.approx(expected_total, rel=1e-5, abs=1e-6)

    def test_optimizer_step_changes_parameters_on_train_path(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        loss_fn = MultitaskLoss.from_config(cfg.loss)
        optimizer = build_optimizer(model, cfg)
        batch = _make_batch(in_channels=model.in_channels)
        params_before = _clone_params(model)

        train_step(
            model,
            batch,
            loss_fn,
            optimizer,
            device="cpu",
            aux_weight=0.0,
        )

        assert _params_changed(params_before, model)

    def test_eval_path_does_not_update_parameters(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        loss_fn = MultitaskLoss.from_config(cfg.loss)
        batch1 = _make_batch(in_channels=model.in_channels)
        batch2 = _make_batch(in_channels=model.in_channels)
        params_before = _clone_params(model)

        summary = evaluate_one_epoch(
            model,
            [batch1, batch2],
            loss_fn,
            device="cpu",
            aux_weight=0.0,
        )

        _assert_summary_keys(summary)
        assert summary["n_batches"] == 2
        assert summary["n_samples"] == 4
        assert _params_unchanged(params_before, model)

    def test_train_one_epoch_returns_aggregated_losses(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        loss_fn = MultitaskLoss.from_config(cfg.loss)
        optimizer = build_optimizer(model, cfg)
        batches = [
            _make_batch(in_channels=model.in_channels),
            _make_batch(in_channels=model.in_channels),
        ]

        summary = train_one_epoch(
            model,
            batches,
            loss_fn,
            optimizer,
            device="cpu",
            aux_weight=0.0,
            gradient_clip_norm=cfg.training.gradient_clip,
        )

        _assert_summary_keys(summary)
        assert summary["n_batches"] == 2
        assert summary["n_samples"] == 4
        expected_total = (
            cfg.loss.extent_weight * summary["extent"]
            + cfg.loss.boundary_weight * summary["boundary"]
            + cfg.loss.distance_weight * summary["distance"]
        )
        assert summary["total"] == pytest.approx(expected_total, rel=1e-5, abs=1e-6)

    def test_eval_uses_aux_outputs_when_aux_weight_enabled(self) -> None:
        cfg = _make_config(encoder_depth=4, base_channels=8)
        model = build_model(cfg)
        loss_fn = MultitaskLoss.from_config(cfg.loss)
        batch = _make_batch(in_channels=model.in_channels)

        no_aux = evaluate_one_epoch(
            model,
            [batch],
            loss_fn,
            device="cpu",
            aux_weight=0.0,
        )
        with_aux = evaluate_one_epoch(
            model,
            [batch],
            loss_fn,
            device="cpu",
            aux_weight=cfg.loss.aux_weight,
        )

        assert no_aux["n_aux"] > 0
        assert with_aux["n_aux"] == no_aux["n_aux"]
        assert with_aux["total"] > no_aux["total"]
        assert with_aux["aux_total"] > 0

    def test_empty_valid_batch_raises_contract_error(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        loss_fn = MultitaskLoss.from_config(cfg.loss)
        optimizer = build_optimizer(model, cfg)
        bad_batch = _make_batch(in_channels=model.in_channels, all_valid=False)

        with pytest.raises(ContractError):
            train_step(
                model,
                bad_batch,
                loss_fn,
                optimizer,
                device="cpu",
            )

        with pytest.raises(ContractError):
            evaluate_one_epoch(
                model,
                [bad_batch],
                loss_fn,
                device="cpu",
            )

    def test_trainer_requires_assembled_image_and_does_not_reassemble(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        loss_fn = MultitaskLoss.from_config(cfg.loss)
        optimizer = build_optimizer(model, cfg)
        batch = _make_batch(in_channels=model.in_channels)

        # "img" is dataset-side naming; trainer requires assembled "image" tensor.
        bad_batch = {
            "img": batch["image"],
            "extent": batch["extent"],
            "boundary": batch["boundary"],
            "distance": batch["distance"],
            "valid": batch["valid"],
        }
        with pytest.raises(ContractError, match="missing required key: 'image'"):
            train_step(
                model,
                bad_batch,  # type: ignore[arg-type]
                loss_fn,
                optimizer,
                device="cpu",
            )

    def test_gradient_accumulation_steps_must_be_positive(self) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        loss_fn = MultitaskLoss.from_config(cfg.loss)
        optimizer = build_optimizer(model, cfg)
        batch = _make_batch(in_channels=model.in_channels)

        with pytest.raises(ContractError, match="gradient_accumulation_steps"):
            train_one_epoch(
                model,
                [batch],
                loss_fn,
                optimizer,
                device="cpu",
                gradient_accumulation_steps=0,
            )
