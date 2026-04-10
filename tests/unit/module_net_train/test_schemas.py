"""Unit tests for module_net_train.schemas.

Verifies that NetTrainConfig and its sub-configs:
  - accept valid baseline values without raising
  - raise the correct ContractError subclass on every known violation

Contract anchors:
  DATA_CONTRACT.md §7.1: feature_mode must be "raw8" or "raw8_idx3".
  DATA_CONTRACT.md §6.4, DEC-002: valid_as_input_channel must be True.
  module_net_train.md §8.6: loss weights must be > 0.
  module_net_train.md §12.2: optimizer "adamw" only in baseline v1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_fields.common.errors import (
    ContractError,
    FeatureModeError,
    ValidPolicyError,
)
from ai_fields.module_net_train.schemas import (
    LossConfig,
    MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1,
    MONITORED_METRIC_INTERIM_VAL_TOTAL_LOSS,
    ModelConfig,
    MonitoringConfig,
    NetTrainConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# NetTrainConfig — baseline construction
# ---------------------------------------------------------------------------


class TestNetTrainConfigDefaults:
    def test_default_construction_does_not_raise(self):
        cfg = NetTrainConfig()
        cfg.validate()

    def test_raw8_accepted(self):
        cfg = NetTrainConfig(feature_mode="raw8")
        cfg.validate()

    def test_raw8_idx3_accepted(self):
        cfg = NetTrainConfig(feature_mode="raw8_idx3")
        cfg.validate()

    def test_is_frozen(self):
        cfg = NetTrainConfig()
        with pytest.raises(Exception):
            cfg.feature_mode = "raw8_idx3"  # type: ignore[misc]


class TestNetTrainFeatureMode:
    def test_unknown_mode_raises_feature_mode_error(self):
        with pytest.raises(FeatureModeError):
            NetTrainConfig(feature_mode="raw11").validate()

    def test_non_string_mode_raises(self):
        with pytest.raises(FeatureModeError):
            NetTrainConfig(feature_mode=8).validate()  # type: ignore[arg-type]


class TestNetTrainValidAsInputChannel:
    def test_true_accepted(self):
        cfg = NetTrainConfig(valid_as_input_channel=True)
        cfg.validate()  # no error

    def test_false_raises_valid_policy_error(self):
        """DEC-002: valid_as_input_channel must be True."""
        with pytest.raises(ValidPolicyError, match="valid_as_input_channel must be True"):
            NetTrainConfig(valid_as_input_channel=False).validate()

    def test_non_bool_raises(self):
        with pytest.raises(ValidPolicyError):
            NetTrainConfig(valid_as_input_channel=1).validate()  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults_valid(self):
        ModelConfig().validate()

    def test_unknown_architecture_raises(self):
        with pytest.raises(ContractError, match="architecture"):
            ModelConfig(architecture="unet_v2").validate()

    def test_encoder_depth_too_small_raises(self):
        with pytest.raises(ContractError, match="encoder_depth"):
            ModelConfig(encoder_depth=1).validate()

    def test_base_channels_zero_raises(self):
        with pytest.raises(ContractError, match="base_channels"):
            ModelConfig(base_channels=0).validate()


# ---------------------------------------------------------------------------
# LossConfig
# ---------------------------------------------------------------------------


class TestLossConfig:
    def test_defaults_valid(self):
        LossConfig().validate()

    def test_extent_weight_zero_raises(self):
        with pytest.raises(ContractError, match="extent_weight"):
            LossConfig(extent_weight=0.0).validate()

    def test_boundary_weight_zero_raises(self):
        with pytest.raises(ContractError, match="boundary_weight"):
            LossConfig(boundary_weight=0.0).validate()

    def test_distance_weight_negative_raises(self):
        with pytest.raises(ContractError, match="distance_weight"):
            LossConfig(distance_weight=-1.0).validate()

    def test_aux_weight_zero_raises(self):
        with pytest.raises(ContractError, match="aux_weight"):
            LossConfig(aux_weight=0.0).validate()

    def test_non_numeric_weight_raises(self):
        with pytest.raises(ContractError):
            LossConfig(extent_weight="high").validate()  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    def test_defaults_valid(self):
        OptimizerConfig().validate()

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ContractError, match="optimizer.name"):
            OptimizerConfig(name="sgd").validate()

    def test_zero_lr_raises(self):
        with pytest.raises(ContractError, match="lr"):
            OptimizerConfig(lr=0.0).validate()

    def test_negative_weight_decay_raises(self):
        with pytest.raises(ContractError, match="weight_decay"):
            OptimizerConfig(weight_decay=-0.01).validate()


# ---------------------------------------------------------------------------
# SchedulerConfig
# ---------------------------------------------------------------------------


class TestSchedulerConfig:
    def test_defaults_valid(self):
        SchedulerConfig().validate()

    def test_all_accepted_names(self):
        for name in ("cosine_with_warmup", "one_cycle", "plateau"):
            SchedulerConfig(name=name).validate()

    def test_unknown_scheduler_raises(self):
        with pytest.raises(ContractError, match="scheduler.name"):
            SchedulerConfig(name="step_lr").validate()

    def test_negative_warmup_raises(self):
        with pytest.raises(ContractError, match="warmup_epochs"):
            SchedulerConfig(warmup_epochs=-1).validate()

    def test_zero_warmup_accepted(self):
        SchedulerConfig(warmup_epochs=0).validate()  # no error

    def test_negative_min_lr_raises(self):
        with pytest.raises(ContractError, match="min_lr"):
            SchedulerConfig(min_lr=-1e-7).validate()


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults_valid(self):
        TrainingConfig().validate()

    def test_zero_batch_size_raises(self):
        with pytest.raises(ContractError, match="batch_size"):
            TrainingConfig(batch_size=0).validate()

    def test_zero_epochs_raises(self):
        with pytest.raises(ContractError, match="num_epochs"):
            TrainingConfig(num_epochs=0).validate()

    def test_zero_gradient_clip_raises(self):
        with pytest.raises(ContractError, match="gradient_clip"):
            TrainingConfig(gradient_clip=0.0).validate()

    def test_unknown_device_raises(self):
        with pytest.raises(ContractError, match="device"):
            TrainingConfig(device="tpu").validate()

    def test_accepted_devices(self):
        for dev in ("cuda", "mps", "cpu"):
            TrainingConfig(device=dev).validate()

    def test_none_device_accepted(self):
        TrainingConfig(device=None).validate()  # auto-detect

    def test_amp_non_bool_raises(self):
        with pytest.raises(ContractError, match="amp"):
            TrainingConfig(amp=1).validate()  # type: ignore[arg-type]

    def test_zero_accumulation_steps_raises(self):
        with pytest.raises(ContractError, match="gradient_accumulation_steps"):
            TrainingConfig(gradient_accumulation_steps=0).validate()


# ---------------------------------------------------------------------------
# MonitoringConfig
# ---------------------------------------------------------------------------


class TestMonitoringConfig:
    def test_defaults_valid(self):
        MonitoringConfig().validate()

    def test_interim_policy_valid(self):
        MonitoringConfig(
            monitored_metric_name=MONITORED_METRIC_INTERIM_VAL_TOTAL_LOSS,
            monitored_metric_mode="min",
        ).validate()

    def test_unknown_metric_raises(self):
        with pytest.raises(ContractError, match="monitored_metric_name"):
            MonitoringConfig(
                monitored_metric_name="mystery_metric",
                monitored_metric_mode="max",
            ).validate()

    def test_mode_mismatch_raises(self):
        with pytest.raises(ContractError, match="inconsistent"):
            MonitoringConfig(
                monitored_metric_name=MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1,
                monitored_metric_mode="min",
            ).validate()


class TestBaselineConfigFiles:
    def _load_cfg_from_yaml(self, path: Path) -> NetTrainConfig:
        yaml = pytest.importorskip("yaml")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return NetTrainConfig(
            feature_mode=payload["feature_mode"],
            valid_as_input_channel=payload["valid_as_input_channel"],
            model=ModelConfig(**payload["model"]),
            loss=LossConfig(**payload["loss"]),
            optimizer=OptimizerConfig(**payload["optimizer"]),
            scheduler=SchedulerConfig(**payload["scheduler"]),
            training=TrainingConfig(**payload["training"]),
            monitoring=MonitoringConfig(**payload["monitoring"]),
        )

    def test_baseline_raw8_yaml_contains_valid_monitoring_policy(self):
        cfg = self._load_cfg_from_yaml(
            Path("configs/module_net_train/baseline.raw8.yaml")
        )
        cfg.validate()
        assert cfg.monitoring.monitored_metric_name == MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1
        assert cfg.monitoring.monitored_metric_mode == "max"

    def test_baseline_raw8_idx3_yaml_contains_valid_monitoring_policy(self):
        cfg = self._load_cfg_from_yaml(
            Path("configs/module_net_train/baseline.raw8_idx3.yaml")
        )
        cfg.validate()
        assert cfg.monitoring.monitored_metric_name == MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1
        assert cfg.monitoring.monitored_metric_mode == "max"
