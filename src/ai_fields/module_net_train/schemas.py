"""Dataclass schemas for module_net_train configuration.

These dataclasses represent the fully-validated, resolved configuration for a
module_net_train run.  They enforce the data contract invariants defined in:

  - DATA_CONTRACT.md  §7 (feature contract), §9 (ignore/valid policy)
  - module_net_train.md  §3–§13
  - DECISIONS.md  DEC-002, DEC-003, DEC-005, DEC-008

Rules that code elsewhere may not override:
  - feature_mode must be "raw8" or "raw8_idx3"  (DATA_CONTRACT.md §7.1).
  - valid_as_input_channel must be True  (DATA_CONTRACT.md §6.4, DEC-002).
  - Loss weights must all be > 0.
  - optimizer.name must be an accepted value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import ClassVar, Optional

from ai_fields.common.constants import FEATURE_MODES
from ai_fields.common.errors import (
    ContractError,
    FeatureModeError,
    ValidPolicyError,
)

# Canonical monitored metric policy name for module_net_train baseline.
MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1 = "composite_boundary_extent_f1"

# Deprecated names are kept only to emit explicit migration errors.
MONITORED_METRIC_INTERIM_VAL_TOTAL_LOSS = "interim_val_total_loss"
MONITORED_METRIC_COMPOSITE_BOUNDARY_F1_EXTENT_F1 = "composite_boundary_f1_extent_f1"

DEPRECATED_MONITORED_METRIC_NAMES = {
    MONITORED_METRIC_INTERIM_VAL_TOTAL_LOSS,
    MONITORED_METRIC_COMPOSITE_BOUNDARY_F1_EXTENT_F1,
}

MONITORED_METRIC_EXPECTED_MODE: dict[str, str] = {
    MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1: "max",
}


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Architecture configuration.

    module_net_train.md §5–§6.
    """

    architecture: str = "edge_aware_multitask_v1"
    """Baseline architecture name.  Only "edge_aware_multitask_v1" in v1."""

    encoder_depth: int = 4
    """Number of encoder levels (4–5 recommended)."""

    base_channels: int = 32
    """Base channel width at the first encoder level."""

    def validate(self) -> None:
        if not isinstance(self.architecture, str):
            raise ContractError(
                f"model.architecture must be a string, got {type(self.architecture).__name__}."
            )
        if self.architecture != "edge_aware_multitask_v1":
            raise ContractError(
                f"model.architecture '{self.architecture}' is not supported.  "
                "Only 'edge_aware_multitask_v1' is accepted in v1 "
                "(module_net_train.md §5)."
            )
        if isinstance(self.encoder_depth, bool) or not isinstance(self.encoder_depth, int):
            raise ContractError(
                f"model.encoder_depth must be an integer, got {type(self.encoder_depth).__name__}."
            )
        if self.encoder_depth < 2:
            raise ContractError(
                f"model.encoder_depth must be >= 2, got {self.encoder_depth}."
            )
        if isinstance(self.base_channels, bool) or not isinstance(self.base_channels, int):
            raise ContractError(
                f"model.base_channels must be an integer, got {type(self.base_channels).__name__}."
            )
        if self.base_channels < 1:
            raise ContractError(
                f"model.base_channels must be >= 1, got {self.base_channels}."
            )


# ---------------------------------------------------------------------------
# Loss config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossConfig:
    """Loss composition configuration.

    module_net_train.md §8.  Baseline weights DEC-005.
    """

    extent_weight: float = 1.0
    """Weight for the extent loss component.  Baseline v1: 1.0."""

    boundary_weight: float = 2.5
    """Weight for the boundary loss component.  Baseline v1: 2.5 (module_net_train.md §8.6)."""

    distance_weight: float = 1.0
    """Weight for the distance regression loss.  Baseline v1: 1.0."""

    aux_weight: float = 0.4
    """Weight applied to each deep supervision auxiliary output (0.3–0.5 recommended).
    Value 0.0 is allowed for explicit auxiliary-supervision ablations.
    """

    extent_aux_weight: Optional[float] = None
    """Optional per-head aux weight override for extent.
    When null, falls back to aux_weight.
    """

    boundary_aux_weight: Optional[float] = None
    """Optional per-head aux weight override for boundary.
    When null, falls back to aux_weight.
    """

    distance_aux_weight: Optional[float] = None
    """Optional per-head aux weight override for distance.
    When null, falls back to aux_weight.
    """

    boundary_lambda_skel_dice: float = 0.5
    """Weight for the soft Dice skeleton term within boundary loss.
    module_net_train.md §22.1:
      boundary_loss = focal_CE_boundary + boundary_lambda_skel_dice * soft_dice_skeleton
    """

    extent_focal_alpha: float = 0.25
    """Foreground balance weight in the focal BCE component of the extent loss.
    Baseline v1: 0.25 (background receives 3× more gradient than foreground).
    alpha=0.50 (symmetric): foreground and background receive equal gradient weight.
    Valid range: (0.0, 1.0) exclusive.
    Only used when extent_loss_mode == "current".
    """

    extent_loss_mode: str = "current"
    """Extent loss computation mode.  Controls which loss formula is used for the extent head.
    Accepted values:
      "current"         — focal BCE (alpha, gamma) + soft Dice.  Baseline v1.
      "legacy_bce_dice" — plain (non-focal) BCE + soft Dice, equal weights on valid pixels.
                          Matches the old model's extent_loss() in my_project/.
    Changing this is an experiment-level ablation; do not change without updating the
    train manifest (export.py records extent_loss_mode under the loss section).
    """

    # Accepted extent loss modes.  ClassVar so dataclass does not treat it as a field.
    _ACCEPTED_EXTENT_LOSS_MODES: ClassVar[frozenset] = frozenset({"current", "legacy_bce_dice"})

    def validate(self) -> None:
        for attr in (
            "extent_weight", "boundary_weight", "distance_weight", "boundary_lambda_skel_dice"
        ):
            val = getattr(self, attr)
            if isinstance(val, bool) or not isinstance(val, Real):
                raise ContractError(
                    f"loss.{attr} must be a number, got {type(val).__name__}."
                )
            if val <= 0:
                raise ContractError(
                    f"loss.{attr} must be > 0, got {val}.  "
                    "All loss weights must be positive (module_net_train.md §8.6)."
                )
        aux_val = self.aux_weight
        if isinstance(aux_val, bool) or not isinstance(aux_val, Real):
            raise ContractError(
                f"loss.aux_weight must be a number, got {type(aux_val).__name__}."
            )
        if aux_val < 0:
            raise ContractError(
                f"loss.aux_weight must be >= 0, got {aux_val}.  "
                "Use 0.0 only for explicit deep-supervision ablations."
            )
        for attr in ("extent_aux_weight", "boundary_aux_weight", "distance_aux_weight"):
            val = getattr(self, attr)
            if val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, Real):
                raise ContractError(
                    f"loss.{attr} must be a number or null, got {type(val).__name__}."
                )
            if val < 0:
                raise ContractError(
                    f"loss.{attr} must be >= 0 when provided, got {val}."
                )
        val = self.extent_focal_alpha
        if isinstance(val, bool) or not isinstance(val, Real):
            raise ContractError(
                f"loss.extent_focal_alpha must be a number, got {type(val).__name__}."
            )
        if not (0.0 < float(val) < 1.0):
            raise ContractError(
                f"loss.extent_focal_alpha must be in (0, 1) exclusive, got {val}."
            )
        if not isinstance(self.extent_loss_mode, str):
            raise ContractError(
                f"loss.extent_loss_mode must be a string, got {type(self.extent_loss_mode).__name__}."
            )
        if self.extent_loss_mode not in self._ACCEPTED_EXTENT_LOSS_MODES:
            raise ContractError(
                f"loss.extent_loss_mode '{self.extent_loss_mode}' is not accepted.  "
                f"Accepted values: {sorted(self._ACCEPTED_EXTENT_LOSS_MODES)}."
            )


# ---------------------------------------------------------------------------
# Optimizer config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration.

    module_net_train.md §12.2.  Baseline: AdamW.
    """

    name: str = "adamw"
    """Optimizer name.  Only "adamw" is accepted in baseline v1."""

    lr: float = 1e-4
    """Initial learning rate."""

    weight_decay: float = 1e-2
    """L2 regularization coefficient."""

    def validate(self) -> None:
        if not isinstance(self.name, str):
            raise ContractError(
                f"optimizer.name must be a string, got {type(self.name).__name__}."
            )
        accepted = {"adamw"}
        if self.name not in accepted:
            raise ContractError(
                f"optimizer.name '{self.name}' is not accepted.  "
                f"Accepted values: {sorted(accepted)} (module_net_train.md §12.2)."
            )
        for attr in ("lr", "weight_decay"):
            val = getattr(self, attr)
            if isinstance(val, bool) or not isinstance(val, Real):
                raise ContractError(
                    f"optimizer.{attr} must be a number, got {type(val).__name__}."
                )
            if val <= 0:
                raise ContractError(
                    f"optimizer.{attr} must be > 0, got {val}."
                )


# ---------------------------------------------------------------------------
# Scheduler config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate scheduler configuration.

    module_net_train.md §12.3.  Baseline: cosine with warmup.
    """

    name: str = "cosine_with_warmup"
    """Scheduler name.  Accepted: "cosine_with_warmup", "one_cycle", "plateau"."""

    warmup_epochs: int = 5
    """Number of linear warmup epochs (used by cosine_with_warmup)."""

    min_lr: float = 1e-6
    """Minimum learning rate at the end of cosine decay."""

    def validate(self) -> None:
        if not isinstance(self.name, str):
            raise ContractError(
                f"scheduler.name must be a string, got {type(self.name).__name__}."
            )
        accepted = {"cosine_with_warmup", "one_cycle", "plateau"}
        if self.name not in accepted:
            raise ContractError(
                f"scheduler.name '{self.name}' is not accepted.  "
                f"Accepted values: {sorted(accepted)} (module_net_train.md §12.3)."
            )
        if isinstance(self.warmup_epochs, bool) or not isinstance(self.warmup_epochs, int):
            raise ContractError(
                f"scheduler.warmup_epochs must be an integer, got {type(self.warmup_epochs).__name__}."
            )
        if self.warmup_epochs < 0:
            raise ContractError(
                f"scheduler.warmup_epochs must be >= 0, got {self.warmup_epochs}."
            )
        if isinstance(self.min_lr, bool) or not isinstance(self.min_lr, Real):
            raise ContractError(
                f"scheduler.min_lr must be a number, got {type(self.min_lr).__name__}."
            )
        if self.min_lr < 0:
            raise ContractError(
                f"scheduler.min_lr must be >= 0, got {self.min_lr}."
            )


# ---------------------------------------------------------------------------
# Training runtime config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageAwareSamplingConfig:
    """Optional train-split sampling policy that reweights extent-coverage buckets.

    The policy is train-only and does not modify dataset contents, validation
    sampling, or any model/loss contract. It is disabled by default.
    """

    enabled: bool = False
    """Enable extent-coverage-aware weighted sampling on train split only."""

    coverage_quantile_low: float = 1.0 / 3.0
    """Lower quantile boundary for low/medium bucket split."""

    coverage_quantile_high: float = 2.0 / 3.0
    """Upper quantile boundary for medium/high bucket split."""

    bucket_weights: Mapping[str, float] = field(
        default_factory=lambda: {"low": 1.0, "medium": 1.0, "high": 1.0}
    )
    """Sampling weights per coverage bucket: low, medium, high."""

    replacement: bool = True
    """WeightedRandomSampler replacement policy (baseline: True)."""

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ContractError(
                f"training.coverage_aware_sampling.enabled must be a boolean, "
                f"got {type(self.enabled).__name__}."
            )
        for attr in ("coverage_quantile_low", "coverage_quantile_high"):
            val = getattr(self, attr)
            if isinstance(val, bool) or not isinstance(val, Real):
                raise ContractError(
                    f"training.coverage_aware_sampling.{attr} must be numeric, "
                    f"got {type(val).__name__}."
                )
        q_low = float(self.coverage_quantile_low)
        q_high = float(self.coverage_quantile_high)
        if not (0.0 < q_low < q_high < 1.0):
            raise ContractError(
                "training.coverage_aware_sampling quantiles must satisfy "
                f"0 < low < high < 1, got low={q_low}, high={q_high}."
            )

        if not isinstance(self.bucket_weights, Mapping):
            raise ContractError(
                "training.coverage_aware_sampling.bucket_weights must be a mapping."
            )
        expected_keys = {"low", "medium", "high"}
        keys = set(self.bucket_weights.keys())
        if keys != expected_keys:
            raise ContractError(
                "training.coverage_aware_sampling.bucket_weights must contain exactly "
                f"{sorted(expected_keys)}, got {sorted(keys)}."
            )
        for key in ("low", "medium", "high"):
            val = self.bucket_weights.get(key)
            if isinstance(val, bool) or not isinstance(val, Real):
                raise ContractError(
                    "training.coverage_aware_sampling.bucket_weights values must be numeric, "
                    f"got {type(val).__name__} for key '{key}'."
                )
            if float(val) <= 0:
                raise ContractError(
                    f"training.coverage_aware_sampling.bucket_weights['{key}'] "
                    f"must be > 0, got {val}."
                )

        if not isinstance(self.replacement, bool):
            raise ContractError(
                "training.coverage_aware_sampling.replacement must be a boolean."
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop runtime configuration.

    module_net_train.md §12.
    """

    batch_size: int = 8
    """Per-device batch size."""

    num_epochs: int = 100
    """Total training epochs."""

    gradient_clip: float = 1.0
    """Max gradient norm for clipping (module_net_train.md §12.5)."""

    amp: bool = True
    """Request mixed-precision training.  Actual use subject to hardware-adaptive policy."""

    seed: int = 42
    """Global random seed for reproducibility."""

    num_workers: int = 4
    """DataLoader worker count."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for effective batch size scaling."""

    device: Optional[str] = None
    """Requested device: "cuda", "mps", "cpu", or None for auto-detect.
    None triggers hardware-adaptive auto-selection (CUDA -> MPS -> CPU).
    """

    augment: bool = True
    """Enable spatial augmentation on the training split (module_net_train.md §11.2).
    When True, horizontal/vertical flips and 90° rotations are applied.
    Must be True/False; defaults to True for baseline training."""

    coverage_aware_sampling: CoverageAwareSamplingConfig | Mapping[str, object] = field(
        default_factory=CoverageAwareSamplingConfig
    )
    """Optional train-only coverage-bucket weighted sampling policy.
    Accepts CoverageAwareSamplingConfig or an equivalent mapping loaded from YAML.
    Disabled by default to preserve baseline behavior.
    """

    def validate(self) -> None:
        for int_attr in ("batch_size", "num_epochs", "seed", "num_workers", "gradient_accumulation_steps"):
            val = getattr(self, int_attr)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ContractError(
                    f"training.{int_attr} must be an integer, got {type(val).__name__}."
                )
            if val < 1:
                raise ContractError(
                    f"training.{int_attr} must be >= 1, got {val}."
                )
        if isinstance(self.gradient_clip, bool) or not isinstance(self.gradient_clip, Real):
            raise ContractError(
                f"training.gradient_clip must be a number, got {type(self.gradient_clip).__name__}."
            )
        if self.gradient_clip <= 0:
            raise ContractError(
                f"training.gradient_clip must be > 0, got {self.gradient_clip}."
            )
        if not isinstance(self.amp, bool):
            raise ContractError(
                f"training.amp must be a boolean, got {type(self.amp).__name__}."
            )
        if self.device is not None:
            if not isinstance(self.device, str):
                raise ContractError(
                    f"training.device must be a string or null, got {type(self.device).__name__}."
                )
            accepted_devices = {"cuda", "mps", "cpu"}
            if self.device not in accepted_devices:
                raise ContractError(
                    f"training.device '{self.device}' is not accepted.  "
                    f"Accepted values: {sorted(accepted_devices)} or null for auto-detect."
                )
        if not isinstance(self.augment, bool):
            raise ContractError(
                f"training.augment must be a boolean, got {type(self.augment).__name__}."
            )
        coverage_cfg = self.coverage_aware_sampling
        if isinstance(coverage_cfg, CoverageAwareSamplingConfig):
            coverage_cfg.validate()
        elif isinstance(coverage_cfg, Mapping):
            CoverageAwareSamplingConfig(**dict(coverage_cfg)).validate()
        else:
            raise ContractError(
                "training.coverage_aware_sampling must be a mapping/object or "
                "CoverageAwareSamplingConfig."
            )


# ---------------------------------------------------------------------------
# Monitoring policy config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringConfig:
    """Monitored metric selection policy for checkpoint ranking."""

    monitored_metric_name: str = MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1
    """Metric name used for best-checkpoint selection."""

    monitored_metric_mode: str = "max"
    """Improvement direction: 'max' for score metrics, 'min' for loss metrics."""

    def validate(self) -> None:
        if not isinstance(self.monitored_metric_name, str) or self.monitored_metric_name.strip() == "":
            raise ContractError("monitoring.monitored_metric_name must be a non-empty string.")
        if self.monitored_metric_name in DEPRECATED_MONITORED_METRIC_NAMES:
            raise ContractError(
                f"monitoring.monitored_metric_name '{self.monitored_metric_name}' is deprecated "
                f"and no longer supported. Use '{MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1}'."
            )
        expected_mode = MONITORED_METRIC_EXPECTED_MODE.get(self.monitored_metric_name)
        if expected_mode is None:
            raise ContractError(
                f"monitoring.monitored_metric_name '{self.monitored_metric_name}' is not supported. "
                f"Accepted values: {sorted(MONITORED_METRIC_EXPECTED_MODE)}."
            )

        if not isinstance(self.monitored_metric_mode, str):
            raise ContractError(
                "monitoring.monitored_metric_mode must be a string."
            )
        if self.monitored_metric_mode not in {"min", "max"}:
            raise ContractError(
                "monitoring.monitored_metric_mode must be 'min' or 'max'."
            )
        if self.monitored_metric_mode != expected_mode:
            raise ContractError(
                f"monitoring.monitored_metric_mode '{self.monitored_metric_mode}' is inconsistent "
                f"with monitoring.monitored_metric_name '{self.monitored_metric_name}'. "
                f"Expected '{expected_mode}'."
            )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetTrainConfig:
    """Fully-validated configuration for a module_net_train run.

    Passed into every stage of the module.  Constructed by build_config()
    and must not be mutated at runtime.

    Contract invariants (must all hold after validate()):
      - feature_mode in ("raw8", "raw8_idx3")  — DATA_CONTRACT.md §7.1
      - valid_as_input_channel is True  — DATA_CONTRACT.md §6.4, DEC-002
      - All sub-configs pass their own validate()
    """

    feature_mode: str = "raw8"
    """Dataset-side feature mode.  "raw8" or "raw8_idx3" only."""

    valid_as_input_channel: bool = True
    """valid must always be an input channel (DATA_CONTRACT.md §6.4, DEC-002).
    Setting this to False raises ValidPolicyError.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def validate(self) -> None:
        """Run all contract checks.  Raises a ContractError subclass on failure."""
        # feature_mode — DATA_CONTRACT.md §7.1
        if not isinstance(self.feature_mode, str):
            raise FeatureModeError(
                f"feature_mode must be a string, got {type(self.feature_mode).__name__}."
            )
        if self.feature_mode not in FEATURE_MODES:
            raise FeatureModeError(
                f"feature_mode '{self.feature_mode}' is not supported.  "
                f"Accepted values: {list(FEATURE_MODES)}.  "
                "(DATA_CONTRACT.md §7.1, DECISIONS.md DEC-003)"
            )

        # valid_as_input_channel — DATA_CONTRACT.md §6.4, DEC-002
        if not isinstance(self.valid_as_input_channel, bool):
            raise ValidPolicyError(
                "valid_as_input_channel must be a boolean, "
                f"got {type(self.valid_as_input_channel).__name__}."
            )
        if not self.valid_as_input_channel:
            raise ValidPolicyError(
                "valid_as_input_channel must be True.  "
                "valid must remain both a service mask and an input channel "
                "(DATA_CONTRACT.md §6.4, DECISIONS.md DEC-002)."
            )

        # Sub-config validation
        self.model.validate()
        self.loss.validate()
        self.optimizer.validate()
        self.scheduler.validate()
        self.training.validate()
        self.monitoring.validate()
