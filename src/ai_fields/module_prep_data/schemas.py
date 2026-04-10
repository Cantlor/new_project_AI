"""Dataclass schemas for module_prep_data configuration.

These dataclasses represent the fully-validated, resolved configuration for a
module_prep_data run.  They enforce the data contract invariants defined in:

  - DATA_CONTRACT.md  §6 (valid/NoData), §7 (feature contract), §10 (normalization)
  - module_prep_data.md  §9–§14, §17
  - DECISIONS.md  DEC-002, DEC-003, DEC-008

Rules that code elsewhere may not override:
  - feature_mode must be "raw8" or "raw8_idx3" (DATA_CONTRACT.md §7.1).
  - valid_policy.compute_before_fill must be True (DATA_CONTRACT.md §6.2).
  - Supported patch_sizes are 256, 384, 512 (module_prep_data.md §12.1).
  - aoi.buffer_m must be ≥ 0 (main_tech.md §8).
  - normalization.clip_percentiles must be an increasing pair in [0, 100].
  - normalization.scale_range must be an increasing pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Literal, Optional, Tuple

from ai_fields.common.constants import FEATURE_MODES
from ai_fields.common.errors import (
    FeatureModeError,
    ValidPolicyError,
    ContractError,
)


# ---------------------------------------------------------------------------
# Leaf config sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidPolicyConfig:
    """Policy for computing and interpreting the valid / NoData mask.

    DATA_CONTRACT.md §6.2, §6.5.
    """

    nodata_source: str = "metadata_then_config"
    """Priority chain for resolving valid mask.
    Accepted values mirror the YAML baseline:
      - "metadata_then_config"  (sidecar mask > explicit nodata > config rule)
      - "explicit_nodata"
      - "config_rule_only"
    """

    compute_before_fill: bool = True
    """valid MUST be computed before any NoData fill.  If False, build_config
    raises ValidPolicyError (DATA_CONTRACT.md §6.2, DEC-002).
    """

    def validate(self) -> None:
        """Raise ValidPolicyError if the policy violates the contract."""
        if not isinstance(self.compute_before_fill, bool):
            raise ValidPolicyError(
                "valid_policy.compute_before_fill must be a boolean (true/false), "
                f"got {type(self.compute_before_fill).__name__}."
            )
        if not self.compute_before_fill:
            raise ValidPolicyError(
                "valid_policy.compute_before_fill must be True.  "
                "The valid mask must be computed before any NoData fill value is "
                "applied (DATA_CONTRACT.md §6.2, DECISIONS.md DEC-002)."
            )
        if not isinstance(self.nodata_source, str):
            raise ValidPolicyError(
                "valid_policy.nodata_source must be a string, "
                f"got {type(self.nodata_source).__name__}."
            )
        accepted_sources = {
            "metadata_then_config",
            "explicit_nodata",
            "config_rule_only",
        }
        if self.nodata_source not in accepted_sources:
            raise ValidPolicyError(
                f"Unknown nodata_source '{self.nodata_source}'.  "
                f"Accepted values: {sorted(accepted_sources)}."
            )


@dataclass(frozen=True)
class AoiConfig:
    """AOI policy configuration.

    main_tech.md §8, DATA_CONTRACT.md §3.3.
    """

    enabled: bool = False
    aoi_path: Optional[str] = None
    buffer_m: float = 30.0
    """Context buffer around AOI in metres (baseline 30 m, main_tech.md §8)."""
    derive_from_labels_if_missing: bool = False
    """Auto-derive AOI from buffered bbox of label polygons when no aoi_path is given.
    Opt-in only; defaults to False.  Incompatible with enabled=True."""

    def validate(self) -> None:
        """Raise ContractError if the AOI config is invalid."""
        if not isinstance(self.enabled, bool):
            raise ContractError(
                f"aoi.enabled must be a boolean (true/false), got {type(self.enabled).__name__}."
            )
        if self.aoi_path is not None and not isinstance(self.aoi_path, str):
            raise ContractError(
                f"aoi.aoi_path must be a string or null, got {type(self.aoi_path).__name__}."
            )
        if isinstance(self.buffer_m, bool) or not isinstance(self.buffer_m, Real):
            raise ContractError(
                f"aoi.buffer_m must be a number, got {type(self.buffer_m).__name__}."
            )
        if self.buffer_m < 0:
            raise ContractError(
                f"aoi.buffer_m must be >= 0, got {self.buffer_m}.  "
                "A negative buffer would shrink the AOI and lose context "
                "(main_tech.md §8)."
            )
        if self.enabled and self.aoi_path is None:
            raise ContractError(
                "aoi.enabled is True but aoi.aoi_path is null.  "
                "Provide a path to the AOI file."
            )
        if not isinstance(self.derive_from_labels_if_missing, bool):
            raise ContractError(
                "aoi.derive_from_labels_if_missing must be a boolean, "
                f"got {type(self.derive_from_labels_if_missing).__name__}."
            )
        if self.derive_from_labels_if_missing and self.enabled:
            raise ContractError(
                "aoi.derive_from_labels_if_missing and aoi.enabled=True cannot both be set: "
                "use enabled=True with an explicit aoi_path for user-provided AOI, or "
                "derive_from_labels_if_missing=True (without enabled) for auto-derived AOI."
            )


@dataclass(frozen=True)
class PatchesConfig:
    """Patch extraction configuration.

    module_prep_data.md §12.
    """

    patch_size: int = 512
    """Patch size in pixels.  Must be 256, 384, or 512 (module_prep_data.md §12.1)."""

    sampling_policy: str = "strategic"
    """Sampling strategy.  Accepted values: "strategic", "random"."""

    SUPPORTED_PATCH_SIZES: Tuple[int, ...] = field(
        default=(256, 384, 512), init=False, repr=False, compare=False
    )

    def validate(self) -> None:
        """Raise ContractError if the patch config is invalid."""
        if isinstance(self.patch_size, bool) or not isinstance(self.patch_size, int):
            raise ContractError(
                f"patches.patch_size must be an integer, got {type(self.patch_size).__name__}."
            )
        if not isinstance(self.sampling_policy, str):
            raise ContractError(
                "patches.sampling_policy must be a string, "
                f"got {type(self.sampling_policy).__name__}."
            )
        if self.patch_size not in (256, 384, 512):
            raise ContractError(
                f"patches.patch_size must be 256, 384, or 512, got {self.patch_size}.  "
                "(module_prep_data.md §12.1)"
            )
        accepted_policies = {"strategic", "random"}
        if self.sampling_policy not in accepted_policies:
            raise ContractError(
                f"patches.sampling_policy '{self.sampling_policy}' is not accepted.  "
                f"Accepted values: {sorted(accepted_policies)}."
            )


@dataclass(frozen=True)
class BoundaryConfig:
    """Boundary target encoding configuration.

    DATA_CONTRACT.md §8.3, module_prep_data.md §11.2.
    """

    encoding: str = "background_skeleton_buffer"
    """Boundary encoding scheme.
    Only "background_skeleton_buffer" is supported in v1
    (DATA_CONTRACT.md §8.3).
    """

    def validate(self) -> None:
        if not isinstance(self.encoding, str):
            raise ContractError(
                f"boundary.encoding must be a string, got {type(self.encoding).__name__}."
            )
        if self.encoding != "background_skeleton_buffer":
            raise ContractError(
                f"boundary.encoding '{self.encoding}' is not supported.  "
                "Only 'background_skeleton_buffer' is accepted in v1 "
                "(DATA_CONTRACT.md §8.3)."
            )


@dataclass(frozen=True)
class DistanceConfig:
    """Distance target configuration.

    DATA_CONTRACT.md §8.4, module_prep_data.md §11.3.
    """

    target: str = "unsigned_distance_to_boundary"
    """Distance target type.  Only "unsigned_distance_to_boundary" is
    supported in v1 (DATA_CONTRACT.md §8.4).
    """

    def validate(self) -> None:
        if not isinstance(self.target, str):
            raise ContractError(
                f"distance.target must be a string, got {type(self.target).__name__}."
            )
        if self.target != "unsigned_distance_to_boundary":
            raise ContractError(
                f"distance.target '{self.target}' is not supported.  "
                "Only 'unsigned_distance_to_boundary' is accepted in v1 "
                "(DATA_CONTRACT.md §8.4)."
            )


@dataclass(frozen=True)
class NormalizationConfig:
    """Per-band robust normalization configuration.

    DATA_CONTRACT.md §10, module_prep_data.md §10, DECISIONS.md DEC-008.
    """

    name: str = "robust_percentile"
    """Normalization scheme name.  Only "robust_percentile" is supported in
    baseline v1 (DATA_CONTRACT.md §10.2).
    """

    clip_percentiles: Tuple[float, float] = (0.5, 99.5)
    """(low_pct, high_pct) for per-band percentile clipping.
    Both must be in [0, 100] and low < high (main_tech.md §7.3).
    """

    scale_range: Tuple[float, float] = (0.0, 1.0)
    """(min_val, max_val) output scaling range.  min < max required."""

    stats_computed_on: str = "valid_train_pixels"
    """Where the normalization statistics are computed.
    Must be "valid_train_pixels" for baseline v1 (DATA_CONTRACT.md §10.2).
    """

    def validate(self) -> None:
        """Raise ContractError or NormalizationContractError on policy violations."""
        from ai_fields.common.errors import NormalizationContractError

        if not isinstance(self.name, str):
            raise NormalizationContractError(
                f"normalization.name must be a string, got {type(self.name).__name__}."
            )
        if (
            not isinstance(self.clip_percentiles, (tuple, list))
            or len(self.clip_percentiles) != 2
        ):
            raise NormalizationContractError(
                "normalization.clip_percentiles must be a two-element sequence "
                f"(low, high), got {self.clip_percentiles!r}."
            )
        if not isinstance(self.scale_range, (tuple, list)) or len(self.scale_range) != 2:
            raise NormalizationContractError(
                "normalization.scale_range must be a two-element sequence "
                f"(min, max), got {self.scale_range!r}."
            )
        if not isinstance(self.stats_computed_on, str):
            raise NormalizationContractError(
                "normalization.stats_computed_on must be a string, "
                f"got {type(self.stats_computed_on).__name__}."
            )

        if self.name != "robust_percentile":
            raise NormalizationContractError(
                f"normalization.name '{self.name}' is not supported.  "
                "Only 'robust_percentile' is accepted in baseline v1 "
                "(DATA_CONTRACT.md §10.2)."
            )
        lo, hi = self.clip_percentiles
        if isinstance(lo, bool) or not isinstance(lo, Real):
            raise NormalizationContractError(
                "normalization.clip_percentiles[0] must be a number, "
                f"got {type(lo).__name__}."
            )
        if isinstance(hi, bool) or not isinstance(hi, Real):
            raise NormalizationContractError(
                "normalization.clip_percentiles[1] must be a number, "
                f"got {type(hi).__name__}."
            )
        if not (0.0 <= lo < hi <= 100.0):
            raise NormalizationContractError(
                f"normalization.clip_percentiles must satisfy 0 <= lo < hi <= 100, "
                f"got ({lo}, {hi})."
            )
        slo, shi = self.scale_range
        if isinstance(slo, bool) or not isinstance(slo, Real):
            raise NormalizationContractError(
                "normalization.scale_range[0] must be a number, "
                f"got {type(slo).__name__}."
            )
        if isinstance(shi, bool) or not isinstance(shi, Real):
            raise NormalizationContractError(
                "normalization.scale_range[1] must be a number, "
                f"got {type(shi).__name__}."
            )
        if slo >= shi:
            raise NormalizationContractError(
                f"normalization.scale_range must satisfy min < max, "
                f"got ({slo}, {shi})."
            )
        if self.stats_computed_on != "valid_train_pixels":
            raise NormalizationContractError(
                f"normalization.stats_computed_on must be 'valid_train_pixels', "
                f"got '{self.stats_computed_on}'.  "
                "Using non-train-valid stats breaks the train/predict normalization "
                "contract (DATA_CONTRACT.md §10.2, DECISIONS.md DEC-008)."
            )


@dataclass(frozen=True)
class SplitConfig:
    """Train/val/test split policy.

    module_prep_data.md §14.
    """

    policy: str = "spatial_stratified"
    """Split strategy.  Accepted values: "spatial_stratified", "random"."""

    random_seed: Optional[int] = 42
    """Random seed for reproducibility.  None disables seeding (not recommended)."""

    def validate(self) -> None:
        if not isinstance(self.policy, str):
            raise ContractError(
                f"split.policy must be a string, got {type(self.policy).__name__}."
            )
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)
        ):
            raise ContractError(
                "split.random_seed must be an integer or null, "
                f"got {type(self.random_seed).__name__}."
            )
        accepted = {"spatial_stratified", "random"}
        if self.policy not in accepted:
            raise ContractError(
                f"split.policy '{self.policy}' is not accepted.  "
                f"Accepted values: {sorted(accepted)}."
            )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepDataConfig:
    """Fully-validated configuration for a module_prep_data run.

    This is the single source of resolved config truth that is passed into
    every stage of the module.  It is constructed by config.build_config() and
    must not be mutated at runtime.

    Contract invariants (must all hold after validate()):
      - feature_mode in ("raw8", "raw8_idx3")  — DATA_CONTRACT.md §7.1
      - valid_policy.compute_before_fill is True  — DATA_CONTRACT.md §6.2
      - All sub-configs pass their own validate()
    """

    feature_mode: str = "raw8"
    """Dataset-side feature mode.  "raw8" or "raw8_idx3" only."""

    valid_policy: ValidPolicyConfig = field(default_factory=ValidPolicyConfig)
    aoi: AoiConfig = field(default_factory=AoiConfig)
    patches: PatchesConfig = field(default_factory=PatchesConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def validate(self) -> None:
        """Run all contract checks.  Raises a ContractError subclass on failure.

        Call this immediately after constructing a PrepDataConfig instance.
        build_config() calls this automatically.
        """
        # Feature mode check — DATA_CONTRACT.md §7.1
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

        # Delegate to sub-config validators
        self.valid_policy.validate()
        self.aoi.validate()
        self.patches.validate()
        self.boundary.validate()
        self.distance.validate()
        self.normalization.validate()
        self.split.validate()
