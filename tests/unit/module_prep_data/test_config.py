"""Unit tests for module_prep_data schemas and config loader.

Coverage targets (TESTING_STRATEGY.md §5, §9, §12):
  - Happy-path: default config, baseline YAML, raw8_idx3 YAML variant
  - Contract invariants: feature_mode, valid_policy, patch_size, aoi.buffer_m,
    normalization, boundary, distance, split
  - Negative tests: each contract violation raises the correct error type
  - build_config(): unknown-key warnings, correct type coercions
  - load_yaml(): missing file, non-mapping YAML
  - load_config(): round-trip from the checked-in baseline YAML

These tests MUST NOT be edited to accommodate code changes.
If a constant or rule changes, update the source document first
(REPO_CONVENTIONS.md §18.2).
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import pytest

from ai_fields.common.errors import (
    ContractError,
    FeatureModeError,
    NormalizationContractError,
    ValidPolicyError,
)
from ai_fields.module_prep_data.config import build_config, load_config, load_yaml
from ai_fields.module_prep_data.schemas import (
    AoiConfig,
    BoundaryConfig,
    DistanceConfig,
    NormalizationConfig,
    PatchesConfig,
    PrepDataConfig,
    SplitConfig,
    ValidPolicyConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASELINE_YAML = Path(__file__).parents[3] / "configs" / "module_prep_data" / "baseline.raw8.yaml"
BASELINE_YAML_IDX3 = (
    Path(__file__).parents[3] / "configs" / "module_prep_data" / "baseline.raw8_idx3.yaml"
)


def _minimal_raw() -> dict:
    """Return the minimal raw dict that passes build_config without error."""
    return {"feature_mode": "raw8"}


# ---------------------------------------------------------------------------
# PrepDataConfig defaults
# ---------------------------------------------------------------------------


class TestPrepDataConfigDefaults:
    def test_default_feature_mode(self):
        cfg = PrepDataConfig()
        cfg.validate()
        assert cfg.feature_mode == "raw8"

    def test_default_is_frozen(self):
        cfg = PrepDataConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.feature_mode = "raw8_idx3"  # type: ignore[misc]

    def test_raw8_idx3_valid_default(self):
        cfg = PrepDataConfig(feature_mode="raw8_idx3")
        cfg.validate()
        assert cfg.feature_mode == "raw8_idx3"


# ---------------------------------------------------------------------------
# Feature mode validation — DATA_CONTRACT.md §7.1, DEC-003
# ---------------------------------------------------------------------------


class TestFeatureModeValidation:
    def test_raw8_accepted(self):
        cfg = PrepDataConfig(feature_mode="raw8")
        cfg.validate()  # must not raise

    def test_raw8_idx3_accepted(self):
        cfg = PrepDataConfig(feature_mode="raw8_idx3")
        cfg.validate()  # must not raise

    def test_unknown_mode_raises_feature_mode_error(self):
        with pytest.raises(FeatureModeError):
            PrepDataConfig(feature_mode="raw11").validate()

    def test_unknown_mode_is_contract_error(self):
        with pytest.raises(ContractError):
            PrepDataConfig(feature_mode="full_input").validate()

    def test_empty_mode_raises(self):
        with pytest.raises(FeatureModeError):
            PrepDataConfig(feature_mode="").validate()

    def test_case_sensitive(self):
        # "RAW8" is not a valid mode — contract names are lowercase
        with pytest.raises(FeatureModeError):
            PrepDataConfig(feature_mode="RAW8").validate()


# ---------------------------------------------------------------------------
# ValidPolicyConfig — DATA_CONTRACT.md §6.2, DEC-002
# ---------------------------------------------------------------------------


class TestValidPolicyConfig:
    def test_default_compute_before_fill_is_true(self):
        vp = ValidPolicyConfig()
        vp.validate()
        assert vp.compute_before_fill is True

    def test_compute_before_fill_false_raises(self):
        vp = ValidPolicyConfig(compute_before_fill=False)
        with pytest.raises(ValidPolicyError):
            vp.validate()

    def test_valid_policy_error_is_contract_error(self):
        vp = ValidPolicyConfig(compute_before_fill=False)
        with pytest.raises(ContractError):
            vp.validate()

    def test_unknown_nodata_source_raises(self):
        vp = ValidPolicyConfig(nodata_source="guess_it")
        with pytest.raises(ValidPolicyError):
            vp.validate()

    @pytest.mark.parametrize("bad_value", ["false", "0", 0, 1, [], {}, None])
    def test_compute_before_fill_requires_real_bool(self, bad_value):
        vp = ValidPolicyConfig(compute_before_fill=bad_value)  # type: ignore[arg-type]
        with pytest.raises(ValidPolicyError):
            vp.validate()

    def test_accepted_nodata_sources(self):
        for src in ("metadata_then_config", "explicit_nodata", "config_rule_only"):
            ValidPolicyConfig(nodata_source=src).validate()  # must not raise


# ---------------------------------------------------------------------------
# AoiConfig — main_tech.md §8
# ---------------------------------------------------------------------------


class TestAoiConfig:
    def test_default_not_enabled(self):
        aoi = AoiConfig()
        aoi.validate()
        assert aoi.enabled is False

    def test_negative_buffer_raises(self):
        aoi = AoiConfig(buffer_m=-1.0)
        with pytest.raises(ContractError):
            aoi.validate()

    def test_zero_buffer_ok(self):
        aoi = AoiConfig(buffer_m=0.0)
        aoi.validate()  # must not raise

    def test_enabled_without_path_raises(self):
        aoi = AoiConfig(enabled=True, aoi_path=None)
        with pytest.raises(ContractError):
            aoi.validate()

    def test_enabled_with_path_ok(self):
        aoi = AoiConfig(enabled=True, aoi_path="/some/aoi.gpkg", buffer_m=30.0)
        aoi.validate()  # must not raise

    def test_enabled_must_be_real_bool(self):
        with pytest.raises(ContractError):
            AoiConfig(enabled="false").validate()  # type: ignore[arg-type]

    def test_baseline_buffer_is_30(self):
        assert AoiConfig().buffer_m == 30.0


# ---------------------------------------------------------------------------
# PatchesConfig — module_prep_data.md §12.1
# ---------------------------------------------------------------------------


class TestPatchesConfig:
    @pytest.mark.parametrize("size", [256, 384, 512])
    def test_supported_patch_sizes(self, size):
        PatchesConfig(patch_size=size).validate()  # must not raise

    def test_unsupported_patch_size_raises(self):
        with pytest.raises(ContractError):
            PatchesConfig(patch_size=128).validate()

    def test_unsupported_sampling_policy_raises(self):
        with pytest.raises(ContractError):
            PatchesConfig(patch_size=512, sampling_policy="dense_grid").validate()

    def test_default_patch_size_is_512(self):
        assert PatchesConfig().patch_size == 512

    def test_default_sampling_is_strategic(self):
        assert PatchesConfig().sampling_policy == "strategic"

    def test_random_sampling_accepted(self):
        PatchesConfig(patch_size=512, sampling_policy="random").validate()


# ---------------------------------------------------------------------------
# BoundaryConfig — DATA_CONTRACT.md §8.3
# ---------------------------------------------------------------------------


class TestBoundaryConfig:
    def test_default_encoding(self):
        bc = BoundaryConfig()
        bc.validate()
        assert bc.encoding == "background_skeleton_buffer"

    def test_unsupported_encoding_raises(self):
        with pytest.raises(ContractError):
            BoundaryConfig(encoding="binary").validate()


# ---------------------------------------------------------------------------
# DistanceConfig — DATA_CONTRACT.md §8.4
# ---------------------------------------------------------------------------


class TestDistanceConfig:
    def test_default_target(self):
        dc = DistanceConfig()
        dc.validate()
        assert dc.target == "unsigned_distance_to_boundary"

    def test_unsupported_target_raises(self):
        with pytest.raises(ContractError):
            DistanceConfig(target="signed_distance").validate()


# ---------------------------------------------------------------------------
# NormalizationConfig — DATA_CONTRACT.md §10, DEC-008
# ---------------------------------------------------------------------------


class TestNormalizationConfig:
    def test_defaults_valid(self):
        NormalizationConfig().validate()  # must not raise

    def test_wrong_name_raises(self):
        with pytest.raises(NormalizationContractError):
            NormalizationConfig(name="minmax").validate()

    def test_inverted_percentiles_raises(self):
        with pytest.raises(NormalizationContractError):
            NormalizationConfig(clip_percentiles=(99.5, 0.5)).validate()

    def test_equal_percentiles_raises(self):
        with pytest.raises(NormalizationContractError):
            NormalizationConfig(clip_percentiles=(50.0, 50.0)).validate()

    def test_out_of_range_percentile_raises(self):
        with pytest.raises(NormalizationContractError):
            NormalizationConfig(clip_percentiles=(0.0, 101.0)).validate()

    def test_inverted_scale_range_raises(self):
        with pytest.raises(NormalizationContractError):
            NormalizationConfig(scale_range=(1.0, 0.0)).validate()

    def test_non_train_stats_raises(self):
        with pytest.raises(NormalizationContractError):
            NormalizationConfig(stats_computed_on="all_pixels").validate()

    def test_normalization_error_is_contract_error(self):
        with pytest.raises(ContractError):
            NormalizationConfig(name="minmax").validate()

    def test_baseline_clip_percentiles(self):
        nc = NormalizationConfig()
        assert nc.clip_percentiles == (0.5, 99.5)

    def test_baseline_scale_range(self):
        nc = NormalizationConfig()
        assert nc.scale_range == (0.0, 1.0)

    def test_baseline_stats_on_valid_train(self):
        assert NormalizationConfig().stats_computed_on == "valid_train_pixels"


# ---------------------------------------------------------------------------
# SplitConfig — module_prep_data.md §14
# ---------------------------------------------------------------------------


class TestSplitConfig:
    def test_defaults_valid(self):
        SplitConfig().validate()

    def test_unknown_policy_raises(self):
        with pytest.raises(ContractError):
            SplitConfig(policy="chronological").validate()

    def test_random_policy_accepted(self):
        SplitConfig(policy="random").validate()

    def test_none_seed_accepted(self):
        SplitConfig(policy="spatial_stratified", random_seed=None).validate()


# ---------------------------------------------------------------------------
# build_config — happy path
# ---------------------------------------------------------------------------


class TestBuildConfigHappyPath:
    def test_minimal_raw_returns_config(self):
        cfg = build_config(_minimal_raw())
        assert isinstance(cfg, PrepDataConfig)

    def test_feature_mode_preserved(self):
        cfg = build_config({"feature_mode": "raw8_idx3"})
        assert cfg.feature_mode == "raw8_idx3"

    def test_defaults_filled_in(self):
        cfg = build_config(_minimal_raw())
        assert cfg.patches.patch_size == 512
        assert cfg.normalization.clip_percentiles == (0.5, 99.5)
        assert cfg.valid_policy.compute_before_fill is True

    def test_full_raw8_dict(self):
        raw = {
            "feature_mode": "raw8",
            "valid_policy": {
                "nodata_source": "metadata_then_config",
                "compute_before_fill": True,
            },
            "aoi": {"enabled": False, "aoi_path": None, "buffer_m": 30},
            "patches": {"patch_size": 512, "sampling_policy": "strategic"},
            "boundary": {"encoding": "background_skeleton_buffer"},
            "distance": {"target": "unsigned_distance_to_boundary"},
            "normalization": {
                "name": "robust_percentile",
                "clip_percentiles": [0.5, 99.5],
                "scale_range": [0.0, 1.0],
                "stats_computed_on": "valid_train_pixels",
            },
            "split": {"policy": "spatial_stratified", "random_seed": 42},
        }
        cfg = build_config(raw)
        assert cfg.feature_mode == "raw8"
        assert cfg.normalization.name == "robust_percentile"
        assert cfg.split.random_seed == 42

    def test_buffer_m_coerced_to_float(self):
        raw = _minimal_raw()
        raw["aoi"] = {"buffer_m": 30}  # int in YAML → float in dataclass
        cfg = build_config(raw)
        assert isinstance(cfg.aoi.buffer_m, float)

    def test_clip_percentiles_coerced_to_float_tuple(self):
        raw = _minimal_raw()
        raw["normalization"] = {
            "clip_percentiles": [1, 99],
            "scale_range": [0, 1],
        }
        cfg = build_config(raw)
        lo, hi = cfg.normalization.clip_percentiles
        assert isinstance(lo, float)
        assert isinstance(hi, float)


# ---------------------------------------------------------------------------
# build_config — unknown-key warnings
# ---------------------------------------------------------------------------


class TestBuildConfigUnknownKeys:
    def test_unknown_top_level_key_warns(self):
        raw = _minimal_raw()
        raw["extra_field"] = "oops"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_config(raw)
        messages = [str(w.message) for w in caught]
        assert any("extra_field" in m for m in messages)

    def test_unknown_aoi_key_warns(self):
        raw = _minimal_raw()
        raw["aoi"] = {"buffer_m": 30, "unknown_key": True}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_config(raw)
        messages = [str(w.message) for w in caught]
        assert any("unknown_key" in m for m in messages)

    def test_unknown_patches_key_warns(self):
        raw = _minimal_raw()
        raw["patches"] = {"patch_size": 512, "typo_key": 99}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_config(raw)
        messages = [str(w.message) for w in caught]
        assert any("typo_key" in m for m in messages)


# ---------------------------------------------------------------------------
# build_config — contract errors
# ---------------------------------------------------------------------------


class TestBuildConfigContractErrors:
    def test_invalid_feature_mode_raises(self):
        with pytest.raises(FeatureModeError):
            build_config({"feature_mode": "raw11"})

    def test_compute_before_fill_false_raises(self):
        raw = _minimal_raw()
        raw["valid_policy"] = {"compute_before_fill": False}
        with pytest.raises(ValidPolicyError):
            build_config(raw)

    def test_bad_patch_size_raises(self):
        raw = _minimal_raw()
        raw["patches"] = {"patch_size": 64}
        with pytest.raises(ContractError):
            build_config(raw)

    def test_negative_aoi_buffer_raises(self):
        raw = _minimal_raw()
        raw["aoi"] = {"buffer_m": -5}
        with pytest.raises(ContractError):
            build_config(raw)

    def test_bad_normalization_name_raises(self):
        raw = _minimal_raw()
        raw["normalization"] = {"name": "zscore"}
        with pytest.raises(NormalizationContractError):
            build_config(raw)

    def test_clip_percentiles_not_list_raises(self):
        raw = _minimal_raw()
        raw["normalization"] = {"clip_percentiles": 99.5}
        with pytest.raises(ContractError):
            build_config(raw)

    def test_scale_range_not_list_raises(self):
        raw = _minimal_raw()
        raw["normalization"] = {
            "clip_percentiles": [0.5, 99.5],
            "scale_range": 1.0,
        }
        with pytest.raises(ContractError):
            build_config(raw)

    def test_bad_split_policy_raises(self):
        raw = _minimal_raw()
        raw["split"] = {"policy": "leaky_random"}
        with pytest.raises(ContractError):
            build_config(raw)


# ---------------------------------------------------------------------------
# build_config — strict typing and ValueError leakage hardening
# ---------------------------------------------------------------------------


class TestBuildConfigStrictTyping:
    @pytest.mark.parametrize("bad_value", [1, 0, True, False, [], {}, None])
    def test_feature_mode_requires_string(self, bad_value):
        with pytest.raises(FeatureModeError):
            build_config({"feature_mode": bad_value})

    @pytest.mark.parametrize("bad_value", ["false", "true", "0", "1", 0, 1, [], {}, None])
    def test_valid_policy_compute_before_fill_requires_real_bool(self, bad_value):
        raw = _minimal_raw()
        raw["valid_policy"] = {"compute_before_fill": bad_value}
        with pytest.raises(ContractError):
            build_config(raw)

    @pytest.mark.parametrize("bad_value", ["false", "true", "0", "1", 0, 1, [], {}, None])
    def test_aoi_enabled_requires_real_bool(self, bad_value):
        raw = _minimal_raw()
        raw["aoi"] = {"enabled": bad_value}
        with pytest.raises(ContractError):
            build_config(raw)

    @pytest.mark.parametrize(
        "section_name, section_value",
        [
            ("valid_policy", []),
            ("aoi", "..."),
            ("patches", []),
            ("boundary", 1),
            ("distance", 1.0),
            ("normalization", 1),
            ("split", "random"),
        ],
    )
    def test_sections_must_be_mappings(self, section_name, section_value):
        raw = _minimal_raw()
        raw[section_name] = section_value
        with pytest.raises(ContractError):
            build_config(raw)

    def test_non_mapping_root_raises_contract_error(self):
        with pytest.raises(ContractError):
            build_config([])  # type: ignore[arg-type]


class TestBuildConfigNoValueErrorLeakage:
    @pytest.mark.parametrize(
        "raw",
        [
            {"feature_mode": "raw8", "patches": {"patch_size": "abc"}},
            {"feature_mode": "raw8", "aoi": {"buffer_m": "abc"}},
            {"feature_mode": "raw8", "split": {"random_seed": "abc"}},
            {"feature_mode": "raw8", "patches": {"patch_size": 1.5}},
            {"feature_mode": "raw8", "patches": {"sampling_policy": 1}},
            {"feature_mode": "raw8", "normalization": {"name": 1}},
            {"feature_mode": "raw8", "normalization": {"clip_percentiles": [0.5, "99.5"]}},
            {"feature_mode": "raw8", "normalization": {"scale_range": [0, True]}},
            {"feature_mode": "raw8", "aoi": {"aoi_path": 123}},
        ],
    )
    def test_contract_violations_raise_contract_error_not_value_error(self, raw):
        try:
            build_config(raw)
        except ValueError as exc:
            pytest.fail(f"ValueError leaked from contract layer: {exc}")
        except ContractError:
            pass
        else:
            pytest.fail("Expected a ContractError for invalid config input.")


# ---------------------------------------------------------------------------
# Missing feature_mode — hardening (module_prep_data.md §17.1, DATA_CONTRACT.md §7.1)
# ---------------------------------------------------------------------------


class TestMissingFeatureMode:
    """feature_mode is required; no silent default is permitted."""

    def test_empty_dict_raises_feature_mode_error(self):
        with pytest.raises(FeatureModeError):
            build_config({})

    def test_missing_feature_mode_raises_feature_mode_error(self):
        # All other sections present; feature_mode key absent.
        raw = {
            "valid_policy": {"nodata_source": "metadata_then_config", "compute_before_fill": True},
            "patches": {"patch_size": 512, "sampling_policy": "strategic"},
        }
        with pytest.raises(FeatureModeError):
            build_config(raw)

    def test_missing_feature_mode_is_contract_error(self):
        with pytest.raises(ContractError):
            build_config({})

    def test_missing_feature_mode_message_names_the_key(self):
        with pytest.raises(FeatureModeError, match="feature_mode"):
            build_config({})

    def test_missing_feature_mode_message_lists_accepted_values(self):
        with pytest.raises(FeatureModeError, match="raw8"):
            build_config({})

    def test_typo_key_featuremode_raises_feature_mode_error(self):
        # "featuremode" (no underscore) must not silently become "raw8".
        with pytest.raises(FeatureModeError):
            build_config({"featuremode": "raw8"})

    def test_typo_key_also_warns_about_unknown_key(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                build_config({"featuremode": "raw8"})
            except FeatureModeError:
                pass
        messages = [str(w.message) for w in caught]
        assert any("featuremode" in m for m in messages)

    def test_load_config_from_yaml_without_feature_mode_raises(self, tmp_path):
        f = tmp_path / "no_mode.yaml"
        f.write_text("patches:\n  patch_size: 512\n", encoding="utf-8")
        with pytest.raises(FeatureModeError):
            load_config(f)


# ---------------------------------------------------------------------------
# load_yaml
# ---------------------------------------------------------------------------


class TestLoadYaml:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_non_mapping_raises_contract_error(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ContractError):
            load_yaml(f)

    def test_valid_yaml_returns_dict(self, tmp_path):
        f = tmp_path / "ok.yaml"
        f.write_text("feature_mode: raw8\n", encoding="utf-8")
        result = load_yaml(f)
        assert isinstance(result, dict)
        assert result["feature_mode"] == "raw8"

    def test_empty_mapping_returns_empty_dict(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("{}\n", encoding="utf-8")
        result = load_yaml(f)
        assert result == {}


# ---------------------------------------------------------------------------
# load_config — round-trip through the checked-in baseline YAML
# ---------------------------------------------------------------------------


class TestLoadConfigBaseline:
    @pytest.mark.skipif(
        not BASELINE_YAML.exists(),
        reason="baseline.raw8.yaml not present in repo",
    )
    def test_baseline_yaml_loads_without_error(self):
        cfg = load_config(BASELINE_YAML)
        assert isinstance(cfg, PrepDataConfig)

    @pytest.mark.skipif(
        not BASELINE_YAML.exists(),
        reason="baseline.raw8.yaml not present in repo",
    )
    def test_baseline_feature_mode_is_raw8(self):
        cfg = load_config(BASELINE_YAML)
        assert cfg.feature_mode == "raw8"

    @pytest.mark.skipif(
        not BASELINE_YAML.exists(),
        reason="baseline.raw8.yaml not present in repo",
    )
    def test_baseline_compute_before_fill_true(self):
        cfg = load_config(BASELINE_YAML)
        assert cfg.valid_policy.compute_before_fill is True

    @pytest.mark.skipif(
        not BASELINE_YAML.exists(),
        reason="baseline.raw8.yaml not present in repo",
    )
    def test_baseline_patch_size_512(self):
        cfg = load_config(BASELINE_YAML)
        assert cfg.patches.patch_size == 512

    @pytest.mark.skipif(
        not BASELINE_YAML.exists(),
        reason="baseline.raw8.yaml not present in repo",
    )
    def test_baseline_clip_percentiles(self):
        cfg = load_config(BASELINE_YAML)
        assert cfg.normalization.clip_percentiles == (0.5, 99.5)

    @pytest.mark.skipif(
        not BASELINE_YAML.exists(),
        reason="baseline.raw8.yaml not present in repo",
    )
    def test_baseline_aoi_buffer_30(self):
        cfg = load_config(BASELINE_YAML)
        assert cfg.aoi.buffer_m == 30.0

    def test_load_config_from_tmp_yaml(self, tmp_path):
        content = textwrap.dedent(
            """\
            feature_mode: raw8_idx3
            patches:
              patch_size: 256
              sampling_policy: strategic
            """
        )
        f = tmp_path / "test.yaml"
        f.write_text(content, encoding="utf-8")
        cfg = load_config(f)
        assert cfg.feature_mode == "raw8_idx3"
        assert cfg.patches.patch_size == 256


# ---------------------------------------------------------------------------
# load_config — round-trip through the checked-in baseline.raw8_idx3.yaml
# (DATA_CONTRACT.md §7.3, DECISIONS.md DEC-003)
# ---------------------------------------------------------------------------


class TestLoadConfigBaselineRaw8Idx3:
    @pytest.mark.skipif(
        not BASELINE_YAML_IDX3.exists(),
        reason="baseline.raw8_idx3.yaml not present in repo",
    )
    def test_baseline_idx3_yaml_loads_without_error(self):
        cfg = load_config(BASELINE_YAML_IDX3)
        assert isinstance(cfg, PrepDataConfig)

    @pytest.mark.skipif(
        not BASELINE_YAML_IDX3.exists(),
        reason="baseline.raw8_idx3.yaml not present in repo",
    )
    def test_baseline_idx3_feature_mode_is_raw8_idx3(self):
        # DATA_CONTRACT.md §7.3 — raw8_idx3 must be a supported dataset-side feature mode
        cfg = load_config(BASELINE_YAML_IDX3)
        assert cfg.feature_mode == "raw8_idx3"

    @pytest.mark.skipif(
        not BASELINE_YAML_IDX3.exists(),
        reason="baseline.raw8_idx3.yaml not present in repo",
    )
    def test_baseline_idx3_compute_before_fill_true(self):
        # DATA_CONTRACT.md §6.2, DEC-002 — valid must be computed before fill
        cfg = load_config(BASELINE_YAML_IDX3)
        assert cfg.valid_policy.compute_before_fill is True

    @pytest.mark.skipif(
        not BASELINE_YAML_IDX3.exists(),
        reason="baseline.raw8_idx3.yaml not present in repo",
    )
    def test_baseline_idx3_patch_size_512(self):
        cfg = load_config(BASELINE_YAML_IDX3)
        assert cfg.patches.patch_size == 512

    @pytest.mark.skipif(
        not BASELINE_YAML_IDX3.exists(),
        reason="baseline.raw8_idx3.yaml not present in repo",
    )
    def test_baseline_idx3_clip_percentiles(self):
        cfg = load_config(BASELINE_YAML_IDX3)
        assert cfg.normalization.clip_percentiles == (0.5, 99.5)

    @pytest.mark.skipif(
        not BASELINE_YAML_IDX3.exists(),
        reason="baseline.raw8_idx3.yaml not present in repo",
    )
    def test_baseline_idx3_aoi_buffer_30(self):
        cfg = load_config(BASELINE_YAML_IDX3)
        assert cfg.aoi.buffer_m == 30.0
