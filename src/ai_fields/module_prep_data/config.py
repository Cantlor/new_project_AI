"""YAML loader and build_config factory for module_prep_data.

Public API:
    load_yaml(path)         -- read a YAML file, return raw dict
    build_config(raw)       -- validate raw dict, return PrepDataConfig
    load_config(path)       -- load_yaml + build_config in one call

Design rules:
  - No silent fallbacks.  Missing required keys raise ContractError.
  - All validation is delegated to PrepDataConfig.validate() and sub-config
    validate() methods so that the contract lives in one place (schemas.py).
  - Type coercions that are safe (e.g. int → float) are performed explicitly
    and documented.
  - Unexpected top-level keys produce a warning, not a silent ignore, so that
    typos in config files are caught early (REPO_CONVENTIONS.md §8.3).

Sources:
  DATA_CONTRACT.md §7, §10
  module_prep_data.md §17
  DECISIONS.md DEC-003, DEC-008, DEC-015
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ai_fields.common.constants import FEATURE_MODES
from ai_fields.common.errors import ContractError, FeatureModeError
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
# Known top-level keys in a prep_data config
# ---------------------------------------------------------------------------
_KNOWN_TOP_LEVEL_KEYS = frozenset(
    {
        "feature_mode",
        "valid_policy",
        "aoi",
        "patches",
        "boundary",
        "distance",
        "normalization",
        "split",
    }
)


# ---------------------------------------------------------------------------
# Strict scalar/section parsing helpers
# ---------------------------------------------------------------------------


def _ensure_section_mapping(section_name: str, value: Any) -> Dict[str, Any]:
    """Return *value* as dict or raise ContractError for wrong-type sections."""
    if not isinstance(value, dict):
        raise ContractError(
            f"{section_name} section must be a YAML mapping/object, "
            f"got {type(value).__name__}."
        )
    return value


def _sorted_unknown_keys(unknown: set[Any]) -> list[str]:
    """Return unknown keys as stable printable strings without type-order errors."""
    return sorted((str(k) for k in unknown))


def _get_str(section_name: str, d: Dict[str, Any], key: str, default: str) -> str:
    if key not in d:
        return default
    value = d[key]
    if not isinstance(value, str):
        raise ContractError(
            f"{section_name}.{key} must be a string, got {type(value).__name__}."
        )
    return value


def _get_bool(section_name: str, d: Dict[str, Any], key: str, default: bool) -> bool:
    """Strict bool parser: accepts only real bool values, never truthy coercion."""
    if key not in d:
        return default
    value = d[key]
    if isinstance(value, bool):
        return value
    raise ContractError(
        f"{section_name}.{key} must be a boolean (true/false), "
        f"got {value!r} ({type(value).__name__})."
    )


def _get_int(section_name: str, d: Dict[str, Any], key: str, default: int) -> int:
    if key not in d:
        return default
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(
            f"{section_name}.{key} must be an integer, got {value!r} "
            f"({type(value).__name__})."
        )
    return value


def _get_optional_int(
    section_name: str, d: Dict[str, Any], key: str, default: Optional[int]
) -> Optional[int]:
    if key not in d:
        return default
    value = d[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(
            f"{section_name}.{key} must be an integer or null, got {value!r} "
            f"({type(value).__name__})."
        )
    return value


def _get_float(section_name: str, d: Dict[str, Any], key: str, default: float) -> float:
    if key not in d:
        return default
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(
            f"{section_name}.{key} must be a number, got {value!r} "
            f"({type(value).__name__})."
        )
    return float(value)


def _get_two_number_tuple(
    section_name: str,
    d: Dict[str, Any],
    key: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    raw = d[key] if key in d else default
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise ContractError(
            f"{section_name}.{key} must be a two-element list/tuple, got {raw!r}."
        )
    lo, hi = raw
    if isinstance(lo, bool) or not isinstance(lo, (int, float)):
        raise ContractError(
            f"{section_name}.{key}[0] must be a number, got {lo!r} "
            f"({type(lo).__name__})."
        )
    if isinstance(hi, bool) or not isinstance(hi, (int, float)):
        raise ContractError(
            f"{section_name}.{key}[1] must be a number, got {hi!r} "
            f"({type(hi).__name__})."
        )
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Read *path* and return the parsed YAML content as a plain dict.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ContractError:     if the file does not parse to a YAML mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ContractError(
            f"Config file '{path}' must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}."
        )
    return raw


# ---------------------------------------------------------------------------
# Sub-config builders
# ---------------------------------------------------------------------------


def _build_valid_policy(d: Optional[Dict[str, Any]]) -> ValidPolicyConfig:
    if d is None:
        return ValidPolicyConfig()
    d = _ensure_section_mapping("valid_policy", d)
    unknown = set(d) - {"nodata_source", "compute_before_fill"}
    if unknown:
        warnings.warn(
            f"Unknown keys in valid_policy section: {_sorted_unknown_keys(unknown)}.  "
            "They will be ignored.",
            stacklevel=4,
        )
    return ValidPolicyConfig(
        nodata_source=_get_str(
            "valid_policy", d, "nodata_source", "metadata_then_config"
        ),
        compute_before_fill=_get_bool(
            "valid_policy", d, "compute_before_fill", True
        ),
    )


def _build_aoi(d: Optional[Dict[str, Any]]) -> AoiConfig:
    if d is None:
        return AoiConfig()
    d = _ensure_section_mapping("aoi", d)
    unknown = set(d) - {"enabled", "aoi_path", "buffer_m"}
    if unknown:
        warnings.warn(
            f"Unknown keys in aoi section: {_sorted_unknown_keys(unknown)}.  "
            "They will be ignored.",
            stacklevel=4,
        )
    aoi_path = d.get("aoi_path", None)
    if aoi_path is not None and not isinstance(aoi_path, str):
        raise ContractError(
            f"aoi.aoi_path must be a string or null, got {type(aoi_path).__name__}."
        )
    return AoiConfig(
        enabled=_get_bool("aoi", d, "enabled", False),
        aoi_path=aoi_path,
        buffer_m=_get_float("aoi", d, "buffer_m", 30.0),
    )


def _build_patches(d: Optional[Dict[str, Any]]) -> PatchesConfig:
    if d is None:
        return PatchesConfig()
    d = _ensure_section_mapping("patches", d)
    unknown = set(d) - {"patch_size", "sampling_policy"}
    if unknown:
        warnings.warn(
            f"Unknown keys in patches section: {_sorted_unknown_keys(unknown)}.  "
            "They will be ignored.",
            stacklevel=4,
        )
    return PatchesConfig(
        patch_size=_get_int("patches", d, "patch_size", 512),
        sampling_policy=_get_str("patches", d, "sampling_policy", "strategic"),
    )


def _build_boundary(d: Optional[Dict[str, Any]]) -> BoundaryConfig:
    if d is None:
        return BoundaryConfig()
    d = _ensure_section_mapping("boundary", d)
    unknown = set(d) - {"encoding"}
    if unknown:
        warnings.warn(
            f"Unknown keys in boundary section: {_sorted_unknown_keys(unknown)}.  "
            "They will be ignored.",
            stacklevel=4,
        )
    return BoundaryConfig(
        encoding=_get_str(
            "boundary", d, "encoding", "background_skeleton_buffer"
        )
    )


def _build_distance(d: Optional[Dict[str, Any]]) -> DistanceConfig:
    if d is None:
        return DistanceConfig()
    d = _ensure_section_mapping("distance", d)
    unknown = set(d) - {"target"}
    if unknown:
        warnings.warn(
            f"Unknown keys in distance section: {_sorted_unknown_keys(unknown)}.  "
            "They will be ignored.",
            stacklevel=4,
        )
    return DistanceConfig(
        target=_get_str(
            "distance", d, "target", "unsigned_distance_to_boundary"
        )
    )


def _build_normalization(d: Optional[Dict[str, Any]]) -> NormalizationConfig:
    if d is None:
        return NormalizationConfig()
    d = _ensure_section_mapping("normalization", d)
    unknown = set(d) - {"name", "clip_percentiles", "scale_range", "stats_computed_on"}
    if unknown:
        warnings.warn(
            f"Unknown keys in normalization section: {_sorted_unknown_keys(unknown)}.  "
            "They will be ignored.",
            stacklevel=4,
        )
    raw_clip = _get_two_number_tuple(
        "normalization", d, "clip_percentiles", (0.5, 99.5)
    )
    raw_scale = _get_two_number_tuple(
        "normalization", d, "scale_range", (0.0, 1.0)
    )
    return NormalizationConfig(
        name=_get_str("normalization", d, "name", "robust_percentile"),
        clip_percentiles=raw_clip,
        scale_range=raw_scale,
        stats_computed_on=_get_str(
            "normalization", d, "stats_computed_on", "valid_train_pixels"
        ),
    )


def _build_split(d: Optional[Dict[str, Any]]) -> SplitConfig:
    if d is None:
        return SplitConfig()
    d = _ensure_section_mapping("split", d)
    unknown = set(d) - {"policy", "random_seed"}
    if unknown:
        warnings.warn(
            f"Unknown keys in split section: {_sorted_unknown_keys(unknown)}.  "
            "They will be ignored.",
            stacklevel=4,
        )
    seed = _get_optional_int("split", d, "random_seed", 42)
    return SplitConfig(
        policy=_get_str("split", d, "policy", "spatial_stratified"),
        random_seed=seed,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_config(raw: Dict[str, Any]) -> PrepDataConfig:
    """Validate *raw* dict and return a fully-validated PrepDataConfig.

    Steps:
      1. Warn about any unexpected top-level keys (warns, not errors — see below).
      2. Require feature_mode to be explicitly present — raises FeatureModeError if absent.
      3. Build each sub-config from its section dict.
      4. Construct PrepDataConfig.
      5. Call PrepDataConfig.validate() — raises ContractError on failure.

    Unknown top-level keys are warnings, not errors, because:
      - The contract (DATA_CONTRACT.md §17) lists causes for explicit errors; an
        unrecognised key is not one of them.
      - Wrapper tooling may legitimately inject metadata keys (e.g. run_id).
      - After step 2, a key-name typo for feature_mode (e.g. "featuremode") will
        produce both a warning AND a FeatureModeError, so typos are caught.
      - All other sections fall back to documented baseline defaults, which are
        the same values as the checked-in baseline.raw8.yaml.

    Raises:
        FeatureModeError: if feature_mode is absent from *raw*.
        ContractError (or a subclass): on any other contract violation.
    """
    if not isinstance(raw, dict):
        raise ContractError(
            f"Config root must be a YAML mapping/object, got {type(raw).__name__}."
        )

    unknown = set(raw) - _KNOWN_TOP_LEVEL_KEYS
    if unknown:
        warnings.warn(
            f"Unknown top-level config keys: {_sorted_unknown_keys(unknown)}.  "
            "They will be ignored.  Check for typos.",
            stacklevel=2,
        )

    # feature_mode is required — no silent default.
    # module_prep_data.md §17.1 states the config *must* specify feature_mode.
    # Using .get() with a default would silently mask key-name typos.
    if "feature_mode" not in raw:
        raise FeatureModeError(
            "feature_mode is required in the config but is not present.  "
            f"Accepted values: {list(FEATURE_MODES)}.  "
            "Check for key typos (e.g. 'featuremode' instead of 'feature_mode').  "
            "(DATA_CONTRACT.md §7.1, module_prep_data.md §17.1)"
        )

    cfg = PrepDataConfig(
        feature_mode=raw["feature_mode"],
        valid_policy=_build_valid_policy(raw.get("valid_policy")),
        aoi=_build_aoi(raw.get("aoi")),
        patches=_build_patches(raw.get("patches")),
        boundary=_build_boundary(raw.get("boundary")),
        distance=_build_distance(raw.get("distance")),
        normalization=_build_normalization(raw.get("normalization")),
        split=_build_split(raw.get("split")),
    )
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> PrepDataConfig:
    """Load a YAML config file and return a fully-validated PrepDataConfig.

    Convenience wrapper around load_yaml() + build_config().

    Raises:
        FileNotFoundError:  if *path* does not exist.
        ContractError:      on any contract or validation failure.
    """
    return build_config(load_yaml(path))
