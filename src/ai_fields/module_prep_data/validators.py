"""Pure input-contract validators for module_prep_data (01_check_inputs stage).

Scope of this module is intentionally narrow:
  - validate required inputs, paths, metadata shape, and baseline contract rules;
  - raise explicit ContractError hierarchy exceptions on violations;
  - avoid heavy geospatial IO, reprojection, or runtime stage logic.

Source references:
  - module_prep_data.md §3, §5 (01_check_inputs), §6, §7
  - docs/IMPLEMENTATION_PLAN.md §6.2 (validators scope)
  - DATA_CONTRACT.md §3.1, §5, §6.2, §17
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ContractError, SpatialContractError, ValidPolicyError

BASELINE_SOURCE_RASTER_BAND_COUNT = 8
_ALLOWED_VECTOR_GEOMETRY_BASE = {"polygon", "multipolygon"}
_NODATA_SCALAR_TYPES = (int, float, str)


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping/object, got {type(value).__name__}.")
    return value


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a string, got {type(value).__name__}.")
    if value.strip() == "":
        raise ContractError(f"{name} must be a non-empty string.")
    return value


def _require_bool(name: str, value: Any, *, error_cls: type[ContractError] = ContractError) -> bool:
    if not isinstance(value, bool):
        raise error_cls(
            f"{name} must be a boolean (true/false), got {value!r} ({type(value).__name__})."
        )
    return value


def _to_non_empty_path(name: str, value: Any) -> Path:
    if isinstance(value, PathLike):
        as_str = str(value)
    elif isinstance(value, str):
        as_str = value
    else:
        raise ContractError(
            f"{name} must be path-like (str or Path), got {value!r} ({type(value).__name__})."
        )
    if as_str.strip() == "":
        raise ContractError(f"{name} must be a non-empty path-like value.")
    return Path(value)


def _normalize_geometry_kind(kind: str) -> str:
    normalized = kind.strip().lower().replace(" ", "")
    if normalized.endswith("25d"):
        normalized = normalized[: -len("25d")]
    if normalized.endswith("zm"):
        normalized = normalized[:-2]
    elif normalized.endswith("z") or normalized.endswith("m"):
        normalized = normalized[:-1]
    return normalized


def _extract_normalized_crs(metadata_name: str, metadata: Mapping[str, Any], *, required: bool) -> str | None:
    raw = metadata.get("crs")
    if raw is None:
        if required:
            raise SpatialContractError(
                f"{metadata_name}.crs is required for CRS compatibility checks but is missing."
            )
        return None
    if not isinstance(raw, str):
        raise SpatialContractError(
            f"{metadata_name}.crs must be a non-empty string, got {raw!r} ({type(raw).__name__})."
        )
    if raw.strip() == "":
        raise SpatialContractError(f"{metadata_name}.crs must be a non-empty string.")
    return raw.strip()


def _parse_crs_or_raise(*, metadata_name: str, raw_crs: str) -> str:
    try:
        from pyproj import CRS as ProjCRS  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SpatialContractError(
            "pyproj is required to validate CRS parseability and transformability."
        ) from exc

    try:
        parsed = ProjCRS.from_user_input(raw_crs)
    except Exception as exc:  # noqa: BLE001
        raise SpatialContractError(
            f"{metadata_name}.crs is not parseable/valid CRS: {raw_crs!r} ({exc})"
        ) from exc

    normalized = parsed.to_string()
    if not isinstance(normalized, str) or normalized.strip() == "":
        raise SpatialContractError(
            f"{metadata_name}.crs is parseable but cannot be normalized to non-empty CRS string."
        )
    return normalized


def _assert_transformable(*, source_name: str, source_crs: str, target_name: str, target_crs: str) -> None:
    try:
        from pyproj import Transformer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SpatialContractError(
            "pyproj is required to validate CRS transformability."
        ) from exc

    try:
        Transformer.from_crs(source_crs, target_crs, always_xy=True)
    except Exception as exc:  # noqa: BLE001
        raise SpatialContractError(
            "CRS contract violation: "
            f"{source_name} ({source_crs!r}) cannot be transformed to "
            f"{target_name} ({target_crs!r}): {exc}"
        ) from exc


def _validate_optional_readability_hint(metadata_name: str, metadata: Mapping[str, Any]) -> None:
    """Validate optional '<meta>.readable' metadata hint.

    This validator layer is pure and does not do file-system or geospatial I/O.
    Therefore `readable` is treated as an optional contract signal:
      - missing key: validator does not infer readability either way;
      - True: accepted;
      - False: explicit contract failure.
    """
    if "readable" not in metadata:
        return
    readable = _require_bool(f"{metadata_name}.readable", metadata["readable"])
    if not readable:
        raise ContractError(f"{metadata_name} is marked as unreadable.")


def validate_required_inputs_present(
    *,
    raster_input: Any,
    vector_input: Any,
    raster_metadata: Any | None = None,
    vector_metadata: Any | None = None,
    require_metadata: bool = False,
) -> None:
    """Validate that required input handles are present.

    AOI is intentionally not required at this layer.
    """
    if raster_input is None:
        raise ContractError("Required input 'raster_input' is missing.")
    if vector_input is None:
        raise ContractError("Required input 'vector_input' is missing.")

    if require_metadata:
        if raster_metadata is None:
            raise ContractError("Required input 'raster_metadata' is missing.")
        if vector_metadata is None:
            raise ContractError("Required input 'vector_metadata' is missing.")

    if raster_metadata is not None:
        _require_mapping("raster_metadata", raster_metadata)
    if vector_metadata is not None:
        _require_mapping("vector_metadata", vector_metadata)


def validate_input_paths_contract(
    *,
    raster_path: Any,
    vector_path: Any,
    aoi_path: Any | None = None,
) -> dict[str, Path | None]:
    """Validate path-like contract for required/optional file references."""
    resolved_raster = _to_non_empty_path("raster_path", raster_path)
    resolved_vector = _to_non_empty_path("vector_path", vector_path)
    resolved_aoi: Path | None = None
    if aoi_path is not None:
        resolved_aoi = _to_non_empty_path("aoi_path", aoi_path)
    return {
        "raster_path": resolved_raster,
        "vector_path": resolved_vector,
        "aoi_path": resolved_aoi,
    }


def validate_band_count_contract(
    *,
    raster_metadata: Any,
    expected_band_count: int = BASELINE_SOURCE_RASTER_BAND_COUNT,
) -> int:
    """Validate baseline source raster band-count contract (v1 requires 8 bands)."""
    md = _require_mapping("raster_metadata", raster_metadata)

    if isinstance(expected_band_count, bool) or not isinstance(expected_band_count, int):
        raise ContractError(
            "expected_band_count must be an integer, "
            f"got {expected_band_count!r} ({type(expected_band_count).__name__})."
        )
    if expected_band_count <= 0:
        raise ContractError(f"expected_band_count must be > 0, got {expected_band_count}.")

    if "band_count" not in md:
        raise ContractError("raster_metadata.band_count is required.")
    band_count = md["band_count"]
    if isinstance(band_count, bool) or not isinstance(band_count, int):
        raise ContractError(
            "raster_metadata.band_count must be an integer, "
            f"got {band_count!r} ({type(band_count).__name__})."
        )
    if band_count != expected_band_count:
        raise ContractError(
            f"Source raster band-count contract violation: expected {expected_band_count}, "
            f"got {band_count}."
        )
    return band_count


def validate_vector_geometry_contract(
    *,
    vector_metadata: Any,
    metadata_name: str = "vector_metadata",
) -> None:
    """Validate basic vector geometry contract for labels/AOI metadata."""
    md = _require_mapping(metadata_name, vector_metadata)

    if "feature_count" not in md:
        raise ContractError(f"{metadata_name}.feature_count is required.")
    feature_count = md["feature_count"]
    if isinstance(feature_count, bool) or not isinstance(feature_count, int):
        raise ContractError(
            f"{metadata_name}.feature_count must be an integer, "
            f"got {feature_count!r} ({type(feature_count).__name__})."
        )
    if feature_count <= 0:
        raise ContractError(f"{metadata_name}.feature_count must be > 0, got {feature_count}.")

    if "geometry_types" not in md:
        raise ContractError(f"{metadata_name}.geometry_types is required.")
    geometry_types = md["geometry_types"]
    if isinstance(geometry_types, (str, bytes)) or not isinstance(geometry_types, Sequence):
        raise ContractError(
            f"{metadata_name}.geometry_types must be a non-empty sequence of strings."
        )
    if len(geometry_types) == 0:
        raise ContractError(f"{metadata_name}.geometry_types must not be empty.")

    for idx, kind in enumerate(geometry_types):
        if not isinstance(kind, str):
            raise ContractError(
                f"{metadata_name}.geometry_types[{idx}] must be a string, "
                f"got {type(kind).__name__}."
            )
        base_kind = _normalize_geometry_kind(kind)
        if base_kind not in _ALLOWED_VECTOR_GEOMETRY_BASE:
            raise ContractError(
                f"{metadata_name}.geometry_types[{idx}] '{kind}' is not supported.  "
                "Only polygon/multipolygon family is accepted."
            )


def validate_crs_contract(
    *,
    raster_metadata: Any,
    vector_metadata: Any,
    aoi_metadata: Any | None = None,
) -> dict[str, Any]:
    """Validate CRS availability and compatibility for raster/vector/(optional AOI)."""
    raster_md = _require_mapping("raster_metadata", raster_metadata)
    vector_md = _require_mapping("vector_metadata", vector_metadata)
    aoi_md = None if aoi_metadata is None else _require_mapping("aoi_metadata", aoi_metadata)

    raster_crs_raw = _extract_normalized_crs("raster_metadata", raster_md, required=True)
    vector_crs_raw = _extract_normalized_crs("vector_metadata", vector_md, required=True)
    assert raster_crs_raw is not None  # for type checker; required=True above
    assert vector_crs_raw is not None

    raster_crs = _parse_crs_or_raise(metadata_name="raster_metadata", raw_crs=raster_crs_raw)
    vector_crs = _parse_crs_or_raise(metadata_name="vector_metadata", raw_crs=vector_crs_raw)

    vector_reprojection_required = raster_crs != vector_crs
    if vector_reprojection_required:
        _assert_transformable(
            source_name="vector_metadata.crs",
            source_crs=vector_crs,
            target_name="raster_metadata.crs",
            target_crs=raster_crs,
        )

    aoi_crs: str | None = None
    aoi_reprojection_required = False
    if aoi_md is not None:
        aoi_crs_raw = _extract_normalized_crs("aoi_metadata", aoi_md, required=True)
        assert aoi_crs_raw is not None
        aoi_crs = _parse_crs_or_raise(metadata_name="aoi_metadata", raw_crs=aoi_crs_raw)
        aoi_reprojection_required = aoi_crs != raster_crs
        if aoi_reprojection_required:
            _assert_transformable(
                source_name="aoi_metadata.crs",
                source_crs=aoi_crs,
                target_name="raster_metadata.crs",
                target_crs=raster_crs,
            )

    return {
        "raster_crs": raster_crs,
        "vector_crs": vector_crs,
        "aoi_crs": aoi_crs,
        "crs_match": (not vector_reprojection_required and not aoi_reprojection_required),
        "vector_reprojection_required": vector_reprojection_required,
        "aoi_reprojection_required": aoi_reprojection_required,
        "reprojection_required": vector_reprojection_required or aoi_reprojection_required,
    }


def validate_nodata_valid_contract(
    *,
    has_valid_mask: Any,
    nodata_value: Any = None,
    config_override_present: Any = False,
) -> None:
    """Validate that valid/NoData interpretation is resolvable without guessing."""
    mask_flag = _require_bool(
        "has_valid_mask",
        has_valid_mask,
        error_cls=ValidPolicyError,
    )
    override_flag = _require_bool(
        "config_override_present",
        config_override_present,
        error_cls=ValidPolicyError,
    )

    nodata_known = nodata_value is not None
    if nodata_known:
        if isinstance(nodata_value, bool) or not isinstance(nodata_value, _NODATA_SCALAR_TYPES):
            raise ValidPolicyError(
                "nodata_value must be a number/string or null, "
                f"got {nodata_value!r} ({type(nodata_value).__name__})."
            )
        if isinstance(nodata_value, str) and nodata_value.strip() == "":
            raise ValidPolicyError(
                "nodata_value string must be non-empty when provided."
            )

    if mask_flag or nodata_known or override_flag:
        return

    raise ValidPolicyError(
        "Unable to resolve valid/NoData policy.  Expected at least one of: "
        "valid mask, nodata metadata value, or explicit config override."
    )


def validate_check_inputs_contract(
    *,
    raster_path: Any,
    vector_path: Any,
    raster_metadata: Any,
    vector_metadata: Any,
    aoi_path: Any | None = None,
    aoi_metadata: Any | None = None,
    config_override_present: bool = False,
) -> dict[str, Any]:
    """Aggregate contract checks for stage `01_check_inputs`.

    Readability checks here are metadata-signal-only (`<meta>.readable`) and do
    not perform actual file reads at this layer.
    """
    validate_required_inputs_present(
        raster_input=raster_path,
        vector_input=vector_path,
        raster_metadata=raster_metadata,
        vector_metadata=vector_metadata,
        require_metadata=True,
    )
    paths = validate_input_paths_contract(
        raster_path=raster_path,
        vector_path=vector_path,
        aoi_path=aoi_path,
    )

    raster_md = _require_mapping("raster_metadata", raster_metadata)
    vector_md = _require_mapping("vector_metadata", vector_metadata)
    aoi_md = None if aoi_metadata is None else _require_mapping("aoi_metadata", aoi_metadata)

    if paths["aoi_path"] is not None and aoi_md is None:
        raise ContractError("aoi_metadata is required when aoi_path is provided.")

    _validate_optional_readability_hint("raster_metadata", raster_md)
    _validate_optional_readability_hint("vector_metadata", vector_md)
    if aoi_md is not None:
        _validate_optional_readability_hint("aoi_metadata", aoi_md)

    band_count = validate_band_count_contract(raster_metadata=raster_md)
    validate_vector_geometry_contract(vector_metadata=vector_md, metadata_name="vector_metadata")
    if aoi_md is not None:
        validate_vector_geometry_contract(vector_metadata=aoi_md, metadata_name="aoi_metadata")

    crs_summary = validate_crs_contract(
        raster_metadata=raster_md,
        vector_metadata=vector_md,
        aoi_metadata=aoi_md,
    )

    validate_nodata_valid_contract(
        has_valid_mask=raster_md.get("has_valid_mask", False),
        nodata_value=raster_md.get("nodata"),
        config_override_present=config_override_present,
    )

    return {
        **paths,
        "band_count": band_count,
        "crs": crs_summary,
        "valid_resolution": {
            "has_valid_mask": raster_md.get("has_valid_mask", False),
            "nodata_value": raster_md.get("nodata"),
            "config_override_present": config_override_present,
        },
    }
