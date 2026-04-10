"""Minimal manifest/summary JSON utilities.

This layer intentionally provides only baseline object-level contract checks
for manifests and deterministic JSON read/write helpers. It does not implement
full recursive schema validation or stage-specific manifest builders.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from ai_fields.common.errors import ManifestError

_MANIFEST_REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_name",
    "schema_version",
    "module_name",
    "module_version",
    "data_contract_version",
    "run_id",
    "stage_name",
    "created_at_utc",
    "status",
)
_MANIFEST_REQUIRED_NON_NULL_STR_FIELDS = (
    "schema_name",
    "schema_version",
    "module_name",
    "data_contract_version",
    "run_id",
    "created_at_utc",
    "status",
)
_MANIFEST_NULLABLE_STR_FIELDS = ("module_version", "stage_name")
_MANIFEST_ALLOWED_STATUS_VALUES = {"success", "partial", "failed"}


def _normalize_path(path: Any, *, name: str) -> Path:
    if isinstance(path, PathLike):
        as_text = str(path)
    elif isinstance(path, str):
        as_text = path
    else:
        raise ManifestError(
            f"{name} must be path-like (str or Path), got {path!r} ({type(path).__name__})."
        )
    if as_text.strip() == "":
        raise ManifestError(f"{name} must be a non-empty path-like value.")
    return Path(path)


def _normalize_payload(payload: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ManifestError(
            f"{name} must be a mapping/object, got {payload!r} ({type(payload).__name__})."
        )
    normalized = dict(payload)
    if not all(isinstance(k, str) for k in normalized):
        raise ManifestError(f"{name} top-level keys must be strings.")
    return normalized


def _require_non_empty_string_field(payload: Mapping[str, Any], field_name: str) -> None:
    value = payload[field_name]
    if not isinstance(value, str) or value.strip() == "":
        raise ManifestError(
            f"{field_name} must be a non-empty string, got {value!r} ({type(value).__name__})."
        )


def _require_nullable_string_field(payload: Mapping[str, Any], field_name: str) -> None:
    value = payload[field_name]
    if value is not None and not isinstance(value, str):
        raise ManifestError(
            f"{field_name} must be a string or null, got {value!r} ({type(value).__name__})."
        )


def _validate_manifest_top_level_contract(payload: Mapping[str, Any]) -> None:
    missing = [field for field in _MANIFEST_REQUIRED_TOP_LEVEL_FIELDS if field not in payload]
    if missing:
        raise ManifestError(
            f"Manifest payload is missing required top-level fields: {sorted(missing)}."
        )

    for field_name in _MANIFEST_REQUIRED_NON_NULL_STR_FIELDS:
        _require_non_empty_string_field(payload, field_name)

    for field_name in _MANIFEST_NULLABLE_STR_FIELDS:
        _require_nullable_string_field(payload, field_name)

    if payload["status"] not in _MANIFEST_ALLOWED_STATUS_VALUES:
        raise ManifestError(
            "status must be one of "
            f"{sorted(_MANIFEST_ALLOWED_STATUS_VALUES)}, got {payload['status']!r}."
        )


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            f"Failed to serialize {kind} JSON at '{path}': {exc}"
        ) from exc
    except OSError as exc:
        raise ManifestError(
            f"Failed to write {kind} JSON at '{path}': {exc}"
        ) from exc


def write_manifest(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a manifest JSON file with minimal top-level contract validation."""
    normalized_path = _normalize_path(path, name="path")
    payload = _normalize_payload(data, name="manifest payload")
    _validate_manifest_top_level_contract(payload)
    _write_json(normalized_path, payload, kind="manifest")


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest JSON file and validate minimal top-level contract."""
    normalized_path = _normalize_path(path, name="path")
    if not normalized_path.exists():
        raise ManifestError(f"Manifest file does not exist: {normalized_path}")

    try:
        with normalized_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"Manifest file '{normalized_path}' is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise ManifestError(
            f"Failed to read manifest file '{normalized_path}': {exc}"
        ) from exc

    payload = _normalize_payload(payload, name="manifest payload")
    _validate_manifest_top_level_contract(payload)
    return payload


def write_summary(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a summary JSON file.

    Summary is intentionally allowed to be incomplete vs manifest-level contract.
    """
    normalized_path = _normalize_path(path, name="path")
    payload = _normalize_payload(data, name="summary payload")
    _write_json(normalized_path, payload, kind="summary")
