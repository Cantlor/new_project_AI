"""Unit tests for _extract_aoi_policy_from_postprocess_manifest in module_eval.export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.module_eval.export import _extract_aoi_policy_from_postprocess_manifest


def _write_postprocess_manifest(path: Path, *, resolved_policy: dict | None = None) -> Path:
    """Write a postprocess_manifest.json fixture that satisfies read_manifest validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        # Required top-level fields for read_manifest
        "schema_name": "postprocess_vectorize.postprocess_manifest",
        "schema_version": "v1",
        "module_name": "module_postprocess_vectorize",
        "module_version": None,
        "data_contract_version": "v1",
        "run_id": "test-run-001",
        "stage_name": "postprocess_scene",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "status": "success",
    }
    if resolved_policy is not None:
        payload["resolved_policy"] = resolved_policy
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestExtractAoiPolicyNone:
    def test_none_path_returns_none(self):
        result = _extract_aoi_policy_from_postprocess_manifest(None)
        assert result is None

    def test_manifest_without_resolved_policy_returns_none(self, tmp_path):
        p = _write_postprocess_manifest(tmp_path / "postprocess_manifest.json")
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert result is None

    def test_manifest_with_resolved_policy_no_aoi_returns_none(self, tmp_path):
        p = _write_postprocess_manifest(
            tmp_path / "postprocess_manifest.json",
            resolved_policy={"valid_suppression": "applied", "aoi_policy": None},
        )
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert result is None

    def test_nonexistent_file_returns_none_best_effort(self, tmp_path):
        p = tmp_path / "does_not_exist.json"
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert result is None

    def test_invalid_json_returns_none_best_effort(self, tmp_path):
        p = tmp_path / "broken.json"
        p.write_text("NOT VALID JSON {{{", encoding="utf-8")
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert result is None


class TestExtractAoiPolicyReturnsDict:
    def test_aoi_policy_dict_is_returned_structured(self, tmp_path):
        p = _write_postprocess_manifest(
            tmp_path / "postprocess_manifest.json",
            resolved_policy={
                "aoi_policy": {
                    "mode": "aoi_suppression_applied",
                    "aoi_path": "/some/aoi.gpkg",
                    "suppression_applied": True,
                }
            },
        )
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert result is not None
        assert isinstance(result, dict)

    def test_result_has_source_field(self, tmp_path):
        p = _write_postprocess_manifest(
            tmp_path / "postprocess_manifest.json",
            resolved_policy={"aoi_policy": {"mode": "aoi_suppression_applied", "suppression_applied": True}},
        )
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert result["source"] == "upstream_postprocess_manifest"

    def test_result_has_postprocess_manifest_path(self, tmp_path):
        p = _write_postprocess_manifest(
            tmp_path / "postprocess_manifest.json",
            resolved_policy={"aoi_policy": {"mode": "aoi_suppression_applied", "suppression_applied": True}},
        )
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert result["postprocess_manifest_path"] == str(p)

    def test_result_contains_original_aoi_fields(self, tmp_path):
        p = _write_postprocess_manifest(
            tmp_path / "postprocess_manifest.json",
            resolved_policy={
                "aoi_policy": {
                    "mode": "aoi_suppression_applied",
                    "aoi_path": "/data/aoi.gpkg",
                    "aoi_manifest_path": "/data/aoi_manifest.json",
                    "suppression_applied": True,
                }
            },
        )
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert result["mode"] == "aoi_suppression_applied"
        assert result["aoi_path"] == "/data/aoi.gpkg"
        assert result["aoi_manifest_path"] == "/data/aoi_manifest.json"
        assert result["suppression_applied"] is True

    def test_result_has_aoi_policy_summary_string(self, tmp_path):
        p = _write_postprocess_manifest(
            tmp_path / "postprocess_manifest.json",
            resolved_policy={
                "aoi_policy": {"mode": "aoi_suppression_applied", "suppression_applied": True}
            },
        )
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        summary = result["aoi_policy_summary"]
        assert isinstance(summary, str)
        assert "mode=aoi_suppression_applied" in summary
        assert "suppression_applied=True" in summary

    def test_suppression_not_applied_mode(self, tmp_path):
        p = _write_postprocess_manifest(
            tmp_path / "postprocess_manifest.json",
            resolved_policy={
                "aoi_policy": {
                    "mode": "aoi_path_provided_suppression_not_applied",
                    "suppression_applied": False,
                }
            },
        )
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert result is not None
        assert result["mode"] == "aoi_path_provided_suppression_not_applied"
        assert "suppression_applied=False" in result["aoi_policy_summary"]

    def test_result_is_machine_readable_dict(self, tmp_path):
        """The returned value must be a plain dict, not a custom object."""
        p = _write_postprocess_manifest(
            tmp_path / "postprocess_manifest.json",
            resolved_policy={"aoi_policy": {"mode": "aoi_suppression_applied", "suppression_applied": True}},
        )
        result = _extract_aoi_policy_from_postprocess_manifest(p)
        assert type(result) is dict  # noqa: E721
        # Must be JSON-serialisable
        json.dumps(result)  # should not raise
