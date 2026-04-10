"""Unit tests for _resolve_predict_aoi_policy in module_target_predict.predict_run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.common.errors import ContractError
from ai_fields.module_target_predict.predict_run import _resolve_predict_aoi_policy


def _write_aoi_manifest(
    path: Path,
    *,
    buffer_m: float | None = 30.0,
    schema_name: str = "prep_data.aoi_manifest",
) -> Path:
    """Write an aoi_manifest.json fixture that satisfies read_manifest validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Required top-level fields for read_manifest
        "schema_name": schema_name,
        "schema_version": "v1",
        "module_name": "module_prep_data",
        "module_version": None,
        "data_contract_version": "v1",
        "run_id": "test-run-001",
        "stage_name": "02_prepare_spatial_context",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "status": "success",
        # Domain fields
        "aoi_present": True,
        "buffer_m": buffer_m,
        "aoi_output_path": str(path.parent / "aoi_in_raster_crs.gpkg"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestResolvePredictAoiPolicyNoAOI:
    def test_both_none_returns_aoi_present_false(self):
        result = _resolve_predict_aoi_policy(aoi_path=None, aoi_manifest_path=None)
        assert result["aoi_present"] is False
        assert result["buffer_m"] is None
        assert result["output_extent_mode"] == "full_raster"

    def test_both_none_has_no_aoi_path_key(self):
        result = _resolve_predict_aoi_policy(aoi_path=None, aoi_manifest_path=None)
        assert "aoi_path" not in result
        assert "aoi_manifest_path" not in result
        assert "aoi_source_type" not in result


class TestResolvePredictAoiPolicyUserProvided:
    def test_aoi_path_only_returns_user_provided(self):
        result = _resolve_predict_aoi_policy(aoi_path="some/aoi.gpkg", aoi_manifest_path=None)
        assert result["aoi_present"] is True
        assert result["aoi_source_type"] == "user_provided"
        assert result["aoi_path"] == "some/aoi.gpkg"
        assert result["aoi_manifest_path"] is None
        assert result["buffer_m"] is None
        assert result["output_extent_mode"] == "full_raster"

    def test_aoi_path_as_pathlib_object(self):
        result = _resolve_predict_aoi_policy(aoi_path=Path("my/aoi.gpkg"), aoi_manifest_path=None)
        assert result["aoi_present"] is True
        assert result["aoi_source_type"] == "user_provided"

    def test_note_field_is_present(self):
        result = _resolve_predict_aoi_policy(aoi_path="aoi.gpkg", aoi_manifest_path=None)
        assert "note" in result


class TestResolvePredictAoiPolicyUpstreamManifest:
    def test_valid_manifest_resolves_upstream_type(self, tmp_path):
        manifest_path = _write_aoi_manifest(tmp_path / "aoi_manifest.json", buffer_m=30.0)
        result = _resolve_predict_aoi_policy(aoi_path=None, aoi_manifest_path=manifest_path)
        assert result["aoi_present"] is True
        assert result["aoi_source_type"] == "upstream_prep_data_resolved"
        assert result["buffer_m"] == 30.0
        assert result["aoi_manifest_path"] == str(manifest_path)

    def test_manifest_buffer_m_none_is_preserved(self, tmp_path):
        manifest_path = _write_aoi_manifest(tmp_path / "aoi_manifest.json", buffer_m=None)
        result = _resolve_predict_aoi_policy(aoi_path=None, aoi_manifest_path=manifest_path)
        assert result["buffer_m"] is None
        assert result["aoi_source_type"] == "upstream_prep_data_resolved"

    def test_manifest_aoi_output_path_used_when_aoi_path_not_given(self, tmp_path):
        manifest_path = _write_aoi_manifest(tmp_path / "aoi_manifest.json")
        result = _resolve_predict_aoi_policy(aoi_path=None, aoi_manifest_path=manifest_path)
        # Should pick up aoi_output_path from manifest
        assert result["aoi_path"] == str(tmp_path / "aoi_in_raster_crs.gpkg")

    def test_explicit_aoi_path_takes_precedence_over_manifest_output_path(self, tmp_path):
        manifest_path = _write_aoi_manifest(tmp_path / "aoi_manifest.json")
        result = _resolve_predict_aoi_policy(
            aoi_path="explicit/override.gpkg",
            aoi_manifest_path=manifest_path,
        )
        assert result["aoi_path"] == "explicit/override.gpkg"
        assert result["aoi_source_type"] == "upstream_prep_data_resolved"


class TestResolvePredictAoiPolicyErrors:
    def test_nonexistent_manifest_path_raises_contract_error(self, tmp_path):
        bad_path = tmp_path / "nonexistent.json"
        with pytest.raises(ContractError, match="Failed to read aoi_manifest_path"):
            _resolve_predict_aoi_policy(aoi_path=None, aoi_manifest_path=bad_path)

    def test_wrong_schema_name_raises_contract_error(self, tmp_path):
        bad_manifest = _write_aoi_manifest(
            tmp_path / "bad_manifest.json",
            schema_name="some.other_schema",
        )
        with pytest.raises(ContractError, match="does not point to a"):
            _resolve_predict_aoi_policy(aoi_path=None, aoi_manifest_path=bad_manifest)

    def test_invalid_json_raises_contract_error(self, tmp_path):
        bad_path = tmp_path / "broken.json"
        bad_path.write_text("NOT VALID JSON {{{{", encoding="utf-8")
        with pytest.raises(ContractError, match="Failed to read aoi_manifest_path"):
            _resolve_predict_aoi_policy(aoi_path=None, aoi_manifest_path=bad_path)

    def test_error_wraps_original_exception(self, tmp_path):
        bad_path = tmp_path / "missing.json"
        with pytest.raises(ContractError) as exc_info:
            _resolve_predict_aoi_policy(aoi_path=None, aoi_manifest_path=bad_path)
        # The ContractError must chain the original exception
        assert exc_info.value.__cause__ is not None
