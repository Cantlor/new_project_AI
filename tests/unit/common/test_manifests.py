"""Unit tests for ai_fields.common.manifests."""

from __future__ import annotations

import json

import pytest

from ai_fields.common.errors import ContractError, ManifestError
from ai_fields.common.manifests import read_manifest, write_manifest, write_summary


def _valid_manifest_payload(**overrides):
    data = {
        "schema_name": "prep_data.check_inputs_manifest",
        "schema_version": "v1",
        "module_name": "module_prep_data",
        "module_version": None,
        "data_contract_version": "v1",
        "run_id": "run-001",
        "stage_name": "01_check_inputs",
        "created_at_utc": "2026-04-01T00:00:00Z",
        "status": "success",
    }
    data.update(overrides)
    return data


def test_manifest_error_is_contract_error():
    assert issubclass(ManifestError, ContractError)


class TestWriteManifest:
    def test_happy_path_writes_json_and_creates_parent_dirs(self, tmp_path):
        out_path = tmp_path / "nested" / "check_inputs_manifest.json"
        payload = _valid_manifest_payload(extra={"note": "ok"})

        write_manifest(out_path, payload)

        assert out_path.exists()
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))
        assert on_disk == payload
        assert out_path.read_text(encoding="utf-8").endswith("\n")

    def test_deterministic_output_for_different_key_order(self, tmp_path):
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        payload_a = _valid_manifest_payload()
        payload_b = {
            "status": "success",
            "created_at_utc": "2026-04-01T00:00:00Z",
            "stage_name": "01_check_inputs",
            "run_id": "run-001",
            "data_contract_version": "v1",
            "module_version": None,
            "module_name": "module_prep_data",
            "schema_version": "v1",
            "schema_name": "prep_data.check_inputs_manifest",
        }

        write_manifest(path_a, payload_a)
        write_manifest(path_b, payload_b)

        assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")

    def test_non_mapping_payload_raises_manifest_error(self, tmp_path):
        with pytest.raises(ManifestError):
            write_manifest(tmp_path / "m.json", [])  # type: ignore[arg-type]

    def test_missing_required_top_level_field_raises_manifest_error(self, tmp_path):
        payload = _valid_manifest_payload()
        payload.pop("run_id")
        with pytest.raises(ManifestError):
            write_manifest(tmp_path / "m.json", payload)

    def test_invalid_status_value_raises_manifest_error(self, tmp_path):
        payload = _valid_manifest_payload(status="done")
        with pytest.raises(ManifestError):
            write_manifest(tmp_path / "m.json", payload)

    def test_invalid_path_type_raises_manifest_error(self):
        with pytest.raises(ManifestError):
            write_manifest(123, _valid_manifest_payload())  # type: ignore[arg-type]

    def test_unserializable_payload_raises_manifest_error_not_type_error(self, tmp_path):
        payload = _valid_manifest_payload(extra={"bad": {1, 2, 3}})
        try:
            write_manifest(tmp_path / "m.json", payload)
        except TypeError as exc:  # pragma: no cover - defensive guard
            pytest.fail(f"TypeError leaked from manifest writer: {exc}")
        except ManifestError:
            pass
        else:  # pragma: no cover - defensive guard
            pytest.fail("Expected ManifestError for unserializable payload.")


class TestReadManifest:
    def test_happy_path_roundtrip(self, tmp_path):
        path = tmp_path / "manifest.json"
        payload = _valid_manifest_payload()
        write_manifest(path, payload)

        loaded = read_manifest(path)

        assert loaded == payload

    def test_missing_file_raises_manifest_error(self, tmp_path):
        with pytest.raises(ManifestError):
            read_manifest(tmp_path / "missing_manifest.json")

    def test_invalid_json_raises_manifest_error_not_jsondecodeerror(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{bad json", encoding="utf-8")
        try:
            read_manifest(path)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            pytest.fail(f"JSONDecodeError leaked from manifest reader: {exc}")
        except ManifestError:
            pass
        else:  # pragma: no cover - defensive guard
            pytest.fail("Expected ManifestError for invalid JSON.")

    def test_top_level_non_object_raises_manifest_error(self, tmp_path):
        path = tmp_path / "array.json"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ManifestError):
            read_manifest(path)

    def test_missing_required_fields_on_read_raises_manifest_error(self, tmp_path):
        path = tmp_path / "incomplete.json"
        incomplete_payload = {"schema_name": "prep_data.check_inputs_manifest"}
        path.write_text(json.dumps(incomplete_payload), encoding="utf-8")
        with pytest.raises(ManifestError):
            read_manifest(path)

    def test_invalid_path_type_raises_manifest_error(self):
        with pytest.raises(ManifestError):
            read_manifest(123)  # type: ignore[arg-type]


class TestWriteSummary:
    def test_happy_path_writes_summary_json(self, tmp_path):
        path = tmp_path / "summary.json"
        payload = {
            "schema_name": "prep_data.summary",
            "status": "success",
            "feature_mode": "raw8",
            "written_total": 128,
        }

        write_summary(path, payload)

        assert path.exists()
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk == payload

    def test_non_mapping_payload_raises_manifest_error(self, tmp_path):
        with pytest.raises(ManifestError):
            write_summary(tmp_path / "summary.json", "bad-payload")  # type: ignore[arg-type]

    def test_invalid_path_type_raises_manifest_error(self):
        with pytest.raises(ManifestError):
            write_summary(123, {"status": "success"})  # type: ignore[arg-type]

    def test_roundtrip_by_reading_written_summary_as_json(self, tmp_path):
        path = tmp_path / "summary.json"
        payload = {"schema_name": "prep_data.summary", "status": "success"}
        write_summary(path, payload)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == payload
