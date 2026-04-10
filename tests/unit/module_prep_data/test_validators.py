"""Unit tests for module_prep_data input-contract validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_fields.common.errors import ContractError, SpatialContractError, ValidPolicyError
from ai_fields.module_prep_data.validators import (
    validate_band_count_contract,
    validate_check_inputs_contract,
    validate_crs_contract,
    validate_input_paths_contract,
    validate_nodata_valid_contract,
    validate_required_inputs_present,
    validate_vector_geometry_contract,
)


def _raster_meta(**overrides):
    data = {
        "band_count": 8,
        "crs": "EPSG:32642",
        "has_valid_mask": True,
        "nodata": None,
        "readable": True,
    }
    data.update(overrides)
    return data


def _vector_meta(**overrides):
    data = {
        "feature_count": 10,
        "geometry_types": ["Polygon"],
        "crs": "EPSG:32642",
        "readable": True,
    }
    data.update(overrides)
    return data


def _aoi_meta(**overrides):
    data = {
        "feature_count": 1,
        "geometry_types": ["MultiPolygon"],
        "crs": "EPSG:32642",
        "readable": True,
    }
    data.update(overrides)
    return data


class TestValidateRequiredInputsPresent:
    def test_happy_path(self):
        validate_required_inputs_present(
            raster_input="input.tif",
            vector_input="labels.gpkg",
            raster_metadata=_raster_meta(),
            vector_metadata=_vector_meta(),
            require_metadata=True,
        )

    def test_missing_raster_input_raises(self):
        with pytest.raises(ContractError):
            validate_required_inputs_present(raster_input=None, vector_input="labels.gpkg")

    def test_missing_vector_input_raises(self):
        with pytest.raises(ContractError):
            validate_required_inputs_present(raster_input="input.tif", vector_input=None)

    def test_require_metadata_missing_raises(self):
        with pytest.raises(ContractError):
            validate_required_inputs_present(
                raster_input="input.tif",
                vector_input="labels.gpkg",
                raster_metadata=None,
                vector_metadata=_vector_meta(),
                require_metadata=True,
            )

    def test_wrong_metadata_type_raises(self):
        with pytest.raises(ContractError):
            validate_required_inputs_present(
                raster_input="input.tif",
                vector_input="labels.gpkg",
                raster_metadata=[],
                vector_metadata=_vector_meta(),
            )


class TestValidateInputPathsContract:
    def test_happy_path(self):
        result = validate_input_paths_contract(
            raster_path="input.tif",
            vector_path=Path("labels.gpkg"),
            aoi_path=None,
        )
        assert isinstance(result["raster_path"], Path)
        assert isinstance(result["vector_path"], Path)
        assert result["aoi_path"] is None

    @pytest.mark.parametrize("bad_path", ["", "   ", 123, {}, []])
    def test_bad_raster_path_raises(self, bad_path):
        with pytest.raises(ContractError):
            validate_input_paths_contract(
                raster_path=bad_path,
                vector_path="labels.gpkg",
            )


class TestValidateBandCountContract:
    def test_happy_path(self):
        assert validate_band_count_contract(raster_metadata=_raster_meta()) == 8

    def test_invalid_band_count_raises(self):
        with pytest.raises(ContractError):
            validate_band_count_contract(raster_metadata=_raster_meta(band_count=7))

    def test_missing_band_count_raises(self):
        md = _raster_meta()
        md.pop("band_count")
        with pytest.raises(ContractError):
            validate_band_count_contract(raster_metadata=md)

    def test_wrong_type_band_count_raises(self):
        with pytest.raises(ContractError):
            validate_band_count_contract(raster_metadata=_raster_meta(band_count="8"))


class TestValidateVectorGeometryContract:
    def test_happy_polygon_family(self):
        validate_vector_geometry_contract(
            vector_metadata=_vector_meta(geometry_types=["Polygon", "MultiPolygon", "PolygonZ"])
        )

    def test_empty_vector_collection_raises(self):
        with pytest.raises(ContractError):
            validate_vector_geometry_contract(vector_metadata=_vector_meta(feature_count=0))

    def test_empty_geometry_types_raises(self):
        with pytest.raises(ContractError):
            validate_vector_geometry_contract(vector_metadata=_vector_meta(geometry_types=[]))

    def test_unsupported_geometry_kind_raises(self):
        with pytest.raises(ContractError):
            validate_vector_geometry_contract(
                vector_metadata=_vector_meta(geometry_types=["LineString"])
            )

    def test_wrong_type_vector_metadata_raises(self):
        with pytest.raises(ContractError):
            validate_vector_geometry_contract(vector_metadata="not-a-mapping")


class TestValidateCrsContract:
    def test_happy_path_case_insensitive(self):
        result = validate_crs_contract(
            raster_metadata=_raster_meta(crs="epsg:32642"),
            vector_metadata=_vector_meta(crs="EPSG:32642"),
        )
        assert result["raster_crs"] == "EPSG:32642"
        assert result["vector_crs"] == "EPSG:32642"
        assert result["aoi_crs"] is None

    def test_crs_mismatch_transformable_is_allowed(self):
        result = validate_crs_contract(
            raster_metadata=_raster_meta(crs="EPSG:32642"),
            vector_metadata=_vector_meta(crs="EPSG:3857"),
        )
        assert result["raster_crs"] == "EPSG:32642"
        assert result["vector_crs"] == "EPSG:3857"
        assert result["vector_reprojection_required"] is True
        assert result["reprojection_required"] is True

    def test_missing_required_crs_raises_spatial_error(self):
        with pytest.raises(SpatialContractError):
            validate_crs_contract(
                raster_metadata=_raster_meta(crs=None),
                vector_metadata=_vector_meta(crs="EPSG:32642"),
            )

    def test_invalid_crs_type_raises_spatial_error(self):
        with pytest.raises(SpatialContractError):
            validate_crs_contract(
                raster_metadata=_raster_meta(crs=32642),
                vector_metadata=_vector_meta(crs="EPSG:32642"),
            )

    def test_invalid_unparseable_crs_raises_spatial_error(self):
        with pytest.raises(SpatialContractError):
            validate_crs_contract(
                raster_metadata=_raster_meta(crs="EPSG:NOT_A_REAL_CODE"),
                vector_metadata=_vector_meta(crs="EPSG:32642"),
            )

    def test_aoi_crs_mismatch_transformable_is_allowed(self):
        result = validate_crs_contract(
            raster_metadata=_raster_meta(),
            vector_metadata=_vector_meta(),
            aoi_metadata=_aoi_meta(crs="EPSG:3857"),
        )
        assert result["aoi_crs"] == "EPSG:3857"
        assert result["aoi_reprojection_required"] is True


class TestValidateNodataValidContract:
    def test_resolvable_by_valid_mask(self):
        validate_nodata_valid_contract(has_valid_mask=True, nodata_value=None, config_override_present=False)

    def test_resolvable_by_nodata_metadata(self):
        validate_nodata_valid_contract(
            has_valid_mask=False,
            nodata_value=-9999,
            config_override_present=False,
        )

    def test_resolvable_by_config_override(self):
        validate_nodata_valid_contract(
            has_valid_mask=False,
            nodata_value=None,
            config_override_present=True,
        )

    def test_unresolved_contract_raises_valid_policy_error(self):
        with pytest.raises(ValidPolicyError):
            validate_nodata_valid_contract(
                has_valid_mask=False,
                nodata_value=None,
                config_override_present=False,
            )

    def test_wrong_type_has_valid_mask_raises_valid_policy_error(self):
        with pytest.raises(ValidPolicyError):
            validate_nodata_valid_contract(
                has_valid_mask="true",
                nodata_value=None,
                config_override_present=False,
            )

    def test_wrong_type_nodata_value_raises_valid_policy_error(self):
        with pytest.raises(ValidPolicyError):
            validate_nodata_valid_contract(
                has_valid_mask=False,
                nodata_value={"nodata": -9999},
                config_override_present=False,
            )

    def test_empty_string_nodata_value_raises_valid_policy_error(self):
        with pytest.raises(ValidPolicyError):
            validate_nodata_valid_contract(
                has_valid_mask=False,
                nodata_value="",
                config_override_present=False,
            )

    def test_bool_nodata_value_raises_valid_policy_error(self):
        with pytest.raises(ValidPolicyError):
            validate_nodata_valid_contract(
                has_valid_mask=False,
                nodata_value=True,
                config_override_present=False,
            )


class TestValidateCheckInputsContract:
    def test_happy_path_without_aoi(self):
        result = validate_check_inputs_contract(
            raster_path="input.tif",
            vector_path="labels.gpkg",
            raster_metadata=_raster_meta(has_valid_mask=True, nodata=None),
            vector_metadata=_vector_meta(),
        )
        assert result["band_count"] == 8
        assert result["aoi_path"] is None
        assert result["valid_resolution"]["has_valid_mask"] is True

    def test_happy_path_with_aoi_and_nodata(self):
        result = validate_check_inputs_contract(
            raster_path="input.tif",
            vector_path="labels.gpkg",
            aoi_path="aoi.gpkg",
            raster_metadata=_raster_meta(has_valid_mask=False, nodata=-9999),
            vector_metadata=_vector_meta(),
            aoi_metadata=_aoi_meta(),
        )
        assert isinstance(result["aoi_path"], Path)
        assert result["valid_resolution"]["nodata_value"] == -9999

    def test_missing_required_input_raises(self):
        with pytest.raises(ContractError):
            validate_check_inputs_contract(
                raster_path=None,
                vector_path="labels.gpkg",
                raster_metadata=_raster_meta(),
                vector_metadata=_vector_meta(),
            )

    def test_invalid_band_count_raises(self):
        with pytest.raises(ContractError):
            validate_check_inputs_contract(
                raster_path="input.tif",
                vector_path="labels.gpkg",
                raster_metadata=_raster_meta(band_count=6),
                vector_metadata=_vector_meta(),
            )

    def test_empty_vector_collection_raises(self):
        with pytest.raises(ContractError):
            validate_check_inputs_contract(
                raster_path="input.tif",
                vector_path="labels.gpkg",
                raster_metadata=_raster_meta(),
                vector_metadata=_vector_meta(feature_count=0),
            )

    def test_unsupported_geometry_kind_raises(self):
        with pytest.raises(ContractError):
            validate_check_inputs_contract(
                raster_path="input.tif",
                vector_path="labels.gpkg",
                raster_metadata=_raster_meta(),
                vector_metadata=_vector_meta(geometry_types=["LineString"]),
            )

    def test_crs_mismatch_transformable_is_allowed(self):
        result = validate_check_inputs_contract(
            raster_path="input.tif",
            vector_path="labels.gpkg",
            raster_metadata=_raster_meta(crs="EPSG:32642"),
            vector_metadata=_vector_meta(crs="EPSG:3857"),
        )
        assert result["crs"]["vector_reprojection_required"] is True
        assert result["crs"]["reprojection_required"] is True

    def test_unresolved_nodata_valid_raises(self):
        with pytest.raises(ValidPolicyError):
            validate_check_inputs_contract(
                raster_path="input.tif",
                vector_path="labels.gpkg",
                raster_metadata=_raster_meta(has_valid_mask=False, nodata=None),
                vector_metadata=_vector_meta(),
                config_override_present=False,
            )

    def test_wrong_type_metadata_raises(self):
        with pytest.raises(ContractError):
            validate_check_inputs_contract(
                raster_path="input.tif",
                vector_path="labels.gpkg",
                raster_metadata=1,
                vector_metadata=_vector_meta(),
            )

    def test_aoi_path_without_aoi_metadata_raises(self):
        with pytest.raises(ContractError):
            validate_check_inputs_contract(
                raster_path="input.tif",
                vector_path="labels.gpkg",
                aoi_path="aoi.gpkg",
                raster_metadata=_raster_meta(),
                vector_metadata=_vector_meta(),
                aoi_metadata=None,
            )

    def test_readable_false_raises(self):
        with pytest.raises(ContractError):
            validate_check_inputs_contract(
                raster_path="input.tif",
                vector_path="labels.gpkg",
                raster_metadata=_raster_meta(readable=False),
                vector_metadata=_vector_meta(),
            )

    def test_readable_missing_is_allowed_as_hint_only(self):
        raster_md = _raster_meta()
        raster_md.pop("readable")
        vector_md = _vector_meta()
        vector_md.pop("readable")
        result = validate_check_inputs_contract(
            raster_path="input.tif",
            vector_path="labels.gpkg",
            raster_metadata=raster_md,
            vector_metadata=vector_md,
        )
        assert result["band_count"] == 8

    @pytest.mark.parametrize(
        "kwargs, expected_cls",
        [
            (
                {
                    "raster_path": 123,
                    "vector_path": "labels.gpkg",
                    "raster_metadata": _raster_meta(),
                    "vector_metadata": _vector_meta(),
                },
                ContractError,
            ),
            (
                {
                    "raster_path": "input.tif",
                    "vector_path": "labels.gpkg",
                    "raster_metadata": _raster_meta(has_valid_mask="true"),
                    "vector_metadata": _vector_meta(),
                },
                ValidPolicyError,
            ),
            (
                {
                    "raster_path": "input.tif",
                    "vector_path": "labels.gpkg",
                    "raster_metadata": _raster_meta(),
                    "vector_metadata": _vector_meta(crs=None),
                },
                SpatialContractError,
            ),
        ],
    )
    def test_no_valueerror_or_typeerror_leakage(self, kwargs, expected_cls):
        try:
            validate_check_inputs_contract(**kwargs)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive guard
            pytest.fail(f"Raw exception leaked from validators: {type(exc).__name__}: {exc}")
        except ContractError as exc:
            assert isinstance(exc, expected_cls)
        else:  # pragma: no cover - defensive guard
            pytest.fail("Expected a contract-layer exception for invalid input.")
