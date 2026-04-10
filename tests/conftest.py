from pathlib import Path

import numpy as np
import pytest

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_root() -> Path:
    """Root directory for all test fixtures."""
    return FIXTURES_ROOT


@pytest.fixture
def raster_fixtures(fixtures_root: Path) -> Path:
    return fixtures_root / "rasters"


@pytest.fixture
def vector_fixtures(fixtures_root: Path) -> Path:
    return fixtures_root / "vectors"


@pytest.fixture
def config_fixtures(fixtures_root: Path) -> Path:
    return fixtures_root / "configs"


@pytest.fixture
def manifest_fixtures(fixtures_root: Path) -> Path:
    return fixtures_root / "manifests"


@pytest.fixture
def baseline_raw8_config_path(config_fixtures: Path) -> Path:
    """Path to the checked-in baseline.raw8.yaml fixture config."""
    return config_fixtures / "baseline.raw8.yaml"


@pytest.fixture
def tiny_8band_raster_path(tmp_path: Path) -> Path:
    """Tiny synthetic 8-band GeoTIFF in EPSG:32637, suitable for runtime probe tests.

    Covers (599800, 4399800)–(600200, 4400200) at 25 m/pixel (16×16).
    Written with nodata=0.  Skipped automatically if rasterio is not installed.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    out = tmp_path / "raster_fixtures" / "tiny_8band.tif"
    out.parent.mkdir(parents=True, exist_ok=True)

    transform = from_bounds(
        west=599800.0, south=4399800.0, east=600200.0, north=4400200.0,
        width=16, height=16,
    )
    # Non-zero data so valid mask has pixels to work with (nodata=0 → valid=1 here)
    data = np.full((8, 16, 16), fill_value=1000, dtype=np.uint16)

    with rasterio.open(
        out, "w",
        driver="GTiff",
        height=16, width=16, count=8,
        dtype="uint16",
        crs=CRS.from_epsg(32637),
        transform=transform,
        nodata=0,
    ) as ds:
        ds.write(data)

    return out


@pytest.fixture
def tiny_vector_path(tmp_path: Path) -> Path:
    """Tiny synthetic vector layer (1 Polygon, EPSG:32637) for runtime probe tests.

    Polygon at (599900, 4399900)–(600100, 4400100) — centred inside the
    tiny_8band_raster_path extent.
    Skipped automatically if fiona is not installed.
    """
    fiona = pytest.importorskip("fiona")

    out = tmp_path / "vector_fixtures" / "tiny_fields.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)

    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(out, "w", driver="GPKG", schema=schema, crs="EPSG:32637") as dst:
        dst.write({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    (599900.0, 4399900.0),
                    (600100.0, 4399900.0),
                    (600100.0, 4400100.0),
                    (599900.0, 4400100.0),
                    (599900.0, 4399900.0),
                ]],
            },
            "properties": {"id": 1},
        })

    return out


@pytest.fixture
def tiny_feature_stack_dir(tmp_path: Path, tiny_8band_raster_path: Path) -> Path:
    """Directory containing img.tif and valid.tif computed from the tiny raster.

    Skipped automatically if rasterio is not installed.
    """
    pytest.importorskip("rasterio")
    from ai_fields.module_prep_data.features_compute import compute_and_save_features

    out_dir = tmp_path / "feature_stack"
    out_dir.mkdir(parents=True, exist_ok=True)
    compute_and_save_features(
        raster_path=tiny_8band_raster_path,
        output_dir=out_dir,
        feature_mode="raw8",
    )
    return out_dir


@pytest.fixture
def tiny_targets_dir(tmp_path: Path, tiny_8band_raster_path: Path, tiny_vector_path: Path) -> Path:
    """Directory containing all target GeoTIFFs computed from the tiny raster+vector.

    Skipped automatically if rasterio, geopandas, or scipy are not installed.
    """
    pytest.importorskip("rasterio")
    pytest.importorskip("geopandas")
    pytest.importorskip("scipy")
    from ai_fields.module_prep_data.targets_compute import compute_and_save_targets

    out_dir = tmp_path / "targets"
    out_dir.mkdir(parents=True, exist_ok=True)
    compute_and_save_targets(
        raster_path=tiny_8band_raster_path,
        vector_path=tiny_vector_path,
        output_dir=out_dir,
    )
    return out_dir
