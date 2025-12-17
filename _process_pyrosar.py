"""
PyroSAR/SNAP processing with Kartverket and Copernicus DEMs.
Run in conda pyrosar environment.
"""
import json
import tempfile
import os
from _config import (
    OUTPUT_RESOLUTION, COMPARISON_DATES, AOI_FILES, DEM_FILES,
    MULTITEMP_PYROSAR_KART_DIR, MULTITEMP_PYROSAR_COP_DIR,
    get_safe_path, load_aoi
)

try:
    from pyroSAR.snap import geocode
except ImportError:
    geocode = None
    print("Warning: pyroSAR not available")


def create_shapefile(aoi_key):
    """Create temporary geojson shapefile for pyroSAR."""
    #use the first geometry
    gdf = load_aoi(aoi_key).to_crs('EPSG:4326')

    geom = gdf.geometry.iloc[0].__geo_interface__
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": geom,
            "properties": {"name": aoi_key}
        }]
    }

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False)
    json.dump(geojson, tmp)
    tmp.close()
    return tmp.name


def find_output(output_dir):
    """Check if output already exists."""
    if not output_dir.exists():
        return None
    matches = list(output_dir.glob('*_VV_gamma0-rtc.tif'))
    return matches[0] if matches else None


def process_pyrosar(date_str, aoi_key, dem_type='kartverket'):
    """Process with PyroSAR using specified DEM."""
    if geocode is None:
        raise ImportError("pyroSAR required")

    safe = get_safe_path(date_str)
    if not safe:
        raise FileNotFoundError(f"No SAFE file for {date_str}")

    if dem_type == 'kartverket':
        output_dir = MULTITEMP_PYROSAR_KART_DIR / date_str / aoi_key
        dem_path = DEM_FILES[aoi_key]
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM not found: {dem_path}")

        dem_kwargs = {
            'demName': 'External DEM',
            'externalDEMFile': str(dem_path),
            'externalDEMNoDataValue': -9999,
            'externalDEMApplyEGM': True,
        }
    else:
        output_dir = MULTITEMP_PYROSAR_COP_DIR / date_str / aoi_key
        dem_kwargs = {
            'demName': 'Copernicus 30m Global DEM',
        }

    if find_output(output_dir):
        return False  # Already exists

    output_dir.mkdir(parents=True, exist_ok=True)
    shapefile = create_shapefile(aoi_key)

    try:
        geocode(
            infile=str(safe),
            outdir=str(output_dir),
            shapefile=shapefile,
            t_srs=32632,  # UTM 32N
            spacing=OUTPUT_RESOLUTION,
            scaling='linear',
            terrainFlattening=True,
            refarea='gamma0',
            removeS1BorderNoise=True,
            removeS1ThermalNoise=True,
            alignToStandardGrid=True,
            standardGridOriginX=0.0,
            standardGridOriginY=0.0,
            export_extra=[
                'incidenceAngleFromEllipsoid',
                'layoverShadowMask',
                'localIncidenceAngle'
            ],
            cleanup=True,
            **dem_kwargs
        )
        return True
    finally:
        os.remove(shapefile)


def main():
    aoi_keys = list(AOI_FILES.keys())

    for date_str in COMPARISON_DATES:
        for aoi_key in aoi_keys:

            # Kartverket DEM
            try:
                if process_pyrosar(date_str, aoi_key, 'kartverket'):
                    print(f"{date_str}/{aoi_key}/kart: done")
            except Exception as e:
                print(f"{date_str}/{aoi_key}/kart: {e}")

            # Copernicus DEM
            try:
                if process_pyrosar(date_str, aoi_key, 'copernicus'):
                    print(f"{date_str}/{aoi_key}/cop: done")
            except Exception as e:
                print(f"{date_str}/{aoi_key}/cop: {e}")


if __name__ == "__main__":
    main()