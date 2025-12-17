"""
GEE processing: s1_ard (Kartverket + Copernicus) and Standard GRD.
"""
import ee
import requests
from datetime import datetime, timedelta
from _config import (
    GEE_PROJECT, OUTPUT_CRS, OUTPUT_RESOLUTION,
    COMPARISON_DATES, AOI_FILES, GEE_DEM_ASSETS,
    get_gee_path, load_aoi
)

# Import s1_ard wrapper (assuming it's available)
try:
    from s1_ard import wrapper as s1ard
except ImportError:
    s1ard = None
    print("Warning: s1_ard module not found")


def get_ee_geometry(aoi_key):
    """Get EE geometry from geojson AOI."""
    gdf = load_aoi(aoi_key).to_crs('EPSG:4326')
    geom = gdf.geometry.iloc[0]
    
    # Check if geometry is MultiPolygon and extract the Polygon
    if geom.geom_type == 'MultiPolygon':
        # If valid MultiPolygon, grab the first geometry (or the largest)
        # s1_ard usually expects a simple Polygon for the ROI
        geom = max(geom.geoms, key=lambda x: x.area)
        
    coords = list(geom.exterior.coords)
    return ee.Geometry.Polygon([coords])


def download_image(image, band, geometry, output_path):
    """Download a single band from EE image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = image.select(band).getDownloadUrl({
        'region': geometry,
        'crs': OUTPUT_CRS,
        'scale': OUTPUT_RESOLUTION,
        'format': 'GEO_TIFF'
    })
    r = requests.get(url)
    r.raise_for_status()
    with open(output_path, 'wb') as f:
        f.write(r.content)


def process_s1ard(date_str, aoi_key, dem_type='copernicus'):
    """Process with s1_ard using specified DEM."""
    if s1ard is None:
        raise ImportError("s1_ard module required")
    
    geometry = get_ee_geometry(aoi_key)
    dt = datetime.strptime(date_str, '%Y%m%d')
    start = dt.strftime('%Y-%m-%d')
    end = (dt + timedelta(days=1)).strftime('%Y-%m-%d')
    
    if dem_type == 'copernicus':
        # FIX: Properly prepare Copernicus DEM with buffered bounds
        # This ensures we get enough DEM coverage for terrain calculations
        bounds = geometry.bounds()
        buffered_bounds = bounds.buffer(5000)  # 5km buffer
        
        # Get the DEM with proper projection handling
        dem_collection = ee.ImageCollection('COPERNICUS/DEM/GLO30').filterBounds(buffered_bounds)
        dem_proj = dem_collection.first().select('DEM').projection()
        
        dem = (dem_collection
               .mosaic()
               .setDefaultProjection(dem_proj)  # CRITICAL: Set projection after mosaic
               .clip(buffered_bounds)
               .select('DEM')
               .rename('DEM'))
        
        method_key = 'gee_s1ard_copernicus'
    else:
        dem = ee.Image(GEE_DEM_ASSETS[aoi_key])
        if 'DEM' not in dem.bandNames().getInfo():
            dem = dem.select([0]).rename('DEM')
        method_key = 'gee_s1ard_kartverket'
    
    params = {
        'START_DATE': start,
        'STOP_DATE': end,
        'POLARIZATION': 'VVVH',
        'ORBIT': 'BOTH',
        'ROI': geometry,
        'APPLY_BORDER_NOISE_CORRECTION': True,
        'APPLY_SPECKLE_FILTERING': False,

        #Crashes without these even if filtering is disabled
        'SPECKLE_FILTER_FRAMEWORK': 'MULTI',
        'SPECKLE_FILTER': 'GAMMA MAP',
        'SPECKLE_FILTER_KERNEL_SIZE': 7,
        'SPECKLE_FILTER_NR_OF_IMAGES': 10,

        'APPLY_TERRAIN_FLATTENING': True,
        'DEM': dem,
        'TERRAIN_FLATTENING_MODEL': 'DIRECT', # OR VOLUME
        'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
        'FORMAT': 'LINEAR',
        'CLIP_TO_ROI': True,
        'SAVE_ASSET': False,
        'ASSET_ID': 'morten'
    }
    
    image = s1ard.s1_preproc(params).first()
    
    for pol in ['vv', 'vh']:
        out_path = get_gee_path(date_str, aoi_key, method_key, pol)
        download_image(image, pol.upper(), geometry, out_path)

    # Download local incidence angle band from edited terrain flattening
    lia_path = get_gee_path(date_str, aoi_key, method_key, 'lia')
    download_image(image, 'local_incidence_angle', geometry, lia_path)


def process_standard(date_str, aoi_key):
    """Process standard GRD (no terrain correction)."""
    geometry = get_ee_geometry(aoi_key)
    dt = datetime.strptime(date_str, '%Y%m%d')
    start = dt.strftime('%Y-%m-%d')
    end = (dt + timedelta(days=1)).strftime('%Y-%m-%d')
    
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
          .filterBounds(geometry)
          .filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .first())
    
    # Convert dB to linear
    #vv = ee.Image(10).pow(s1.select('VV').divide(10)).rename('VV')
    #vh = ee.Image(10).pow(s1.select('VH').divide(10)).rename('VH')
    vv = s1.select('VV').rename('VV')
    vh = s1.select('VH').rename('VH')
    image = vv.addBands(vh).clip(geometry)
    
    for pol in ['vv', 'vh']:
        out_path = get_gee_path(date_str, aoi_key, 'gee_standard', pol)
        download_image(image, pol.upper(), geometry, out_path)


def main():
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    
    aoi_keys = list(AOI_FILES.keys())
    
    for date_str in COMPARISON_DATES:
        print(f"\nProcessing date: {date_str}")
        for aoi_key in aoi_keys:
            # s1_ard with Copernicus DEM
            if not get_gee_path(date_str, aoi_key, 'gee_s1ard_copernicus', 'vv').exists():
                try:
                    process_s1ard(date_str, aoi_key, 'copernicus')
                    print(f"{date_str}/{aoi_key}/cop: done")
                except Exception as e:
                    print(f"{date_str}/{aoi_key}/cop: {e}")
            
            # s1_ard with Kartverket DEM
            if not get_gee_path(date_str, aoi_key, 'gee_s1ard_kartverket', 'vv').exists():
                try:
                    process_s1ard(date_str, aoi_key, 'kartverket')
                    print(f"{date_str}/{aoi_key}/kart: done")
                except Exception as e:
                    print(f"{date_str}/{aoi_key}/kart: {e}")
            
            # Standard GRD
            if not get_gee_path(date_str, aoi_key, 'gee_standard', 'vv').exists():
                try:
                    process_standard(date_str, aoi_key)
                    print(f"{date_str}/{aoi_key}/std: done")
                except Exception as e:
                    print(f"{date_str}/{aoi_key}/std: {e}")


if __name__ == "__main__":
    main()