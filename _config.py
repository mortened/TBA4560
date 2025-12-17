from pathlib import Path
import geopandas as gpd
import os
from dotenv import load_dotenv
load_dotenv()

GEE_PROJECT = os.getenv('GEE_PROJECT')
OUTPUT_CRS = 'EPSG:32632'  # UTM 32N
OUTPUT_RESOLUTION = 10

BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures'

AOI_DIR = DATA_DIR / 'aoi'
DEM_DIR = DATA_DIR / 'dem'
ALL_RAW_DIR = DATA_DIR / 'all_dates_raw'

MULTITEMP_DIR = DATA_DIR / 'multitemporal'
MULTITEMP_HYP3_DIR = MULTITEMP_DIR / 'hyp3'
MULTITEMP_GEE_DIR = MULTITEMP_DIR / 'gee'
MULTITEMP_PYROSAR_KART_DIR = MULTITEMP_DIR / 'pyrosar_kartverket'
MULTITEMP_PYROSAR_COP_DIR = MULTITEMP_DIR / 'pyrosar_copernicus'

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, AOI_DIR, DEM_DIR,
          MULTITEMP_DIR, MULTITEMP_HYP3_DIR, MULTITEMP_GEE_DIR,
          MULTITEMP_PYROSAR_KART_DIR, MULTITEMP_PYROSAR_COP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VEG_CORRECTION_COEFS = {
    'A': -10,
    'B': 4,
    'C': 7
}

AOI_FILES = {
    'jorde': AOI_DIR / 'aoi_jorde.geojson',
    'skog_flatt': AOI_DIR / 'aoi_skog_flatt.geojson',
    'skog_bratt': AOI_DIR / 'aoi_skog_bratt.geojson',
}

DEM_FILES = {
    'jorde': DEM_DIR / 'kartverket_dem_aas.tif',
    'skog_flatt': DEM_DIR / 'kartverket_dem_aarnes.tif',
    'skog_bratt': DEM_DIR / 'kartverket_dem_aarnes.tif',
}

GEE_DEM_ASSETS = {
    'jorde': f'projects/{GEE_PROJECT}/assets/kartverket_dem_aas',
    'skog_flatt': f'projects/{GEE_PROJECT}/assets/kartverket_dem_arnes',
    'skog_bratt': f'projects/{GEE_PROJECT}/assets/kartverket_dem_arnes',
}

COMPARISON_DATES = [
    '20170508', '20170515', '20170526', #'20170531',
    '20170607', '20170613', '20170614', '20170620', '20170624', '20170630', 
    '20170707', '20170714', '20170720',
    '20170801', '20170807', '20170823', '20170831',
    '20170924', '20170928',
    '20171010', '20171016'
]

PRIMARY_DATE = '20170613'

DATE_METADATA = {
    '20170508':'Descending',
    '20170515':'Descending',
    '20170526':'Descending',
    
    '20170607':'Descending',
    '20170613':'Ascending',
    '20170614':'Descending', 
    '20170620':'Descending',
    '20170624':'Ascending',
    '20170630':'Ascending',
    
    '20170707':'Ascending',
    '20170714':'Descending',
    '20170720':'Descending',
    
    '20170801':'Descending',
    '20170807':'Descending',
    '20170823':'Ascending',
    '20170831':'Descending',
    
    '20170924':'Descending',
    '20170928':'Ascending',
    
    '20171010':'Ascending',
    '20171016':'Ascending'
}

METHODS = {
    'hyp3_gamma': {
        'name': 'HyP3 GAMMA (Cop. 30m)',
        'dem': 'copernicus_30m',
        'is_reference': True,
        'color': 'black',
        'marker': 'o'
    },
    'pyrosar_kartverket': {
        'name': 'PyroSAR/SNAP (Kart. 10m)',
        'dem': 'kartverket_10m',
        'is_reference': False,
        'color': 'purple',
        'marker': 's'
    },
    'pyrosar_copernicus': {
        'name': 'PyroSAR/SNAP (Cop. 30m)',
        'dem': 'copernicus_30m',
        'is_reference': False,
        'color': 'magenta',
        'marker': '^'
    },
    'gee_s1ard_kartverket': {
        'name': 'GEE s1_ard (Kart. 10m)',
        'dem': 'kartverket_10m',
        'is_reference': False,
        'color': 'blue',
        'marker': 'D'
    },
    'gee_s1ard_copernicus': {
        'name': 'GEE s1_ard (Cop. 30m)',
        'dem': 'copernicus_30m',
        'is_reference': False,
        'color': 'green',
        'marker': 'v'
    },
    'gee_standard': {
        'name': 'GEE Standard GRD',
        'dem': None,
        'is_reference': False,
        'color': 'red',
        'marker': 'x'
    }
}

def load_aoi(aoi_key):
    """Load AOI geometry from geojson file."""
    path = AOI_FILES[aoi_key]
    if not path.exists():
        raise FileNotFoundError(f"AOI file not found: {path}")
    return gpd.read_file(path)

def get_aoi_bounds(aoi_key, crs=None):
    """Get AOI bounds, optionally reprojected."""
    gdf = load_aoi(aoi_key)
    if crs:
        gdf = gdf.to_crs(crs)
    return gdf.total_bounds

def get_aoi_geometry(aoi_key, crs=None):
    """Get AOI geometry, optionally reprojected."""
    gdf = load_aoi(aoi_key)
    if crs:
        gdf = gdf.to_crs(crs)
    return gdf.geometry.iloc[0]

def get_safe_path(date_str):
    """Find SAFE file for a given date."""
    pattern = f"S1*_{date_str}T*.SAFE"
    matches = list(ALL_RAW_DIR.glob(pattern))
    return matches[0] if matches else None

def get_hyp3_path(date_str, aoi_key, pol):
    """Path to HyP3 output."""
    return MULTITEMP_HYP3_DIR / date_str / f'{aoi_key}_hyp3_{pol}.tif'

def get_gee_path(date_str, aoi_key, method_key, pol):
    """Path to GEE output."""
    suffix = {
        'gee_s1ard_kartverket': 's1ard_kartverket',
        'gee_s1ard_copernicus': 's1ard_copernicus',
        'gee_standard': 'standard'
    }.get(method_key, method_key)
    return MULTITEMP_GEE_DIR / date_str / f'{aoi_key}_{suffix}_{pol}.tif'

def get_pyrosar_path(date_str, aoi_key, pol, dem_type='kartverket'):
    """Path to PyroSAR cropped output."""
    base = MULTITEMP_PYROSAR_KART_DIR if dem_type == 'kartverket' else MULTITEMP_PYROSAR_COP_DIR
    return base / date_str / aoi_key / f'{aoi_key}_{pol.upper()}_gamma0-rtc_cropped.tif'

def find_pyrosar_file(date_str, aoi_key, pol, dem_type='kartverket'):
    """Find PyroSAR output file (cropped)."""
    path = get_pyrosar_path(date_str, aoi_key, pol, dem_type)
    return path if path.exists() else None

def print_status():
    """Print processing status for all dates and methods."""
    print("=" * 70)
    print("PROCESSING STATUS")
    print("=" * 70)
    
    for date_str in COMPARISON_DATES:
        print(f"\n{date_str}:")
        safe = get_safe_path(date_str)
        print(f"  SAFE: {'OK' if safe else 'MISSING'}")
        
        for aoi_key in AOI_FILES.keys():
            status = []
            if get_hyp3_path(date_str, aoi_key, 'vv').exists():
                status.append('hyp3')
            if get_gee_path(date_str, aoi_key, 'gee_s1ard_copernicus', 'vv').exists():
                status.append('gee_cop')
            if get_gee_path(date_str, aoi_key, 'gee_s1ard_kartverket', 'vv').exists():
                status.append('gee_kart')
            if get_gee_path(date_str, aoi_key, 'gee_standard', 'vv').exists():
                status.append('grd')
            if find_pyrosar_file(date_str, aoi_key, 'vv', 'kartverket'):
                status.append('pyr_kart')
            if find_pyrosar_file(date_str, aoi_key, 'vv', 'copernicus'):
                status.append('pyr_cop')
            print(f"  {aoi_key}: {', '.join(status) if status else '-'}")