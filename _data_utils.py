"""
Fixed data utilities - ensures we load CROPPED PyroSAR files.
Replace your _data_utils.py with this or update the relevant functions.
"""
import numpy as np
import rasterio
from pathlib import Path
from scipy.ndimage import zoom

# Import your config
from _config import (
    METHODS, AOI_FILES,
    MULTITEMP_HYP3_DIR, MULTITEMP_GEE_DIR,
    MULTITEMP_PYROSAR_KART_DIR, MULTITEMP_PYROSAR_COP_DIR,
    load_aoi
)


def find_pyrosar_file(date_str, aoi_key, pol, dem_type='kartverket'):
    """
    Find PyroSAR output file 
    """
    base = MULTITEMP_PYROSAR_KART_DIR if dem_type == 'kartverket' else MULTITEMP_PYROSAR_COP_DIR
    target_dir = base / date_str / aoi_key
    
    # try cropped file (
    cropped_pattern = f"{aoi_key}_V{pol.upper()[1]}_gamma0-rtc_cropped.tif"
    cropped_path = target_dir / cropped_pattern
    
    if cropped_path.exists():
        return cropped_path
    
    # Alternative cropped pattern
    alt_pattern = f"{aoi_key}_{pol.upper()}_gamma0-rtc_cropped.tif"
    alt_path = target_dir / alt_pattern
    
    if alt_path.exists():
        return alt_path
    
    # fallback: search for any cropped file
    cropped_matches = list(target_dir.glob(f"*{pol.upper()}*cropped*.tif"))
    if cropped_matches:
        return cropped_matches[0]
    
    # last resort: original uncropped
    original_matches = list(target_dir.glob(f"*_{pol.upper()}_gamma0-rtc.tif"))
    if original_matches:
        print(f"WARNING: Using uncropped file for {aoi_key}/{pol} - shapes may not match!")
        return original_matches[0]
    
    return None


def find_pyrosar_lia(date_str, aoi_key, dem_type='kartverket'):
    """Find PyroSAR LIA file"""
    base = MULTITEMP_PYROSAR_KART_DIR if dem_type == 'kartverket' else MULTITEMP_PYROSAR_COP_DIR
    target_dir = base / date_str / aoi_key
    
    # try cropped LIA
    cropped_path = target_dir / f"{aoi_key}_localIncidenceAngle_cropped.tif"
    if cropped_path.exists():
        return cropped_path
    
    # fallback to original
    original_matches = list(target_dir.glob("*localIncidenceAngle*.tif"))
    if original_matches:
        # Filter out cropped if present
        uncropped = [p for p in original_matches if 'cropped' not in p.name]
        if uncropped:
            print(f"WARNING: Using uncropped LIA for {aoi_key}")
            return uncropped[0]
    
    return None


def load_raster(path, aoi_key=None, as_db=False):
    """Load raster, optionally convert to dB."""
    if path is None or not Path(path).exists():
        return None, None
    
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
    
    # Mask invalid
    data[data <= 0] = np.nan
    
    if as_db:
        data = 10 * np.log10(data)
    
    return data, transform


def load_all_methods(date_str, aoi_key, pol='vv', as_db=True):
    """
    Load backscatter from all methods
    Returns dict of {method_key: data_array} and reference transform.
    """
    data_dict = {}
    ref_transform = None
    ref_shape = None
    
    # HyP3 (reference)
    hyp3_path = MULTITEMP_HYP3_DIR / date_str / f'{aoi_key}_hyp3_{pol}.tif'
    if hyp3_path.exists():
        data, transform = load_raster(hyp3_path, as_db=as_db)
        if data is not None:
            data_dict['hyp3_gamma'] = data
            ref_transform = transform
            ref_shape = data.shape
    
    # GEE methods
    gee_methods = [
        ('gee_s1ard_kartverket', f'{aoi_key}_s1ard_kartverket_{pol}.tif'),
        ('gee_s1ard_copernicus', f'{aoi_key}_s1ard_copernicus_{pol}.tif'),
        ('gee_standard', f'{aoi_key}_standard_{pol}.tif'),
    ]
    
    for method_key, filename in gee_methods:
        path = MULTITEMP_GEE_DIR / date_str / filename
        if path.exists():
            data, _ = load_raster(path, as_db=as_db)
            if data is not None:
                data_dict[method_key] = data
                if ref_shape is None:
                    ref_shape = data.shape
    
    # PyroSAR methods
    for dem_type, method_key in [('kartverket', 'pyrosar_kartverket'), 
                                  ('copernicus', 'pyrosar_copernicus')]:
        path = find_pyrosar_file(date_str, aoi_key, pol, dem_type)
        if path and path.exists():
            data, _ = load_raster(path, as_db=as_db)
            if data is not None:
                # Resample if shape mismatch
                if ref_shape and data.shape != ref_shape:
                    print(f"  Resampling {method_key}: {data.shape} → {ref_shape}")
                    zoom_factors = (ref_shape[0] / data.shape[0], 
                                   ref_shape[1] / data.shape[1])
                    data = zoom(data, zoom_factors, order=1)
                data_dict[method_key] = data
    
    return data_dict, ref_transform


def verify_file_availability(date_str, aoi_key):
    """Check which files exist and their properties."""
    print(f"\n{'='*60}")
    print(f"File verification: {date_str} / {aoi_key}")
    print(f"{'='*60}")
    
    files = {
        'HyP3': MULTITEMP_HYP3_DIR / date_str / f'{aoi_key}_hyp3_vv.tif',
        'GEE s1_ard Kart': MULTITEMP_GEE_DIR / date_str / f'{aoi_key}_s1ard_kartverket_vv.tif',
        'GEE s1_ard Cop': MULTITEMP_GEE_DIR / date_str / f'{aoi_key}_s1ard_copernicus_vv.tif',
        'PyroSAR Kart (cropped)': find_pyrosar_file(date_str, aoi_key, 'vv', 'kartverket'),
        'PyroSAR Cop (cropped)': find_pyrosar_file(date_str, aoi_key, 'vv', 'copernicus'),
    }
    
    shapes = {}
    for name, path in files.items():
        if path and Path(path).exists():
            with rasterio.open(path) as src:
                shapes[name] = src.shape
                origin = (src.transform.c, src.transform.f)
                print(f"✓ {name:25s}: {src.shape}, origin=({origin[0]:.0f}, {origin[1]:.0f})")
        else:
            print(f"✗ {name:25s}: NOT FOUND")
    
    # Check shape consistency
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        print(f"\nWARNING: Shape mismatch detected!")
        print(f"   Unique shapes: {unique_shapes}")
    else:
        print(f"\n✓ All files have consistent shape: {list(unique_shapes)[0] if unique_shapes else 'N/A'}")
    
    return files, shapes


if __name__ == "__main__":
    # Test
    from _config import COMPARISON_DATES, AOI_FILES
    
    for aoi_key in AOI_FILES.keys():
        verify_file_availability('20170526', aoi_key)