"""
Standalone script for cropping PyroSAR geocoded output files (.tif)
to the exact AOI geometry defined in _config.py.
Handles VV, VH, incidence angles, and masks similar to the HyP3 workflow.
"""
import os
from pathlib import Path
import rasterio
from rasterio.mask import mask

try:
    from _config import (
        COMPARISON_DATES, AOI_FILES,
        MULTITEMP_PYROSAR_KART_DIR, MULTITEMP_PYROSAR_COP_DIR,
        load_aoi
    )
except ImportError:
    print("ERROR: Could not import _config.py.")
    exit()

def get_pyrosar_filename_map(output_dir: Path):
    """
    Scans the directory for PyroSAR outputs and maps them to standard keys.
    Returns a dict like {'vv': Path(...), 'vh': Path(...), 'inc': Path(...), ...}
    """
    if not output_dir.exists():
        return {}

    file_map = {}
    
    # Define patterns to search for based on PyroSAR naming conventions
    # Note: Adjust the glob patterns if your file naming varies slightly
    patterns = {
        'vv': '*_VV_gamma0-rtc.tif',
        'vh': '*_VH_gamma0-rtc.tif',
        'inc': '*_incidenceAngleFromEllipsoid.tif',
        'ls': '*_layoverShadowMask.tif',
        'local_inc': '*_localIncidenceAngle.tif'
    }

    for key, pattern in patterns.items():
        matches = list(output_dir.glob(pattern))
        if matches:
            # Take the first match (there should usually be only one per folder/date)
            file_map[key] = matches[0]

    return file_map

def crop_raster_to_aoi(input_path: Path, output_path: Path, aoi_key: str):
    """
    Crops a single raster file to the AOI geometry.
    """
    if output_path.exists():
        return  # Skip if already cropped

    try:
        # Load AOI
        gdf = load_aoi(aoi_key)

        with rasterio.open(input_path) as src:
            # Reproject AOI to match the raster's CRS
            gdf_proj = gdf.to_crs(src.crs)
            
            # Perform the crop
            out_image, out_transform = mask(src, gdf_proj.geometry, crop=True, filled=True, nodata=0)
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": 0,
                "compress": "lzw"
            })

            # Save cropped file
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
                
        print(f"    [Cropped] {output_path.name}")

    except Exception as e:
        print(f"    [Error] Failed to crop {input_path.name}: {e}")
        if output_path.exists():
            os.remove(output_path)

def main():
    processing_targets = [
        (MULTITEMP_PYROSAR_KART_DIR, "Kartverket"),
        (MULTITEMP_PYROSAR_COP_DIR, "Copernicus")
    ]

    for date in COMPARISON_DATES:
        for aoi_key in AOI_FILES.keys():
            for base_dir, label in processing_targets:
                
                # The folder where PyroSAR put the original results
                target_dir = base_dir / date / aoi_key
                
                # Find all available files in that folder
                file_map = get_pyrosar_filename_map(target_dir)

                if not file_map:
                    # Silent skip or minimal log if folder is empty/missing
                    continue

                print(f"--- Processing {label}: {date} / {aoi_key} ---")

                # Iterate through found files and crop them
                for file_type, input_path in file_map.items():
                    
                    # Define output name. 
                    # Naming convention: {aoi}_{type}_cropped.tif
                    # You can adjust suffix as needed.
                    suffix_map = {
                        'vv': 'VV_gamma0-rtc',
                        'vh': 'VH_gamma0-rtc',
                        'inc': 'incidenceAngleFromEllipsoid',
                        'ls': 'layoverShadowMask',
                        'local_inc': 'localIncidenceAngle'
                    }
                    
                    suffix = suffix_map.get(file_type, file_type)
                    output_filename = f"{aoi_key}_{suffix}_cropped.tif"
                    output_path = target_dir / output_filename

                    # Don't crop if the input file IS the cropped file (safety check)
                    if "cropped" in input_path.name:
                        continue

                    crop_raster_to_aoi(input_path, output_path, aoi_key)

if __name__ == "__main__":
    main()