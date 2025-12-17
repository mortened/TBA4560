"""
HyP3 RTC processing script.
"""
import os
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np

load_dotenv()

from hyp3_sdk import HyP3


try:
    import asf_search as asf
except ImportError:
    asf = None

from _config import (
    COMPARISON_DATES, AOI_FILES, MULTITEMP_HYP3_DIR,
    get_safe_path, get_hyp3_path, load_aoi
)


def get_credentials():
    """Get HyP3 credentials from environment."""
    username = os.getenv('HYP3USERNAME')
    password = os.getenv('HYP3PASSWORD')
    if not username or not password:
        raise ValueError("Set HYP3USERNAME and HYP3PASSWORD in .env")
    return username, password


def search_asf_scene(date_str, aoi_key='jorde'):
    """Search ASF for Sentinel-1 scene on given date."""
    if asf is None:
        raise ImportError("asf_search required")
    
    # Load AOI in WGS84 for ASF search
    aoi_gdf = load_aoi(aoi_key).to_crs('EPSG:4326')
    
    # Date range: +/- 1 day
    date_dt = datetime.strptime(date_str, '%Y%m%d')
    start = (date_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    end = (date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    
    results = asf.search(
        platform=[asf.PLATFORM.SENTINEL1],
        intersectsWith=str(aoi_gdf.geometry.iloc[0]),
        start=start,
        end=end,
        processingLevel='GRD_HD',
        beamMode='IW',
    )
    
    if not results:
        return None
    
    return results[0].properties['sceneName']


def submit_jobs(dates=None, overwrite=False):
    """Submit HyP3 RTC jobs."""
    
    username, password = get_credentials()
    hyp3 = HyP3(username=username, password=password)
    
    dates = dates or COMPARISON_DATES
    existing_jobs = {job.name: job for job in hyp3.find_jobs()}
    
    for date_str in dates:
        if date_str == '20170707':
            overwrite = True
        job_name = f"Thesis_RTC_{date_str}"
        
        if job_name in existing_jobs and not overwrite:
            print(f"{date_str}: exists")
            continue
        
        # First try to get scene name from local SAFE file
        safe = get_safe_path(date_str)
        if safe:
            scene_name = safe.stem
        else:
            # If no local SAFE, search ASF
            try:
                scene_name = search_asf_scene(date_str)
                if not scene_name:
                    print(f"{date_str}: no scene found in ASF")
                    continue
            except Exception as e:
                print(f"{date_str}: ASF search failed - {e}")
                continue
        
        try:
            hyp3.submit_rtc_job(
                granule=scene_name,
                name=job_name,
                resolution=10,
                scale='power',
                speckle_filter=False,
                dem_name='copernicus',
                dem_matching=False,
                radiometry='gamma0',
                include_dem=True,
                include_inc_map=True,
                include_scattering_area=True,
            )
            print(f"{date_str}: submitted ({scene_name})")
        except Exception as e:
            print(f"{date_str}: {e}")


def download_jobs(dates=None):
    """Download completed HyP3 jobs."""
    if HyP3 is None:
        raise ImportError("hyp3_sdk required")
    
    username, password = get_credentials()
    hyp3 = HyP3(username=username, password=password)
    
    dates = dates or COMPARISON_DATES
    jobs = {job.name: job for job in hyp3.find_jobs()}
    
    for date_str in dates:

        job_name = f"Thesis_RTC_{date_str}"
        
        if job_name not in jobs:
            print(f"{date_str}: no job")
            continue
        
        job = jobs[job_name]
        if job.status_code != 'SUCCEEDED':
            print(f"{date_str}: {job.status_code}")
            continue
        
        output_dir = MULTITEMP_HYP3_DIR / date_str
        if list(output_dir.glob('*/*.tif')) if output_dir.exists() else []:
            print(f"{date_str}: exists")
            continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        job.download_files(str(output_dir))
        
        # Extract zips
        for zf in output_dir.glob('*.zip'):
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(output_dir)
            zf.unlink()
        
        print(f"{date_str}: downloaded")


def crop_to_aois(dates=None, overwrite=False):
    """Crop downloaded HyP3 data to AOIs, reprojecting to OUTPUT_CRS."""
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import reproject, Resampling
    from _config import OUTPUT_CRS
    
    dates = dates or COMPARISON_DATES
    
    for date_str in dates:
        date_dir = MULTITEMP_HYP3_DIR / date_str
        if not date_dir.exists():
            continue
        
        scene_dirs = [d for d in date_dir.iterdir() if d.is_dir() and d.name.startswith('S1')]
        if not scene_dirs:
            continue
        
        scene_dir = scene_dirs[0]
        vv_files = list(scene_dir.glob('*_VV.tif'))
        if not vv_files:
            continue
        
        vv_input = vv_files[0]
        scene_name = vv_input.stem.replace('_VV', '')
        
        input_files = {
            'vv': vv_input,
            'vh': scene_dir / f'{scene_name}_VH.tif',
            'inc': scene_dir / f'{scene_name}_inc_map.tif',
            'ls': scene_dir / f'{scene_name}_ls_map.tif',
            'dem': scene_dir / f'{scene_name}_dem.tif',
        }
        
        for aoi_key in AOI_FILES.keys():
            gdf = load_aoi(aoi_key)
            
            for file_type, input_path in input_files.items():
                if not input_path.exists():
                    continue
                
                output_path = get_hyp3_path(date_str, aoi_key, file_type)
                
                if output_path.exists() and not overwrite:
                    continue
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    with rasterio.open(input_path) as src:
                        needs_reproject = str(src.crs) != OUTPUT_CRS
                        
                        if needs_reproject:
                            print(f"  {date_str}/{aoi_key}/{file_type}: reprojecting {src.crs} -> {OUTPUT_CRS}")
                        
                        gdf_src = gdf.to_crs(src.crs)
                        data, transform = mask(src, gdf_src.geometry, crop=True, 
                                              filled=True, nodata=0)
                        
                        if needs_reproject:
                            gdf_dst = gdf.to_crs(OUTPUT_CRS)
                            dst_bounds = gdf_dst.total_bounds
                            
                            dst_width = int(np.ceil((dst_bounds[2] - dst_bounds[0]) / 10))
                            dst_height = int(np.ceil((dst_bounds[3] - dst_bounds[1]) / 10))
                            dst_transform = rasterio.transform.from_bounds(
                                dst_bounds[0], dst_bounds[1], 
                                dst_bounds[2], dst_bounds[3],
                                dst_width, dst_height
                            )
                            
                            dst_data = np.zeros((data.shape[0], dst_height, dst_width), dtype=data.dtype)
                            reproject(
                                source=data,
                                destination=dst_data,
                                src_transform=transform,
                                src_crs=src.crs,
                                dst_transform=dst_transform,
                                dst_crs=OUTPUT_CRS,
                                resampling=Resampling.bilinear
                            )
                            data = dst_data
                            transform = dst_transform
                            crs = OUTPUT_CRS
                        else:
                            crs = src.crs
                        
                        profile = src.profile.copy()
                        profile.update(
                            height=data.shape[1],
                            width=data.shape[2],
                            transform=transform,
                            crs=crs,
                            nodata=0
                        )
                        
                        with rasterio.open(output_path, 'w', **profile) as dst:
                            dst.write(data)
                            
                except Exception as e:
                    print(f"Error cropping {file_type} for {date_str}: {e}")
        
        print(f"{date_str}: cropped")

def check_status():
    """Print status of HyP3 processing."""
    print("HyP3 File Availability:")
    for date_str in COMPARISON_DATES:
        hyp3_exists = any(get_hyp3_path(date_str, aoi, 'vv').exists() 
                         for aoi in AOI_FILES.keys())
        status = 'OK' if hyp3_exists else 'MISSING'
        print(f"  {date_str}: {status}")

    hyp3 = HyP3(username=os.getenv('HYP3USERNAME'), password=os.getenv('HYP3PASSWORD'))
    print("\nRecent HyP3 Jobs:")
    jobs = hyp3.find_jobs()
    for job in jobs[:20]:
        print(f"  {job.name:<30} | {job.status_code:<10} | {job.job_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    
    if args.submit:
        submit_jobs()
    elif args.download:
        hyp3 = HyP3(username=os.getenv('HYP3USERNAME'), password=os.getenv('HYP3PASSWORD'))
        download_jobs()
    elif args.crop:
        crop_to_aois()
    else:
        check_status()