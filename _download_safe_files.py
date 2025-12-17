"""
Download Sentinel-1 SAFE files from ASF for specified dates.
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

try:
    import asf_search as asf
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install asf_search")
    exit(1)

from _config import (
    COMPARISON_DATES, ALL_RAW_DIR, load_aoi, get_safe_path
)

# config
HYP3_USERNAME = os.getenv('HYP3USERNAME')
HYP3_PASSWORD = os.getenv('HYP3PASSWORD')

DATES_TO_DOWNLOAD = COMPARISON_DATES
OVERWRITE = False


def download_safe_files(dates_to_download, overwrite=False):
    """Download Sentinel-1 SAFE files for specified dates."""
    
    print("Authenticating with ASF...")
    session = asf.ASFSession().auth_with_creds(HYP3_USERNAME, HYP3_PASSWORD)
    
    # Use jorde AOI for search (covers the area)
    aoi_gdf = load_aoi('jorde').to_crs('EPSG:4326')
    
    print("=" * 70)
    print("DOWNLOADING SAFE FILES")
    print("=" * 70)
    print(f"Dates: {dates_to_download}")
    print(f"Output: {ALL_RAW_DIR}")
    
    ALL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded, skipped, failed = [], [], []
    
    for date_str in dates_to_download:
        print(f"\n{'─' * 50}")
        print(f"DATE: {date_str}")
        print(f"{'─' * 50}")
        
        # Check if already exists
        existing = get_safe_path(date_str)
        if existing and existing.exists() and not overwrite:
            print(f"  ✓ Already exists: {existing.name}")
            skipped.append(date_str)
            continue
        
        # Search for scene
        date_dt = datetime.strptime(date_str, '%Y%m%d')
        start = (date_dt - timedelta(days=1)).strftime('%Y-%m-%d')
        end = (date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"  Searching ASF for {start} to {end}...")
        
        results = asf.search(
            platform=[asf.PLATFORM.SENTINEL1],
            intersectsWith=str(aoi_gdf.geometry.iloc[0]),
            start=start,
            end=end,
            processingLevel='GRD_HD',
            beamMode='IW',
        )
        
        if not results:
            print(f"  ✗ No scene found")
            failed.append(date_str)
            continue
        
        granule = results[0]
        scene_name = granule.properties['sceneName']
        print(f"  Found: {scene_name}")
        print(f"  Downloading (~1 GB)...")
        
        try:
            granule.download(path=str(ALL_RAW_DIR), session=session)
            
            zip_path = ALL_RAW_DIR / f"{scene_name}.zip"
            if zip_path.exists():
                print(f"  Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(ALL_RAW_DIR)
                zip_path.unlink()
                print(f"  ✓ Done: {scene_name}.SAFE")
                downloaded.append(date_str)
            else:
                print(f"  ✗ Zip file not found")
                failed.append(date_str)
                
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            failed.append(date_str)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Downloaded: {len(downloaded)} - {downloaded}")
    print(f"  Skipped:    {len(skipped)} - {skipped}")
    print(f"  Failed:     {len(failed)} - {failed}")
    
    return downloaded, skipped, failed


def verify_downloads():
    """Check which dates have SAFE files available."""
    print("\n" + "=" * 70)
    print("AVAILABLE SAFE FILES")
    print("=" * 70)
    
    for date_str in COMPARISON_DATES:
        safe_path = get_safe_path(date_str)
        if safe_path and safe_path.exists():
            size_gb = sum(f.stat().st_size for f in safe_path.rglob('*')) / (1024**3)
            print(f"  ✓ {date_str}: {safe_path.name} ({size_gb:.2f} GB)")
        else:
            print(f"  ✗ {date_str}: Not found")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOAD SAFE FILES FROM ASF")
    print("=" * 70)
    
    verify_downloads()
    download_safe_files(DATES_TO_DOWNLOAD, overwrite=OVERWRITE)
    verify_downloads()