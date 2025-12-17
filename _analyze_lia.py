"""LIA-analyse for RTC-kvalitet. God RTC: slope ≈ 0"""
import numpy as np
import pandas as pd
from scipy import stats, ndimage
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import rasterio

from _config import (
    METHODS, AOI_FILES, PRIMARY_DATE, RESULTS_DIR, FIGURES_DIR,
    MULTITEMP_HYP3_DIR, MULTITEMP_PYROSAR_KART_DIR, MULTITEMP_PYROSAR_COP_DIR, MULTITEMP_GEE_DIR
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
AOI_NO = {'jorde': 'Jordbruk', 'skog_flatt': 'Skog (flatt)', 'skog_bratt': 'Skog (bratt)'}


def load_raster(path):
    if not path or not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def get_bs_path(date, aoi, method, pol='vv'):
    if method == 'hyp3_gamma':
        return MULTITEMP_HYP3_DIR / date / f'{aoi}_hyp3_{pol}.tif'
    elif method == 'gee_standard':
        return MULTITEMP_GEE_DIR / date / f'{aoi}_standard_{pol}.tif'
    elif method == 'gee_s1ard_copernicus':
        return MULTITEMP_GEE_DIR / date / f'{aoi}_s1ard_copernicus_{pol}.tif'
    elif method == 'gee_s1ard_kartverket':
        return MULTITEMP_GEE_DIR / date / f'{aoi}_s1ard_kartverket_{pol}.tif'
    elif method == 'pyrosar_kartverket':
        matches = list((MULTITEMP_PYROSAR_KART_DIR / date / aoi).glob(f'*{pol.upper()}*cropped*.tif'))
        return matches[0] if matches else None
    elif method == 'pyrosar_copernicus':
        matches = list((MULTITEMP_PYROSAR_COP_DIR / date / aoi).glob(f'*{pol.upper()}*cropped*.tif'))
        return matches[0] if matches else None
    return None


def get_lia_path(date, aoi, method):
    if method == 'hyp3_gamma':
        return MULTITEMP_HYP3_DIR / date / f'{aoi}_hyp3_inc.tif', True
    elif method == 'pyrosar_kartverket':
        return MULTITEMP_PYROSAR_KART_DIR / date / aoi / f'{aoi}_localIncidenceAngle_cropped.tif', False
        #return MULTITEMP_HYP3_DIR / date / f'{aoi}_hyp3_inc.tif', True
    elif method == 'pyrosar_copernicus':
        return MULTITEMP_PYROSAR_COP_DIR / date / aoi / f'{aoi}_localIncidenceAngle_cropped.tif', False
        #return MULTITEMP_HYP3_DIR / date / f'{aoi}_hyp3_inc.tif', True
    elif method == 'gee_s1ard_kartverket':
        return MULTITEMP_GEE_DIR / date / f'{aoi}_s1ard_kartverket_lia.tif', False
        #return MULTITEMP_HYP3_DIR / date / f'{aoi}_hyp3_inc.tif', True
    elif method == 'gee_s1ard_copernicus':
        return MULTITEMP_GEE_DIR / date / f'{aoi}_s1ard_copernicus_lia.tif', False;
        #return MULTITEMP_HYP3_DIR / date / f'{aoi}_hyp3_inc.tif', True
    elif method == 'gee_standard':
        return MULTITEMP_HYP3_DIR / date / f'{aoi}_hyp3_inc.tif', True
    return None, False


def find_offset(ref, test, max_shift=3):
    ref, test = ref.copy(), test.copy()
    ref[~np.isfinite(ref)] = 0
    test[~np.isfinite(test)] = 0
    if ref.std() < 1e-6 or test.std() < 1e-6:
        return (0, 0)
    ref = (ref - ref.mean()) / ref.std()
    test = (test - test.mean()) / test.std()
    
    best = (-999, (0, 0))
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = ndimage.shift(test, (dy, dx), order=0, mode='constant')
            valid = (ref != 0) & (shifted != 0)
            if valid.sum() >= 50:
                corr = np.corrcoef(ref[valid], shifted[valid])[0, 1]
                if corr > best[0]:
                    best = (corr, (dy, dx))
    return best[1]


def calc_slope(bs, lia):
    if bs is None or lia is None:
        return {'slope': np.nan, 'r2': np.nan, 'n': 0}
    
    bs_f, lia_f = bs.flatten(), lia.flatten()
    valid = (np.isfinite(bs_f) & np.isfinite(lia_f) & 
             (lia_f >= 15) & (lia_f <= 60) & (bs_f > -40) & (bs_f < 5))
    
    if valid.sum() < 20:
        return {'slope': np.nan, 'r2': np.nan, 'n': valid.sum()}
    
    slope, intercept, r, _, _ = stats.linregress(lia_f[valid], bs_f[valid])
    return {'slope': slope, 'intercept': intercept, 'r2': r**2, 'n': valid.sum(),
            '_lia': lia_f[valid], '_bs': bs_f[valid]}


def run_analysis(date, aoi, pol='vv'):
    hyp3_bs = load_raster(get_bs_path(date, aoi, 'hyp3_gamma', pol))
    if hyp3_bs is None:
        return None, None, None, None
    
    hyp3_bs[hyp3_bs <= 0] = np.nan
    hyp3_db = 10 * np.log10(hyp3_bs)
    ref_shape = hyp3_db.shape
    
    results, data_dict, lia_cache, reg_cache = [], {}, {}, {}
    
    for method in METHODS:
        bs = load_raster(get_bs_path(date, aoi, method, pol))
        if bs is None:
            continue
        bs[bs <= 0] = np.nan
        bs_db = 10 * np.log10(bs)
        
        lia_path, is_rad = get_lia_path(date, aoi, method)
        if not lia_path or not lia_path.exists():
            continue
        lia = load_raster(lia_path)
        if lia is None:
            continue
        if is_rad:
            lia = np.rad2deg(lia)
        
        # Align
        if bs_db.shape != ref_shape:
            bs_db = zoom(bs_db, (ref_shape[0]/bs_db.shape[0], ref_shape[1]/bs_db.shape[1]), order=1)
        if lia.shape != ref_shape:
            lia = zoom(lia, (ref_shape[0]/lia.shape[0], ref_shape[1]/lia.shape[1]), order=1)
        
        # Offset-korreksjon
        # offset = (0, 0)
        # if method != 'hyp3_gamma':
        #     offset = find_offset(hyp3_db, bs_db)
        #     if offset != (0, 0):
        #         print(f"  {method} / {aoi}: offset={offset}")
        #         bs_db = ndimage.shift(bs_db, offset, order=0, cval=np.nan)
                #lia = ndimage.shift(lia, offset, order=0, cval=np.nan)
        
        data_dict[method] = bs_db
        lia_cache[method] = lia
        reg = calc_slope(bs_db, lia)
        reg_cache[method] = reg
        
        results.append({'date': date, 'aoi': aoi, 'pol': pol, 'method': method,
                       'slope': reg['slope'], 'r2': reg['r2'], 'n': reg['n']})
    
    return pd.DataFrame(results) if results else None, data_dict, lia_cache, reg_cache


def plot_scatter(data_dict, lia_cache, reg_cache, aoi, pol, date):
    methods = [m for m in data_dict if lia_cache.get(m) is not None and reg_cache[m]['n'] >= 20]
    if not methods:
        return
    
    ncols = min(3, len(methods))
    nrows = (len(methods) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        reg = reg_cache[method]
        lia_v, bs_v = reg['_lia'], reg['_bs']
        
        if len(bs_v) > 2000:
            idx_s = np.random.choice(len(bs_v), 2000, replace=False)
            lia_v, bs_v = lia_v[idx_s], bs_v[idx_s]
        
        ax.hexbin(lia_v, bs_v, gridsize=30, cmap='viridis', mincnt=1)
        x = np.array([15, 60])
        color = 'green' if abs(reg['slope']) < 0.05 else ('orange' if abs(reg['slope']) < 0.1 else 'red')
        ax.plot(x, reg['slope']*x + reg['intercept'], color=color, lw=2,
               label=f"helning={reg['slope']:.3f}\nr²={reg['r2']:.3f}")
        ax.set_xlabel('LIA [°]')
        ax.set_ylabel('σ° [dB]')
        ax.set_title(METHODS[method]['name'])
        ax.legend(fontsize=8)
        ax.set_xlim(15, 60)
        ax.grid(alpha=0.3)
    
    for idx in range(len(methods), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'{AOI_NO[aoi]} - {pol.upper()} - {date}')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f'lia_scatter_{aoi}_{pol}_{date}.png', dpi=150)
    plt.close()


def main():
    print("LIA-ANALYSE\n")
    all_results = []
    
    for aoi in AOI_FILES:
        for pol in ['vv', 'vh']:
            print(f"{aoi}/{pol}...", end=" ")
            res = run_analysis(PRIMARY_DATE, aoi, pol)
            if res[0] is not None:
                all_results.append(res[0])
                plot_scatter(res[1], res[2], res[3], aoi, pol, PRIMARY_DATE)
                print("✓")
            else:
                print("✗")
    
    if not all_results:
        return None
    
    df = pd.concat(all_results, ignore_index=True)
    df.to_csv(RESULTS_DIR / f'lia_{PRIMARY_DATE}.csv', index=False)
    
    print("\n=== Resultater (|slope| i dB/°) ===")
    print(df.pivot_table(values='slope', index='method', columns=['aoi', 'pol'],
                        aggfunc=lambda x: x.abs().mean()).round(3))
    
    return df

if __name__ == "__main__":
    main()