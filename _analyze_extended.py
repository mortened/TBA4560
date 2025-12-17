"""Extended RTC metrics: CV, RMSE - stratified by AOI (not slope within AOI)."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

from _config import (
    METHODS, AOI_FILES, PRIMARY_DATE, RESULTS_DIR, FIGURES_DIR
)
from _data_utils import load_all_methods

EXPECTED_DB = {'vv': -6.5, 'vh': -12.5}


def calc_cv(data_linear):
    d = data_linear[np.isfinite(data_linear) & (data_linear > 0)]
    if len(d) == 0:
        return np.nan
    return (np.std(d) / np.mean(d)) * 100


def calc_rmse(test, ref):
    valid = np.isfinite(test) & np.isfinite(ref)
    if valid.sum() < 10:
        return np.nan
    return np.sqrt(np.mean((test[valid] - ref[valid])**2))


def calc_r2(test, ref):
    valid = np.isfinite(test) & np.isfinite(ref)
    if valid.sum() < 10:
        return np.nan
    r, _ = stats.pearsonr(test[valid], ref[valid])
    return r**2


def run_extended(date_str, aoi_key, pol='vv'):
    data_linear, _ = load_all_methods(date_str, aoi_key, pol, as_db=False)
    data_dict, _ = load_all_methods(date_str, aoi_key, pol, as_db=True)
    if not data_dict:
        return None
    
    ref_data = data_dict.get('hyp3_gamma')
    if ref_data is None:
        ref_shape = list(data_dict.values())[0].shape
    else:
        ref_shape = ref_data.shape
    
    for k, v in data_dict.items():
        if v.shape != ref_shape:
            zf = (ref_shape[0] / v.shape[0], ref_shape[1] / v.shape[1])
            data_dict[k] = zoom(v, zf, order=1)
    
    results = []
    for method, bs in data_dict.items():
        bs_linear = data_linear.get(method)
        mean_db = np.nanmean(bs[(bs > -50) & (bs < 10)])
        
        row = {
            'date': date_str,
            'aoi': aoi_key,
            'pol': pol,
            'method': method,
            'mean_db': mean_db,
            'bias': mean_db - EXPECTED_DB[pol],
            'cv': calc_cv(bs_linear) if bs_linear is not None else np.nan
        }
        
        if ref_data is not None and method != 'hyp3_gamma':
            row['rmse_vs_ref'] = calc_rmse(bs, ref_data)
            row['r2_vs_ref'] = calc_r2(bs.flatten(), ref_data.flatten())
        
        results.append(row)
    
    return pd.DataFrame(results)


def plot_cv_by_aoi(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    aoi_colors = {
        'jorde': 'gold',
        'skog_flatt': 'forestgreen', 
        'skog_bratt': 'saddlebrown'
    }
    aoi_labels = {
        'jorde': 'Jorde',
        'skog_flatt': 'Skog (flatt)',
        'skog_bratt': 'Skog (bratt)'
    }
    
    for idx, pol in enumerate(['vv', 'vh']):
        ax = axes[idx]
        data = df[df['pol'] == pol]
        methods = data['method'].unique()
        x = np.arange(len(methods))
        n_aois = len(aoi_colors)
        w = 0.8 / n_aois
        
        for i, (aoi, color) in enumerate(aoi_colors.items()):
            vals = data[data['aoi'] == aoi].groupby('method')['cv'].mean()
            ax.bar(x + i*w - 0.4 + w/2, [vals.get(m, 0) for m in methods], 
                   w, label=aoi_labels[aoi], color=color, alpha=0.8)
        
        ax.axhline(100, color='red', ls='--', alpha=0.5, label='CV=100%')
        ax.set_ylabel('CV (%)')
        ax.set_title(pol.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([METHODS[m]['name'] for m in methods], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 50)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'cv_by_aoi.png', dpi=150)
    plt.close()


def plot_rmse_by_aoi(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    aoi_colors = {
        'jorde': 'gold',
        'skog_flatt': 'forestgreen', 
        'skog_bratt': 'saddlebrown'
    }
    aoi_labels = {
        'jorde': 'Jorde',
        'skog_flatt': 'Skog (flatt)',
        'skog_bratt': 'Skog (bratt)'
    }
    
    for idx, pol in enumerate(['vv', 'vh']):
        ax = axes[idx]
        data = df[(df['pol'] == pol) & (df['rmse_vs_ref'].notna())]
        if data.empty:
            continue
        
        methods = data['method'].unique()
        x = np.arange(len(methods))
        n_aois = len(aoi_colors)
        w = 0.8 / n_aois
        
        for i, (aoi, color) in enumerate(aoi_colors.items()):
            vals = data[data['aoi'] == aoi].groupby('method')['rmse_vs_ref'].mean()
            ax.bar(x + i*w - 0.4 + w/2, [vals.get(m, 0) for m in methods],
                   w, label=aoi_labels[aoi], color=color, alpha=0.8)
        
        ax.set_ylabel('RMSE vs GAMMA (dB)')
        ax.set_title(pol.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([METHODS[m]['name'] for m in methods], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'rmse_by_aoi.png', dpi=150)
    plt.close()


def plot_bias(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, pol in enumerate(['vv', 'vh']):
        ax = axes[idx]
        data = df[df['pol'] == pol]
        bias = data.groupby('method')['bias'].mean()
        colors = [METHODS[m]['color'] for m in bias.index]
        
        ax.bar(range(len(bias)), bias.values, color=colors)
        ax.axhline(0, color='green', lw=2)
        ax.axhspan(-1, 1, alpha=0.1, color='green')
        ax.set_xticks(range(len(bias)))
        ax.set_xticklabels([METHODS[m]['name'] for m in bias.index], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Bias fra forventet (dB)')
        ax.set_title(f'{pol.upper()} (forventet: {EXPECTED_DB[pol]} dB)')
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'calibration_bias.png', dpi=150)
    plt.close()


def main():
    all_results = []
    
    for aoi_key in AOI_FILES.keys():
        for pol in ['vv', 'vh']:
            print(f"{aoi_key}/{pol}...", end=" ")
            df = run_extended(PRIMARY_DATE, aoi_key, pol)
            if df is not None:
                all_results.append(df)
                print("✓")
            else:
                print("✗")
    
    if not all_results:
        return
    
    df = pd.concat(all_results, ignore_index=True)
    df.to_csv(RESULTS_DIR / 'extended_metrics.csv', index=False)
    
    print("\n=== CV by AOI (%) ===")
    print(df.pivot_table(values='cv', index='method', columns='aoi').round(1))
    
    print("\n=== RMSE vs GAMMA by AOI (dB) ===")
    print(df.pivot_table(values='rmse_vs_ref', index='method', columns='aoi').round(3))
    
    print("\n=== Radiometric Bias (dB) ===")
    print(df.groupby(['method', 'pol'])['bias'].mean().unstack().round(2))
    
    plot_cv_by_aoi(df)
    plot_rmse_by_aoi(df)
    plot_bias(df)
    
    return df


if __name__ == "__main__":
    main()