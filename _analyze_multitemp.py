"""Multi-temporal extended metrics - stratified by AOI."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _config import METHODS, AOI_FILES, COMPARISON_DATES, RESULTS_DIR, FIGURES_DIR
from _analyze_extended import run_extended


def main():
    all_results = []
    for date_str in COMPARISON_DATES:
        print(f"{date_str}...", end=" ")
        date_ok = False
        for aoi_key in AOI_FILES.keys():
            for pol in ['vv', 'vh']:
                df = run_extended(date_str, aoi_key, pol)
                if df is not None:
                    all_results.append(df)
                    date_ok = True
        print("OK" if date_ok else "NO DATA")
    
    if not all_results:
        return
    
    df = pd.concat(all_results, ignore_index=True)
    df.to_csv(RESULTS_DIR / 'extended_multitemporal.csv', index=False)
    
    print("\n=== Mean CV by method/AOI (%) ===")
    print(df.pivot_table(values='cv', index='method', columns='aoi').round(1))
    
    print("\n=== Mean RMSE vs GAMMA by AOI (dB) ===")
    print(df.pivot_table(values='rmse_vs_ref', index='method', columns='aoi').round(3))
    
    # Time series plots
    all_dates = sorted(df['date'].unique())
    date_idx = {d: i for i, d in enumerate(all_dates)}
    
    aoi_list = list(AOI_FILES.keys())
    fig, axes = plt.subplots(len(aoi_list), 2, figsize=(14, 4*len(aoi_list)))
    
    for row, aoi_key in enumerate(aoi_list):
        aoi_data = df[df['aoi'] == aoi_key]
        
        # CV over time
        ax = axes[row, 0]
        for method in df['method'].unique():
            mdata = aoi_data[aoi_data['method'] == method]
            daily = mdata.groupby('date')['cv'].mean()
            x_vals = [date_idx[d] for d in daily.index]
            ax.plot(x_vals, daily.values,
                    marker=METHODS[method]['marker'],
                    color=METHODS[method]['color'],
                    label=METHODS[method]['name'], alpha=0.8)
        
        ax.set_xticks(range(len(all_dates)))
        ax.set_xticklabels(all_dates, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'CV - {aoi_key}')
        ax.set_ylabel('CV (%)')
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(alpha=0.3)
        ax.axhline(100, color='red', ls='--', alpha=0.5)
        
        # RMSE over time
        ax = axes[row, 1]
        for method in df['method'].unique():
            mdata = aoi_data[(aoi_data['method'] == method) & (aoi_data['rmse_vs_ref'].notna())]
            if mdata.empty:
                continue
            daily = mdata.groupby('date')['rmse_vs_ref'].mean()
            x_vals = [date_idx[d] for d in daily.index]
            ax.plot(x_vals, daily.values,
                    marker=METHODS[method]['marker'],
                    color=METHODS[method]['color'],
                    label=METHODS[method]['name'], alpha=0.8)
        
        ax.set_xticks(range(len(all_dates)))
        ax.set_xticklabels(all_dates, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'RMSE vs GAMMA - {aoi_key}')
        ax.set_ylabel('RMSE (dB)')
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'extended_multitemporal.png', dpi=150)
    plt.close()
    
    return df

if __name__ == "__main__":
    main()