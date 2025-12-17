"""Multitemporal LIA-analyse."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _analyze_lia import run_analysis
from _config import METHODS, AOI_FILES, COMPARISON_DATES, RESULTS_DIR, FIGURES_DIR, DATE_METADATA

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AOI_NO = {'jorde': 'Jordbruk', 'skog_flatt': 'Skog (flatt)', 'skog_bratt': 'Skog (bratt)'}

def main():
    print("MULTITEMPORAL LIA-ANALYSE\n")
    
    all_results = []
    for date_str in COMPARISON_DATES:
        print(f"{date_str}...", end=" ")
        ok = False
        for aoi in AOI_FILES:
            for pol in ['vv', 'vh']:
                res = run_analysis(date_str, aoi, pol)
                if res[0] is not None:
                    all_results.append(res[0])
                    ok = True
        print("OK" if ok else "FAIL")
    
    if not all_results:
        return None
    
    df = pd.concat(all_results, ignore_index=True)
    df['orbit'] = df['date'].map(DATE_METADATA)
    df.to_csv(RESULTS_DIR / 'lia_multitemporal.csv', index=False)
    
    # Resultater
    print("\n=== Mean |slope| per metode ===")
    summary = df.groupby('method')['slope'].agg(
        mean=lambda x: x.abs().mean(),
        std=lambda x: x.abs().std(),
        n='count'
    ).sort_values('mean').round(3)
    print(summary)
    
    print("\n=== Per AOI ===")
    print(df.groupby(['method', 'aoi'])['slope'].apply(lambda x: x.abs().mean()).unstack().round(3))
    
    # Hovedfunn
    steep = df[df['aoi'] == 'skog_bratt']
    rtc = steep[steep['method'] != 'gee_standard']['slope'].abs().mean()
    no_rtc = steep[steep['method'] == 'gee_standard']['slope'].abs().mean()
    print(f"\nBratt terreng: RTC={rtc:.3f}, Uten RTC={no_rtc:.3f}, Forbedring={no_rtc/rtc:.1f}x")
    
    # --- FIGUR 1: Tidsserie ---
    fig, ax = plt.subplots(figsize=(12, 5))
    dates = sorted(df['date'].unique())
    
    for method in summary.index:
        mdata = df[df['method'] == method].groupby('date')['slope'].apply(lambda x: x.abs().mean())
        ax.plot(range(len(mdata)), mdata.values, marker=METHODS[method]['marker'],
                color=METHODS[method]['color'], label=METHODS[method]['name'], lw=1.5, ms=5)
    
    ax.axhline(0.05, color='green', ls='--', alpha=0.7)
    ax.axhline(0.1, color='orange', ls='--', alpha=0.7)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('|Helning| [dB/°]')
    ax.set_xlabel('Dato')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.25)
    ax.set_title('Temporal konsistens av RTC-kvalitet')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'lia_tidsserie.png', dpi=150)
    plt.close()
    
    # --- FIGUR 2: Boxplot per terrengtype ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    methods_order = summary.index.tolist()
    
    for idx, aoi in enumerate(AOI_FILES.keys()):
        ax = axes[idx]
        aoi_df = df[df['aoi'] == aoi]
        
        # Samle data per metode
        box_data = []
        colors = []
        labels = []
        for m in methods_order:
            vals = aoi_df[aoi_df['method'] == m]['slope'].abs().dropna().values
            if len(vals) > 0:
                box_data.append(vals)
                colors.append(METHODS[m]['color'])
                labels.append(METHODS[m]['name'].replace(' (Cop. 30m)', '').replace(' (Kart. 10m)', ''))
        
        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True)
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        ax.axhline(0.05, color='green', ls='--', alpha=0.7)
        ax.axhline(0.1, color='orange', ls='--', alpha=0.7)
        ax.set_ylabel('|Helning| [dB/°]')
        ax.set_title(AOI_NO[aoi])
        ax.set_ylim(0, 0.3)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('LIA-avhengighet per terrengtype', fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'lia_boxplot.png', dpi=150)
    plt.close()
    
    # --- FIGUR 3: Baneretning ---
    orbit_tbl = df.groupby(['method', 'orbit'])['slope'].apply(lambda x: x.abs().mean()).unstack()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods_order))
    w = 0.35
    
    asc = [orbit_tbl.loc[m, 'Ascending'] if m in orbit_tbl.index else np.nan for m in methods_order]
    desc = [orbit_tbl.loc[m, 'Descending'] if m in orbit_tbl.index else np.nan for m in methods_order]
    
    ax.bar(x - w/2, asc, w, label='Stigende', color='steelblue')
    ax.bar(x + w/2, desc, w, label='Synkende', color='darkorange')
    ax.axhline(0.05, color='green', ls='--', alpha=0.7)
    ax.axhline(0.1, color='orange', ls='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([METHODS[m]['name'] for m in methods_order], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('|Helning| [dB/°]')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('LIA-avhengighet per baneretning')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'lia_baneretning.png', dpi=150)
    plt.show()
    
    print(f"\nFigurer lagret: lia_tidsserie.png, lia_boxplot.png, lia_baneretning.png")
    return df



if __name__ == "__main__":
    main()