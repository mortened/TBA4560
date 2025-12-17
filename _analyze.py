"""
SAR RTC comparison analysis.
Implements metrics from Flores-Anderson et al. (2023).
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from _config import (
    COMPARISON_DATES, AOI_FILES, METHODS, PRIMARY_DATE,
    RESULTS_DIR, FIGURES_DIR
)
from _data_utils import (
    load_all_methods, 
    get_valid_mask, 
    extract_common_pixels,
    to_db
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_cv(data):
    """Coefficient of Variation (%)."""
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return np.nan
    return (np.std(valid) / np.abs(np.mean(valid))) * 100


def calculate_stats(data):
    """Basic statistics for backscatter array."""
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return {'mean': np.nan, 'std': np.nan, 'cv': np.nan, 'n': 0}
    return {
        'mean': np.mean(valid),
        'std': np.std(valid),
        'cv': calculate_cv(data),
        'n': len(valid)
    }


def calculate_rmse(data1, data2):
    """RMSE between two arrays (must be same shape)."""
    mask = np.isfinite(data1) & np.isfinite(data2)
    if np.sum(mask) == 0:
        return np.nan
    diff = data1[mask] - data2[mask]
    return np.sqrt(np.mean(diff ** 2))


def calculate_bias(data1, data2):
    """Mean bias (data1 - data2)."""
    mask = np.isfinite(data1) & np.isfinite(data2)
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(data1[mask] - data2[mask])


def calculate_correlation(data1, data2):
    """Pearson correlation between two arrays."""
    mask = np.isfinite(data1) & np.isfinite(data2)
    if np.sum(mask) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(data1[mask], data2[mask])
    return r, p

def analyze_single_date(date_str, aoi_key, pol='vv'):
    """
    Analyze all methods for a single date/AOI.
    """
    data_dict, profile = load_all_methods(date_str, aoi_key, pol, as_db=True)
    
    if not data_dict:
        print(f"  Warning: No data for {date_str}/{aoi_key}/{pol}")
        return None
    
    print(f"  Loaded {len(data_dict)} methods for {aoi_key}/{pol}")
    
    # Verify all same shape
    shapes = {k: v.shape for k, v in data_dict.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        print(f"  Warning: Shape mismatch after alignment: {shapes}")
    
    results = []
    for method_key, data in data_dict.items():
        stats_dict = calculate_stats(data)
        results.append({
            'date': date_str,
            'aoi': aoi_key,
            'pol': pol,
            'method': method_key,
            'method_name': METHODS[method_key]['name'],
            **stats_dict
        })
    
    return pd.DataFrame(results)


def analyze_inter_product(date_str, aoi_key, pol='vv', ref_method='hyp3_gamma'):
    """
    Compare all methods against reference
    """
    data_dict, profile = load_all_methods(date_str, aoi_key, pol, as_db=True)
    
    if not data_dict:
        return None
    
    # Check if reference exists
    if ref_method not in data_dict:
        print(f"  Warning: Reference {ref_method} not available, skipping comparison")
        return None
    
    ref_data = data_dict[ref_method]
    
    results = []
    for method_key, data in data_dict.items():
        if method_key == ref_method:
            continue
        
        # Verify same shape
        if data.shape != ref_data.shape:
            print(f"  Warning: Shape mismatch {method_key}:{data.shape} vs ref:{ref_data.shape}")
            continue
        
        rmse = calculate_rmse(data, ref_data)
        bias = calculate_bias(data, ref_data)
        r, p = calculate_correlation(data, ref_data)
        
        results.append({
            'date': date_str,
            'aoi': aoi_key,
            'pol': pol,
            'method': method_key,
            'method_name': METHODS[method_key]['name'],
            'ref': ref_method,
            'rmse': rmse,
            'bias': bias,
            'r': r,
            'p': p
        })
    
    return pd.DataFrame(results) if results else None

def run_single_date_analysis(date_str=None):
    """Run analysis for single date (primary date if not specified)."""
    if date_str is None:
        date_str = PRIMARY_DATE
    
    print(f"\nAnalyzing date: {date_str}")
    
    all_stats = []
    all_comparisons = []
    
    for aoi_key in AOI_FILES.keys():
        for pol in ['vv', 'vh']:
            # Statistics
            df_stats = analyze_single_date(date_str, aoi_key, pol)
            if df_stats is not None:
                all_stats.append(df_stats)
            
            # Comparison to reference
            df_comp = analyze_inter_product(date_str, aoi_key, pol)
            if df_comp is not None:
                all_comparisons.append(df_comp)
    
    # Save and display results
    stats_df = None
    comp_df = None
    
    if all_stats:
        stats_df = pd.concat(all_stats, ignore_index=True)
        stats_df.to_csv(RESULTS_DIR / f'stats_{date_str}.csv', index=False)
        
        print(f"\n{'='*60}")
        print(f"STATISTICS FOR {date_str}")
        print('='*60)
        
        # Pivot table for display
        pivot = stats_df.pivot_table(
            values=['mean', 'cv'], 
            index='method_name', 
            columns=['aoi', 'pol'],
            aggfunc='first'
        ).round(2)
        print(pivot)
    
    if all_comparisons:
        comp_df = pd.concat(all_comparisons, ignore_index=True)
        comp_df.to_csv(RESULTS_DIR / f'comparison_{date_str}.csv', index=False)
        
        print(f"\n{'='*60}")
        print(f"COMPARISON TO REFERENCE FOR {date_str}")
        print('='*60)
        
        pivot = comp_df.pivot_table(
            values=['rmse', 'r', 'bias'], 
            index='method_name', 
            columns=['aoi', 'pol'],
            aggfunc='first'
        ).round(3)
        print(pivot)
    
    return stats_df, comp_df


def run_timeseries_analysis():
    """Run analysis across all dates."""
    print("\n" + "="*60)
    print("TIME SERIES ANALYSIS")
    print("="*60)
    
    all_stats = []
    all_comparisons = []
    
    for date_str in COMPARISON_DATES:
        print(f"\nProcessing {date_str}...", end=" ")
        
        date_has_data = False
        for aoi_key in AOI_FILES.keys():
            for pol in ['vv', 'vh']:
                df_stats = analyze_single_date(date_str, aoi_key, pol)
                if df_stats is not None:
                    all_stats.append(df_stats)
                    date_has_data = True
                
                df_comp = analyze_inter_product(date_str, aoi_key, pol)
                if df_comp is not None:
                    all_comparisons.append(df_comp)
        
        print("✓" if date_has_data else "✗ (no data)")
    
    # Save and summarize
    if all_stats:
        stats_df = pd.concat(all_stats, ignore_index=True)
        stats_df.to_csv(RESULTS_DIR / 'stats_timeseries.csv', index=False)
        
        # Summary by method and AOI
        summary = stats_df.groupby(['method_name', 'aoi', 'pol']).agg({
            'mean': ['mean', 'std'],
            'cv': ['mean', 'std'],
            'n': 'mean'
        }).round(3)
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary.to_csv(RESULTS_DIR / 'stats_summary.csv')
        
        print("\n" + "="*60)
        print("TIME SERIES SUMMARY - STATISTICS")
        print("="*60)
        print(summary)
    
    if all_comparisons:
        comp_df = pd.concat(all_comparisons, ignore_index=True)
        comp_df.to_csv(RESULTS_DIR / 'comparison_timeseries.csv', index=False)
        
        comp_summary = comp_df.groupby(['method_name', 'aoi', 'pol']).agg({
            'rmse': ['mean', 'std'],
            'r': ['mean', 'std'],
            'bias': ['mean', 'std']
        }).round(3)
        comp_summary.columns = ['_'.join(col).strip() for col in comp_summary.columns]
        comp_summary.to_csv(RESULTS_DIR / 'comparison_summary.csv')
        
        print("\n" + "="*60)
        print("TIME SERIES SUMMARY - COMPARISON TO REFERENCE")
        print("="*60)
        print(comp_summary)
    
    return (stats_df if all_stats else None, 
            comp_df if all_comparisons else None)


def check_data_availability():
    """Check what data is available before running analysis."""
    print("\n" + "="*60)
    print("DATA AVAILABILITY CHECK")
    print("="*60)
    
    from _config import print_status
    print_status()

def main():
    print("=" * 60)
    print("SAR RTC COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Check data availability
    check_data_availability()
    
    # Single date analysis (primary date)
    print(f"\n--- Primary Date Analysis ({PRIMARY_DATE}) ---")
    stats_single, comp_single = run_single_date_analysis(PRIMARY_DATE)
    
    # Time series analysis
    print("\n--- Time Series Analysis ---")
    stats_ts, comp_ts = run_timeseries_analysis()
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()