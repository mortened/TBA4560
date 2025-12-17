import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from rasterio.transform import rowcol

from _config import METHODS, COMPARISON_DATES, FIGURES_DIR, DATE_METADATA
from _config import VEG_CORRECTION_COEFS as VCC
from _data_utils import load_all_methods


def match_dates(sar_dates, field_dates, max_days=3):
    """Match SAR-datoer til nærmeste feltmåling."""
    matches = []
    for s in sar_dates:
        sd = datetime.strptime(s, '%Y%m%d')
        closest = min(field_dates, key=lambda f: abs((f - sd).days))
        if abs((closest - sd).days) <= max_days:
            matches.append((s, closest))
    return matches


def filter_dates(date_list, months=None, orbit=None):
    """Filtrer datoer på måned og/eller baneretning."""
    result = date_list
    if months:
        result = [d for d in result 
                  if datetime.strptime(d[0] if isinstance(d, tuple) else d, '%Y%m%d').month in months]
    if orbit:
        result = [d for d in result 
                  if DATE_METADATA.get(d[0] if isinstance(d, tuple) else d) == orbit]
    return result


def extract_at_points(raster, transform, x_coords, y_coords):
    """Ekstraher rasterverdier ved punktkoordinater."""
    values = []
    for x, y in zip(x_coords, y_coords):
        row, col = rowcol(transform, x, y)
        if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
            values.append(raster[row, col])
        else:
            values.append(np.nan)
    return np.array(values)


def validate_method(method_key, field_df, matched_dates, aoi='jorde', pol='vv', veg_correction=False):
    """Valider en preprosesseringsmetode mot feltdata."""
    all_bs = []
    for sar_date, _ in matched_dates:
        data_dict, _ = load_all_methods(sar_date, aoi, pol, as_db=True)
        if method_key in data_dict:
            gamma = data_dict[method_key]
            all_bs.append(gamma[np.isfinite(gamma)])
    
    if not all_bs:
        return None
    
    all_bs = np.concatenate(all_bs)

    if all_bs.size == 0:
        print(f"Ingen gyldige bakkspredningsverdier funnet for kalibrering av {method_key}.")
        return None
    
    # find percentile limits
    gamma_min = np.percentile(all_bs, 5)
    gamma_max = np.percentile(all_bs, 95)
    sm_min = field_df['theta_median'].min()
    sm_max = field_df['theta_median'].max()
    
    all_field, all_sar = [], []
    
    for sar_date, field_date in matched_dates:
        data_dict, transform = load_all_methods(sar_date, aoi, pol, as_db=True)
        if method_key not in data_dict or transform is None:
            continue
        
        field_subset = field_df[field_df['dato'] == field_date]
        if len(field_subset) == 0:
            continue
        if veg_correction:
            print("Using vegetation correction")
            ndvi = field_subset['NDVI_S2'].mean()
            sigma_range_corr = VCC['A'] * ndvi**2 + VCC['B'] * ndvi + VCC['C']
            gamma_max_veg = gamma_min + sigma_range_corr
        else:
            print("Without vegetation correction")
            gamma_max_veg = gamma_max
        
        if gamma_max_veg <= gamma_min:
            gamma_max_veg = gamma_max
        
        gamma = data_dict[method_key]
        if(veg_correction):
            gamma_norm = np.clip((gamma - gamma_min) / (gamma_max_veg - gamma_min), 0, 1)
        else:
            gamma_norm = (gamma - gamma_min) / (gamma_max_veg - gamma_min)

        sm_map = gamma_norm * (sm_max - sm_min) + sm_min
        
        sm_sar = extract_at_points(sm_map, transform, 
                                   field_subset['x'].values, 
                                   field_subset['y'].values)
        sm_field = field_subset['theta_median'].values
        
        valid = np.isfinite(sm_sar) & np.isfinite(sm_field)
        if valid.sum() >= 3:
            all_field.extend(sm_field[valid])
            all_sar.extend(sm_sar[valid])
    
    if not all_field:
        return None
    
    all_field, all_sar = np.array(all_field), np.array(all_sar)
    r, _ = stats.pearsonr(all_field, all_sar)
    
    return {
        'method': method_key,
        'r': r,
        'rmse': np.sqrt(np.mean((all_field - all_sar)**2)),
        'bias': np.mean(all_sar - all_field),
        'n': len(all_field),
        'field': all_field,
        'sar': all_sar
    }


def plot_results(results, suffix='', title=''):
    """Plot comparison of methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    
    df = pd.DataFrame([{k: r[k] for k in ['method', 'r', 'rmse', 'bias']} for r in results])
    df = df.sort_values('rmse')
    
    colors = [METHODS[m]['color'] for m in df['method']]
    names = [METHODS[m]['name'] for m in df['method']]
    
    for ax, col, xlabel in zip(axes, ['r', 'rmse', 'bias'], 
                                ['Korrelasjon (r)', 'RMSE [m³/m³]', 'Bias [m³/m³]']):
        ax.barh(range(len(df)), df[col], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', alpha=0.3)
        if col == 'bias':
            ax.axvline(0, color='black', linestyle='--', lw=1)
    
    plt.tight_layout()
    plt.show()
    
    # Scatter plots
    n = len(results)
    fig, axes = plt.subplots(1, min(n, 5), figsize=(4*min(n, 5), 4))
    if n == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        ax.scatter(res['field'], res['sar'], alpha=0.5, s=20, 
                   color=METHODS[res['method']]['color'])
        lim = max(res['field'].max(), res['sar'].max())
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
        ax.set_xlabel('Felt SM [m³/m³]')
        ax.set_ylabel('SAR SM [m³/m³]')
        ax.set_title(f"{METHODS[res['method']]['name']}\nr={res['r']:.2f}, RMSE={res['rmse']:.3f}")
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main(months=None, orbit=None, veg_correction=False):
    """Run soil moisture validation."""
    field_df = pd.read_csv('data/input_cleaned.csv')
    field_df['dato'] = pd.to_datetime(field_df['dato'])
    
    matched = match_dates(COMPARISON_DATES, field_df['dato'].unique(), max_days=3)
    matched = filter_dates(matched, months=months, orbit=orbit)
    
    suffix = ''
    title_parts = []
    if months:
        suffix += f"_m{''.join(map(str, months))}"
        title_parts.append(f"Måned {months}")
    if orbit:
        suffix += f"_{orbit[:3].lower()}"
        title_parts.append(orbit)
    title = ' - '.join(title_parts) if title_parts else 'Alle datoer'
    
    print(f"\n{title}: {len(matched)} datopar")
    
    if not matched:
        return None
    
    methods = ['hyp3_gamma', 'pyrosar_kartverket', 'pyrosar_copernicus',
               'gee_s1ard_kartverket', 'gee_s1ard_copernicus']
    
    results = []
    for m in methods:
        res = validate_method(m, field_df, matched, aoi='jorde', pol='vv', veg_correction=veg_correction)
        if res:
            results.append(res)
            print(f"  {METHODS[m]['name']}: r={res['r']:.3f}, RMSE={res['rmse']:.3f}, n={res['n']}")
    
    if results:
        plot_results(results, suffix=suffix, title=title)
    
    return results


if __name__ == "__main__":

    # # All months, both orbits
    # print("Validering for alle datoer:")
    #main(veg_correction=False)
    # # Juni, per baneretning


    # main(months=[6])

    # main(months=[7])
    
    # main(months=[8])
    # main(months=[9])
    
    # main(months=[10])
    #main()
    
    # # # Hele sesongen, per baneretning
    # main(orbit='Ascending')
    # main(orbit='Descending')
    
    # # Baseline
    main(months=[6], orbit='Ascending', veg_correction=True)
    main(months=[6], orbit='Descending', veg_correction=True)