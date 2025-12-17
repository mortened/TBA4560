import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# load cleaned dataset
df = pd.read_csv('data/input_cleaned.csv')
df['dato'] = pd.to_datetime(df['dato'])

plot_cols = ['theta_median', 'plantehøyde', 'NDVI_S2', 'VC']
titles = ['Jordfuktighet trend (theta_median)', 'Plantehøyde trend', 'NDVI trend', 'Vegetasjonsdekning (VC) trend']
ylabels = ['Volumetrisk jordvanninnhold', 'Høyde (cm)', 'Indeksverdi', 'Fraksjon']

# 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()


dates = sorted(df['dato'].unique())
dates_str = [d.strftime('%d.%m.%Y') for d in dates]

# iterate and plot each variable
for i, col in enumerate(plot_cols):
    ax = axes[i]
    

    df['dato_str'] = df['dato'].dt.strftime('%d.%m.%Y')
    
    # Boxplot
    sns.boxplot(x='dato_str', y=col, data=df, ax=ax, 
                color='lightblue', showfliers=True, order=dates_str)
    
    # trendline (median)
    daily_median = df.groupby('dato')[col].median()
    y_values = [daily_median.get(d, float('nan')) for d in dates]
    ax.plot(range(len(dates)), y_values, marker='o', color='red', linewidth=2, label='Median Trend')
    
    # Formatting
    ax.set_title(titles[i], fontsize=14)
    ax.set_ylabel(ylabels[i], fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    

    if i >= 2:
        # Show x-axis labels only on bottom plots
        ax.set_xticklabels(dates_str, rotation=45, ha='right')
        ax.set_xlabel('Dato', fontsize=12)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False) # Hide labels

    if i == 0:
        ax.legend(loc='upper right')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()