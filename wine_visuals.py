"""Explore wine2.csv and save a chart for every variable.

Usage: python3 wine_visuals.py [path/to/data.csv] [output_directory]
Requires pandas and matplotlib. Rows (including duplicates) are retained.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().with_name('wine2.csv')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().with_name('wine_visuals')
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)
plt.style.use('ggplot')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches='tight')
    plt.close(fig)


# One figure per column; discrete quality scores are shown as counts.
for column in df.columns:
    series = df[column].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    if pd.api.types.is_numeric_dtype(series) and column != 'quality':
        ax.hist(series, bins=40, color='#497eaa', edgecolor='white', linewidth=0.4)
        ax.axvline(series.median(), color='#b63838', linestyle='--', label=f'Median: {series.median():.3g}')
        ax.legend()
        ax.set_ylabel('Number of wines')
        ax.set_xlabel(column.replace('_', ' ').title())
    else:
        counts = series.value_counts().sort_index()
        if column == 'quality_class':
            counts = counts.reindex([v for v in ['Low', 'Medium', 'High'] if v in counts.index])
        ax.bar(counts.index.astype(str), counts.values, color='#497eaa')
        ax.set_ylabel('Number of wines')
        ax.set_xlabel(column.replace('_', ' ').title())
        ax.tick_params(axis='x', labelrotation=30)
    ax.set_title(f'{column.replace("_", " ").title()} distribution (n={len(series):,})')
    save(fig, f'{column}.png')


# Compare measured properties across quality scores, with sample sizes shown.
for column in ['alcohol', 'volatile_acidity']:
    if column not in df or 'quality' not in df:
        continue
    grouped = df.dropna(subset=[column, 'quality']).groupby('quality')[column]
    scores = sorted(grouped.groups)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([grouped.get_group(score) for score in scores], tick_labels=[str(score) for score in scores], showfliers=False)
    ax.set_xlabel('Quality score (number of wines)')
    ax.set_xticklabels([f'{score}\n(n={len(grouped.get_group(score)):,})' for score in scores])
    ax.set_ylabel(column.replace('_', ' ').title())
    ax.set_title(f'{column.replace("_", " ").title()} by quality score')
    ax.text(0.99, 0.02, 'Boxplots omit outlier markers; all observations are included.',
            transform=ax.transAxes, ha='right', fontsize=8)
    save(fig, f'{column}_by_quality.png')


numeric = df.select_dtypes(include='number')
if len(numeric.columns) > 1:
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    img = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha='right')
    ax.set_yticks(range(len(corr)), corr.index)
    fig.colorbar(img, ax=ax, label='Pearson correlation')
    ax.set_title('Correlations between numeric variables')
    save(fig, 'correlations.png')


print(f'Read {len(df):,} rows and {len(df.columns)} columns from {DATA}')
print(f'Missing values: {df.isna().sum().sum():,}; duplicate rows retained: {df.duplicated().sum():,}')
if 'quality' in numeric:
    print('\nPearson correlations with quality:')
    print(numeric.corr()['quality'].drop('quality').sort_values().to_string())
print(f'\nCharts saved to {OUT.resolve()}')

# Focused matrices show how associations differ by wine type.
focus = ['volatile_acidity', 'total_sulfur_dioxide', 'residual_sugar', 'density', 'alcohol', 'quality']
if set(focus + ['wine_type']).issubset(df.columns):
    groups = [('All wines', df), ('Red wines', df[df.wine_type == 'red']),
              ('White wines', df[df.wine_type == 'white'])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, (title, group) in zip(axes, groups):
        corr = group[focus].corr()
        img = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        labels = [c.replace('_', ' ') for c in focus]
        ax.set_xticks(range(len(focus)), labels, rotation=55, ha='right')
        ax.set_yticks(range(len(focus)), labels)
        for i in range(len(focus)):
            for j in range(len(focus)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
        ax.set_title(f'{title} (n={len(group):,})')
    fig.colorbar(img, ax=axes, shrink=0.7, label='Pearson correlation')
    fig.savefig(OUT / 'focused_correlations_by_wine_type.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Pooled association reverses when the two wine types are examined separately.
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {'red': '#a64343', 'white': '#b39625'}
    for kind in ['red', 'white']:
        group = df[df.wine_type == kind]
        r = group['volatile_acidity'].corr(group['total_sulfur_dioxide'])
        ax.scatter(group['volatile_acidity'], group['total_sulfur_dioxide'],
                   s=12, alpha=0.25, color=colors[kind], label=f'{kind.title()}: r={r:.2f} (n={len(group):,})')
    pooled = df['volatile_acidity'].corr(df['total_sulfur_dioxide'])
    ax.set_title(f'Volatile acidity vs total sulfur dioxide (all wines: r={pooled:.2f})')
    ax.set_xlabel('Volatile acidity')
    ax.set_ylabel('Total sulfur dioxide')
    ax.legend()
    save(fig, 'volatile_acidity_vs_total_sulfur_by_type.png')

    # Separate boxplots explain the group composition behind the pooled relationship.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, column in zip(axes, ['volatile_acidity', 'total_sulfur_dioxide']):
        data = [df.loc[df.wine_type == kind, column].dropna() for kind in ['red', 'white']]
        ax.boxplot(data, tick_labels=[f'Red\n(n={len(data[0]):,})', f'White\n(n={len(data[1]):,})'], showfliers=False)
        ax.set_ylabel(column.replace('_', ' ').title())
        ax.set_title(column.replace('_', ' ').title() + ' by wine type')
    fig.suptitle('Wine type differences (outlier markers omitted; all rows included)')
    save(fig, 'acidity_and_sulfur_by_wine_type.png')

    # Sugar-density relationship differs substantially between wine types.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, kind in zip(axes, ['red', 'white']):
        group = df[df.wine_type == kind].dropna(subset=['residual_sugar', 'density']).copy()
        group['sugar_quartile'] = pd.qcut(group['residual_sugar'], 4, duplicates='drop')
        bins = list(group.groupby('sugar_quartile', observed=True))
        ax.boxplot([part['density'] for _, part in bins],
                   tick_labels=[f'{interval.left:.1f}–{interval.right:.1f}\n(n={len(part):,})' for interval, part in bins],
                   showfliers=False)
        r = group['residual_sugar'].corr(group['density'])
        ax.set_title(f'{kind.title()} wines: r={r:.2f}')
        ax.set_xlabel('Residual sugar quartile (within type)')
        ax.tick_params(axis='x', labelsize=8)
    axes[0].set_ylabel('Density')
    fig.suptitle('Density by residual sugar (outlier markers omitted; all rows included)')
    save(fig, 'density_by_sugar_and_wine_type.png')


