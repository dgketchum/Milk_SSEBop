"""
journal_figures.py — Consolidated publication figures for GIScience & Remote Sensing.

Produces Figures 2–5 at final print dimensions (175 mm double-column width, 600 DPI).
Self-contained: no imports from project analysis modules.

Usage:
    python journal_figures.py [--data_dir /path/to/milk] [--map_png /path/to/map.png]
"""

import argparse
import json
import os
from calendar import month_abbr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, skew, kurtosis

# ──────────────────────────────────────────────────────────────────────────────
# STYLE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DOUBLE_COL_W = 6.89          # 175 mm in inches
MAX_HEIGHT = 9.06             # 230 mm in inches

FONT_AXIS_LABEL = 8
FONT_TICK = 7
FONT_LEGEND = 7
FONT_PANEL_LABEL = 9
FONT_ANNOT = 6.5

MARKER_DENSE = 3              # ~180 k points (met‐station residuals)
MARKER_EC = 15                # ~680 points (eddy‑covariance)

PALETTE = sns.color_palette('rocket', 6)

plt.rcParams.update({
    'font.size':           FONT_TICK,
    'axes.labelsize':      FONT_AXIS_LABEL,
    'axes.titlesize':      FONT_AXIS_LABEL,
    'xtick.labelsize':     FONT_TICK,
    'ytick.labelsize':     FONT_TICK,
    'legend.fontsize':     FONT_LEGEND,
    'figure.dpi':          150,
    'savefig.dpi':         600,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.08,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'xtick.direction':     'out',
    'ytick.direction':     'out',
    'pdf.fonttype':        42,       # TrueType in PDF
    'ps.fonttype':         42,
    'font.family':         'sans-serif',
    'font.sans-serif':     ['Arial', 'Helvetica', 'DejaVu Sans'],
})
sns.set_style('white', {'axes.linewidth': 0.5})

# ── Variable labels / limits (inlined from eto_error / eto_plots) ────────────

STR_MAP_SIMPLE = {
    'vpd':       'VPD',
    'rn':        'Rn',
    'mean_temp': 'Mean Temp',
    'wind':      'Wind Speed',
    'eto':       'ETo',
}

UNITS_MAP = {
    'rn':        r'[MJ m$^{-2}$ d$^{-1}$]',
    'vpd':       r'[kPa]',
    'mean_temp': r'[C]',
    'wind':      r'[m s$^{-1}$]',
    'eto':       r'[mm day$^{-1}$]',
}

SCATTER_LIMITS = {'vpd': 2.5, 'rn': 7, 'wind': 9, 'mean_temp': 12}

SCATTER_VARS = ['vpd', 'rn', 'wind', 'mean_temp']
HEXBIN_GRIDSIZE = 42

LC_COLORS = {1: '#e69073', 2: '#e0cd88', 3: '#196d12'}
LC_LABELS = {1: 'Cropland', 2: 'Grass/Shrubland', 3: 'Forest'}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def _panel_label(ax, letter, x=-0.06, y=1.06):
    """Place a bold panel label (a), (b), ... in axes coordinates."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=FONT_PANEL_LABEL, fontweight='bold',
            va='top', ha='right')


def _stats_box(ax, text, x=0.05, y=0.95, **kwargs):
    """White‑filled annotation box for statistics text."""
    props = dict(boxstyle='square,pad=0.3', facecolor='white',
                 edgecolor='gray', alpha=0.9)
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=FONT_ANNOT, va='top', ha='left',
            bbox=props, **kwargs)


def _oneto1_line(ax, lo=None, hi=None):
    """Black dashed 1:1 reference line."""
    if lo is None or hi is None:
        lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.5, zorder=0)


def _regression_stats(x, y):
    """Return dict with R2, slope, intercept, RMSE, n."""
    slope, intercept, r_value, _, _ = linregress(x, y)
    rmse = np.sqrt(np.mean((np.asarray(y) - np.asarray(x)) ** 2))
    return dict(r2=r_value ** 2, slope=slope, intercept=intercept,
                rmse=rmse, n=len(x))


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def load_pooled_residuals(data_dir):
    """all_residuals_nldas2.json → dict of lists keyed by variable."""
    p = os.path.join(data_dir, 'weather_station_data_processing',
                     'error_analysis', 'all_residuals_nldas2.json')
    return _load_json(p)


def load_station_residuals(data_dir):
    """station_residuals_nldas2.json → dict[station_id] → dict[var] → [resid, eto_resid, doy]."""
    p = os.path.join(data_dir, 'weather_station_data_processing',
                     'error_analysis', 'station_residuals_nldas2.json')
    return _load_json(p)


def load_monthly_residuals(data_dir):
    """station_residuals_nldas2_month.json → dict[station_id] → dict[var] → dict[month] → list."""
    p = os.path.join(data_dir, 'weather_station_data_processing',
                     'error_analysis', 'station_residuals_nldas2_month.json')
    return _load_json(p)


def load_residual_csvs(data_dir):
    """res_*.csv files → single concatenated DataFrame of residuals."""
    csv_dir = os.path.join(data_dir, 'weather_station_data_processing',
                           'comparison_data')
    frames = []
    for fn in sorted(os.listdir(csv_dir)):
        if fn.startswith('res_') and fn.endswith('.csv'):
            frames.append(pd.read_csv(os.path.join(csv_dir, fn)))
    df = pd.concat(frames, ignore_index=True)
    if 'mean_temp' not in df.columns and 'min_temp' in df.columns:
        df['mean_temp'] = 0.5 * (df['min_temp'] + df['max_temp'])
    return df


def load_nldas_gridded(data_dir):
    """Raw NLDAS-2 gridded CSVs → concatenated DataFrame of model estimates."""
    csv_dir = os.path.join(data_dir, 'weather_station_data_processing',
                           'gridded', 'nldas2')
    frames = []
    for fn in sorted(os.listdir(csv_dir)):
        if fn.endswith('.csv') and not fn.startswith('res_'):
            df = pd.read_csv(os.path.join(csv_dir, fn))
            if 'mean_temp' not in df.columns and 'min_temp' in df.columns:
                df['mean_temp'] = 0.5 * (df['min_temp'] + df['max_temp'])
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_variance_decomp(data_dir):
    """Build per-station variance fractions from MC JSONs.

    var_decomp_par.csv is a summary (one row per variable) so we reconstruct
    per-station data from the individual JSON files for the stacked bar chart.
    """
    metvars = ['vpd', 'rn', 'mean_temp', 'wind']
    json_dir = os.path.join(data_dir, 'weather_station_data_processing',
                            'error_analysis', 'mc_par_variance')

    if os.path.isdir(json_dir) and os.listdir(json_dir):
        rows = {}
        for fn in sorted(os.listdir(json_dir)):
            if not fn.endswith('.json'):
                continue
            station = fn.replace('.json', '').split('_', 3)[-1]
            with open(os.path.join(json_dir, fn)) as f:
                sim = json.load(f)
            variances = {}
            for var in metvars:
                try:
                    variances[var] = sum(v[1] for v in sim[var])
                except (KeyError, IndexError):
                    variances[var] = np.nan
            rows[station] = variances

        df = pd.DataFrame.from_dict(rows, orient='index', columns=metvars)
        df.dropna(how='any', inplace=True)
        df['sum'] = df.sum(axis=1)
        df = df.div(df['sum'], axis=0).drop(columns='sum')
        return df

    # Fallback: old per-station CSV
    base = os.path.join(data_dir, 'weather_station_data_processing',
                        'error_analysis')
    for name in ('var_decomp_par.csv', 'old_json/var_decomp_stations.csv'):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            df = pd.read_csv(p, index_col=0)
            if not df.empty and len(df) > 5:
                df = df.rename(columns={'tmean': 'mean_temp', 'u2': 'wind'})
                return df

    raise FileNotFoundError('No variance decomposition data found')


def load_ec_comparison(data_dir, time_step='daily'):
    """Parse ec_comparison JSON → concatenated DataFrame with 'fid' column.

    time_step: 'daily' → ec_comparison.json (eta_obs, eta_ssebop, eto_obs, eto_nldas)
               'monthly' → ec_comparison_monthly.json (eta_obs, eta_ssebop)
    """
    if time_step == 'daily':
        p = os.path.join(data_dir, 'validation', 'error_analysis',
                         'ec_comparison.json')
    else:
        p = os.path.join(data_dir, 'validation', 'error_analysis',
                         'ec_comparison_monthly.json')
    raw = _load_json(p)

    frames = []
    for fid, data in raw.items():
        rec = {'eta_obs': data['eta_obs'], 'eta_ssebop': data['eta_ssebop']}
        rec['fid'] = [fid] * len(data['eta_obs'])
        frames.append(pd.DataFrame(rec))
    return pd.concat(frames, ignore_index=True)


def load_et_extracts(data_dir):
    """Read CSV dir of monthly ET extracts, merge OBJECTIDs 1+2, compute volumes.

    Returns DataFrame indexed by month with et_1, et_2, et_3 in km³.
    """
    csv_dir = os.path.join(data_dir, 'results', 'et_extracts')
    file_list = sorted([os.path.join(csv_dir, x) for x in os.listdir(csv_dir)
                        if x.endswith('.csv')])

    first, adf = True, None
    for f in file_list:
        splt = os.path.basename(f).split('.')[0].split('_')
        dt = pd.to_datetime(f'{splt[-2]}-{splt[-1]}-01')
        if first:
            adf = pd.read_csv(f)
            adf['date'] = dt
            first = False
        else:
            c = pd.read_csv(f)
            c['date'] = dt
            adf = pd.concat([adf, c], ignore_index=False, axis=0)

    dct = {}
    for fid in [1, 2]:
        df = adf.loc[adf['OBJECTID'] == fid].copy()
        df.index = pd.DatetimeIndex(df['date'])
        df.drop(columns=['date', 'OBJECTID'], inplace=True)
        df = df.sort_index()
        df = df.resample('ME').sum()

        for p in ['all', 1, 2, 3]:
            df[f'et_{p}'] = df[f'et_{p}'] * df[f'lc_{p}'] / 1e9
        for p in ['all', 1, 2, 3]:
            df[f'lc_{p}'] = df[f'lc_{p}'] / 1e6

        dct[fid] = df.copy()

    vals = dct[1].values + dct[2].values
    df = pd.DataFrame(data=vals, index=dct[1].index, columns=dct[1].columns)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PANEL DRAWING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def draw_eto_histogram(ax, pooled_resids):
    """Fig 2a – ETo residual histogram + KDE + descriptive stats."""
    resids = np.asarray(pooled_resids.get('eto', []))

    sns.histplot(resids, kde=False, stat='density', color='grey', ax=ax,
                 linewidth=0.4)
    sns.kdeplot(resids, color='red', ax=ax, linewidth=0.8)
    ax.axvline(np.mean(resids), color='red', ls='--', lw=0.6)

    ax.set_xlabel(r'ETo Residual [mm day$^{-1}$]' + '\n(Observed minus NLDAS-2)')
    ax.set_ylabel('Density')
    ax.set_xlim(-5, 5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    n = len(resids)
    mu = float(np.mean(resids))
    var_ = float(np.var(resids))
    skew_ = float(skew(resids))
    kurt_ = float(kurtosis(resids))

    stats_text = '\n'.join([
        f'n = {n:,}',
        f'\u03bc = {mu:.2f}',
        f'\u03c3\u00b2 = {var_:.2f}',
        f'\u03b3\u2081 = {skew_:.2f}',
        f'\u03b3\u2082 = {kurt_:.2f}',
    ])
    _stats_box(ax, stats_text, x=0.05, y=0.95)

    return dict(n=n, mean=mu, variance=var_, skewness=skew_, kurtosis=kurt_)


def draw_monthly_boxplot(ax, monthly_data, daily_data):
    """Fig 2b – Monthly boxplot of per‑station mean ETo residuals + daily median overlay."""
    rows = []
    for station_id, station_data in monthly_data.items():
        for month_str, res_list in station_data.get('eto', {}).items():
            if res_list:
                rows.append({'month': int(month_str), 'mean_resid': np.mean(res_list)})
    df = pd.DataFrame(rows)

    n_stations = len(monthly_data)
    monthly_medians = {}
    for m in range(1, 13):
        vals = df.loc[df['month'] == m, 'mean_resid'].values
        monthly_medians[month_abbr[m]] = float(np.median(vals)) if len(vals) else np.nan

    bp = ax.boxplot([df.loc[df['month'] == m, 'mean_resid'].values for m in range(1, 13)],
                    positions=range(1, 13), widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='.', markersize=2,
                                                     markerfacecolor='gray', alpha=0.5))
    for patch in bp['boxes']:
        patch.set_facecolor('#d9d9d9')
        patch.set_alpha(0.9)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_linewidth(0.6)

    # Daily median ETo residual overlay
    eto_by_doy = {d: [] for d in range(1, 366)}
    for sid, data in daily_data.items():
        eto_vals = data.get('eto', [])
        if len(eto_vals) == 3:
            for resid, eto_resid, doy in zip(*eto_vals):
                if 1 <= doy <= 365:
                    eto_by_doy[doy].append(eto_resid)

    # Map DOY medians to fractional month positions for overlay
    doy_medians = [np.nanmedian(eto_by_doy[d]) if eto_by_doy[d] else np.nan
                   for d in range(1, 366)]
    doys = np.arange(1, 366)
    month_frac = doys / 30.44 + 0.5   # approximate DOY → month position
    ax.plot(month_frac, doy_medians, color='red', lw=0.8, label='Daily median residual')

    ax.axhline(0, color='black', lw=0.5, ls='-')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([month_abbr[m][0] for m in range(1, 13)])
    ax.set_xlabel('Month')
    ax.set_ylabel(r'ETo Residual [mm day$^{-1}$]')
    ax.legend(loc='lower right', framealpha=0.9)

    return dict(n_stations=n_stations, monthly_median_residual=monthly_medians)


def draw_correlation_heatmap(ax, nldas_df, resid_csv_df):
    """Fig 2c – 5×5 split‑triangle correlation heatmap.

    Upper triangle: NLDAS-2 estimate data correlations.
    Lower triangle: residual (obs minus NLDAS-2) correlations.
    """
    col_order = ['ETo', 'Rn', 'Mean Temp', 'VPD', 'Wind Speed']

    # Upper triangle: NLDAS-2 model estimates
    df_nldas = nldas_df.rename(columns=STR_MAP_SIMPLE)
    df_nldas = df_nldas[[v for v in col_order if v in df_nldas.columns]]
    corr_nldas = df_nldas[col_order].corr()

    # Lower triangle: residuals (obs minus NLDAS-2)
    df_resid = resid_csv_df.rename(columns=STR_MAP_SIMPLE)
    df_resid = df_resid[[v for v in col_order if v in df_resid.columns]]
    corr_resid = df_resid[col_order].corr()

    # Merge: upper = NLDAS-2 data, lower = residuals, diagonal = NaN
    joined = corr_resid.values.copy()
    ur = np.triu_indices_from(joined)
    joined[ur] = corr_nldas.values[ur]
    diag = np.diag_indices(joined.shape[0])
    joined[diag] = np.nan

    sns.heatmap(joined, annot=True, fmt='.2f', cmap='rocket', square=True,
                cbar=False, annot_kws={'size': FONT_ANNOT}, ax=ax,
                linewidths=0.5, linecolor='white')
    short_labels = ['ETo', 'Rn', 'Temp', 'VPD', 'Wind']
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_yticklabels(short_labels, rotation=0)
    # Re‑enable spines for heatmap box
    for spine in ax.spines.values():
        spine.set_visible(True)

    corr_dict = {}
    for i, row_var in enumerate(col_order):
        for j, col_var in enumerate(col_order):
            if i != j:
                tri = 'upper' if j > i else 'lower'
                corr_dict[f'{row_var}-{col_var} ({tri})'] = float(f'{joined[i, j]:.2f}')
    return corr_dict


def collect_met_scatter_data(var, station_resids):
    """Collect pooled residual pairs for one Fig 3 panel."""
    target_all, eto_all = [], []
    for sid, data in station_resids.items():
        vals = data.get(var, [])
        if len(vals) == 3:
            target_all.extend(vals[0])
            eto_all.extend(vals[1])

    return np.asarray(eto_all), np.asarray(target_all)


def _max_binned_count(eto, target, ylim, gridsize=HEXBIN_GRIDSIZE):
    """Approximate peak density for a shared log-scale legend."""
    counts, _, _ = np.histogram2d(
        eto, target,
        bins=(gridsize, gridsize),
        range=[[-6, 6], [-ylim, ylim]],
    )
    return max(1, int(counts.max()))


def draw_met_scatter(ax, var, eto, target, norm, show_n=False):
    """Fig 3a‑d – Log-density hexbin of met-variable residual vs. ETo residual."""
    hb = ax.hexbin(
        eto, target,
        gridsize=HEXBIN_GRIDSIZE,
        extent=(-6, 6, -SCATTER_LIMITS[var], SCATTER_LIMITS[var]),
        mincnt=1,
        cmap='rocket',
        norm=norm,
        linewidths=0,
        rasterized=True,
    )

    # White contour lines preserve the density shape in print and grayscale copies.
    counts, xedges, yedges = np.histogram2d(
        eto, target,
        bins=(HEXBIN_GRIDSIZE, HEXBIN_GRIDSIZE),
        range=[[-6, 6], [-SCATTER_LIMITS[var], SCATTER_LIMITS[var]]],
    )
    positive = counts[counts > 0]
    if positive.size:
        levels = np.unique(np.quantile(positive, [0.60, 0.80, 0.93]))
        if levels.size:
            xcent = 0.5 * (xedges[:-1] + xedges[1:])
            ycent = 0.5 * (yedges[:-1] + yedges[1:])
            ax.contour(xcent, ycent, counts.T, levels=levels,
                       colors='white', linewidths=0.35, alpha=0.8)

    ax.axvline(0, color='gray', alpha=0.5, lw=0.4)
    ax.axhline(0, color='gray', alpha=0.5, lw=0.4)

    _, _, r_value, _, _ = linregress(eto, target)
    r2 = r_value ** 2

    n = len(eto)
    if show_n:
        txt = f'n = {n:,}\nr\u00b2 = {r2:.2f}'
    else:
        txt = f'r\u00b2 = {r2:.2f}'
    _stats_box(ax, txt, x=0.05, y=0.95)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-SCATTER_LIMITS[var], SCATTER_LIMITS[var])
    ax.set_ylabel(f'{STR_MAP_SIMPLE[var]} {UNITS_MAP[var]}')

    return hb, dict(variable=var, n=n, r2=float(f'{r2:.4f}'))


def draw_summary_bar(ax, decomp_df):
    """Fig 3e – Single horizontal stacked bar of mean variance fractions."""
    df = decomp_df.rename(columns=STR_MAP_SIMPLE)
    cols = ['VPD', 'Wind Speed', 'Mean Temp', 'Rn']
    df = df[[c for c in cols if c in df.columns]]
    colors = sns.color_palette('rocket', n_colors=len(df.columns))

    means = df.mean()
    left = 0.0
    for col, c in zip(cols, colors):
        val = means[col]
        ax.barh(0, val, left=left, color=c, edgecolor='white', linewidth=0.5,
                height=0.5, label=col)
        pct = f'{val:.0%}'
        if val < 0.05:
            ax.text(left + val + 0.008, 0, pct, ha='left', va='center',
                    fontsize=FONT_TICK, fontweight='bold', color='0.3')
        else:
            ax.text(left + val / 2, 0, pct, ha='center', va='center',
                    fontsize=FONT_TICK, fontweight='bold', color='white')
        left += val

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Mean Variance Fraction (n = {} stations)'.format(len(df)))
    ax.legend(loc='upper center', ncol=4, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.45), fontsize=FONT_LEGEND)

    mean_fracs = {col: float(f'{df[col].mean():.4f}') for col in df.columns}
    return dict(n_stations=len(df), mean_variance_fraction=mean_fracs)


def draw_eta_scatter(ax, ec_df, time_step='daily'):
    """Fig 4a/b – Observed vs. SSEBop ETa scatter, colored by station."""
    stations = ec_df['fid'].unique()
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h']
    for i, fid in enumerate(stations):
        sub = ec_df[ec_df['fid'] == fid]
        m = markers[i % len(markers)]
        ax.scatter(sub['eta_obs'], sub['eta_ssebop'], s=MARKER_EC,
                   marker=m, alpha=0.8, label=fid, edgecolors='none')

    x, y = ec_df['eta_obs'].values, ec_df['eta_ssebop'].values
    st = _regression_stats(x, y)

    lo = min(x.min(), y.min()) - 0.2
    hi = max(x.max(), y.max()) + 0.2
    _oneto1_line(ax, lo, hi)

    txt = (f'R\u00b2 = {st["r2"]:.2f}\nSlope = {st["slope"]:.2f}\n'
           f'RMSE = {st["rmse"]:.2f}\nn = {st["n"]}')
    _stats_box(ax, txt, x=0.05, y=0.95)

    unit = r'mm day$^{-1}$' if time_step == 'daily' else r'mm month$^{-1}$'
    ax.set_xlabel(f'Flux Tower ET [{unit}]')
    ax.set_ylabel(f'SSEBop ET [{unit}]')
    ax.legend(title='Station', fontsize=5, title_fontsize=5.5,
              loc='lower right', framealpha=0.9, markerscale=0.6)

    n_stations = len(stations)
    return dict(time_step=time_step, n_stations=n_stations, n=st['n'],
                r2=float(f'{st["r2"]:.4f}'), slope=float(f'{st["slope"]:.4f}'),
                rmse=float(f'{st["rmse"]:.4f}'),
                bias=float(f'{st["intercept"]:.4f}'))


def draw_et_timeseries(ax, et_df):
    """Fig 5a – Monthly ETa volume by land cover."""
    for key in [1, 2, 3]:
        col = f'et_{key}'
        if col in et_df.columns:
            ax.plot(et_df.index, et_df[col], color=LC_COLORS[key],
                    label=LC_LABELS[key], lw=0.7)

    ax.set_ylabel(r'ETa [km$^3$]')
    ax.set_xlabel('Time')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(base=6))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    period = f'{et_df.index.min().strftime("%Y-%m")} to {et_df.index.max().strftime("%Y-%m")}'
    annual = et_df.groupby(et_df.index.year).sum()
    lc_stats = {}
    for key, lbl in LC_LABELS.items():
        col = f'et_{key}'
        if col in annual.columns:
            lc_stats[lbl] = dict(
                mean_annual_km3=float(f'{annual[col].mean():.4f}'),
                total_km3=float(f'{et_df[col].sum():.2f}'))
    return dict(period=period, n_months=len(et_df), land_cover=lc_stats)


def draw_map_image(ax, map_path):
    """Fig 5b – Embedded GIS‑produced map image, auto-cropped."""
    img = plt.imread(map_path)
    # Auto-crop whitespace: find bounding box of non-white content
    if img.dtype == np.float32:
        non_white = np.any(img[:, :, :3] < 0.99, axis=2)
    else:
        non_white = np.any(img[:, :, :3] < 252, axis=2)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    if len(rows) and len(cols):
        pad = 20
        r0 = max(rows[0] - pad, 0)
        r1 = min(rows[-1] + pad, img.shape[0])
        c0 = max(cols[0] - pad, 0)
        c1 = min(cols[-1] + pad, img.shape[1])
        img = img[r0:r1, c0:c1]
    ax.imshow(img)
    ax.axis('off')


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE ASSEMBLY
# ──────────────────────────────────────────────────────────────────────────────


def figure_2(data_dir, out_dir):
    """ETo Error Characterization — 3 panels."""
    pooled = load_pooled_residuals(data_dir)
    monthly = load_monthly_residuals(data_dir)
    daily = load_station_residuals(data_dir)
    resid_csv = load_residual_csvs(data_dir)
    nldas = load_nldas_gridded(data_dir)

    fig = plt.figure(figsize=(DOUBLE_COL_W, 2.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1],
                           wspace=0.38, figure=fig)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    stats_a = draw_eto_histogram(ax_a, pooled)
    stats_b = draw_monthly_boxplot(ax_b, monthly, daily)
    stats_c = draw_correlation_heatmap(ax_c, nldas, resid_csv)

    _panel_label(ax_a, 'a')
    _panel_label(ax_b, 'b')
    # (c) placed at same figure-space y as (a) since heatmap square=True
    # shrinks the axes height
    ax_a_top = ax_a.get_position().y1
    ax_c_pos = ax_c.get_position()
    y_in_c = (ax_a_top + 0.02 - ax_c_pos.y0) / ax_c_pos.height
    _panel_label(ax_c, 'c', y=y_in_c)

    _save(fig, out_dir, 'figure_2')
    return {'2a_histogram': stats_a, '2b_monthly_boxplot': stats_b,
            '2c_correlation': stats_c}


def figure_3(data_dir, out_dir):
    """Variance Decomposition — 4 density panels."""
    station_res = load_station_residuals(data_dir)

    fig = plt.figure(figsize=(DOUBLE_COL_W, 3.0))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 0.08],
                           hspace=0.28, wspace=0.55,
                           bottom=0.16, figure=fig)

    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax = fig.add_subplot(gs[1, 1:3])

    scatter_data = {
        var: collect_met_scatter_data(var, station_res)
        for var in SCATTER_VARS
    }
    vmax = max(
        _max_binned_count(eto, target, SCATTER_LIMITS[var])
        for var, (eto, target) in scatter_data.items()
    )
    shared_norm = LogNorm(vmin=1, vmax=vmax)

    scatter_stats = {}
    hb = None
    for i, var in enumerate(SCATTER_VARS):
        lbl = chr(ord('a') + i)
        eto, target = scatter_data[var]
        hb, scatter_stats[f'3{lbl}_{var}'] = draw_met_scatter(
            axes[i], var, eto, target, shared_norm, show_n=(i == 0))
        _panel_label(axes[i], lbl)
        axes[i].set_xlabel('')

    cb = fig.colorbar(hb, cax=cax, orientation='horizontal')
    cb.set_label('Hex-Bin Count [log]', labelpad=1)
    cb.outline.set_linewidth(0.4)

    fig.text(0.5, 0.01, r'ETo Residual [mm day$^{-1}$]',
             ha='center', va='bottom', fontsize=FONT_AXIS_LABEL)

    _save(fig, out_dir, 'figure_3')
    return scatter_stats


def figure_4(data_dir, out_dir):
    """SSEBop ETa Validation — 2 panels (daily + monthly)."""
    ec_daily = load_ec_comparison(data_dir, 'daily')
    ec_monthly = load_ec_comparison(data_dir, 'monthly')

    fig = plt.figure(figsize=(DOUBLE_COL_W, 3.2))
    gs = gridspec.GridSpec(1, 2, wspace=0.35, figure=fig)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    stats_a = draw_eta_scatter(ax_a, ec_daily, 'daily')
    stats_b = draw_eta_scatter(ax_b, ec_monthly, 'monthly')

    _panel_label(ax_a, 'a')
    _panel_label(ax_b, 'b')

    _save(fig, out_dir, 'figure_4')
    return {'4a_daily_eta': stats_a, '4b_monthly_eta': stats_b}


def figure_5(data_dir, out_dir, map_png=None):
    """39‑Year ETa Record — 2 panels (timeseries + map)."""
    et_df = load_et_extracts(data_dir)

    if map_png and os.path.isfile(map_png):
        fig = plt.figure(figsize=(DOUBLE_COL_W, 5.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5],
                               hspace=0.30, figure=fig)
        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1])

        stats_a = draw_et_timeseries(ax_a, et_df)
        draw_map_image(ax_b, map_png)

        _panel_label(ax_a, 'a')
        _panel_label(ax_b, 'b')
    else:
        fig = plt.figure(figsize=(DOUBLE_COL_W, 2.5))
        ax_a = fig.add_subplot(111)
        stats_a = draw_et_timeseries(ax_a, et_df)
        _panel_label(ax_a, 'a')

    _save(fig, out_dir, 'figure_5')
    return {'5a_timeseries': stats_a}


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT HELPER
# ──────────────────────────────────────────────────────────────────────────────


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{name}.{ext}')
        fig.savefig(p)
        print(f'  saved {p}')
    plt.close(fig)


def _write_stats_doc(all_stats, out_dir):
    """Write figure_stats.txt — flat text document of every statistic shown in figures."""
    p = os.path.join(out_dir, 'figure_stats.txt')
    lines = ['JOURNAL FIGURE STATISTICS',
             '=' * 60, '']

    def _fmt(d, indent=0):
        pfx = '  ' * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f'{pfx}{k}:')
                _fmt(v, indent + 1)
            else:
                lines.append(f'{pfx}{k}: {v}')

    for fig_key in sorted(all_stats.keys()):
        lines.append(f'Figure {fig_key}')
        lines.append('-' * 40)
        _fmt(all_stats[fig_key], indent=1)
        lines.append('')

    with open(p, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  saved {p}')


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_data_dir(cli_arg=None):
    if cli_arg and os.path.isdir(cli_arg):
        return cli_arg
    raise FileNotFoundError('Could not locate data directory. '
                            'Pass --data_dir explicitly.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default=None,
                        help='Root data directory (milk)')
    parser.add_argument('--map_png', default=None,
                        help='Path to mean annual ETa map PNG for Fig 5b')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: <data_dir>/figures)')
    parser.add_argument('--figure', type=int, nargs='+', default=None,
                        help='Generate only specific figures, e.g. --figure 2 3')
    args = parser.parse_args()

    d = _resolve_data_dir(args.data_dir)
    out = args.out_dir or os.path.join(d, 'figures')
    print(f'Data directory: {d}')
    print(f'Output directory: {out}')

    figs = args.figure or [2, 3, 4, 5]
    all_stats = {}
    if 2 in figs:
        all_stats['2'] = figure_2(d, out)
    if 3 in figs:
        all_stats['3'] = figure_3(d, out)
    if 4 in figs:
        all_stats['4'] = figure_4(d, out)
    if 5 in figs:
        map_png = args.map_png
        if map_png is None:
            default_map = os.path.join(d, 'figures', 'et_results',
                                       'SMM_ET_Large.png')
            if os.path.isfile(default_map):
                map_png = default_map
        all_stats['5'] = figure_5(d, out, map_png=map_png)

    _write_stats_doc(all_stats, out)
    print('\nDone.')

# ========================= EOF ====================================================================
