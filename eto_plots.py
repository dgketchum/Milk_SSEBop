import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.stats import linregress

from nldas_eto_error import LIMITS

STR_MAP_SIMPLE = {
    'rn': r'Rn',
    'vpd': r'VPD',
    'mean_temp': r'Mean Temp',
    'wind': r'Wind Speed',
    'eto': r'ETo'
}

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
          'xtick.color': 'black',
          'ytick.color': 'black',
          'xtick.direction': 'out',
          'ytick.direction': 'out',
          'xtick.bottom': True,
          'xtick.top': False,
          'ytick.left': True,
          'ytick.right': False,
          }
plt.rcParams.update(params)
sns.set_style('white', {'axes.linewidth': 0.5})

LIMITS = {'vpd': 3,
          'rn': 10,
          'wind': 12,
          'mean_temp': 15}


def plot_residual_met_histograms(resids_file, plot_dir):
    with open(resids_file, 'r') as f:
        res_dct = json.load(f)

    for var, residuals in res_dct.items():

        mean_ = np.mean(residuals)
        variance = np.var(residuals).item()
        data_skewness = skew(residuals).item()
        data_kurtosis = kurtosis(residuals).item()

        plt.figure(figsize=(10, 6))

        ax = sns.histplot(residuals, kde=True, color='grey')
        kde_line = ax.lines[0]
        kde_line.set_color('red')

        plt.axvline(mean_, color='black', linestyle='dashed', linewidth=1)
        plt.xlabel(f'{STR_MAP[var]} Residuals\n(Observed minus NLDAS)')
        plt.ylabel('Frequency')
        plt.grid(True)

        if var in LIMITS:
            plt.xlim([-1 * LIMITS[var], LIMITS[var]])

        textstr = '\n'.join((
            r'$n={:,}$'.format(len(residuals)),
            r'$\mu={:.2f}$'.format(mean_),
            r'$\sigma^2={:.2f}$'.format(variance),
            r'$\gamma_1={:.2f}$'.format(data_skewness),
            r'$\gamma_2={:.2f}$'.format(data_kurtosis)))
        props = dict(boxstyle='round', facecolor='white')
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                       verticalalignment='top', bbox=props)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f'{var}_residuals_histogram.png')
        plt.savefig(plot_path)


def plot_eto_var_scatter_histograms(resid_json, plot_dir):
    with open(resid_json, 'r') as f:
        dct = json.load(f)

    vars = [c for c in LIMITS.keys()]

    fig, axes = plt.subplots(2, 2, figsize=(10, 12), dpi=600)
    axes = axes.flatten()

    dct = {k: v for k, v in dct.items() if len(v['eto']) > 0}

    for i, var in enumerate(vars):

        data = [v[var] for k, v in dct.items()]

        target = [d[0] for d in data]
        target = np.array([d for sub in target for d in sub])

        eto = [d[1] for d in data]
        eto = np.array([d for sub in eto for d in sub])

        print(var)
        ax_main = axes[i]

        ax_main.scatter(eto, target, s=2, alpha=.9,
                        c=sns.color_palette('rocket')[0], edgecolors='gray', linewidths=.5)

        ax_main.scatter(eto.mean(), target.mean(), s=10, alpha=.9, color='red')

        ax_main.set_ylabel(STR_MAP_SIMPLE[var])

        ax_main.axvline(0, color='gray', alpha=0.7)
        ax_main.axhline(0, color='gray', alpha=0.7)

        slope, intercept, r_value, p_value, std_err = linregress(eto, target)
        r_squared = r_value ** 2

        textstr = '\n'.join((
            r'$n={:,}$'.format(target.shape[0]),
            r'$r^2$: {:.2f}'.format(r_squared)))
        props = dict(boxstyle='round', facecolor='white')
        ax_main.text(0.05, 0.95, textstr, transform=ax_main.transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)

        ax_main.set_xlim(-6, 6)
        ax_main.set_ylim(-1 * LIMITS[var], LIMITS[var])

    textstr = 'ETo [mm day$^{-1}$]'
    props = dict(facecolor='white')
    fig.text(0.5, 0.05, textstr, ha='center', va='top', fontsize=16, bbox=props)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plot_path = os.path.join(plot_dir, 'eto_vars_scatter_plot.png'.format(var))
    plt.savefig(plot_path)
    plt.close()


def plot_resid_corr_heatmap(joined_resid_csv, plot_dir):
    df = pd.read_csv(joined_resid_csv, index_col=0)
    df = df.rename(columns=STR_MAP_SIMPLE)
    corr_matrix = df.corr()

    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='rocket',
                square=True,
                mask=mask,
                cbar=False,
                annot_kws={'size': 16})

    plot_path = os.path.join(plot_dir, 'correlation_matrix_heatmap.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def station_barplot(csv, out_file):
    plt.figure(figsize=(10, 4))

    df = pd.read_csv(csv, index_col=0)
    df = df.rename(columns=STR_MAP_SIMPLE)
    df = df[['VPD', 'Wind Speed', 'Mean Temp', 'Rn', 'sum']]
    colors = sns.color_palette('rocket', n_colors=len(df.columns) - 1)
    fig, ax = plt.subplots()

    bottom = np.zeros(df.shape[0])
    for v, c in zip(df.columns[:4], colors):
        p = ax.bar(df.index, df[v], 0.9, label=v, bottom=bottom, color=c)
        bottom += df[v]

    plt.xlabel('Station', fontsize=24)
    plt.ylabel('Variance Accounted For', fontsize=24)
    # plt.title('Stacked Bar Plot of Measurement Fractions by Station')
    ax.set_xticks([])

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels,
                       loc='lower left', bbox_to_anchor=(0.1, 0),
                       facecolor='white', fontsize=18)
    plt.tight_layout()
    plt.savefig(out_file)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    met_residuals = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residuals.json')

    model_ = 'gridmet'
    residuals = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                             'station_residuals_{}.json'.format(model_))

    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_hist')
    # plot_residual_met_histograms(met_residuals, hist)

    scatter = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_scatter', model_)
    plot_eto_var_scatter_histograms(residuals, scatter)

    heat = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'heatmap')
    # plot_resid_corr_heatmap(joined_resid, heat)

    decomp = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'var_decomp_stations_tprop.csv')
    decomp_plt = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                              'decomp_barplot', 'var_decomp_stations_tprop.png')
    # station_barplot(decomp, decomp_plt)
# ========================= EOF ====================================================================
