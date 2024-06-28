import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis

from nldas_eto_error import LIMITS, STR_MAP

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
sns.set_style("white", {'axes.linewidth': 0.5})


def plot_residual_met_histograms(resids_file, plot_dir):
    with open(resids_file, 'r') as f:
        res_dct = json.load(f)

    for var, residuals in res_dct.items():

        mean_ = np.mean(residuals)
        variance = np.var(residuals).item()
        data_skewness = skew(residuals).item()
        data_kurtosis = kurtosis(residuals).item()

        # Create a histogram plot
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='blue')
        plt.axvline(mean_, color='red', linestyle='dashed', linewidth=1)
        plt.title(f'{STR_MAP[var]} Residuals Histogram')
        plt.xlabel(f'{STR_MAP[var]} Residuals\nObserved minus NLDAS')
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

        plot_path = os.path.join(plot_dir, f'{var}_residuals_histogram.png')
        plt.savefig(plot_path)
        # plt.show()


def plot_residual_scatter_histograms(error_json, plot_dir):

    with open(error_json, 'r') as f:
        meta = json.load(f)

    eto_residuals = []
    first, df = True, None
    for sid, data in meta.items():

        if data == 'exception':
            continue

        _file = data['resid']
        c = pd.read_csv(_file, parse_dates=True, index_col='date')

        eto = c['eto'].copy()
        eto.dropna(how='any', inplace=True)
        eto_residuals += list(eto.values)

        c.dropna(how='any', axis=0, inplace=True)
        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c], ignore_index=True, axis=0)

    print(df.shape[0])

    vars = [c for c in df.columns if c != 'eto']

    for v in vars:
        fig = plt.figure(figsize=(16, 10), dpi=600)
        grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

        ax_main = fig.add_subplot(grid[:-1, :-1])
        ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
        ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

        sns.jointplot(x=df['eto'], y=df[v], kind='scatter', marginal_kws={'kde': True},
                      s=1, color='black')
        ax_main.scatter(df[v].values, df['eto'].values, s=2, alpha=.9,
                        cmap="tab10", edgecolors='gray', linewidths=.5)

        ax_main.scatter([df[v].values.mean()], [df['eto'].values.mean()], s=10, alpha=.9, color='red')

        sns.kdeplot(data=df, ax=ax_right, y='eto')
        ax_right.set_xlabel(None)
        ax_right.set_ylabel(None)
        ax_bottom.invert_yaxis()

        sns.kdeplot(data=df, ax=ax_bottom, x=v)
        ax_bottom.set_xlabel(None)
        ax_bottom.set_ylabel(None)

        plt.axvline(0, color='gray', alpha=0.3)
        plt.axhline(0, color='gray', alpha=0.3)

        # plt.title('Scatter Plot of {} vs. {}'.format('eto', v))

        plot_path = os.path.join(plot_dir, 'eto_{}_scatter_plot.png'.format(v))
        plt.savefig(plot_path)
        print(plot_path)
        plt.close()

    corr_matrix = df.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title('Correlation Matrix Heatmap')
    plot_path = os.path.join(plot_dir, 'correlation_matrix_heatmap.png')
    plt.savefig(plot_path)
    print(plot_path)
    plt.close()

    mean_ = np.array(eto_residuals).mean()
    variance = np.array(eto_residuals).var()

    plt.figure(figsize=(10, 6))
    sns.histplot(eto_residuals, kde=True, color='blue')
    plt.axvline(mean_, color='red', linestyle='dashed', linewidth=1)
    plt.title(f'{STR_MAP["eto"]} Residuals Histogram')
    plt.xlabel(f'{STR_MAP["eto"]} Residuals\nObserved minus NLDAS')
    plt.ylabel('Frequency')
    plt.grid(True)

    textstr = '\n'.join((
        r'$n={:,}$'.format(len(eto_residuals)),
        r'$\mu={:.2f}$'.format(mean_),
        r'$\sigma^2={:.2f}$'.format(variance)))
    props = dict(boxstyle='round', facecolor='white')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)

    plot_path = os.path.join(plot_dir, 'eto_residuals_histogram.png')
    plt.savefig(plot_path)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    met_residuals = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residuals.json')

    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residual_histograms')

    plot_residual_met_histograms(met_residuals, hist)

    error_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_distributions.json')

    plot_residual_scatter_histograms(error_json, hist)

# ========================= EOF ====================================================================
