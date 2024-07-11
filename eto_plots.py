import json
import os
from calendar import month_abbr

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
from scipy.stats import skew, kurtosis

from eto_error import LIMITS, STR_MAP

STR_MAP_SIMPLE = {
    'vpd': r'VPD',
    'rn': r'Rn',
    'mean_temp': r'Mean Temp',
    'wind': r'Wind Speed',
    'eto': r'ETo'
}

UNITS_MAP = {
    'rn': r'[MJ m$^{-2}$ d$^{-1}$]',
    'vpd': r'[kPa]',
    'mean_temp': r'[C]',
    'wind': r'[m s$^{-1}$]',
    'eto': r'[mm day$^{-1}$]'
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

LIMITS = {'vpd': 2.5,
          'rn': 7,
          'wind': 9,
          'mean_temp': 12}


def plot_residuals_comparison_histograms(resids_file1, resids_file2, plot_dir, desc_1='NLDAS-2', desc_2='GridMET'):
    try:
        with open(resids_file1, 'r') as f1, open(resids_file2, 'r') as f2:
            res_dct1 = json.load(f1)
            res_dct2 = json.load(f2)

        all_vars = set(res_dct1.keys()) | set(res_dct2.keys())
        num_vars = len(all_vars)
        num_rows = int(np.ceil(num_vars / 3))

        fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))

        fig.delaxes(axes[1, 0])
        fig.delaxes(axes[1, 2])

        if num_vars % 3 != 0:
            for i in range(num_vars % 3, 3):
                fig.delaxes(axes[-1, i])

        palette = sns.color_palette('rocket', 2)

        first = True
        for i, var in enumerate(STR_MAP_SIMPLE.keys()):
            row = i // 3
            col = i % 3
            ax = axes[row, col]

            residuals1 = res_dct1.get(var, [])
            residuals2 = res_dct2.get(var, [])

            try:
                sns.histplot(residuals1, kde=True, stat='density', color=palette[0],
                             label=f'{STR_MAP_SIMPLE[var]} {desc_1}',
                             ax=ax)
                sns.histplot(residuals2, kde=True, stat='density', color=palette[1],
                             label=f'{STR_MAP_SIMPLE[var]} {desc_2}', ax=ax)

                ax.axvline(np.mean(residuals1), color='blue', linestyle='dashed', linewidth=1)
                ax.axvline(np.mean(residuals2), color='orange', linestyle='dashed', linewidth=1)

                if col == 0 and row == 0:
                    ax.set_xlabel(f'{STR_MAP[var]}\n(Observed minus Gridded)')
                else:
                    ax.set_xlabel(f'{STR_MAP[var]}')

                bbox = [0.02, 0.75, 0.4, 0.2]
                cell_text = create_table_text(residuals1, residuals2, first, desc_1, desc_2)
                table_obj = ax.table(cellText=cell_text, bbox=bbox, colWidths=[1, 2, 2], edges='horizontal')

                ax.set_ylabel('Frequency' if col == 0 else '')
                ax.axvline(0, color='black', linestyle='solid', linewidth=0.7)

                if first:
                    patch1 = mpatches.Patch(color=palette[0], label=desc_1)
                    patch2 = mpatches.Patch(color=palette[1], label=desc_2)
                    ax.legend(handles=[patch1, patch2], loc='upper right')
                    first = False

                if var in LIMITS:
                    ax.set_xlim([-1 * LIMITS[var], LIMITS[var]])

            except (ZeroDivisionError, ValueError, OverflowError) as e:
                print(f"Error processing variable '{var}': {e}")

        plt.tight_layout()

        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        plot_path = os.path.join(plot_dir, f'{desc_1}_{desc_2}_histogram.png')
        plt.savefig(plot_path)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing files: {e}")


def create_table_text(residuals1, residuals2, first, desc_1, desc_2):
    labels = ['', f'{desc_1}', f'{desc_2}']
    if first:
        stats = ['n', 'μ', 'σ²', 'γ₁', 'γ₂']
        data = [[label] + [create_textstr_value(res, stat) for res in [residuals1, residuals2]] for label, stat in
                zip(stats, ['n', 'mean', 'var', 'skew', 'kurtosis'])]
    else:
        stats = ['μ', 'σ²', 'γ₁', 'γ₂']
        data = [[label] + [create_textstr_value(res, stat) for res in [residuals1, residuals2]] for label, stat in
                zip(stats, ['mean', 'var', 'skew', 'kurtosis'])]

    return [labels] + data


def create_textstr_value(residuals, stat_func):
    if residuals:
        if stat_func == 'n':
            return '{:,}'.format(len(residuals))
        else:
            try:
                func = getattr(np, stat_func)
            except AttributeError:
                func = getattr(stats, stat_func)
            return '{:.2f}'.format(func(residuals).item())
    else:
        return 'No Data'


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
        plt.xlabel(f'{STR_MAP_SIMPLE[var]} Residuals\n(Observed minus NLDAS)')
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
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        plot_path = os.path.join(plot_dir, f'{var}_residuals_histogram.png')
        plt.savefig(plot_path)


def plot_eto_var_scatter_histograms(resid_json, plot_dir):
    with open(resid_json, 'r') as f:
        dct = json.load(f)

    vars = [c for c in LIMITS.keys()]

    fig, axes = plt.subplots(2, 2, figsize=(10, 12), dpi=600)
    axes = axes.flatten()

    dct = {k: v for k, v in dct.items() if len(v['eto']) > 0}

    first = True
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

        if first:
            textstr = '\n'.join((
                r'$n={:,}$'.format(eto.shape[0]),
                r'$r^2$: {:.2f}'.format(r_squared)))
            props = dict(boxstyle='round', facecolor='white')
            ax_main.text(0.05, 0.95, textstr, transform=ax_main.transAxes, fontsize=14,
                         verticalalignment='top', bbox=props)

            first = False

        else:
            textstr = r'$r^2$: {:.2f}'.format(r_squared)
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
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plt.savefig(plot_path)
    plt.close()


def plot_resid_corr_heatmap(joined_resid_csv, plot_dir):
    with open(joined_resid_csv, 'r') as f:
        dct = json.load(f)

    df = pd.DataFrame.from_dict(dct, orient='columns')
    df = df.rename(columns=STR_MAP_SIMPLE)
    df = df[['ETo', 'Rn', 'Mean Temp', 'VPD', 'Wind Speed']]
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


def plot_monthly_residuals_by_station(station_residuals_file, out_fig, variable_name='eto'):
    with open(station_residuals_file, 'r') as f:
        data = json.load(f)

    for station_id, station_data in data.items():
        residuals_data = []
        for month, res in station_data.get(variable_name, {}).items():
            if res:
                residuals_data.extend([(month, r) for r in res])

        if residuals_data:
            df = pd.DataFrame(residuals_data, columns=['Month', 'Residual'])
            plt.figure(figsize=(10, 6))

            sns.boxplot(x='Month', y='Residual', data=df, showmeans=True, meanprops={"marker": "o",
                                                                                     "markerfacecolor": "white",
                                                                                     "markeredgecolor": "black",
                                                                                     "markersize": "10"})

            plt.axhline(0, color='red', linestyle='-', linewidth=1)

            plt.ylim(-6, 6)
            plt.ylabel(f"{STR_MAP_SIMPLE[variable_name]} Residual", fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(out_fig, '{}.png'.format(station_id)))
            plt.close()


def plot_monthly_residuals(station_residuals_file, variable_name, output_dir):
    with open(station_residuals_file, 'r') as f:
        data = json.load(f)

    all_data = []
    for station_id, station_data in data.items():
        for month, res in station_data.get(variable_name, {}).items():
            if res:
                mean_val = np.mean(res)
                q25 = np.percentile(res, 25)
                q75 = np.percentile(res, 75)
                all_data.append({'Station': station_id, 'Month': month, 'Mean': mean_val, '25th': q25, '75th': q75})

    df = pd.DataFrame(all_data)

    # growing season stats:
    idx = [i for i, r in df.iterrows() if r['Month'] in [str(m) for m in range(4, 11)]]
    gsdf = df.loc[idx].copy()
    gsdf = gsdf[['Station', 'Mean']].groupby('Station').mean()

    plt.figure(figsize=(15, 6))
    ax = plt.gca()
    # ax.set_xlim([-0.5, 11.5])
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_abbr[1:])

    num_stations = df['Station'].nunique()
    x_offset = 1.0 / num_stations
    start_offset = -0.8 + (0.8 / num_stations) / 2

    for m in range(1, 13):
        vals = df[df['Month'] == str(m)]['Mean'].values
        pos = np.count_nonzero(vals >= 0)
        neg = np.count_nonzero(vals < 0)
        print('Month {}: {} positive bias, {} negative bias: {:.2f} positive'.format(m, pos, neg, pos / (pos + neg)))

    for station in df['Station'].unique():
        station_df = df[df['Station'] == station]

        ax.errorbar(
            station_df['Month'].astype(int) - 0.5 + start_offset,
            station_df['Mean'],
            yerr=[station_df['Mean'] - station_df['25th'], station_df['75th'] - station_df['Mean']],
            fmt='none',
            elinewidth=1,
            ecolor='black',
            capsize=0
        )

        ax.plot(
            station_df['Month'].astype(int) - 0.5 + start_offset,
            station_df['Mean'],
            marker='o', linestyle='none', markersize=2, color='black',
            label=station if start_offset == 0 else ''  # Only label first point for each station
        )

        start_offset += x_offset

    plt.axhline(0, color='red', linestyle='-', linewidth=1)
    plt.ylabel(f"{variable_name} Residual", fontsize=12)
    plt.xlabel("Month", fontsize=12)
    # plt.legend(title='Station', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{variable_name}_all_stations_residuals_boxplot.png'))
    plt.show()


def annual_residuals(station_residuals_file, variable_name, output_dir):
    with open(station_residuals_file, 'r') as f:
        data = json.load(f)

    all_data = []
    for station_id, station_data in data.items():
        res = station_data['eto']
        mean_val = np.mean(res)
        all_data.append({'Station': station_id, 'Mean': mean_val})

    df = pd.DataFrame(all_data)
    pass


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    # GridMET - NLDAS-2 comparison ==============
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_nldas2_south.json')
    res_json2 = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                             'all_residuals_gridmet.json')
    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_hist')
    # plot_residuals_comparison_histograms(res_json, res_json2, hist, desc_1='NLDAS-2', desc_2='GridMET')

    # NDLAS-2 USA - CAN comparison ==============
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_nldas2_south.json')
    res_json2 = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                             'all_residuals_nldas2_north.json')
    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_hist')
    # plot_residuals_comparison_histograms(res_json, res_json2, hist, desc_1='United States', desc_2='Canada')

    model_ = 'nldas2'
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_{}.json'.format(model_))
    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_hist')
    # plot_residual_met_histograms(res_json, hist)

    residuals = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                             'station_residuals_{}.json'.format(model_))
    scatter = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_scatter', model_)
    # plot_eto_var_scatter_histograms(residuals, scatter)

    heat = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'heatmap')
    # plot_resid_corr_heatmap(res_json, heat)

    decomp = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'var_decomp_stations_tprop.csv')
    decomp_plt = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                              'decomp_barplot', 'var_decomp_stations_notprop.png')
    # station_barplot(decomp, decomp_plt)

    sta_res = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                           'station_residuals_{}_month.json'.format(model_))
    whisker = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'box_whisker')
    plot_monthly_residuals(sta_res, 'eto', whisker)

    sta_res = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                           'station_residuals_{}_annual.json'.format(model_))
    # annual_residuals(sta_res, 'eto', None)

# ========================= EOF ====================================================================
