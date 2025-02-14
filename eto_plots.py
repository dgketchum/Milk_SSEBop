import json
import os
from calendar import month_abbr, month_name
from datetime import datetime
from matplotlib.gridspec import GridSpec
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


def plot_eto_histogram(res_data, plot_dir, desc='NLDAS-2'):
    with open(res_data, 'r') as f1:
        res_dct = json.load(f1)

    resids = res_dct.get('eto', [])

    plt.figure(figsize=(14, 7))

    ax = plt.subplot()

    sns.kdeplot(resids, color='red', common_norm=False)

    sns.histplot(resids, kde=False, stat='density', color='grey',
                 label=f'{desc} ETo', ax=ax)

    ax.axvline(np.mean(resids), color='red', linestyle='dashed', linewidth=1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(True, color='lightgrey', linestyle='-', linewidth=0.5, axis='both')
    ax.set_xlabel(f'ETo Residuals\n(Observed minus NLDAS-2)')

    stats = ['n', 'μ', 'σ²', 'γ₁', 'γ₂']
    data = [[label] + [create_textstr_value(resids, stat)] for label, stat in zip(stats,
                                                                                  ['n', 'mean', 'var', 'skew',
                                                                                   'kurtosis'])]
    stats_text = '\n'.join([f'{label} = {value}' for label, value in data])

    ax.text(0.05, 0.8, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=14)

    plt.xlabel('ETo Residuals [mm day$^{-1}$]\n(Observed minus NLDAS-2)')
    plt.ylabel('Frequency')
    plt.xlim([-5, 5])
    plt.tight_layout()

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plot_path = os.path.join(plot_dir, f'{desc}_eto_histogram.png')
    plt.savefig(plot_path)
    # plt.show()


def plot_residuals_comparison_histograms(resids_file1, resids_file2, eto_file1, eto_file2, plot_dir,
                                         palette_idx=(0, 1), desc_1='NLDAS-2', desc_2='GridMET', extra_desc=None,
                                         baseline_estimate=None):
    try:
        with open(resids_file1, 'r') as f1, open(resids_file2, 'r') as f2:
            res_dct1 = json.load(f1)
            res_dct2 = json.load(f2)

        with open(eto_file1, 'r') as f1, open(eto_file2, 'r') as f2:
            eto_data1 = json.load(f1)
            eto_data2 = json.load(f2)

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax4 = plt.subplot(gs[1, 0])
        ax5 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[1:, 1])
        ax7 = plt.subplot(gs[1:, 2])

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        letters = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.']

        palette = sns.color_palette('rocket', 10)

        first = True
        for i, var in enumerate(STR_MAP_SIMPLE.keys()):
            ax = axes[i]

            residuals1 = res_dct1.get(var, [])
            residuals2 = res_dct2.get(var, [])

            try:
                sns.histplot(residuals1, kde=True, stat='density', color=palette[palette_idx[0]],
                             label=f'{STR_MAP_SIMPLE[var]}',
                             ax=ax)
                sns.histplot(residuals2, kde=True, stat='density', color=palette[palette_idx[1]],
                             label=f'{STR_MAP_SIMPLE[var]}', ax=ax)

                ax.axvline(np.mean(residuals1), color=palette[palette_idx[0]], linestyle='dashed', linewidth=1)
                ax.axvline(np.mean(residuals2), color=palette[palette_idx[1]], linestyle='dashed', linewidth=1)

                if baseline_estimate is None:
                    baseline_estimate = 'Gridded'

                if i == 0:
                    ax.set_xlabel(f'{STR_MAP[var]} Residuals\n(Observed minus {baseline_estimate})')
                elif var == 'eto':
                    ax.set_xlabel(f'{STR_MAP[var]}\nResiduals')
                else:
                    ax.set_xlabel(f'{STR_MAP[var]} Residuals')

                bbox = [0.02, 0.5, 0.3, 0.33]
                cell_text = create_table_text(residuals1, residuals2, first, desc_1, desc_2)
                ax.table(cellText=cell_text, bbox=bbox, colWidths=[1, 2, 2], edges='horizontal')

                ax.text(0.015, 0.95, letters[i],
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor=None, linewidth=1),
                        fontsize=12)

                ax.set_ylabel('Density' if i == 0 else '')
                ax.axvline(0, color='black', linestyle='solid', linewidth=0.7)

                if first:
                    patch1 = mpatches.Patch(color=palette[palette_idx[0]], label=desc_1)
                    patch2 = mpatches.Patch(color=palette[palette_idx[1]], label=desc_2)
                    ax.legend(handles=[patch1, patch2], loc='upper right')
                    first = False

                if var in LIMITS:
                    ax.set_xlim([-1 * LIMITS[var], LIMITS[var]])

            except (ZeroDivisionError, ValueError, OverflowError) as e:
                print(f"Error processing variable '{var}': {e}")

        if desc_2 == 'GridMET':
            x_labels = [r'NLDAS-2 ETo [mm day$^{-1}$]', r'GridMET ETo [mm day$^{-1}$]']
            keys = ['nldas2', 'gridmet']
        elif desc_2 == 'Winter':
            x_labels = [r'Summer (JJA) NLDAS-2 ETo [mm day$^{-1}$]', r'Winter (DJF) NLDAS-2 ETo [mm day$^{-1}$]']
            keys = ['nldas2', 'nldas2']
        else:
            x_labels = [r'United States ETo [mm day$^{-1}$]', r'Canada ETo [mm day$^{-1}$]']
            keys = ['nldas2', 'nldas2']

        for i, (eto_data, ax, label) in enumerate(zip([eto_data1, eto_data2], [axes[-2], axes[-1]], x_labels)):
            x = eto_data['station']
            y = eto_data[keys[i]]
            slope, intercept, r_value, _, _ = linregress(x, y)
            r_squared = r_value ** 2

            sns.scatterplot(x=x, y=y, color=palette[palette_idx[i]], s=2, alpha=.9, ax=ax)

            stats_text = f"R²: {r_squared:.2f}\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}"

            ax.text(
                0.05, 0.93, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', linewidth=1),
                fontsize=12
            )

            ax.text(0.03, 0.98, letters[i + 5],
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor=None, linewidth=1),
                    fontsize=12)

            min_x = np.min(x)
            max_x = np.max(x)
            regression_line_y = slope * np.array([min_x, max_x]) + intercept
            ax.plot([min_x, max_x], regression_line_y, linestyle='--',
                    color=palette[palette_idx[i]], label='Line of Best Fit')
            ax.plot([0.0, 12.0], [0.0, 12.0], 'k--', lw=1, label='1:1 Line')
            ax.set_facecolor("white")
            ax.set_xlabel("Station ETo [mm day$^{-1}$]")
            ax.set_ylabel(label)
            ax.legend()

        plt.tight_layout()

        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        if extra_desc:
            plot_path = os.path.join(plot_dir, f'{desc_1}_{desc_2}_{extra_desc}_histogram.png')
        else:
            plot_path = os.path.join(plot_dir, f'{desc_1}_{desc_2}_histogram.png')

        # plt.show()
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


def eto_vars_scatter_plot(resid_json, plot_dir):
    with open(resid_json, 'r') as f:
        dct = json.load(f)

    vars = [c for c in LIMITS.keys()]

    fig, axes = plt.subplots(2, 2, figsize=(10, 13))
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
                f'n = {eto.shape[0]:,}',
                f'$\mathrm{{r^2}}$: {r_squared:.2f}'
            ))
            props = dict(boxstyle='round', facecolor='white')
            ax_main.text(0.05, 0.95, textstr, transform=ax_main.transAxes, fontsize=14,
                         verticalalignment='top', bbox=props)
            first = False

        else:
            textstr = f'$\mathrm{{r^2}}$: {r_squared:.2f}'
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


def plot_resid_corr_heatmap(joined_resid_json, resid_csv_dir, plot_dir):
    with open(joined_resid_json, 'r') as f_json:
        dct_json = json.load(f_json)
    df_json = pd.DataFrame.from_dict(dct_json, orient='columns')
    df_json = df_json.rename(columns=STR_MAP_SIMPLE)

    csv_files = [f for f in os.listdir(resid_csv_dir) if f.endswith('.csv')]
    df_csv_list = []
    for file in csv_files:
        filepath = os.path.join(resid_csv_dir, file)
        df_csv_list.append(pd.read_csv(filepath))

    df_csv = pd.concat(df_csv_list, ignore_index=True)
    df_csv['mean_temp'] = 0.5 * (df_csv['min_temp'] + df_csv['max_temp'])
    df_csv = df_csv.rename(columns=STR_MAP_SIMPLE)
    df_csv = df_csv[[v for k, v in STR_MAP_SIMPLE.items()]]

    corr_matrix_json = df_json[['ETo', 'Rn', 'Mean Temp', 'VPD', 'Wind Speed']].corr()
    corr_matrix_csv = df_csv[['ETo', 'Rn', 'Mean Temp', 'VPD', 'Wind Speed']].corr()

    joined_corr_matrix = corr_matrix_csv.values.copy()
    ur = np.triu_indices_from(joined_corr_matrix)
    joined_corr_matrix[ur] = corr_matrix_json.values[ur]
    id_ = np.diag_indices(joined_corr_matrix.shape[0])
    joined_corr_matrix[id_] = np.nan

    plt.figure(figsize=(12, 12))
    sns.heatmap(joined_corr_matrix, annot=True, fmt=".2f", cmap="rocket", square=True, mask=None,
                cbar=False, annot_kws={'size': 16})
    variables = list(corr_matrix_json.columns)
    plt.xticks(np.arange(len(variables)) + 0.5, labels=variables, rotation=45)
    plt.yticks(np.arange(len(variables)) + 0.5, labels=variables, rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'correlation_matrix_heatmap.png')
    plt.savefig(plot_path)
    a = 1


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


def plot_monthly_residuals(monthly_residuals, daily_residuals, variable_name, output_dir):
    with open(monthly_residuals, 'r') as f:
        monthly_data = json.load(f)

    with open(daily_residuals, 'r') as f_eto:
        daily_data = json.load(f_eto)

    all_data = []
    for station_id, station_data in monthly_data.items():
        for month, res in station_data.get(variable_name, {}).items():
            if res:
                mean_val = np.mean(res)
                q25 = np.percentile(res, 25)
                q75 = np.percentile(res, 75)
                all_data.append({'Station': station_id, 'Month': month, 'Mean': mean_val, '25th': q25, '75th': q75})

    df = pd.DataFrame(all_data)

    month_to_doy = {1: 15, 2: 46, 3: 75, 4: 105, 5: 135, 6: 166, 7: 196, 8: 227, 9: 258, 10: 288, 11: 319, 12: 349}
    doy_to_month = {v: k for k, v in month_to_doy.items()}

    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.set_xticks(range(1, 366))

    ticks = []
    for doy in range(1, 366):
        if doy in doy_to_month.keys():
            ticks.append(month_abbr[doy_to_month[doy]])
        else:
            ticks.append('')

    ax.set_xticklabels(ticks)

    num_stations = df['Station'].nunique()
    x_offset = 366 / 12 / num_stations
    start_offset = -12 + (0.8 / num_stations) / 2

    for m in range(1, 13):
        vals = df[df['Month'] == str(m)]['Mean'].values
        pos = np.count_nonzero(vals >= 0)
        neg = np.count_nonzero(vals < 0)
        print('Month {}: {} positive bias, {} negative bias: {:.2f} positive'.format(m, pos, neg, pos / (pos + neg)))

    for station in df['Station'].unique():
        station_df = df[df['Station'] == station]
        station_df['DOY'] = station_df['Month'].astype(int).map(month_to_doy)

        ax.errorbar(
            station_df['DOY'] - 0.5 + start_offset,
            station_df['Mean'],
            yerr=[station_df['Mean'] - station_df['25th'], station_df['75th'] - station_df['Mean']],
            fmt='none',
            elinewidth=1,
            ecolor='black',
            capsize=0
        )
        ax.plot(
            station_df['DOY'] - 0.5 + start_offset,
            station_df['Mean'],
            marker='o', linestyle='none', markersize=2, color='black',
            label='ETo Residual'
        )

        start_offset += x_offset

    eto_by_doy = {d: [] for d in range(1, 366)}
    for doy in range(1, 366):
        for sid, data in daily_data.items():
            eto = [b for a, b, c in zip(*data['eto']) if c == doy]
            eto_by_doy[doy].extend(eto)

    eto_medians = [np.nanmedian(eto_by_doy[doy]) for doy in range(1, 366)]
    doys = list(range(1, 366))

    ax.plot(
        doys,
        eto_medians,
        color="red",
        linewidth=1.5,
        label="Median ETo Daily Residual"
    )

    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel("Month", fontsize=14)
    plt.ylabel(f"ETo Residual", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{variable_name}_all_stations_residuals_boxplot.png'))


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    # All NLDAS-2 residual histogram ===============================================================
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_nldas2.json')

    eto_json = os.path.join(d, 'weather_station_data_processing', 'comparison_data',
                            'eto_all_nldas2.json')

    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residual_histograms')

    # plot_eto_histogram(res_json, hist, desc='NLDAS-2')

    # GridMET - NLDAS-2 comparison ===============================================================
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_nldas2_south.json')

    res_json2 = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                             'all_residuals_gridmet.json')

    eto_json = os.path.join(d, 'weather_station_data_processing', 'comparison_data',
                            'eto_all_nldas2_south.json')

    eto_json2 = os.path.join(d, 'weather_station_data_processing', 'comparison_data',
                             'eto_all_gridmet.json')

    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_hist')

    # plot_residuals_comparison_histograms(res_json, res_json2, eto_json, eto_json2, hist,
    #                                      desc_1='NLDAS-2', desc_2='GridMET', palette_idx=(4, 6),
    #                                      baseline_estimate='Gridded', extra_desc=None)

    # NDLAS-2 USA - CAN comparison ===============================================================
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_nldas2_south.json')

    res_json2 = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                             'all_residuals_nldas2_north.json')

    eto_json = os.path.join(d, 'weather_station_data_processing', 'comparison_data',
                            'eto_all_nldas2_south.json')

    eto_json2 = os.path.join(d, 'weather_station_data_processing', 'comparison_data',
                             'eto_all_nldas2_north.json')

    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_hist')

    # plot_residuals_comparison_histograms(res_json, res_json2, eto_json, eto_json2, hist,
    #                                      desc_1='USA', desc_2='CAN', palette_idx=(2, 8),
    #                                      baseline_estimate='NLDAS-2')

    # Summer Winter NLDAS-2 comparison ===============================================================
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_nldas2_summer.json')

    res_json2 = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                             'all_residuals_nldas2_winter.json')

    eto_json = os.path.join(d, 'weather_station_data_processing', 'comparison_data',
                            'eto_all_nldas2_summer.json')

    eto_json2 = os.path.join(d, 'weather_station_data_processing', 'comparison_data',
                             'eto_all_nldas2_winter.json')

    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_hist')

    # plot_residuals_comparison_histograms(res_json, res_json2, eto_json, eto_json2, hist,
    #                                      desc_1='Summer', desc_2='Winter', palette_idx=(7, 1),
    #                                      baseline_estimate='NLDAS-2')

    # ETo - Met Varaible scatter shows error correlation ==========================================
    model_ = 'nldas2'
    residuals = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                             'station_residuals_{}.json'.format(model_))
    scatter = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_resid_scatter', model_)
    # eto_vars_scatter_plot(residuals, scatter)

    # Data and Residual correlation ===============================================================
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_nldas2.json')
    gridded = os.path.join(d, 'weather_station_data_processing', 'gridded', 'nldas2')
    heat = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'heatmap')
    # plot_resid_corr_heatmap(res_json, gridded, heat)

    # Varince Decomposition Barplot  ===============================================================
    decomp = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'var_decomp_par.csv')
    decomp_plt = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                              'decomp_barplot', 'var_decomp_stations_par.png')
    station_barplot(decomp, decomp_plt)

    # Monthly Station Residual, median daily  ===============================================================
    monthly_ = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'station_residuals_{}_month.json'.format(model_))

    daily_ = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                          'station_residuals_{}.json'.format(model_))

    whisker = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'box_whisker')
    # plot_monthly_residuals(monthly_, daily_, 'eto', whisker)

    # Count pos/neg mean resid annually, no plot  ===============================================================
    sta_res = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                           'station_residuals_{}_annual.json'.format(model_))
    # annual_residuals(sta_res, 'eto', None)

# ========================= EOF ====================================================================
