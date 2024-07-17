import json
import os
import warnings
from calendar import month_abbr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from refet import Daily, calcs
from scipy.stats import linregress
from eto_error import _vpd, _rn, station_par_map

warnings.simplefilter(action='ignore', category=FutureWarning)

VAR_MAP = {'rs': 'Rs (w/m2)',
           'ea': 'Compiled Ea (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'mean_temp': 'TAvg (C)',
           'wind': 'ws_2m (m/s)',
           'eto': 'ETo (mm)'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

COMPARISON_VARS = ['vpd', 'rn', 'mean_temp', 'wind', 'eto']

STR_MAP = {
    'rn': r'Net Radiation [MJ m$^{-2}$ d$^{-1}$]',
    'vpd': r'Vapor Pressure Deficit [kPa]',
    'mean_temp': r'Mean Daily Temperature [C]',
    'wind': r'Wind Speed at 2 m [m s$^{-1}$]',
    'eto': r'ASCE Grass Reference Evapotranspiration [mm day$^{-1}$]'
}

LIMITS = {'vpd': 3,
          'rs': 0.8,
          'u2': 12,
          'mean_temp': 12.5,
          'eto': 5}

PACIFIC = pytz.timezone('US/Pacific')


def corrected_eto(stations, station_data, gridded_data, comparison_out, model='nldas2', apply_correction=False):
    kw = station_par_map('agri')
    station_list = pd.read_csv(stations, index_col=kw['index'])

    eto_estimates = {'station': [], model: []}

    for i, (fid, row) in enumerate(station_list.iterrows()):

        try:

            print('{} of {}: {}'.format(i + 1, station_list.shape[0], fid))

            sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(fid))
            sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')
            sdf.rename(columns=RENAME_MAP, inplace=True)
            sdf['doy'] = [i.dayofyear for i in sdf.index]
            sdf['rs'] *= 0.0864
            sdf['vpd'] = sdf.apply(_vpd, axis=1)
            sdf['rn'] = sdf.apply(_rn, lat=row['latitude'], elev=row['elev_m'], zw=row['anemom_height_m'], axis=1)
            sdf = sdf[COMPARISON_VARS]

            grid_file = os.path.join(gridded_data, '{}.csv'.format(fid))
            gdf = pd.read_csv(grid_file, index_col='date_str', parse_dates=True)
            gdf['mean_temp'] = (gdf['min_temp'] + gdf['max_temp']) * 0.5
            gdf = gdf[COMPARISON_VARS]

            if apply_correction:
                for j, m in enumerate(month_abbr[1:], start=1):
                    idx = [i for i in gdf.index if i.month == j]
                    print('\nMean {}, factor: {:.2f}, pre-correction: {:.2f}'.format(m, row[m],
                                                                                     gdf.loc[idx, 'eto'].mean()))
                    gdf.loc[idx, 'eto'] *= row[m]
                    print('Mean {}, factor: {:.2f}, post-correction: {:.2f}'.format(m, row[m],
                                                                                    gdf.loc[idx, 'eto'].mean()))

            s_var, n_var = 'eto_station', 'eto_{}'.format(model)
            df = pd.DataFrame(columns=[s_var], index=sdf.index, data=sdf['eto'].values)
            df['eto_station'] = sdf['eto']
            df.dropna(how='any', axis=0, inplace=True)

            df[n_var] = gdf.loc[df.index, 'eto'].values

            df[f'eto_{model}'] = gdf.loc[df.index, 'eto'].values

            eto_estimates[model].extend(df[f'eto_{model}'].to_list())
            eto_estimates['station'].extend(df['eto_station'].to_list())

        except Exception as e:
            print('Exception raised on {}, {}'.format(fid, e))

    if apply_correction:
        comparison_out = comparison_out.replace('.json', '_corrected.json')
    else:
        comparison_out = comparison_out.replace('.json', '_uncorrected.json')

    with open(comparison_out, 'w') as dst:
        json.dump(eto_estimates, dst, indent=4)


def plot_corrected(eto_uncorr, eto_corr, plot_dir, palette_idx=(8, 2)):
    palette = sns.color_palette('rocket', 10)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=600)

    x_labels = [r'Uncorrected NLDAS-2 ETo [mm day$^{-1}$]', r'Bias-Corrected NLDAS-2 ETo [mm day$^{-1}$]']
    keys = ['nldas2', 'nldas2']

    with open(eto_uncorr, 'r') as f1, open(eto_corr, 'r') as f2:
        eto_data1 = json.load(f1)
        eto_data2 = json.load(f2)

    for i, (eto_data, ax, label) in enumerate(zip([eto_data1, eto_data2], [axes[-2], axes[-1]], x_labels)):
        x = eto_data['station']
        y = eto_data[keys[i]]
        rmse = np.sqrt(np.mean((np.array(x) - np.array(y))**2))

        slope, intercept, r_value, _, _ = linregress(x, y)
        r_squared = r_value ** 2

        sns.scatterplot(x=x, y=y, color=palette[palette_idx[i]], s=2, alpha=.9, ax=ax)

        stats_text = (f"n: {len(x)}\nR²: {r_squared:.2f}\nSlope: {slope:.2f}\nIntercept:"
                      f" {intercept:.2f}\nRMSE: {rmse:.2f} [mm day$^{-1}$]")

        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', linewidth=1),
            fontsize=12
        )

        min_x = np.min(x)
        max_x = np.max(x)
        regression_line_y = slope * np.array([min_x, max_x]) + intercept
        ax.plot([min_x, max_x], regression_line_y, linestyle='--', color=palette[palette_idx[i]])
        ax.set_facecolor("white")
        ax.set_xlabel("Station ETo [mm day$^{-1}$]")
        ax.set_ylabel(label)

        ax.plot([0.0, 12.0], [0.0, 12.0], 'k--', lw=1)

    plt.tight_layout()

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(plot_dir, f'corrected_comparison_scatter.png')
    plt.savefig(plot_path)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    # pandarallel.initialize(nb_workers=4)

    station_meta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                                   'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')

    sta_data = os.path.join(d, 'weather_station_data_processing', 'corrected_data')

    model_ = 'nldas2'
    grid_data = os.path.join(d, 'weather_station_data_processing', 'gridded', model_)
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_{}.json'.format(model_))
    sta_res = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                           'station_residuals_{}.json'.format(model_))

    comparison_js = os.path.join(d, 'weather_station_data_processing', 'comparison_data', 'eto_{}.json'.format(model_))

    # corrected_eto(station_meta, sta_data, grid_data, comparison_js, apply_correction=True)

    eto_json = os.path.join(d, 'weather_station_data_processing', 'comparison_data',
                            'eto_{}_uncorrected.json'.format(model_))

    eto_json2 = os.path.join(d, 'weather_station_data_processing', 'comparison_data',
                             'eto_{}_corrected.json'.format(model_))

    plot = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'uncorr_vs_corrected_eto')

    plot_corrected(eto_json, eto_json2, plot)

# ========================= EOF ====================================================================
