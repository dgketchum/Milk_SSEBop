import datetime
import json
import os
import warnings
from pprint import pprint

import torch

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandarallel import pandarallel
from refet import Daily
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer
from statsmodels.tsa.stattools import ccf

from eto_error import COMPARISON_VARS
from eto_error import station_par_map

warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mc_timeseries_draw(station_meta, gridded, outfile, residuals_dir, station_type='ec', num_samples=1000):
    kw = station_par_map(station_type)

    results, metadata = {}, None

    station_list = pd.read_csv(station_meta, index_col=kw['index']).sample(n=3)

    for j, (station, row) in enumerate(station_list.iterrows()):

        print('\n{} of {}: {}'.format(j + 1, station_list.shape[0], station))

        file_ = os.path.join(gridded, '{}.csv'.format(station))

        gdf = pd.read_csv(file_, parse_dates=True, index_col='date_str')

        gdf = gdf.loc['2018-01-01': '2019-12-31']

        gdf['doy'] = [i.dayofyear for i in gdf.index]
        gdf['year'] = [i.year for i in gdf.index]
        gdf['mean_temp'] = (gdf['min_temp'] + gdf['max_temp']) / 2
        metvars = COMPARISON_VARS[:4]
        gdf = gdf[metvars + ['eto', 'doy', 'year']]

        doy_medians = gdf.groupby('doy')[metvars].median()
        doy_medians = doy_medians.to_dict()

        anomalies = gdf.copy()
        for var in metvars:
            anomalies[var] = gdf[var].subtract(gdf['doy'].map(doy_medians[var]))

        if j == 0:
            metadata = Metadata.detect_from_dataframes(data={'met_data': gdf[metvars + ['year']]})
            metadata.update_column('year', sdtype='id')
            metadata.set_sequence_key('year')
            metadata.validate()

        resid_file = os.path.join(residuals_dir, 'res_{}.csv'.format(station))
        if not os.path.isfile(resid_file):
            continue

        # res_df = pd.read_csv(resid_file, parse_dates=True, index_col='date')
        # res_df.index = [datetime.date(i.year, i.month, i.day) for i in res_df.index]
        # res_df.dropna(how='any', axis=0, inplace=True)
        # res_df = res_df.rename(columns={'u2': 'wind', 'tmean': 'mean_temp'})
        # eto_arr = gdf.loc[res_df.index, 'eto'].values

        result = {k: [] for k in metvars}

        synthesizer = PARSynthesizer(metadata, cuda=True, verbose=True)
        synthesizer.fit(anomalies[metvars + ['year']])

        for i in range(num_samples):

            synthetic_anomalies = synthesizer.sample(num_sequences=1, sequence_length=gdf.shape[0])
            synthetic_anomalies['doy'] = gdf['doy']
            synthetic_anomalies.index = gdf.index

            sim_df = gdf.copy()
            sim_df[metvars] += synthetic_anomalies[metvars]

            if i == 0:
                for var in metvars:
                    sim_db = sm.stats.durbin_watson(sim_df[var])
                    obs_db = sm.stats.durbin_watson(gdf[var])
                    print('Durbin-Watson {}: obs {:.2f}, sim {:.2f}'.format(var, obs_db, sim_db))

                    gdf_mean, sim_mean = gdf[var].mean(), sim_df[var].mean()
                    print('Mean Residuals {}: obs {:.2f}, sim {:.2f}'.format(var, gdf_mean, sim_mean))

            for var in metvars:
                eto_sim = perturbed_nldas.parallel_apply(calc_eto, mod_var=var,
                                                         mod_vals=perturbed_nldas[var].values, axis=1).values
                res = eto_sim - eto_arr
                variance = np.var(res, ddof=1)
                result[var].append((res.mean(), variance))

                mean_ = res.mean()
                if i == 0:
                    print('Mean ETo residual for {}: {:.2f}\n'.format(var, mean_))

        results[station] = result

    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)


def cross_autocorrelation_multivariate(df, lags=10, demean=True):
    if not isinstance(df, pd.DataFrame):
        return {}

    if df.empty:
        return {}

    series_names = df.columns
    n_series = len(series_names)
    results = {}

    for i in range(n_series):
        for j in range(n_series):
            name_i = series_names[i]
            name_j = series_names[j]
            series_i = df[name_i]
            series_j = df[name_j]

            if demean:
                series_i = series_i - series_i.mean()
                series_j = series_j - series_j.mean()

            correlation = ccf(series_i, series_j, adjusted=False)[:lags + 1]
            lags_used = np.arange(lags + 1)

            results[(name_i, name_j)] = pd.DataFrame(index=lags_used, data={"correlation": correlation})

    return results


def variance_decomposition(sim_results, station_meta, decomp_out, station_type='ec'):
    kw = station_par_map(station_type)

    with open(sim_results, 'r') as f:
        sim_results = json.load(f)

    metvars = COMPARISON_VARS[:4]
    station_list = pd.read_csv(station_meta, index_col=kw['index'])
    var_sums = {k: 0. for k in metvars}
    all = 0.0

    df = pd.DataFrame(index=list(sim_results.keys()), columns=metvars)

    for j, (station, row) in enumerate(station_list.iterrows()):

        if station not in sim_results.keys():
            continue

        station_var, station_sum = {}, 0.0

        for var in metvars:
            try:

                sum_var = sum(np.array([i[1] for i in sim_results[station][var]]))

                station_var[var] = sum_var
                df.loc[station, var] = sum_var
                station_sum += sum_var

                var_sums[var] += sum_var
                all += sum_var

            except KeyError:
                print('Error: {}'.format(station))

        df.loc[station, 'sum'] = station_sum

    df.dropna(how='any', axis=0, inplace=True)
    df = df.div(df['sum'], axis=0)
    df.to_csv(decomp_out)
    print(decomp_out)

    decomp = {k: '{:.2f}'.format(v / all) for k, v in var_sums.items()}
    pprint('summary variance decomposition: {}'.format(decomp))


def calc_eto(r, mod_var, mod_vals):
    # modify the error-perturbed values with setattr
    asce = Daily(
        tmin=r['min_temp'],
        tmax=r['max_temp'],
        ea=r['ea'],
        rs=r['rsds'] * 0.0864,
        uz=r['wind'],
        zw=10.0,
        doy=r['doy'],
        elev=row[kw['elev']],
        lat=row[kw['lat']])

    setattr(asce, mod_var, mod_vals)

    _eto = asce.eto()[0]

    return _eto


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    sta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                          'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')

    model_ = 'nldas2'
    grid_data = os.path.join(d, 'weather_station_data_processing', 'gridded', model_)
    sta_res = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                           'station_residuals_{}.json'.format(model_))

    pandarallel.initialize(nb_workers=10)

    num_sampl_ = 100
    variance_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                                 'eto_variance_{}.json'.format(num_sampl_))

    res_file_dir = os.path.join(d, 'weather_station_data_processing', 'comparison_data')
    mc_timeseries_draw(sta, grid_data, variance_json, res_file_dir, station_type='agri',
                       num_samples=num_sampl_)

    decomp = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'var_decomp_stations_notprop.csv')
    variance_decomposition(variance_json, sta, decomp, station_type='agri')

# ========================= EOF ====================================================================
