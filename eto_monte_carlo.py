import datetime
import json
import os
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from refet import Daily
from scipy.stats import rankdata

from eto_error import COMPARISON_VARS
from eto_error import station_par_map

warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mc_timeseries_draw(station_meta, gridded, outdir, residuals_dir, station_type='ec', num_samples=1000,
                       overwrite=False):
    kw = station_par_map(station_type)

    station_list = pd.read_csv(station_meta, index_col=kw['index'])

    for j, (station, row) in enumerate(station_list.iterrows()):

        # if station not in ['bftm', 'comt', 'bfam']:
        #     continue

        outfile = os.path.join(outdir, f'eto_variance_{num_samples}_{station}.json')

        if os.path.exists(outfile) and not overwrite:
            print(f'{station} file exists, skipping')
            continue

        print('\n{} of {}: {}'.format(j + 1, station_list.shape[0], station))

        file_ = os.path.join(gridded, '{}.csv'.format(station))

        gdf = pd.read_csv(file_, parse_dates=True, index_col='date_str')

        gdf['doy'] = [i.dayofyear for i in gdf.index]
        gdf['year'] = [i.year for i in gdf.index]
        gdf['mean_temp'] = (gdf['min_temp'] + gdf['max_temp']) / 2
        metvars = COMPARISON_VARS[:4]

        resid_file = os.path.join(residuals_dir, 'res_{}.csv'.format(station))
        if not os.path.isfile(resid_file):
            continue

        res_df = pd.read_csv(resid_file, parse_dates=True, index_col='date')
        res_df.index = pd.DatetimeIndex([datetime.date(i.year, i.month, i.day) for i in res_df.index])
        res_df.dropna(how='any', axis=0, inplace=True)
        res_df['year'] = res_df.index.year
        res_df = res_df.rename(columns={'u2': 'wind', 'tmean': 'mean_temp'})
        residuals = res_df[metvars].values

        match_idx = [i for i in res_df.index if i in gdf.index]
        eto_arr = gdf.loc[match_idx, 'eto'].values

        gdf = gdf.loc[match_idx]
        doy = gdf.index.dayofyear

        # this ordering is critical
        addnl_vars = ['ea', 'rsds', 'eto']
        model_estimates = gdf[metvars + addnl_vars].values

        result = {k: [] for k in metvars}
        result['stats'] = {var: {} for var in metvars}

        pert_data, unmod_data = correlated_residual_sampling(residuals, model_estimates, num_samples)

        for var_idx, var in enumerate(metvars):

            sim = np.zeros((num_samples, 1 + len(metvars) + len(addnl_vars)))

            sim_idx = np.random.choice(np.arange(num_samples), size=num_samples, replace=True)
            date_idx = np.random.choice(np.arange(residuals.shape[0]), size=num_samples, replace=True)

            for e, (i, j) in enumerate(zip(sim_idx, date_idx)):
                sim[e, 0] = doy[j]
                sim[e, 1:] = unmod_data[i, j, :]
                sim[e, var_idx + 1] = pert_data[i, j, var_idx]

            gdf_mean, sim_mean = gdf[var].mean(), sim[:, var_idx + 1].mean()

            print('Mean Residuals {}: obs {:.2f}, sim {:.2f}'.format(var, gdf_mean, sim_mean))
            result['stats'][var] = {'obs_mean_res': [], 'sim_mean_res': []}

            result['stats'][var]['obs_mean_res'].append(gdf_mean)
            result['stats'][var]['sim_mean_res'].append(sim_mean)

            sim_df = pd.DataFrame(data=sim, columns=['doy'] + metvars + addnl_vars)

            # duplicate 'mean_temp' to 'min_temp' and 'max_temp' for convenience
            sim_df['min_temp'] = sim_df['mean_temp'].copy()
            sim_df['max_temp'] = sim_df['mean_temp'].copy()

            eto_sim = sim_df.apply(calc_eto, mod_var=var, elev=row[kw['elev']], lat=row[kw['lat']],
                                   mod_vals=sim_df[var].values, axis=1).values

            sim_df['eto_sim'] = eto_sim

            sim_df['res'] = sim_df['eto'] - sim_df['eto_sim']

            res = sim_df['res'].values

            variance = np.var(res, ddof=1)
            result[var].append((res.mean(), variance))

            mean_ = res.mean()
            print('Mean ETo residual for {}: {:.2f}\n'.format(var, mean_))

        with open(outfile, 'w') as f:
            json.dump(result, f, indent=4)


def correlated_residual_sampling(residuals, model_estimates, n_samples):
    """"""
    n_time, n_vars = residuals.shape

    perturbed_estimates = np.zeros((n_samples, n_time, model_estimates.shape[1]))
    unperturbed_estimates = np.zeros((n_samples, n_time, model_estimates.shape[1]))

    correlated_residual_samples = empirical_copula_sample(residuals, n_samples)
    for i in range(n_samples):
        for j in range(n_vars):
            perturbed_data = model_estimates.copy()

            unperturbed_estimates[i, :, :] = perturbed_data

            perturbed_data[:, j] = model_estimates[:, j] + correlated_residual_samples[i, j]
            perturbed_estimates[i, :, :] = perturbed_data

    return perturbed_estimates, unperturbed_estimates


def empirical_copula_sample(data, n_samples):
    n_time_steps, n_variables = data.shape
    uniform_data = np.zeros_like(data, dtype=float)
    for j in range(n_variables):
        uniform_data[:, j] = rankdata(data[:, j]) / (n_time_steps + 1)
    sample_indices = np.random.choice(n_time_steps, size=n_samples, replace=True)
    uniform_samples = uniform_data[sample_indices, :]
    samples = np.zeros_like(uniform_samples)
    for j in range(n_variables):
        for i in range(n_samples):
            idx = np.argmin(np.abs(rankdata(data[:, j]) / (n_time_steps + 1) - uniform_samples[i, j]))
            samples[i, j] = data[idx, j]
    return samples


def variance_decomposition(sim_dir, station_meta, decomp_out, station_type='ec', n_sample=100):
    kw = station_par_map(station_type)

    metvars = COMPARISON_VARS[:4]
    station_list = pd.read_csv(station_meta, index_col=kw['index'])
    var_sums = {k: 0. for k in metvars}
    all = 0.0

    df = pd.DataFrame(index=list(station_list.index), columns=metvars)

    for j, (station, row) in enumerate(station_list.iterrows()):

        sim_results = os.path.join(sim_dir, f'eto_variance_{n_sample}_{station}.json')
        if os.path.exists(sim_results):
            with open(sim_results, 'r') as f:
                sim_results = json.load(f)
        else:
            print(f'{os.path.basename(sim_results)} does not exist, skipping')
            continue

        station_var, station_sum = {}, 0.0

        for var in metvars:
            try:

                sum_var = sum(np.array([i[1] for i in sim_results[var]]))

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


def calc_eto(r, mod_var, mod_vals, elev, lat):
    # modify the error-perturbed values with setattr
    asce = Daily(
        tmin=r['min_temp'],
        tmax=r['max_temp'],
        ea=r['ea'],
        rs=r['rsds'] * 0.0864,
        uz=r['wind'],
        zw=10.0,
        doy=r['doy'],
        elev=elev,
        lat=lat)

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

    # pandarallel.initialize(nb_workers=10)

    num_sampl_ = 10000
    variance_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'mc_par_variance')
    res_file_dir = os.path.join(d, 'weather_station_data_processing', 'comparison_data')
    mc_timeseries_draw(sta, grid_data, variance_json, res_file_dir, station_type='agri',
                       num_samples=num_sampl_, overwrite=True)

    decomp = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'var_decomp_par.csv')
    variance_decomposition(variance_json, sta, decomp, station_type='agri', n_sample=num_sampl_)

# ========================= EOF ====================================================================
