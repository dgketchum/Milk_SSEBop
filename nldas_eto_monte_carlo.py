import datetime
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.fft import fft, ifft
from pandarallel import pandarallel
from refet import Daily
from statsmodels.tsa.stattools import acf

from nldas_eto_sensitivity import COMPARISON_VARS
from nldas_eto_sensitivity import station_par_map

warnings.simplefilter(action='ignore', category=FutureWarning)


def mc_timeseries_draw(json_file, station_meta, outfile, station_type='ec', num_samples=1000):
    kw = station_par_map(station_type)

    with open(json_file, 'r') as f:
        error_distributions = json.load(f)

    results = {}

    station_list = pd.read_csv(station_meta, index_col=kw['index'])

    first, out_vars = True, []

    for j, (station, row) in enumerate(station_list.iterrows()):

        errors = error_distributions[station]
        if errors == 'exception':
            print('Skipping station {} due to previous exception.'.format(station))
            continue

        print('\n{} of {}: {}'.format(j + 1, len(station_list.keys()), station))

        file_ = errors.pop('file')
        resid_file = errors.pop('resid')
        if not os.path.exists(file_):
            file_ = file_.replace('/media/research', '/home/dgketchum/data')
            resid_file = resid_file.replace('/media/research', '/home/dgketchum/data')

        nldas = pd.read_csv(file_, parse_dates=True, index_col='date')
        nldas.index = pd.DatetimeIndex([i.strftime('%Y-%m-%d') for i in nldas.index])

        station_results = {var: [] for var in COMPARISON_VARS}

        cross_corr_obs = np.corrcoef(nldas[COMPARISON_VARS[:4]].values, rowvar=False)
        auto_corrs_obs = [acf(nldas[COMPARISON_VARS[:4]].values[:, i]) for i in range(4)]

        def calc_eto(r, mod_var, mod_vals):
            # modify the error-perturbed values with setattr
            asce = Daily(
                tmin=r['min_temp'],
                tmax=r['max_temp'],
                ea=r['ea'],
                rs=r['rsds'] * 0.0036,
                uz=r['u2'],
                zw=2.0,
                doy=r['doy'],
                elev=row[kw['elev']],
                lat=row[kw['lat']])

            setattr(asce, mod_var, mod_vals)

            return asce.eto()[0]

        res_df = pd.read_csv(resid_file, parse_dates=True, index_col='date')
        res_df.index = [datetime.date(i.year, i.month, i.day) for i in res_df.index]
        res_df.dropna(how='any', axis=0, inplace=True)

        eto_arr = nldas.loc[res_df.index, 'eto'].values

        for j, var in enumerate(COMPARISON_VARS[:4]):

            if first:
                out_vars.append(var)

            result = []

            for i in range(num_samples):
                perturbed_nldas = nldas.loc[res_df.index].copy()
                fft_residuals = fft(res_df[var].values)
                random_phases = np.exp(2j * np.pi * np.random.rand(len(res_df.index)))
                fft_randomized = fft_residuals * random_phases
                error = ifft(fft_randomized).real
                perturbed_nldas[var] += error

                if i == 0:
                    sim_db = sm.stats.durbin_watson(perturbed_nldas[var])
                    obs_db = sm.stats.durbin_watson(nldas[var])
                    print('Durbin-Watson {}: obs {:.2f}, sim {:.2f}'.format(var, obs_db, sim_db))

                eto_sim = perturbed_nldas.parallel_apply(calc_eto, mod_var=var,
                                                         mod_vals=perturbed_nldas[var].values, axis=1).values

                res = eto_sim - eto_arr
                variance = np.var(res, ddof=1)

                result.append((eto_sim.mean(), perturbed_nldas[var].mean()))

            station_results[var] = result

        results[station] = station_results

        first = False

        break

    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)


def variance_decomposition(json_file, resids, station_meta, outfile, station_type='ec'):
    kw = station_par_map(station_type)

    with open(resids, 'r') as f:
        residuals = json.load(f)

    with open(json_file, 'r') as f:
        mc_results = json.load(f)

    results = {}

    station_list = pd.read_csv(station_meta, index_col=kw['index'])

    first, out_vars = True, []

    for j, (station, row) in enumerate(station_list.iterrows()):

        nldas_file = residuals[station].pop('file')
        if not os.path.exists(nldas_file):
            nldas_file = nldas_file.replace('/media/research', '/home/dgketchum/data')

        nldas = pd.read_csv(nldas_file, parse_dates=True, index_col='date')
        nldas.index = [datetime.date(i.year, i.month, i.day) for i in nldas.index]
        nldas.dropna(how='any', axis=0, inplace=True)

        errors = mc_results[station]
        if errors == 'exception':
            print('Skipping station {} due to previous exception.'.format(station))
            continue


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    sta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                          'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')
    error_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_distributions.json')

    pandarallel.initialize(nb_workers=4)

    results_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                                'error_propagation_etovar_100_fft.json')
    mc_timeseries_draw(error_json, sta, results_json, station_type='agri', num_samples=10)

    decomp_out = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                              'error_propagation_variance_decomp.json')
    # variance_decomposition(results_json, error_json, sta, decomp_out, station_type='agri')
# ========================= EOF ====================================================================
