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


def error_propagation(json_file, station_meta, outfile, station_type='ec', num_samples=1000):
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

        for j, var in enumerate(COMPARISON_VARS[:4]):

            if first:
                out_vars.append(var)

            result = []

            if var == 'eto':
                eto_arr = nldas.loc[res_df.index, var].values
                station_results[var] = np.mean(eto_arr), np.std(eto_arr)
                continue

            for i in range(num_samples):
                perturbed_nldas = nldas.loc[res_df.index].copy()

                original_acf = auto_corrs_obs[j]
                padded_acf = np.zeros(res_df.shape[0])
                padded_acf[:len(original_acf)] = original_acf
                psd = np.abs(fft(padded_acf))
                random_phases = np.exp(2j * np.pi * np.random.rand(res_df.shape[0]))
                complex_spectrum = np.sqrt(psd) * random_phases
                error = np.real(ifft(complex_spectrum))
                perturbed_nldas[var] += error

                if i == 0:
                    cross_corr_sim = np.corrcoef(perturbed_nldas[COMPARISON_VARS[:4]].values, rowvar=False)
                    sim_db = sm.stats.durbin_watson(perturbed_nldas[var])
                    obs_db = sm.stats.durbin_watson(nldas[var])
                    print('Durbin-Watson {}: obs {:.2f}, sim {:.2f}'.format(var, obs_db, sim_db))

                eto_values = perturbed_nldas.parallel_apply(calc_eto, mod_var=var,
                                                            mod_vals=perturbed_nldas[var].values, axis=1)
                result.append((eto_values.mean(), perturbed_nldas[var].mean()))

            station_results[var] = result

        results[station] = station_results

        first = False

    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)


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
    error_propagation(error_json, sta, results_json, station_type='agri', num_samples=100)
# ========================= EOF ====================================================================
