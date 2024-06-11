import datetime
import json
import os
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from refet import Daily
from scipy.stats import norm
import statsmodels.api as sm
from scipy.linalg import cholesky
from statsmodels.tsa.stattools import acf
from nldas_eto_sensitivity import COMPARISON_VARS
from nldas_eto_sensitivity import station_par_map
from statsmodels.tsa.api import VAR

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

        print('{} of {}: {}'.format(j + 1, len(station_list.keys()), station))

        file_ = errors.pop('file')
        resid_file = errors.pop('resid')
        if not os.path.exists(file_):
            file_ = file_.replace('/home/dgketchum/data', '/media/research')
            resid_file = resid_file.replace('/home/dgketchum/data', '/media/research')

        nldas = pd.read_csv(file_, parse_dates=True, index_col='date')
        nldas.index = pd.DatetimeIndex([i.strftime('%Y-%m-%d') for i in nldas.index])
        station_results = {var: [] for var in COMPARISON_VARS}

        cross_corr_obs = np.corrcoef(nldas[COMPARISON_VARS].values, rowvar=False)
        auto_corrs_obs = [acf(nldas[COMPARISON_VARS].values[:, i]) for i in range(4)]

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
        model = VAR(res_df)
        model = model.fit(maxlags=1, ic='aic')

        for var in COMPARISON_VARS:

            if first:
                out_vars.append(var)

            result = []

            if var == 'eto':
                eto_arr = nldas.loc[res_df.index, var].values
                station_results[var] = np.mean(eto_arr), np.std(eto_arr)
                continue

            for i in range(num_samples):
                perturbed_nldas = nldas.loc[res_df.index].copy()
                error = model.simulate_var(res_df.shape[0])
                error = pd.DataFrame(columns=res_df.columns, data=error, index=res_df.index)

                if i == 0:
                    perturbed_nldas += error
                    cross_corr_sim = np.corrcoef(perturbed_nldas[COMPARISON_VARS].values, rowvar=False)
                    auto_corrs_sim = [acf(perturbed_nldas[COMPARISON_VARS].values[:, i]) for i in range(4)]
                    sim_db = sm.stats.durbin_watson(error[var])
                    obs_db = sm.stats.durbin_watson(nldas[var])
                    print('Durbin-Watson {}: obs {:.2f}, sim {:.2f}'.format(var, obs_db, sim_db))

                else:
                    perturbed_nldas[var] += error[var]

                eto_values = perturbed_nldas.parallel_apply(calc_eto, mod_var=var,
                                                            mod_vals=perturbed_nldas[var].values, axis=1)
                result.append((eto_values.mean(), perturbed_nldas[var].mean()))

            station_results[var] = result

        results[station] = station_results

        first = False

        if j > 1:
            break

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
                                'error_propagation_etovar_10.json')
    error_propagation(error_json, sta, results_json, station_type='agri', num_samples=10)
# ========================= EOF ====================================================================
