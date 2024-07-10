import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from pandarallel import pandarallel
import scipy.stats as st
from eto_error import station_par_map

warnings.simplefilter(action='ignore', category=FutureWarning)

PACIFIC = pytz.timezone('US/Pacific')

TARGET_COLS = ['ET_BIAS_CORR', 'NLDAS_REFERENCE_ET_BIAS_CORR']


def mc_timeseries_draw(json_file, station_meta, outfile, num_samples=1000):
    kw = station_par_map('ec')

    with open(json_file, 'r') as f:
        error_distributions = json.load(f)

    results = {}

    station_list = pd.read_csv(station_meta, index_col=kw['index'])

    for j, (station, row) in enumerate(station_list.iterrows()):

        # if station != 'US-MSR':
        #     continue

        try:
            errors = error_distributions[station]
        except KeyError as er:
            print(station, er)
            continue

        print('\n\n{} of {}: {}'.format(j + 1, station_list.shape[0], station))

        df = pd.DataFrame().from_dict(errors)
        df.index = df['dates']
        df['eta_res'] = df['eta_obs'] - df['eta_ssebop']
        df['eto_res'] = df['eto_obs'] - df['eto_nldas']

        df['etf_obs'] = df['eta_obs'] / df['eto_obs']
        df['etf_ssebop'] = df['eta_ssebop'] / df['eto_nldas']
        df['etf_res'] = df['etf_obs'] - df['etf_ssebop']

        print('Mean ETof Error (obs - ssebop):{:.2f}'.format(df['etf_res'].mean()))
        print('Mean ETo Error (obs - NLDAS-2): {:.2f}\n'.format(df['eto_res'].mean()))

        target_vars = ['etf', 'eto']
        target_cols = ['etf_ssebop', 'eto_nldas']
        eta_obs = df['eta_obs']

        result = {k: [] for k in target_vars}

        if df.shape[0] < 20:
            too_small = True
        else:
            too_small = False

        for var, col in zip(target_vars, target_cols):

            drawn_values, drawn_errors = [], []
            residuals = df['{}_res'.format(var)].values

            a, b, loc, scale = st.johnsonsu.fit(residuals)
            dist = st.johnsonsu(a, b, loc=loc, scale=scale)

            vals = df[col].values

            for i in range(num_samples):

                error = []

                while len(error) < df.shape[0]:

                    if too_small:
                        e = np.random.choice(list(df[f'{var}_res']))
                    else:
                        e = dist.rvs(1)[0]

                    val = vals[len(error)] + e

                    if var == 'eto':
                        if 0 > val:
                            continue

                        error.append(e)

                    else:
                        if 0 > val or val > 1.25:
                            continue

                        error.append(e)

                perturbed = df.copy()
                perturbed[col] += error

                drawn_errors.extend(error)
                drawn_values.extend(list(perturbed[col]))

                eta_perturbed = (perturbed['etf_ssebop'] * perturbed['eto_nldas']).values

                res = eta_obs - eta_perturbed
                variance = np.var(res, ddof=1)
                result[var].append((df['eta_ssebop'].mean(), eta_perturbed.mean(), res.mean(), variance))

            print('Mean residual {}: {:.2f}, mean drawn error: {:.2f}'.format(var, residuals.mean(),
                                                                              np.array(drawn_errors).mean()))
            print('Mean value {}: {:.2f}, w/ error {:.2f}\n'.format(var, df[col].mean(),
                                                                    np.array(drawn_values).mean()))

        results[station] = result

    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)


def variance_decomposition(sim_results, station_meta, decomp_out):
    kw = station_par_map('ec')

    with open(sim_results, 'r') as f:
        sim_results = json.load(f)

    vars = ['etf', 'eto']
    station_list = pd.read_csv(station_meta, index_col=kw['index'])
    var_sums = {k: 0. for k in vars}
    all = 0.0

    df = pd.DataFrame(index=list(sim_results.keys()), columns=vars)

    for j, (station, row) in enumerate(station_list.iterrows()):

        station_var, station_sum = {}, 0.0

        for var in vars:
            try:
                sum_var = sum(np.array([i[-1] for i in sim_results[station][var]]))

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
    print(decomp)
    # {'etf': '0.60', 'eto': '0.40'}


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    pandarallel.initialize(nb_workers=4)

    sta = os.path.join(d, 'eddy_covariance_data_processing', 'eddy_covariance_stations.csv')

    sta_data = os.path.join(d, 'eddy_covariance_data_processing', 'corrected_data')
    ssebop_data = os.path.join(d, 'validation', 'daily_overpass_date_ssebop_et_at_eddy_covar_sites')

    num_sample_ = 1000
    error_json = os.path.join(d, 'validation', 'error_analysis', 'ec_comparison.json')
    var_json = os.path.join(d, 'validation', 'error_analysis', 'ec_variance_{}.json'.format(num_sample_))

    # mc_timeseries_draw(error_json, sta, var_json, num_samples=num_sample_)

    decomp = os.path.join(d, 'validation', 'error_analysis', 'var_decomp_stations.csv')
    sta = os.path.join(d, 'eddy_covariance_data_processing', 'eddy_covariance_stations.csv')
    variance_decomposition(var_json, sta, decomp)

# ========================= EOF ====================================================================
