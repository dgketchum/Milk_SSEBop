import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from pandarallel import pandarallel
import scipy.stats as st
from nldas_eto_error import station_par_map

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

        try:
            errors = error_distributions[station]
        except KeyError as er:
            print(station, er)
            continue

        print('{} of {}: {}'.format(j + 1, station_list.shape[0], station))

        df = pd.DataFrame().from_dict(errors)
        res_df = df.copy()
        res_df.index = res_df['dates']
        res_df['eta_res'] = res_df['eta_obs'] - res_df['eta_ssebop']
        res_df['eto_res'] = res_df['eto_obs'] - res_df['eto_nldas']

        res_df['etf_obs'] = res_df['eta_obs'] / res_df['eto_obs']
        res_df['etf_ssebop'] = res_df['eta_ssebop'] / res_df['eto_nldas']
        res_df['etf_res'] = res_df['etf_obs'] - res_df['etf_ssebop']

        target_vars = ['etf', 'eto']
        target_cols = ['etf_ssebop', 'eto_nldas']
        eta_obs = res_df['eta_obs']

        result = {k: [] for k in target_vars}

        for var, col in zip(target_vars, target_cols):

            residuals = res_df['{}_res'.format(var)].values

            if var == 'eto':
                a, b, loc, scale = st.johnsonsu.fit(residuals)
                dist = st.johnsonsu(a, b, loc=loc, scale=scale)
            else:
                lower_bound = 0.
                upper_bound = 1.2
                mean, std = st.norm.fit(residuals)
                a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
                dist = st.truncnorm(a, b, loc=mean, scale=std)

            for i in range(num_samples):

                error = []
                for j in range(df.shape[0]):
                    e = -1
                    while 0 > e or e > 1.25:
                        e = dist.rvs(1)[0]
                    error.append(e)

                perturbed = res_df.copy()
                perturbed[col] += error

                eta_perturbed = (perturbed['etf_ssebop'] * perturbed['eto_nldas']).values

                res = eta_obs - eta_perturbed
                variance = np.var(res, ddof=1)
                result[var].append((res_df['eta_ssebop'].mean(), eta_perturbed.mean(), res.mean(), variance))

        results[station] = result

    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)


def variance_decomposition(sim_results, station_meta):
    kw = station_par_map('ec')

    with open(sim_results, 'r') as f:
        sim_results = json.load(f)

    vars = ['eta', 'eto']
    station_list = pd.read_csv(station_meta, index_col=kw['index'])
    var_sums = {k: 0. for k in vars}
    all = 0.0

    for j, (station, row) in enumerate(station_list.iterrows()):

        for var in vars:
            try:

                sum_var = sum(np.array([i[1] for i in sim_results[station][var]]))
                var_sums[var] += sum_var
                all += sum_var
            except KeyError:
                print('Error: {}'.format(station))

    decomp = {k: '{:.2f}'.format(v / all) for k, v in var_sums.items()}
    print(decomp)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    pandarallel.initialize(nb_workers=4)

    sta = os.path.join(d, 'eddy_covariance_data_processing', 'eddy_covariance_stations.csv')

    sta_data = os.path.join(d, 'eddy_covariance_data_processing', 'corrected_data')
    ssebop_data = os.path.join(d, 'validation', 'daily_overpass_date_ssebop_et_at_eddy_covar_sites')

    error_json = os.path.join(d, 'validation', 'error_analysis', 'ec_comparison.json')
    var_json = os.path.join(d, 'validation', 'error_analysis', 'ec_variance_{}.json.json')

    num_sample_ = 10

    mc_timeseries_draw(error_json, sta, var_json, num_samples=num_sample_)

    variance_decomposition(var_json, sta)

# ========================= EOF ====================================================================
