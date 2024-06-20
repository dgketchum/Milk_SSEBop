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

from nldas_eto_error import COMPARISON_VARS
from nldas_eto_error import station_par_map

warnings.simplefilter(action='ignore', category=FutureWarning)


def mc_timeseries_draw(json_file, station_meta, outfile, station_type='ec', num_samples=1000):
    kw = station_par_map(station_type)

    with open(json_file, 'r') as f:
        error_distributions = json.load(f)

    results = {}

    station_list = pd.read_csv(station_meta, index_col=kw['index'])

    for j, (station, row) in enumerate(station_list.iterrows()):

        errors = error_distributions[station]
        if errors == 'exception':
            print('Skipping station {} due to previous exception.'.format(station))
            continue

        print('{} of {}: {}'.format(j + 1, station_list.shape[0], station))

        file_ = errors.pop('file')
        resid_file = errors.pop('resid')
        if not os.path.exists(file_):
            file_ = file_.replace('/media/research', '/home/dgketchum/data')
            resid_file = resid_file.replace('/media/research', '/home/dgketchum/data')

        nldas = pd.read_csv(file_, parse_dates=True, index_col='date')
        nldas.index = pd.DatetimeIndex([i.strftime('%Y-%m-%d') for i in nldas.index])

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

        metvars = COMPARISON_VARS[:4]
        result = {k: [] for k in metvars}

        for var in metvars:

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
                result[var].append((res.mean(), variance))

        results[station] = result

    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)


def variance_decomposition(sim_results, station_meta, station_type='ec'):
    kw = station_par_map(station_type)

    with open(sim_results, 'r') as f:
        sim_results = json.load(f)

    metvars = COMPARISON_VARS[:4]
    station_list = pd.read_csv(station_meta, index_col=kw['index'])
    var_sums = {k: 0. for k in metvars}
    all = 0.0

    for j, (station, row) in enumerate(station_list.iterrows()):

        for var in metvars:
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

    sta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                          'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')
    error_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_distributions.json')

    pandarallel.initialize(nb_workers=4)

    num_sampl_ = 2
    variance_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                                 'eto_variance_{}.json'.format(num_sampl_))

    # mc_timeseries_draw(error_json, sta, variance_json, station_type='agri', num_samples=num_sampl_)

    variance_decomposition(variance_json, sta, station_type='agri')

# ========================= EOF ====================================================================
