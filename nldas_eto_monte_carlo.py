import os
import json
import warnings
import datetime
from pprint import pprint

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

        print('\n{} of {}: {}'.format(j + 1, station_list.shape[0], station))

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

            if mod_var == 'tmean':
                err = mod_vals - ((r['min_temp'] + r['max_temp']) / 2)
                asce = Daily(
                    tmin=r['min_temp'] + err,
                    tmax=r['max_temp'] + err,
                    ea=r['ea'],
                    rs=r['rsds'] * 0.0036,
                    uz=r['u2'],
                    zw=2.0,
                    doy=r['doy'],
                    elev=row[kw['elev']],
                    lat=row[kw['lat']])

            setattr(asce, mod_var, mod_vals)

            _eto = asce.eto()[0]

            return _eto

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

                pert_mean, res_mean = perturbed_nldas[var].mean(), res_df[var].mean()
                perturbed_nldas[var] += res_mean - pert_mean

                if i == 0:
                    sim_db = sm.stats.durbin_watson(perturbed_nldas[var])
                    obs_db = sm.stats.durbin_watson(nldas[var])
                    print('Durbin-Watson {}: obs {:.2f}, sim {:.2f}'.format(var, obs_db, sim_db))
                    pert_mean, res_mean = perturbed_nldas[var].mean(), res_df[var].mean()
                    print('Mean Residuals {}: obs {:.2f}, sim {:.2f}'.format(var, pert_mean, res_mean))

                eto_sim = perturbed_nldas.parallel_apply(calc_eto, mod_var=var,
                                                         mod_vals=perturbed_nldas[var].values, axis=1).values

                res = eto_sim - eto_arr
                variance = np.var(res, ddof=1)
                result[var].append((res.mean(), variance))

            mean_ = res.mean()
            print('Mean ETo residual for {}: {:.2f}\n'.format(var, mean_))

        results[station] = result

    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)


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


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    sta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                          'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')

    error_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_distributions.json')

    pandarallel.initialize(nb_workers=4)

    num_sampl_ = 100
    variance_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                                 'eto_variance_{}_tprop.json'.format(num_sampl_))
    mc_timeseries_draw(error_json, sta, variance_json, station_type='agri', num_samples=num_sampl_)

    decomp = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'var_decomp_stations_tprop.csv')
    variance_decomposition(variance_json, sta, decomp, station_type='agri')

# ========================= EOF ====================================================================
