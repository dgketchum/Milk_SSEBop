import json
import os
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from refet import Daily
from scipy.stats import norm

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

        print('{} of {}: {}'.format(j + 1, len(station_list.keys()), station))

        file_ = errors.pop('file')
        if not os.path.exists(file_):
            file_ = file_.replace('/home/dgketchum/data', '/media/research')
        nldas = pd.read_csv(file_, parse_dates=True, index_col='date')
        nldas.index = pd.DatetimeIndex([i.strftime('%Y-%m-%d') for i in nldas.index])
        station_results = {var: [] for var in COMPARISON_VARS}

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

        for var in COMPARISON_VARS:

            if first:
                out_vars.append(var)

            result = []
            mean_, variance, data_skewness, data_kurtosis, dates = errors[var]
            dates = pd.DatetimeIndex(dates)
            stddev = np.sqrt(variance)

            if var == 'eto':
                eto_arr = nldas.loc[dates, var].values
                station_results[var] = np.mean(eto_arr), np.std(eto_arr)
                continue

            for i in range(num_samples):
                perturbed_nldas = nldas.loc[dates].copy()
                error = norm.rvs(loc=mean_, scale=stddev, size=perturbed_nldas.shape[0])
                perturbed_nldas[var] += error
                eto_values = perturbed_nldas.parallel_apply(calc_eto, mod_var=var,
                                                            mod_vals=perturbed_nldas[var].values,
                                                            axis=1)
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

    # sta = os.path.join(d, '/eddy_covariance_data_processing/eddy_covariance_stations.csv'
    sta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                          'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')
    sta_data = os.path.join(d, 'weather_station_data_processing/corrected_data')
    comp_data = os.path.join(d, 'weather_station_data_processing/comparison_data')

    # error_json = os.path.join(d, 'eddy_covariance_nldas_analysis', 'error_distributions.json')
    error_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_distributions.json')
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residuals.json')
    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residual_histograms')

    pandarallel.initialize(nb_workers=4)

    ee_check = os.path.join(d, 'weather_station_data_processing/NLDAS_data_at_stations')
    residuals(sta, sta_data, error_json, comp_data, res_json,
              station_type='agri', check_dir=None, plot_dir=None)

    results_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                                'error_propagation_etovar_1000.json')
    # error_propagation(error_json, sta, results_json, station_type='agri', num_samples=1000)
# ========================= EOF ====================================================================
