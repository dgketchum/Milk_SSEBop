import json
import os
import warnings

import numpy as np
import pandas as pd
import pytz
from pandarallel import pandarallel
from refet import Daily

from eto_error import station_par_map, RENAME_MAP

warnings.simplefilter(action='ignore', category=FutureWarning)

PACIFIC = pytz.timezone('US/Pacific')

TARGET_COLS = ['ET_BIAS_CORR', 'NLDAS_REFERENCE_ET_BIAS_CORR']


def ec_comparison(stations, station_data, ssebop_dir, out_file):
    kw = station_par_map('ec')
    station_list = pd.read_csv(stations, index_col='SITE_ID')
    results_dict = {}
    for i, (fid, row) in enumerate(station_list.iterrows()):

        print('\n{} of {}: {}'.format(i + 1, station_list.shape[0], fid))

        sdf_file = os.path.join(station_data, '{}_daily_data.csv'.format(fid))
        try:
            sdf = pd.read_csv(sdf_file, parse_dates=True, index_col='date')
        except FileNotFoundError:
            print(fid, 'not found', sdf_file)
            continue

        try:
            sdf.index = sdf.index.tz_localize(PACIFIC)
        except TypeError as e:
            print(fid, e)

        sdf = sdf.rename(RENAME_MAP, axis=1)
        sdf['doy'] = [i.dayofyear for i in sdf.index]
        sdf.dropna(how='any', axis=0, inplace=True)

        def calc_asce_params(r, zw):
            asce = Daily(tmin=r['t_avg'],
                         tmax=r['t_avg'],
                         tdew=r['t_dew'],
                         rs=r['rso'] * 0.0864,
                         uz=r['ws'],
                         zw=zw,
                         doy=r['doy'],
                         elev=row[kw['elev']],
                         lat=row[kw['lat']])

            if 'vpd' in r.keys():
                asce.vpd = r['vpd']
            if 'rn' in r.keys():
                asce.rn = r['rn']

            eto = asce.eto()[0]

            return eto

        asce_params = sdf.parallel_apply(calc_asce_params, zw=10.0, axis=1)
        sdf[['eto']] = pd.DataFrame(asce_params.tolist(), index=sdf.index)

        print('elevation', row[kw['elev']])
        print('latitude', row[kw['lat']])
        means_ = sdf.loc[[i for i in sdf.index if i.month == 7], ['t_avg', 't_dew', 'vpd',
                                                                  'rso', 'ws', 'eto']].mean(axis=0)
        [print('Mean {}: {:.2f}'.format(k, v)) for k, v in means_.items()]

        rsdf_file = os.path.join(ssebop_dir, '{}_SSEBOP_v0p2p6_3x3_daily_et_eccc_site_bias_corr.csv'.format(fid))
        if not os.path.isfile(rsdf_file):
            rsdf_file = os.path.join(ssebop_dir,
                                     '{}_SSEBOP_v0p2p6_3x3_daily_et_ameriflux_site_bias_corr.csv'.format(fid))

        try:
            rsdf = pd.read_csv(rsdf_file, parse_dates=True, index_col='DATE')
        except FileNotFoundError:
            print(fid, 'not found', rsdf_file)
            continue

        rsdf.drop(columns=['system:index', '.geo'])
        rsdf.index = rsdf.index.tz_localize(PACIFIC)

        idx = [i for i in rsdf.index if i in sdf.index]
        try:
            sdf.loc[idx, TARGET_COLS] = rsdf.loc[idx, TARGET_COLS]
        except KeyError:
            missing = [c for c in TARGET_COLS if c not in rsdf.columns]
            print('{} has no {}, {}'.format(fid, missing, os.path.basename(rsdf_file)))
            modified_cols = ['ET', 'NLDAS_REFERENCE_ET_BIAS_CORR']
            sdf.loc[idx, TARGET_COLS] = rsdf.loc[idx, modified_cols]

        df = sdf[TARGET_COLS + ['ET', 'eto']].copy()
        df.dropna(how='any', axis=0, inplace=True)
        df.columns = ['eta_ssebop', 'eto_nldas', 'eta_obs', 'eto_obs']
        df = df[['eta_obs', 'eta_ssebop', 'eto_obs', 'eto_nldas']]

        mean_eto_ec = np.array([v for i, v in df['eto_obs'].items() if i.month == 7]).mean()
        mean_eto_nl = np.array([v for i, v in df['eto_nldas'].items() if i.month == 7]).mean()
        print('{} mean ETo July, station: {:.2f}, NLDAS: {:.2f}'.format(fid, mean_eto_ec, mean_eto_nl))

        results_dict[fid] = {
            'eta_obs': df['eta_obs'].tolist(),
            'eta_ssebop': df['eta_ssebop'].tolist(),
            'eto_obs': df['eto_obs'].tolist(),
            'eto_nldas': df['eto_nldas'].tolist(),
            'dates': [i.strftime('%Y-%m-%d') for i in df.index]}

    with open(out_file, 'w') as f:
        json.dump(results_dict, f, indent=4)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    pandarallel.initialize(nb_workers=4)

    sta = os.path.join(d, 'eddy_covariance_data_processing', 'eddy_covariance_stations.csv')

    sta_data = os.path.join(d, 'eddy_covariance_data_processing', 'corrected_data')
    ssebop_data = os.path.join(d, 'validation', 'daily_overpass_date_ssebop_et_at_eddy_covar_sites')

    error_json = os.path.join(d, 'validation', 'error_analysis', 'ec_comparison_woFPE.json')

    ec_comparison(sta, sta_data, ssebop_data, error_json)

# ========================= EOF ====================================================================
