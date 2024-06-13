import os
import json
import pytz
import warnings
from datetime import timedelta

import pandas as pd
from refet import Daily
from pandarallel import pandarallel

from nldas_eto_error import station_par_map, RENAME_MAP

warnings.simplefilter(action='ignore', category=FutureWarning)

PACIFIC = pytz.timezone('US/Pacific')

TARGET_COLS = ['ET_BIAS_CORR', 'NLDAS_REFERENCE_ET_BIAS_CORR']


def residuals(stations, station_data, ssebop_dir, resids, plot_dir=None):
    kw = station_par_map('ec')
    station_list = pd.read_csv(stations, index_col='SITE_ID')
    dct = {}
    for i, (fid, row) in enumerate(station_list.iterrows()):
        print('{} of {}: {}'.format(i + 1, station_list.shape[0], fid))

        sdf_file = os.path.join(station_data, '{}_daily_data.csv'.format(fid))
        try:
            sdf = pd.read_csv(sdf_file, parse_dates=True, index_col='date')
        except FileNotFoundError:
            print(fid, 'not found', sdf_file)

        s = pd.to_datetime(sdf.index[0]) - timedelta(days=2)
        e = pd.to_datetime(sdf.index[-1]) + timedelta(days=2)

        try:
            sdf.index = sdf.index.tz_localize(PACIFIC)
        except TypeError as e:
            print(fid, e)

        sdf = sdf.rename(RENAME_MAP, axis=1)
        sdf['doy'] = [i.dayofyear for i in sdf.index]

        def calc_asce_params(r, zw):
            asce = Daily(tmin=r['t_avg'],
                         tmax=r['t_avg'],
                         tdew=r['t_dew'],
                         rs=r['rso'] * 0.0036,
                         uz=r['ws'],
                         zw=zw,
                         doy=r['doy'],
                         elev=row[kw['elev']],
                         lat=row[kw['lat']])

            eto = asce.eto()[0]

            return eto

        asce_params = sdf.parallel_apply(calc_asce_params, zw=10.0, axis=1)
        sdf[['eto']] = pd.DataFrame(asce_params.tolist(), index=sdf.index)

        rsdf_file = os.path.join(ssebop_dir, '{}_SSEBOP_v0p2p6_3x3_daily_et_eccc_site_bias_corr.csv'.format(fid))
        try:
            rsdf = pd.read_csv(rsdf_file, parse_dates=True, index_col='DATE')
        except FileNotFoundError:
            print(fid, 'not found', rsdf_file)
            continue

        rsdf.drop(columns=['system:index', '.geo'])
        rsdf.index = rsdf.index.tz_localize(PACIFIC)

        idx = [i for i in rsdf.index if i in sdf.index]
        sdf.loc[idx, TARGET_COLS] = rsdf.loc[idx, TARGET_COLS]

        df = sdf[TARGET_COLS + ['ET', 'eto']].copy()
        df.dropna(how='any', axis=0, inplace=True)
        df.columns = ['eta_ssebop', 'eto_nldas', 'eta_obs', 'eto_obs']
        df = df[['eta_obs', 'eta_ssebop', 'eto_obs', 'eto_nldas']]

        pass

    all_res_dict = {k: [item for sublist in v for item in sublist] for k, v in dct.items()}
    with open(resids, 'w') as dst:
        json.dump(all_res_dict, dst, indent=4)


if __name__ == '__main__':

    d = '/media/hdisk/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    pandarallel.initialize(nb_workers=4)

    sta = os.path.join(d, 'eddy_covariance_data_processing', 'eddy_covariance_stations.csv')

    sta_data = os.path.join(d, 'eddy_covariance_data_processing', 'corrected_data')
    ssebop_data = os.path.join(d, 'validation', 'daily_overpass_date_ssebop_et_at_eddy_covar_sites')

    error_json = os.path.join(d, 'eddy_covariance_analysis', 'error_distributions.json')

    residuals(sta, sta_data, ssebop_data, error_json, plot_dir=None)

# ========================= EOF ====================================================================
