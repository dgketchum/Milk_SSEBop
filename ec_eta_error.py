import json
import os
import warnings
import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from pandarallel import pandarallel
from refet import Daily
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy import stats

from eto_error import station_par_map, RENAME_MAP
from gridded_met.extract_gridded import get_nldas

warnings.simplefilter(action='ignore', category=FutureWarning)

PACIFIC = pytz.timezone('US/Pacific')

TARGET_COLS = ['ET_BIAS_CORR', 'NLDAS_REFERENCE_ET_BIAS_CORR']
FILL_VARS_OVERPASS = ['Daily RMSE', 'Daily R2', 'Daily Slope', 'N Days']
FILL_VARS_MONTHLY = ['Monthly RMSE', 'Monthly R2', 'Monthly Slope', 'N Months']


def donwload_ec_nldas(stations, monthly_rs_dir):
    kw = station_par_map('ec')
    station_list = pd.read_csv(stations, index_col='SITE_ID')
    for i, (fid, row) in enumerate(station_list.iterrows()):
        nl_ec_file = os.path.join(monthly_rs_dir, '{}_uncorr.csv'.format(fid))
        nldas = get_nldas(lat=row[kw['lat']], lon=row[kw['lon']], elev=row[kw['elev']])
        nldas.to_csv(nl_ec_file)
        print(nl_ec_file)


def ec_comparison(stations, station_data, daily_rs_dir, monthly_rs_dir, out_file, full_months):
    eta_ct, eto_ct, months_ct = 0, 0, 0

    kw = station_par_map('ec')
    station_list = pd.read_csv(stations, index_col='SITE_ID')

    meta = station_list.copy()

    results_dict, months_dict = {}, {}
    for i, (fid, row) in enumerate(station_list.iterrows()):

        # if fid != 'CA-Let':
        #     continue

        print('\n{} of {}: {}'.format(i + 1, station_list.shape[0], fid))

        sdf_file = os.path.join(station_data, '{}_daily_data.csv'.format(fid))
        try:
            sdf = pd.read_csv(sdf_file, parse_dates=True, index_col='date')
        except FileNotFoundError:
            print(fid, 'not found', sdf_file)
            continue

        sdf.index = sdf.index.tz_localize(PACIFIC)

        sdf = sdf.rename(RENAME_MAP, axis=1)
        sdf['doy'] = [i.dayofyear for i in sdf.index]
        sdf.dropna(subset=['ET'], axis=0, inplace=True)

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

        rsdf_file_daily = os.path.join(daily_rs_dir,
                                       '{}_SSEBOP_v0p2p6_3x3_daily_et_eccc_site_bias_corr.csv'.format(fid))
        if not os.path.isfile(rsdf_file_daily):
            rsdf_file_daily = os.path.join(daily_rs_dir,
                                           '{}_SSEBOP_v0p2p6_3x3_daily_et_ameriflux_site_bias_corr.csv'.format(fid))

        # TODO: bring in daily bias_corrected NDLDAS-2 data for monthly ETo comparison
        rsdf_file_monthly = os.path.join(monthly_rs_dir, 'monthly_{}_model_and_station_data_comparison.csv'.format(fid))

        try:
            daily_rs = pd.read_csv(rsdf_file_daily, parse_dates=True, index_col='DATE')
            monthly_df = pd.read_csv(rsdf_file_monthly, parse_dates=True, index_col='date')
        except FileNotFoundError:
            print(fid, 'not found', rsdf_file_daily)
            for v in FILL_VARS_MONTHLY + FILL_VARS_OVERPASS:
                meta.loc[fid, v] = np.nan
            continue

        daily_rs.index = daily_rs.index.tz_localize(PACIFIC)
        idx = [i for i in daily_rs.index if i in sdf.index]

        try:
            sdf.loc[idx, TARGET_COLS] = daily_rs.loc[idx, TARGET_COLS]
        except KeyError:
            missing = [c for c in TARGET_COLS if c not in daily_rs.columns]
            print('{} has no {}, {}'.format(fid, missing, os.path.basename(rsdf_file_daily)))
            modified_cols = ['ET', 'NLDAS_REFERENCE_ET_BIAS_CORR']
            sdf.loc[idx, TARGET_COLS] = daily_rs.loc[idx, modified_cols]

        df = sdf[TARGET_COLS + ['ET', 'eto']].copy()
        df.columns = ['eta_ssebop', 'eto_nldas', 'eta_obs', 'eto_obs']

        eta_df = df[['eta_obs', 'eta_ssebop']].copy()
        eta_df.dropna(how='any', axis=0, inplace=True)

        eto_df = df[['eto_obs', 'eto_nldas']].copy()
        eto_df.dropna(how='any', axis=0, inplace=True)

        m_idx = get_full_month_indices(df[['eta_obs']])

        if m_idx:
            mdf = df[['eta_obs']].copy().loc[m_idx]
            mdf = mdf.groupby([mdf.index.year, mdf.index.month]).sum()
            mdf.index = [pd.to_datetime('{}-{}-01'.format(i[0], i[1])) for i in mdf.index]
            match_idx = [i for i in monthly_df.index if i in mdf.index]
            mdf['eta_ssebop'] = monthly_df.loc[match_idx, 'ET_BIAS_CORR_3x3']
            mdf.dropna(how='any', axis=0, inplace=True)

        results_dict[fid] = {
            'eta_obs': eta_df['eta_obs'].tolist(),
            'eta_ssebop': eta_df['eta_ssebop'].tolist(),
            'eto_obs': eto_df['eto_obs'].tolist(),
            'eto_nldas': eto_df['eto_nldas'].tolist(),
            'eta_dates': [i.strftime('%Y-%m-%d') for i in eta_df.index],
            'eto_dates': [i.strftime('%Y-%m-%d') for i in eto_df.index]}

        if m_idx:

            months_ct_fid = len(set([(i.month, i.year) for i in m_idx]))
            months_ct += months_ct_fid
            months_dict[fid] = {
                'eta_obs': mdf['eta_obs'].tolist(),
                'eta_ssebop': mdf['eta_ssebop'].tolist(),
            }
            r2 = r2_score(mdf['eta_obs'], mdf['eta_ssebop'])
            rmse = root_mean_squared_error(mdf['eta_obs'], mdf['eta_ssebop'])
            slope_eta, bias_eta, _, _, _ = stats.linregress(mdf['eta_obs'], mdf['eta_ssebop'])
            meta.loc[fid, 'Monthly R2'] = r2
            meta.loc[fid, 'Monthly RMSE'] = rmse
            meta.loc[fid, 'Monthly Slope'] = slope_eta
            meta.loc[fid, 'N Months'] = mdf.shape[0]

        else:
            months_ct_fid = 0
            meta.loc[fid, 'Monthly R2'] = np.nan
            meta.loc[fid, 'Monthly RMSE'] = np.nan
            meta.loc[fid, 'Monthly Slope'] = np.nan
            meta.loc[fid, 'N Months'] = np.nan

        eta_day_ct = len(results_dict[fid]['eta_obs'])
        eto_day_ct = len(results_dict[fid]['eto_obs'])

        r2 = r2_score(eta_df['eta_obs'], eta_df['eta_ssebop'])
        rmse = root_mean_squared_error(eta_df['eta_obs'], eta_df['eta_ssebop'])
        slope_eta, bias_eta, _, _, _ = stats.linregress(eta_df['eta_obs'], eta_df['eta_ssebop'])

        meta.loc[fid, 'Daily R2'] = r2
        meta.loc[fid, 'Daily RMSE'] = rmse
        meta.loc[fid, 'Daily Slope'] = slope_eta
        meta.loc[fid, 'N Days'] = eta_df.shape[0]

        eta_ct += eta_day_ct
        eto_ct += eto_day_ct

        print('{} eta days, {} eto days, {} months at {}'.format(eta_day_ct, eto_day_ct, months_ct_fid, fid))

    print('{} overpass comparisons, {} full month comparisons'.format(eta_ct, months_ct))

    with open(out_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

    with open(full_months, 'w') as f:
        json.dump(months_dict, f, indent=4)


def get_full_month_indices(daily_df):
    months_idx = []
    mdf = daily_df.copy()

    mdf['count'] = 1
    mdf = mdf['count'].groupby(pd.Grouper(freq='M')).count()

    for i, r in mdf.items():
        mlen = calendar.monthrange(i.year, i.month)[1]
        if r >= mlen:
            months_idx.extend([j for j in daily_df.index if j.month == i.month and j.year == i.year])

    return months_idx


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    pandarallel.initialize(nb_workers=4)

    sta = os.path.join(d, 'eddy_covariance_data_processing', 'eddy_covariance_stations.csv')

    sta_data = os.path.join(d, 'eddy_covariance_data_processing', 'corrected_data')

    daily_ssebop = os.path.join(d, 'validation', 'daily_overpass_date_ssebop_et_at_eddy_covar_sites')
    monthly_ssebop = os.path.join(d, 'validation', 'data')

    error_json = os.path.join(d, 'validation', 'error_analysis', 'ec_comparison.json')
    error_json_month = os.path.join(d, 'validation', 'error_analysis', 'ec_comparison_monthly.json')

    ec_comparison(sta, sta_data, daily_ssebop, monthly_ssebop, error_json, error_json_month)

    # donwload_ec_nldas(sta, monthly_ssebop)

# ========================= EOF ====================================================================
