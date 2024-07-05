import json
import os
import pytz
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from refet import Daily, calcs
from scipy.stats import skew, kurtosis

warnings.simplefilter(action='ignore', category=FutureWarning)

VAR_MAP = {'rs': 'Rs (w/m2)',
           'ea': 'Compiled Ea (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'mean_temp': 'TAvg (C)',
           'wind': 'ws_2m (m/s)',
           'eto': 'ETo (mm)'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

COMPARISON_VARS = ['ea', 'rn', 'mean_temp', 'wind', 'eto']

STR_MAP = {
    'rn': r'Net Radiation [MJ m$^{-2}$ d$^{-1}$]',
    'vpd': r'Vapor Pressure Deficit [kPa]',
    'tmean': r'Mean Daily Temperature [K]',
    'u2': r'Wind Speed at 2 m [m s$^{-1}$]',
    'eto': r'ASCE Grass Reference Evapotranspiration [mm day$^{-1}$]'
}

LIMITS = {'vpd': 3,
          'rs': 0.8,
          'u2': 12,
          'mean_temp': 12.5,
          'eto': 5}

PACIFIC = pytz.timezone('US/Pacific')


def residuals(stations, station_data, out_data, resids, model='nldas2', ee_check=None):
    kw = station_par_map('agri')
    station_list = pd.read_csv(stations, index_col=kw['index'])

    if model == 'gridmet':
        station_list = station_list[station_list['latitude'] <= 49.0]

    all_res_dict = {v: [] for v in COMPARISON_VARS}
    for i, (fid, row) in enumerate(station_list.iterrows()):

        if fid != 'bfam':
            continue

        sta_res = {v: [] for v in COMPARISON_VARS}
        print('{} of {}: {}'.format(i + 1, station_list.shape[0], fid))

        # try:
        sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(fid))
        sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')
        sdf.index = sdf.index.tz_localize(PACIFIC)
        sdf.rename(columns=RENAME_MAP, inplace=True)
        sdf['doy'] = [i.dayofyear for i in sdf.index]
        sdf['vpd'] = sdf.apply(_vpd, axis=1)
        sdf['rn'] = sdf.apply(_rn, lat=row['latitude'], elev=row['elev_m'], zw=row['anemom_height_m'], axis=1)
        sdf = sdf[COMPARISON_VARS]

        grid_file = os.path.join(out_data, '{}.csv'.format(fid))
        gdf = pd.read_csv(grid_file, index_col='date_str', parse_dates=True)
        gdf.rename(columns=RENAME_MAP, inplace=True)
        gdf['vpd'] = sdf.apply(_vpd, axis=1)
        gdf = gdf[COMPARISON_VARS]

        res_df = sdf[['eto']].copy()

        for var in COMPARISON_VARS:
            s_var, n_var = '{}_station'.format(var), '{}_nldas'.format(var)
            df = pd.DataFrame(columns=[s_var], index=sdf.index, data=sdf[var].values)
            df.dropna(how='any', axis=0, inplace=True)
            df[n_var] = gdf.loc[df.index, var].values
            residuals = df[s_var] - df[n_var]
            res_df[var] = residuals
            sta_res[var] = list(residuals)
            all_res_dict[var] += list(residuals)

        # except Exception as e:
        #     print('Exception at {}: {}'.format(fid, e))
        #     errors[fid] = 'exception')

    with open(resids, 'w') as dst:
        json.dump(all_res_dict, dst, indent=4)


def station_par_map(station_type):
    if station_type == 'ec':
        return {'index': 'SITE_ID',
                'lat': 'LATITUDE',
                'lon': 'LONGITUDE',
                'elev': 'ELEVATION (METERS)',
                'start': 'START DATE',
                'end': 'END DATE'}
    elif station_type == 'agri':
        return {'index': 'id',
                'lat': 'latitude',
                'lon': 'longitude',
                'elev': 'elev_m',
                'start': 'record_start',
                'end': 'record_end'}
    else:
        raise NotImplementedError


def _vpd(r):
    es = calcs._sat_vapor_pressure(r['mean_temp'])
    vpd = es - r['ea']
    return vpd[0]


def _rn(r, lat, elev, zw):
    asce = Daily(tmin=r['min_temp'],
                 tmax=r['max_temp'],
                 rs=r['rs'],
                 ea=r['ea'],
                 uz=r['wind'],
                 zw=zw,
                 doy=r['doy'],
                 elev=elev,
                 lat=lat)

    rn = asce.rn[0]
    return rn


def concatenate_station_residuals(error_json, out_file):
    with open(error_json, 'r') as f:
        meta = json.load(f)

    eto_residuals = []
    first, df = True, None
    for sid, data in meta.items():

        if data == 'exception':
            continue

        _file = data['resid']
        c = pd.read_csv(_file, parse_dates=True, index_col='date')

        eto = c['eto'].copy()
        eto.dropna(how='any', inplace=True)
        eto_residuals += list(eto.values)

        c.dropna(how='any', axis=0, inplace=True)
        if first:
            df = c.copy()
            first = False
        else:
            df = pd.concat([df, c], ignore_index=True, axis=0)

    df.to_csv(out_file)


def check_file(lat, elev):
    def calc_asce_params(r, zw):
        asce = Daily(tmin=r['temperature_min'],
                     tmax=r['temperature_max'],
                     rs=r['shortwave_radiation'] * 0.0036,
                     ea=r['ea'],
                     uz=r['wind'],
                     zw=zw,
                     doy=r['doy'],
                     elev=elev,
                     lat=lat)

        vpd = asce.vpd[0]
        rn = asce.rn[0]
        u2 = asce.u2[0]
        mean_temp = asce.tmean[0]
        eto = asce.eto()[0]

        return vpd, rn, u2, mean_temp, eto

    check_file = ('/media/research/IrrigationGIS/milk/weather_station_data_processing/'
                  'NLDAS_data_at_stations/bfam_nldas_daily.csv')
    dri = pd.read_csv(check_file, parse_dates=True, index_col='date')
    dri['doy'] = [i.dayofyear for i in dri.index]
    dri['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                             q=dri['specific_humidity'])
    asce_params = dri.apply(calc_asce_params, zw=10, axis=1)
    dri[['vpd_chk', 'rn_chk', 'u2_chk', 'tmean_chk', 'eto_chk']] = pd.DataFrame(asce_params.tolist(),
                                                                                index=dri.index)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    # pandarallel.initialize(nb_workers=4)

    station_meta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                                   'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')
    sta_data = os.path.join(d, 'weather_station_data_processing', 'corrected_data')

    model_ = 'nldas2'
    grid_data = os.path.join(d, 'weather_station_data_processing', 'gridded', model_)
    error_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_distributions.json')
    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residual_histograms')

    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'residuals_{}_test.json'.format(model_))
    ee_check = os.path.join(d, 'weather_station_data_processing/NLDAS_data_at_stations')

    residuals(station_meta, sta_data, grid_data, res_json, model=model_)

    joined_resid = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_residuals.csv')
    # concatenate_station_residuals(error_json, joined_resid)
# ========================= EOF ====================================================================
