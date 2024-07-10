import json
import os
import pytz
import warnings

import numpy as np
import pandas as pd

from refet import Daily, calcs

warnings.simplefilter(action='ignore', category=FutureWarning)

VAR_MAP = {'rs': 'Rs (w/m2)',
           'ea': 'Compiled Ea (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'mean_temp': 'TAvg (C)',
           'wind': 'ws_2m (m/s)',
           'eto': 'ETo (mm)'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

COMPARISON_VARS = ['vpd', 'rn', 'mean_temp', 'wind', 'eto']

STR_MAP = {
    'rn': r'Net Radiation [MJ m$^{-2}$ d$^{-1}$]',
    'vpd': r'Vapor Pressure Deficit [kPa]',
    'mean_temp': r'Mean Daily Temperature [C]',
    'wind': r'Wind Speed at 2 m [m s$^{-1}$]',
    'eto': r'ASCE Grass Reference Evapotranspiration [mm day$^{-1}$]'
}


LIMITS = {'vpd': 3,
          'rs': 0.8,
          'u2': 12,
          'mean_temp': 12.5,
          'eto': 5}

PACIFIC = pytz.timezone('US/Pacific')


def residuals(stations, station_data, gridded_data, station_residuals, all_residuals, model='nldas2',
              location=None):
    kw = station_par_map('agri')
    station_list = pd.read_csv(stations, index_col=kw['index'])

    if model == 'gridmet':
        station_list = station_list[station_list['latitude'] <= 49.0]

    if location == 'south' and model == 'nldas2':
        station_list = station_list[station_list['latitude'] <= 49.0]

    elif location == 'north' and model == 'nldas2':
        station_list = station_list[station_list['latitude'] >= 49.0]

    errors, all_res_dict = {}, {v: [] for v in COMPARISON_VARS}

    for i, (fid, row) in enumerate(station_list.iterrows()):
        try:
            sta_res = {v: [] for v in COMPARISON_VARS}
            print('{} of {}: {}'.format(i + 1, station_list.shape[0], fid))

            sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(fid))
            sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')
            sdf.rename(columns=RENAME_MAP, inplace=True)
            sdf['doy'] = [i.dayofyear for i in sdf.index]
            sdf['rs'] *= 0.0864
            sdf['vpd'] = sdf.apply(_vpd, axis=1)
            sdf['rn'] = sdf.apply(_rn, lat=row['latitude'], elev=row['elev_m'], zw=row['anemom_height_m'], axis=1)
            sdf = sdf[COMPARISON_VARS]

            grid_file = os.path.join(gridded_data, '{}.csv'.format(fid))
            gdf = pd.read_csv(grid_file, index_col='date_str', parse_dates=True)
            gdf['mean_temp'] = (gdf['min_temp'] + gdf['max_temp']) * 0.5
            gdf = gdf[COMPARISON_VARS]

            res_df = sdf[['eto']].copy()

            for var in COMPARISON_VARS:
                s_var, n_var = '{}_station'.format(var), '{}_{}'.format(var, model)
                df = pd.DataFrame(columns=[s_var], index=sdf.index, data=sdf[var].values)
                df['eto_station'] = sdf['eto']
                df.dropna(how='any', axis=0, inplace=True)

                df[n_var] = gdf.loc[df.index, var].values
                residuals = df[s_var] - df[n_var]
                res_df[var] = residuals

                df[f'eto_{model}'] = gdf.loc[df.index, 'eto'].values
                eto_residuals = df['eto_station'] - df[f'eto_{model}']
                res_df[var] = eto_residuals

                sta_res[var] = [list(residuals), list(eto_residuals)]
                all_res_dict[var] += list(residuals)

            errors[fid] = sta_res.copy()

            res_df['eto'] = sdf['eto'] - gdf['eto']
            _file = os.path.join(gridded_data, 'res_{}.csv'.format(fid))
            res_df.to_csv(_file)

        except Exception as e:
            print('Exception raised on {}, {}'.format(fid, e))

    with open(station_residuals, 'w') as dst:
        json.dump(errors, dst, indent=4)

    with open(all_residuals, 'w') as dst:
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
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                            'all_residuals_{}.json'.format(model_))
    sta_res = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                           'station_residuals_{}.json'.format(model_))

    residuals(station_meta, sta_data, grid_data, sta_res, res_json, model=model_)

# ========================= EOF ====================================================================
