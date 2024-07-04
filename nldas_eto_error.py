import json
import os
import pytz
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pynldas2 as nld
from utils.thredds import GridMet
from utils.thredds import air_pressure, actual_vapor_pressure

from refet import Daily, calcs
from scipy.stats import skew, kurtosis

warnings.simplefilter(action='ignore', category=FutureWarning)

VAR_MAP = {'rsds': 'Rs (w/m2)',
           'ea': 'Compiled Ea (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'temp': 'TAvg (C)',
           'wind': 'Windspeed (m/s)',
           'eto': 'ETo (mm)'}

RESAMPLE_MAP = {'rsds': 'mean',
                'humidity': 'mean',
                'min_temp': 'min',
                'max_temp': 'max',
                'wind': 'mean',
                'doy': 'first'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

COMPARISON_VARS = ['vpd', 'rn', 'tmean', 'u2', 'eto']

STR_MAP = {
    'rn': r'Net Radiation [MJ m$^{-2}$ d$^{-1}$]',
    'vpd': r'Vapor Pressure Deficit [kPa]',
    'tmean': r'Mean Daily Temperature [K]',
    'u2': r'Wind Speed at 2 m [m s$^{-1}$]',
    'eto': r'ASCE Grass Reference Evapotranspiration [mm day$^{-1}$]'
}

STR_MAP_SIMPLE = {
    'rn': r'Rn',
    'vpd': r'VPD',
    'tmean': r'Mean Temp',
    'u2': r'Wind Speed',
    'eto': r'ETo'
}

LIMITS = {'vpd': 3,
          'rn': 0.8,
          'u2': 12,
          'tmean': 12.5,
          'eto': 5}

PACIFIC = pytz.timezone('US/Pacific')


def residuals(stations, station_data, results, out_data, resids, station_type='ec', model='nldas2', check_dir=None):

    kw = station_par_map(station_type)
    station_list = pd.read_csv(stations, index_col=kw['index'])

    if model == 'gridmet':
        station_list = station_list[station_list['latitude'] <= 49.0]

    errors, all_res_dict = {}, {v: [] for v in COMPARISON_VARS}
    for i, (fid, row) in enumerate(station_list.iterrows()):

        sta_res = {v: [] for v in COMPARISON_VARS}
        print('{} of {}: {}'.format(i + 1, station_list.shape[0], fid))

        # try:
        sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(fid))
        sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')

        s = pd.to_datetime(sdf.index[0]) - timedelta(days=2)
        e = pd.to_datetime(sdf.index[-1]) + timedelta(days=2)

        sdf.index = sdf.index.tz_localize(PACIFIC)
        sdf = sdf.rename(RENAME_MAP, axis=1)
        sdf['doy'] = [i.dayofyear for i in sdf.index]

        _zw = 10.0 if station_type == 'ec' else row['anemom_height_m']

        def calc_asce_params(r, zw):
            asce = Daily(tmin=r['min_temp'],
                         tmax=r['max_temp'],
                         ea=r['ea'],
                         rs=r['rsds'] * 0.0036,
                         uz=r['wind'],
                         zw=zw,
                         doy=r['doy'],
                         elev=row[kw['elev']],
                         lat=row[kw['lat']])

            vpd = asce.vpd[0]
            rn = asce.rn[0]
            u2 = asce.u2[0]
            mean_temp = asce.tmean[0]
            eto = asce.eto()[0]

            return vpd, rn, u2, mean_temp, eto

        asce_params = sdf.apply(calc_asce_params, zw=_zw, axis=1)
        sdf[['vpd', 'rn', 'u2', 'tmean', 'eto']] = pd.DataFrame(asce_params.tolist(), index=sdf.index)

        if model == 'nldas2':
            grid = get_nldas(row[kw['lon']], row[kw['lat']], row[kw['elev']], start=s, end=e)

        elif model == 'gridmet':
            grid = get_gridmet(row[kw['lon']], row[kw['lat']], start=s, end=e)

        else:
            raise NotImplementedError('Model {} is not available'.format(model))

        asce_params = grid.apply(calc_asce_params, zw=_zw, axis=1)
        grid[['vpd', 'rn', 'u2', 'tmean', 'eto']] = pd.DataFrame(asce_params.tolist(), index=grid.index)

        # TODO: gridmet ETo is not right
        res_df = sdf[['eto']].copy()

        if check_dir:
            check_file = os.path.join(check_dir, '{}_nldas_daily.csv'.format(fid))
            cdf = pd.read_csv(check_file, parse_dates=True, index_col='date')
            cdf.index = cdf.index.tz_localize(PACIFIC)
            indx = [i for i in cdf.index if i in grid.index]
            rsq = np.corrcoef(grid.loc[indx, 'eto'], cdf.loc[indx, 'eto_asce'])[0, 0]
            print('{} PyNLDAS/Earth Engine r2: {:.3f}'.format(row['station_name'], rsq))

        dct = {}
        for var in COMPARISON_VARS:
            s_var, n_var = '{}_station'.format(var), '{}_nldas'.format(var)
            df = pd.DataFrame(columns=[s_var], index=sdf.index, data=sdf[var].values)
            df.dropna(how='any', axis=0, inplace=True)
            df[n_var] = grid.loc[df.index, var].values
            residuals = df[s_var] - df[n_var]
            res_df[var] = residuals
            sta_res[var] = list(residuals)
            all_res_dict[var] += list(residuals)
            mean_ = np.mean(residuals).item()
            variance = np.var(residuals).item()
            data_skewness = skew(residuals).item()
            data_kurtosis = kurtosis(residuals).item()
            var_dt = [i.strftime('%Y-%m-%d') for i in residuals.index]
            dct[var] = (mean_, variance, data_skewness, data_kurtosis, var_dt)

        dct['file'] = os.path.join(out_data, '{}.csv'.format(fid))
        grid = grid.loc[sdf.index]
        grid['obs_eto'] = sdf['eto']
        grid.to_csv(dct['file'])

        res_df['eto'] = sdf['eto'] - grid['eto']
        dct['resid'] = os.path.join(out_data, 'res_{}.csv'.format(fid))
        res_df.to_csv(dct['resid'])

        errors[fid] = dct.copy()

        # except Exception as e:
        #     print('Exception at {}: {}'.format(fid, e))
        #     errors[fid] = 'exception'

    with open(results, 'w') as dst:
        json.dump(errors, dst, indent=4)

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


def gridmet_par_map():
    return {
        'pet': 'eto',
        'srad': 'rsds',
        'tmmx': 'max_temp',
        'tmmn': 'min_temp',
        'vs': 'wind',
        'vpd': 'q',
    }

def get_nldas(lon, lat, elev, start, end):
    nldas = nld.get_bycoords((lon, lat), start_date=start, end_date=end,
                             variables=['temp', 'wind_u', 'wind_v', 'humidity', 'rsds'])

    nldas = nldas.tz_convert(PACIFIC)

    wind_u = nldas['wind_u']
    wind_v = nldas['wind_v']
    nldas['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

    nldas['min_temp'] = nldas['temp'] - 273.15
    nldas['max_temp'] = nldas['temp'] - 273.15
    nldas['doy'] = [i.dayofyear for i in nldas.index]

    nldas = nldas.resample('D').agg(RESAMPLE_MAP)
    nldas['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                               q=nldas['humidity'])

    return nldas


def get_gridmet(lon, lat, start, end):
    first = True
    df, cols = pd.DataFrame(), gridmet_par_map()

    for thredds_var, variable in cols.items():

        if not thredds_var:
            continue

        try:
            g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
            s = g.get_point_timeseries()
        except OSError as e:
            print('Error on {}, {}'.format(variable, e))

        df[variable] = s[thredds_var]

        if first:
            df['date'] = [i.strftime('%Y-%m-%d') for i in df.index]
            df['year'] = [i.year for i in df.index]
            df['month'] = [i.month for i in df.index]
            df['day'] = [i.day for i in df.index]
            df['centroid_lat'] = [lat for _ in range(df.shape[0])]
            df['centroid_lon'] = [lon for _ in range(df.shape[0])]
            g = GridMet('elev', lat=lat, lon=lon)
            elev = g.get_point_elevation()
            df['elev_m'] = [elev for _ in range(df.shape[0])]
            first = False

    df['doy'] = [i.dayofyear for i in df.index]

    df['min_temp'] = df['min_temp'] - 273.15
    df['max_temp'] = df['max_temp'] - 273.15

    p_air = air_pressure(df['elev_m'])
    ea_kpa = actual_vapor_pressure(df['q'], p_air)
    df['ea'] = ea_kpa.copy()

    df.index = df.index.tz_localize(PACIFIC)

    return df


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


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    # sta = os.path.join(d, '/eddy_covariance_data_processing/eddy_covariance_stations.csv'
    sta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                          'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')
    sta_data = os.path.join(d, 'weather_station_data_processing', 'corrected_data')
    comp_data = os.path.join(d, 'weather_station_data_processing', 'comparison_data')

    # error_json = os.path.join(d, 'eddy_covariance_nldas_analysis', 'error_distributions.json')
    error_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_distributions.json')
    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residual_histograms')

    # pandarallel.initialize(nb_workers=4)

    model_ = 'gridmet'
    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residuals_{}.json'.format(model_))
    ee_check = os.path.join(d, 'weather_station_data_processing/NLDAS_data_at_stations')
    residuals(sta, sta_data, error_json, comp_data, res_json, station_type='agri', model=model_)

    results_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis',
                                'error_propagation_etovar_1000.json')
    # error_propagation(error_json, sta, results_json, station_type='agri', num_samples=1000)

    joined_resid = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'joined_residuals.csv')
    # concatenate_station_residuals(error_json, joined_resid)
# ========================= EOF ====================================================================
