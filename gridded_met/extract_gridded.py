import os

import numpy as np
import pandas as pd
import pynldas2 as nld
import pytz
from pandarallel import pandarallel
from refet import Daily, calcs, Hourly

from eto_error import station_par_map

from gridded_met.thredds import GridMet

PACIFIC = pytz.timezone('US/Pacific')
MOUNTAIN = pytz.timezone('US/Mountain')

pandarallel.initialize(nb_workers=4)

NLDAS_RESAMPLE_MAP = {'rsds': 'sum',
                      'rlds': 'sum',
                      'psurf': 'mean',
                      'humidity': 'mean',
                      'min_temp': 'min',
                      'max_temp': 'max',
                      'wind': 'mean',
                      'ea': 'mean',
                      # 'rn': 'sum',
                      # 'vpd': 'mean',
                      # 'u2': 'mean',
                      # 'eto': 'sum',
                      }


def extract_gridded(stations, out_dir, model='nldas2'):
    kw = station_par_map('agri')
    station_list = pd.read_csv(stations, index_col=kw['index'])

    for i, (fid, row) in enumerate(station_list.iterrows()):

        lat, lon, elv = row[kw['lon']], row[kw['lat']], row[kw['elev']]

        try:
            if model == 'nldas2':
                df = get_nldas(lat, lon, elv)
            elif model == 'gridmet':
                df = get_gridmet(lat, lon, elv)
            else:
                raise NotImplementedError('Choose "nldas2" or "gridmet" model')

            file_unproc = os.path.join(out_dir, model, '{}.csv'.format(fid))
            df.to_csv(file_unproc, index=False)
            print(file_unproc)

        except Exception as e:
            print(model, fid, e)


def get_nldas(lon, lat, elev, start='1989-01-01', end='2023-12-31'):
    df = nld.get_bycoords((lon, lat), start_date=start, end_date=end, source='netcdf',
                          variables=['prcp', 'pet', 'temp', 'wind_u', 'wind_v', 'rlds', 'rsds', 'humidity', 'psurf'])

    df = df.tz_convert(PACIFIC)
    wind_u = df['wind_u']
    wind_v = df['wind_v']
    df['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

    df['temp'] = df['temp'] - 273.15

    df['rsds'] *= 0.0036
    df['rlds'] *= 0.0036

    df['hour'] = [i.hour for i in df.index]

    df['ea'] = calcs._actual_vapor_pressure(pair=df['psurf'] / 1000,
                                            q=df['humidity'])

    df['max_temp'] = df['temp'].copy()
    df['min_temp'] = df['temp'].copy()

    df = df.resample('D').agg(NLDAS_RESAMPLE_MAP)

    df['doy'] = [i.dayofyear for i in df.index]

    def calc_asce_params(r, zw, lat, lon, elev):
        asce = Daily(tmin=r['min_temp'],
                     tmax=r['max_temp'],
                     rs=r['rsds'],
                     ea=r['ea'],
                     uz=r['wind'],
                     zw=zw,
                     doy=r['doy'],
                     elev=elev,
                     lat=lat,
                     method='asce')

        vpd = asce.vpd[0]
        rn = asce.rn[0]
        u2 = asce.u2[0]
        eto = asce.eto()[0]

        return vpd, rn, u2, eto

    asce_params = df.parallel_apply(calc_asce_params, lat=lat, lon=lon, elev=elev, zw=10, axis=1)
    df[['vpd', 'rn', 'u2', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                  index=df.index)
    df['year'] = [i.year for i in df.index]
    df['date_str'] = [i.strftime('%Y-%m-%d') for i in df.index]
    return df


def get_gridmet(lon, lat, elev, start='1989-01-01', end='2023-12-31'):
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

    df['min_temp'] = df['min_temp'] - 273.15
    df['max_temp'] = df['max_temp'] - 273.15

    df['year'] = [i.year for i in df.index]
    df['doy'] = [i.dayofyear for i in df.index]
    df.index = df.index.tz_localize(PACIFIC)

    df['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                            q=df['q'])
    df['rsds'] *= 0.0864

    def calc_asce_params(r, lat, elev, zw):
        asce = Daily(tmin=r['min_temp'],
                     tmax=r['max_temp'],
                     rs=r['rsds'],
                     ea=r['ea'],
                     uz=r['wind'],
                     zw=zw,
                     doy=r['doy'],
                     elev=elev,
                     lat=lat,
                     method='asce')

        vpd = asce.vpd[0]
        rn = asce.rn[0]
        u2 = asce.u2[0]
        mean_temp = asce.tmean[0]
        eto = asce.eto()[0]

        return vpd, rn, u2, mean_temp, eto

    asce_params = df.parallel_apply(calc_asce_params, lat=lat, elev=elev, zw=10, axis=1)
    df[['vpd', 'rn', 'u2', 'mean_temp', 'eto']] = pd.DataFrame(asce_params.tolist(),
                                                               index=df.index)

    df['date_str'] = [i.strftime('%Y-%m-%d') for i in df.index]

    return df


def gridmet_par_map():
    return {
        'pet': 'eto',
        'srad': 'rsds',
        'tmmx': 'max_temp',
        'tmmn': 'min_temp',
        'vs': 'wind',
        'sph': 'q',
    }


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    pandarallel.initialize(nb_workers=8)

    station_meta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                                   'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')

    grid_data_dir = os.path.join(d, 'weather_station_data_processing', 'gridded')

    extract_gridded(station_meta, grid_data_dir, model='nldas2')
    # extract_gridded(station_meta, grid_data_dir, model='gridmet')
# ========================= EOF ====================================================================
