import json
import os
from calendar import month_abbr
from datetime import datetime

import pandas as pd
from refet import Daily, calcs, Hourly

from gridded_met.extract_gridded import get_nldas
from eto_error import RENAME_MAP
from eto_error import _vpd, _rn, station_par_map

COMPARISON_VARS = ['vpd', 'rn', 'mean_temp', 'wind', 'eto']


def corrected_eto(stations, station_data, gridded_data, dri_data, model='nldas2', apply_correction=False):
    kw = station_par_map('agri')
    station_list = pd.read_csv(stations, index_col=kw['index'])

    eto_estimates = {'station': [], model: []}

    for i, (fid, row) in enumerate(station_list.iterrows()):

        if fid != 'bfam':
            continue

        try:

            print('{} of {}: {}'.format(i + 1, station_list.shape[0], fid))

            sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(fid))
            sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')
            sdf.rename(columns=RENAME_MAP, inplace=True)
            sdf['doy'] = [i.dayofyear for i in sdf.index]
            sdf['rs'] *= 0.0864
            sdf['vpd'] = sdf.apply(_vpd, axis=1)
            sdf['rn'] = sdf.apply(_rn, lat=row['latitude'], elev=row['elev_m'], zw=row['anemom_height_m'], axis=1)
            # sdf = sdf[COMPARISON_VARS]

            dri_file = os.path.join(dri_data, '{}_nldas_daily.csv'.format(fid))
            ddf = pd.read_csv(dri_file, parse_dates=True, index_col='date')
            ddf = dri_refet(ddf, elev=row['elev_m'], lat=row['latitude'])

            grid_file = os.path.join(gridded_data, '{}.csv'.format(fid))
            gdf = pd.read_csv(grid_file, index_col='date_str', parse_dates=True)
            gdf['mean_temp'] = (gdf['min_temp'] + gdf['max_temp']) * 0.5
            gdf['rsds_wm2'] = gdf['rsds'] / 0.0864
            # gdf = gdf[COMPARISON_VARS]

            chk = get_nldas(lon=row['longitude'], lat=row['latitude'], elev=row['elev_m'], start='2018-01-01',
                            end='2018-12-31')

            chk['rsds_wm2'] = chk['rsds'] / 0.0864
            chk['rlds_wm2'] = chk['rlds'] / 0.0864

            cols = [c for c in chk.columns if c in ddf.columns]

            if apply_correction:
                for j, m in enumerate(month_abbr[1:], start=1):
                    idx = [i for i in gdf.index if i.month == j]
                    print('\nMean {}, factor: {:.2f}, pre-correction: {:.2f}'.format(m, row[m],
                                                                                     gdf.loc[idx, 'eto'].mean()))
                    gdf.loc[idx, 'eto'] *= row[m]
                    print('Mean {}, factor: {:.2f}, post-correction: {:.2f}'.format(m, row[m],
                                                                                    gdf.loc[idx, 'eto'].mean()))

            s_var, n_var = 'eto_station', 'eto_{}'.format(model)
            df = pd.DataFrame(columns=[s_var], index=sdf.index, data=sdf['eto'].values)
            df['eto_station'] = sdf['eto']
            df.dropna(how='any', axis=0, inplace=True)

            df[n_var] = gdf.loc[df.index, 'eto'].values

            df[f'eto_{model}'] = gdf.loc[df.index, 'eto'].values

            eto_estimates[model].extend(df[f'eto_{model}'].to_list())
            eto_estimates['station'].extend(df['eto_station'].to_list())

        except Exception as e:
            print('Exception raised on {}, {}'.format(fid, e))


def dri_refet(df, elev, lat):
    df['doy'] = [i.dayofyear for i in df.index]
    df['mean_temp'] = (df['temperature_min'] + df['temperature_max']) * 0.5
    df['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                            q=df['specific_humidity'])
    df['rsds'] = df['shortwave_radiation'] * 0.0864
    df['rlds'] = df['longwave_radiation'] * 0.0864

    def calc_asce_params(r, lat, elev, zw):
        asce = Daily(tmin=r['temperature_min'],
                     tmax=r['temperature_max'],
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


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    station_meta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                                   'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')

    sta_data = os.path.join(d, 'weather_station_data_processing', 'corrected_data')

    model_ = 'nldas2'
    grid_data = os.path.join(d, 'weather_station_data_processing', 'gridded', model_)
    dri_data_ = os.path.join(d, 'weather_station_data_processing', 'NLDAS_data_at_stations')

    corrected_eto(station_meta, sta_data, grid_data, dri_data_, apply_correction=True)

# ========================= EOF ====================================================================
