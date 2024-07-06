import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import DatetimeTickFormatter, Span
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save
import matplotlib.dates as mdates


def compile_data(csv_dir, plot_dir):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]

    first, adf = True, None
    for f in l:
        splt = os.path.basename(f).split('.')[0].split('_')
        dt = pd.to_datetime('{}-{}-01'.format(splt[-2], splt[-1]))
        if first:
            adf = pd.read_csv(f)
            adf['date'] = dt
            first = False
        else:
            c = pd.read_csv(f)
            c['date'] = dt
            adf = pd.concat([adf, c], ignore_index=False, axis=0)

    dct = {1: None, 2: None}
    for fid in [1, 2]:
        df = adf.loc[adf['OBJECTID'] == fid].copy()
        df.index = pd.DatetimeIndex(df['date'])
        df.drop(columns=['date', 'OBJECTID'], inplace=True)
        df = df.sort_index()
        df = df.resample('M').sum()

        for p in ['all', 1, 2, 3]:
            df[f'et_{p}'] = df[f'et_{p}'] * df[f'lc_{p}'] / 1e9

        dct[fid] = df.copy()

    vals = dct[1].values + dct[2].values
    df = pd.DataFrame(data=vals, index=df.index, columns=df.columns)

    plots = []
    colors = Category10[10]

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=6))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for lc, key in zip(['Agriculture', 'Grass/Shrubland', 'Forest'], [1, 2, 3]):
        ax.plot(df.index, df['et_{}'.format(key)].values, color=colors[key], label=lc)

    plt.ylabel('ET km$^3$')
    plt.xlabel('Time')
    ax.legend(loc='upper left')
    plt.title('Milk - St Mary Evapotranspiration')
    _fig_file = os.path.join(plot_dir, 'timeseries_crop_et.png')
    plt.savefig(_fig_file)

    p = figure(title='Milk - St Mary Evapotranspiration', x_axis_label='Time', y_axis_label='ET km$^3$',
               width=3600, height=800,
               x_axis_type='datetime')

    for lc, key in zip(['Agriculture', 'Grass/Shrubland', 'Forest'], [1, 2, 3]):
        p.line(df.index, df['et_{}'.format(key)].values, line_width=1.0, color=colors[key], legend_label=lc)

    p.legend.location = 'top_left'
    p.xaxis.formatter = DatetimeTickFormatter(days=['%Y-%m-%d'], months=['%Y-%m-%d'], years=['%Y-%m-%d'])
    plots.append(p)

    _fig_file = os.path.join(plot_dir, 'timeseries_crop_et.html')
    output_file(_fig_file)
    save(column(*plots))


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS', 'milk')

    extracts = os.path.join(d, 'results', 'et_extracts')
    ts_out = os.path.join(d, 'results', 'timeseries_plots')

    compile_data(extracts, ts_out)
# ========================= EOF ====================================================================
