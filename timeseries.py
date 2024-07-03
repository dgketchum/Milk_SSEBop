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

    first, df = True, None
    for f in l:
        splt = os.path.basename(f).split('.')[0].split('_')
        dt = pd.to_datetime('{}-{}-01'.format(splt[-2], splt[-1]))
        if first:
            df = pd.read_csv(f)
            df['date'] = dt
            first = False
        else:
            c = pd.read_csv(f)
            c['date'] = dt
            df = pd.concat([df, c], ignore_index=False)

    df.index = pd.DatetimeIndex(df['date'])
    df.drop(columns=['FID'])
    df = df.sort_index()

    plots = []
    colors = Category10[10]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=6))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for lc, key in zip(['Agriculture', 'Grass/Shrubland', 'Forest'], [1, 2, 3]):
        ax.plot(df.index, df['et_{}'.format(key)].values, color=colors[key], label=lc)

    ax.legend(loc='upper left')

    _fig_file = os.path.join(plot_dir, 'timeseries_crop_et.png')
    plt.savefig(_fig_file)

    p = figure(title='ET', x_axis_label='Time', y_axis_label='Value', width=2400, height=800,
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
