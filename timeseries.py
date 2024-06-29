import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import DatetimeTickFormatter, Span
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save


def compile_data(csv_dir, plot_dir):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]

    first = True
    dates, et, crops = [], [], []
    for f in l:
        c = pd.read_csv(f)
        # smm_crop_et_1985_1.csv
        splt = os.path.basename(f).split('.')[0].split('_')[-2:]
        dt = '{}-{}-01'.format(splt[0], splt[1].rjust(2, '0'))
        dates.append(dt)
        et.append(c.loc[0, 'et'] / 1e9)
        crops.append(c.loc[0, 'crops'] / 1e6)

    df = pd.DataFrame(index=pd.DatetimeIndex(dates), data=np.array([et, crops]).T, columns=['et', 'crops'])
    df = df.sort_index()

    plots = []
    colors = Category10[10]

    p = figure(title='ET', x_axis_label='Time', y_axis_label='Value', width=2400, height=800,
               x_axis_type="datetime")

    p.line(df.index, df['et'].values, line_width=0.5, color='blue', legend_label='Prior')

    p.legend.location = "top_left"
    p.xaxis.formatter = DatetimeTickFormatter(days=["%Y-%m-%d"], months=["%Y-%m-%d"], years=["%Y-%m-%d"])
    plots.append(p)

    p = figure(title='Cropped Area', x_axis_label='Time', y_axis_label='Area [km 2]', width=2400, height=800,
               x_axis_type="datetime")

    p.line(df.index, df['crops'].values, line_width=0.5, color='green', legend_label='Prior')

    p.legend.location = "top_left"
    p.xaxis.formatter = DatetimeTickFormatter(days=["%Y-%m-%d"], months=["%Y-%m-%d"], years=["%Y-%m-%d"])
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
