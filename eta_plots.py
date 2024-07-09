import json
import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bokeh.layouts import column
from bokeh.models import DatetimeTickFormatter
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save
from scipy import stats

warnings.simplefilter(action='ignore', category=FutureWarning)


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
    plt.tight_layout()
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



def plot_results(results_file, fig_file):
    with open(results_file, 'r') as f:
        results_dict = json.load(f)

    all_data = []

    for j, (fid, data) in enumerate(results_dict.items()):
        df = pd.DataFrame({
            'eta_obs': data['eta_obs'],
            'eta_ssebop': data['eta_ssebop'],
            'eto_obs': data['eto_obs'],
            'eto_nldas': data['eto_nldas'],
            'fid': fid,
        })
        all_data.append(df)

    df = pd.concat(all_data)

    r_squared_eta = stats.pearsonr(df['eta_obs'], df['eta_ssebop'])[0] ** 2
    r_squared_eto = stats.pearsonr(df['eto_obs'], df['eto_nldas'])[0] ** 2
    slope_eta, bias_eta, _, _, _ = stats.linregress(df['eta_obs'], df['eta_ssebop'])
    slope_eto, bias_eto, _, _, _ = stats.linregress(df['eto_obs'], df['eto_nldas'])
    rmse_eta = ((df['eta_obs'] - df['eta_ssebop']) ** 2).mean() ** 0.5
    rmse_eto = ((df['eto_obs'] - df['eto_nldas']) ** 2).mean() ** 0.5

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    sns.scatterplot(x='eta_obs', y='eta_ssebop', hue='fid', data=df, ax=axes[0], legend=True, style='fid')

    axes[0].set_title('Station Model Comparison')
    annotation_text_eta = (f'R²: {r_squared_eta:.2f}\nSlope: {slope_eta:.2f}\nBias: {bias_eta:.2f}'
                           f'\nRMSE: {rmse_eta:.2f}\nn: {df.shape[0]}')

    axes[0].text(0.95, 0.05, annotation_text_eta, transform=axes[0].transAxes, fontsize=10,
                 horizontalalignment='right', verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title='Station', loc='upper right')

    min_val = df[['eta_obs', 'eta_ssebop']].min().min()
    max_val = df[['eta_obs', 'eta_ssebop']].max().max()
    axes[0].plot([min_val, max_val], [min_val, max_val], '--', color='red')

    sns.scatterplot(x='eto_obs', y='eto_nldas', hue='fid', data=df, ax=axes[1], legend=True, style='fid')

    annotation_text_eto = (
        f'R²: {r_squared_eto:.2f}\nSlope: {slope_eto:.2f}\nBias: {bias_eto:.2f}'
        f'\nRMSE: {rmse_eto:.2f}\nn: {df.shape[0]}')

    axes[1].text(0.95, 0.05, annotation_text_eto, transform=axes[1].transAxes, fontsize=10,
                 horizontalalignment='right', verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, title='Station', loc='upper right')

    min_val = df[['eto_obs', 'eto_nldas']].min().min()
    max_val = df[['eto_obs', 'eto_nldas']].max().max()
    axes[1].plot([min_val, max_val], [min_val, max_val], '--', color='red')

    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_file)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    error_json = os.path.join(d, 'validation', 'error_analysis', 'ec_comparison.json')

    out_fig = os.path.join(d, 'validation', 'plots', 'all_ec_woFPe.png')
    plot_results(error_json, out_fig)

    extracts = os.path.join(d, 'results', 'et_extracts')
    ts_out = os.path.join(d, 'results', 'timeseries_plots')

    compile_data(extracts, ts_out)

# ========================= EOF ====================================================================
