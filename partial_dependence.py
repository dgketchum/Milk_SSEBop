import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

VARS = ['rn', 'vpd', 'tmean', 'u2', 'eto']


def write_partial_dependence_plot(json_filepath):
    with open(json_filepath, 'r') as file:
        dct = json.load(file)

    first, df = True, None
    for fid, data in dct.items():

        for v in VARS:

            arr = np.array(data[v])
            if first:
                c = pd.DataFrame(columns=[f'eto_f({v})', v], data=arr)
                first = False
            else:
                c[[f'eto_f({v})', v]] = arr
                if v == VARS[-1]:
                    c['fid'] = [fid for _ in range(c.shape[0])]
        if not isinstance(df, pd.DataFrame):
            df = c.copy()
            break
        else:
            df = pd.concat([df, c], axis=0)

    mean_obs, std = df[['eto_f(eto)', 'eto']].iloc[0].values
    min_eto = df[[f'eto_f({v})' for v in VARS]].min().min()
    max_eto = df[[f'eto_f({v})' for v in VARS]].max().max()

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", len(df['fid'].unique()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, v in enumerate(VARS[:-1]):
        row = i // 2
        col = i % 2
        sns.scatterplot(x=v, y=f'eto_f({v})', data=df, hue='fid', palette=palette, ax=axes[row, col])
        sns.regplot(x=v, y=f'eto_f({v})', data=df, ax=axes[row, col], scatter=False, color="red")
        axes[row, col].set_ylim([min_eto, max_eto])
        axes[row, col].set_xlabel(v, fontsize=12)
        axes[row, col].set_ylabel('eto_f(' + v + ')', fontsize=12)

        mean_v = df[v].mean()
        axes[row, col].plot(mean_v, mean_obs, marker='o', markersize=8, color='blue',
                            label='Observed Mean')

        if row == 0 and col == 0:
            axes[row, col].legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig(f'partial_dependence_plot_{fid}.png')  # Save for each fid


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    results_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_propagation_etovar.json')

    write_partial_dependence_plot(results_json)

# ========================= EOF ====================================================================
