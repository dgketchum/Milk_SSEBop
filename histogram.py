import os
import json

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

LIMITS = {'vpd': 3,
          'rn': 0.8,
          'u2': 12,
          'tmean': 12.5,
          'eto': 5}

STR_MAP = {'rn': 'Net Radiation [MJ m-2 d-1]',
           'vpd': 'Vapor Pressure Deficit [kPa]',
           'tmean': 'Mean Daily Temperature [K]',
           'u2': 'Wind Speed at 2 m [m sec-1]',
           'eto': 'ASCE Grass Reference Evapotranspiration [mm]'}


def plot_residual_histograms(resids_file, plot_dir):

    with open(resids_file, 'r') as f:
        res_dct = json.load(f)

    for var, residuals in res_dct.items():

        mean_ = np.mean(residuals)
        variance = np.var(residuals).item()
        data_skewness = skew(residuals).item()
        data_kurtosis = kurtosis(residuals).item()

        # Create a histogram plot
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='blue')
        plt.axvline(mean_, color='red', linestyle='dashed', linewidth=1)
        plt.title(f'{STR_MAP[var]} Residuals Histogram')
        plt.xlabel(f'{STR_MAP[var]} Residuals\nObserved minus NLDAS')
        plt.ylabel('Frequency')
        plt.grid(True)

        if var in LIMITS:
            plt.xlim([-1 * LIMITS[var], LIMITS[var]])

        textstr = '\n'.join((
            r'$n={:,}$'.format(len(residuals)),
            r'$\mu={:.2f}$'.format(mean_),
            r'$\sigma^2={:.2f}$'.format(variance),
            r'$\gamma_1={:.2f}$'.format(data_skewness),
            r'$\gamma_2={:.2f}$'.format(data_kurtosis)))
        props = dict(boxstyle='round', facecolor='white')
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                       verticalalignment='top', bbox=props)

        plot_path = os.path.join(plot_dir, f'{var}_residuals_histogram.png')
        plt.savefig(plot_path)
        # plt.show()


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/milk'
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'

    res_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residuals.json')
    hist = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'residual_histograms')

    plot_residual_histograms(res_json, hist)

# ========================= EOF ====================================================================
