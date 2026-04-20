Mapping Transboundary Evapotranspiration: Methodological Implications
for Water Management in the Milk River Basin
======================================================================

Ketchum, D., Minor, B., Morton, C., Dunkerly, C., Spence, C., Sando, R.

This repository contains the analysis code for the manuscript submitted
to GIScience & Remote Sensing.


DATA
----

Eddy covariance flux tower data collected by Environment and Climate
Change Canada (ECCC) in the Milk River basin are available via:

    Sando, R., Spence, C., Minor, B., Morton, C. (2026).
    Evapotranspiration and associated meteorological data collected
    with eddy-covariance flux towers in the Milk River basin, Canada,
    2022-2024. U.S. Geological Survey data release.
    https://doi.org/10.5066/P1AEOEBW

All other data required to reproduce the analysis (weather station
observations, NLDAS-2 extracts, SSEBop ET extracts, bias correction
metadata, and eddy covariance validation comparisons) will be provided
in a companion data release accompanying the published manuscript.


OVERVIEW
--------

The SSEBop evapotranspiration (ETa) rasters were produced in Google
Earth Engine using the open-source openet-core and openet-ssebop
libraries (Melton et al. 2022). The code in this repository performs
the post-hoc error analysis, Monte Carlo variance decomposition, bias
correction evaluation, eddy covariance validation, and figure
generation described in the paper. It does NOT re-run the SSEBop
model itself.


DATA REQUIREMENTS
-----------------

All scripts expect a root data directory whose path is supplied via
the --data-dir command-line argument. The required layout is:

    <data-dir>/
    ├── bias_ratio_data_processing/
    │   └── ETo/
    │       └── final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv
    │           (48 weather stations with id, latitude, longitude, elev_m,
    │            anemom_height_m, and monthly bias correction ratios)
    │
    ├── weather_station_data_processing/
    │   ├── corrected_data/
    │   │   └── {station_id}_output.xlsx   (one per station, QC'd daily obs)
    │   ├── gridded/
    │   │   ├── nldas2/                    (extracted NLDAS-2 CSVs per station)
    │   │   └── gridmet/                   (extracted gridMET CSVs per station)
    │   ├── error_analysis/                (output: residual JSONs, variance JSONs)
    │   │   └── mc_par_variance/           (output: per-station MC variance JSONs)
    │   └── comparison_data/               (output: per-station residual CSVs)
    │
    ├── eddy_covariance_data_processing/
    │   ├── eddy_covariance_stations.csv   (8 EC stations: SITE_ID, LATITUDE, etc.)
    │   └── corrected_data/                (daily EC observation CSVs)
    │
    ├── validation/
    │   ├── data/                           (monthly/daily SSEBop-station comparisons)
    │   ├── daily_overpass_date_ssebop_et_at_eddy_covar_sites/
    │   ├── error_analysis/                 (output: EC comparison JSONs)
    │   └── plots/                          (output: scatter PNGs)
    │
    ├── results/
    │   ├── et_extracts/                    (monthly ET summary CSVs by land cover)
    │   └── timeseries_plots/               (output: timeseries figures)
    │
    └── figures/                            (output: publication figures)
        └── et_results/
            └── SMM_ET_Large.png            (optional map for Figure 5b)

All input data are provided in the companion data release. Directories
marked "(output)" are created by the analysis scripts.


ENVIRONMENT SETUP
-----------------

The analysis was run with Python 3.12 under a Conda environment named
"milk". No environment.yml is provided; install the key packages with:

    conda create -n milk python=3.12
    conda activate milk
    pip install refet==0.4.2 pynldas2==0.18.0 pandarallel==1.6.5 \
                earthengine-api==0.1.412 rasterio==1.3.10 \
                xarray==2024.10.0 scipy==1.14.0 numpy==1.26.4 \
                pandas==2.2.2 scikit-learn==1.5.1 matplotlib==3.9.1 \
                seaborn==0.13.2 bokeh==3.5.0 statsmodels==0.14.4 \
                openpyxl==3.1.5

The SSEBop model was run in Google Earth Engine using openet-core and
openet-ssebop (not installed locally; GEE server-side only).


REPRODUCING THE MAIN FINDINGS
------------------------------

All commands assume `conda activate milk` and that the working
directory is the repository root. Replace DATA_DIR with the path
to your local copy of the companion data release.

Step 0. Extract gridded meteorology at station locations (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-station NLDAS-2 and gridMET CSV files are included in the data
release. To regenerate them from source:

    python gridded_met/extract_gridded.py --data-dir DATA_DIR --model nldas2
    python gridded_met/extract_gridded.py --data-dir DATA_DIR --model gridmet

This downloads NLDAS-2 hourly data via pynldas2 (or gridMET via
THREDDS), resamples to daily, computes ASCE ETo, and writes one CSV
per station. Requires internet access and, for NLDAS-2, an Earthdata
login configured for pynldas2.

Step 1. ETo residual analysis (Section 3.1, Figure 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python eto_error.py --data-dir DATA_DIR

Reads the 48 station observations and matched NLDAS-2 estimates,
computes residuals for ETo and its four input variables (VPD, Rn,
Tmean, wind speed), and writes:
  - station_residuals_nldas2.json   (per-station residual summaries)
  - all_residuals_nldas2.json       (pooled residuals)
  - eto_all_nldas2.json             (paired obs-model ETo comparisons)
  - res_{station_id}.csv            (per-station daily residual CSVs)

Step 2. Monte Carlo variance decomposition of ETo (Section 3.1, Figure 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python eto_monte_carlo.py --data-dir DATA_DIR

Uses empirical copula sampling of correlated residuals from Step 1 to
run 10,000 Monte Carlo perturbations per station. For each draw, one
input variable is perturbed while the others are held at observed
values; ETo is recomputed via the ASCE Penman-Monteith equation
(refet.Daily). Produces the variance decomposition showing VPD
dominates ETo error (68%). Outputs:
  - mc_par_variance/eto_variance_10000_{station}.json
  - var_decomp_par.csv

Runtime: several hours on a multicore workstation. To run in the
background:
    nohup python eto_monte_carlo.py --data-dir DATA_DIR > mc.log 2>&1 &

Step 3. ETo bias correction evaluation (Section 3.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python eto_biascorr_comparison.py --data-dir DATA_DIR

Applies the monthly IDW bias correction factors (stored in the station
metadata CSV) to NLDAS-2 ETo and computes before/after RMSE, producing
the corrected-vs-uncorrected scatter plot and the 20% RMSE reduction
statistic (0.83 to 0.66 mm/day).

Step 4. Eddy covariance ETa validation (Section 3.2, Figure 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python ec_eta_error.py --data-dir DATA_DIR

Compares SSEBop daily overpass-date and monthly ETa against eight eddy
covariance stations. Computes RMSE, r-squared, and slope for both
temporal aggregations. Outputs ec_comparison.json and
ec_comparison_monthly.json.

Step 5. ETa variance decomposition into ETf and ETo (Section 3.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python ec_monte_carlo.py --data-dir DATA_DIR

Performs 1,000-draw Monte Carlo perturbation of the ETf and ETo
components at each EC station to estimate the relative error
contribution (~60% ETf, ~40% ETo). Outputs:
  - ec_variance_1000.json
  - var_decomp_stations.csv

Step 6. ETa time series and volumetric summaries (Section 3.2, Figure 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python timeseries.py --data-dir DATA_DIR

Compiles monthly ET extracts by land cover class (forest, grassland,
cropland) and generates time series plots.

Step 7. Publication figures (Figures 2-5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python journal_figures.py --data_dir DATA_DIR

Reads the outputs from Steps 1-6 and generates four multi-panel
figures at 600 DPI. Options:
  --data_dir DIR       Root data directory (required)
  --out_dir  DIR       Override the output directory (default: milk/figures)
  --figure N [N ...]   Generate only specific figures (2, 3, 4, or 5)
  --map_png  FILE      Path to mean annual ETa map PNG for Figure 5b


ADDITIONAL SCRIPTS
-------------------

eto_plots.py               Exploratory ETo residual plots (not in paper)
eta_plots.py               Interactive Bokeh ETa time series
ec_scatter.py              EC validation scatter plot (draft version)
partial_dependence.py      Partial dependence of ETo on input variables
harmonize_comparison_units.py  Unit harmonization for EC comparison CSVs
nldas_eto_error.py         Alternate NLDAS-2 error analysis (FFT variant)
nldas_eto_monte_carlo.py   Alternate MC with FFT-based synthetic errors

ee_api/call_ee.py and ee_api/landcover.py were used internally to
produce the SSEBop ETa zonal statistics in Google Earth Engine. The
pre-computed outputs are included in the companion data release.


WEATHER STATION QA/QC
----------------------

The qaqc/ module contains the interactive QA/QC workflow used to
process raw weather station data prior to analysis, following
Allen (1996, 2008), ASCE, and FAO-56 guidelines, implemented via
agweather-qaqc (Dunkerly et al. 2024). This step precedes all
analysis scripts and its outputs (corrected_data/*.xlsx) are the
station data inputs for Steps 1-3.

Usage:
    python -m qaqc.agweatherqaqc [station_config.ini]


REPRODUCIBILITY SUMMARY
-------------------------

To reproduce the main quantitative findings from the companion data
release, set D to your data directory path and run:

    conda activate milk
    D=/path/to/data
    python eto_error.py --data-dir $D                 # ~180k residuals, Fig 2
    python eto_monte_carlo.py --data-dir $D           # VPD=68%, Tmean=17%, wind=11%, Rn=3%
    python eto_biascorr_comparison.py --data-dir $D   # RMSE 0.83 -> 0.66 mm/day
    python ec_eta_error.py --data-dir $D              # monthly RMSE=23.86 mm, r2=0.65
    python ec_monte_carlo.py --data-dir $D            # ETf~60%, ETo~40%
    python timeseries.py --data-dir $D                # land-cover ET volumes
    python journal_figures.py --data_dir $D            # Figures 2-5

These steps are sequential: each script depends on the outputs of
earlier scripts as indicated in Steps 1-7 above.
