import os
import copy
import warnings
from shutil import rmtree
from tempfile import mkdtemp
from datetime import datetime
from urllib.parse import urlunparse

import numpy as np
from numpy import empty, float32, datetime64, timedelta64, argmin, abs, array
from rasterio import open as rasopen
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.warp import calculate_default_transform as cdt
from xarray import open_dataset
from pandas import date_range, DataFrame

warnings.simplefilter(action='ignore', category=DeprecationWarning)


class Thredds:
    """  Unidata's Thematic Real-time Environmental Distributed Data Services (THREDDS)
    
    """

    def __init__(self, start=None, end=None, date=None,
                 bounds=None, target_profile=None, lat=None, lon=None, clip_feature=None):
        self.start = start
        self.end = end
        self.date = date

        self.src_bounds_wsen = None

        self.target_profile = target_profile
        self.bbox = bounds
        self.lat = lat
        self.lon = lon
        self.clip_feature = clip_feature
        self._is_masked = False

    def conform(self, subset, out_file=None):
        if subset.dtype != float32:
            subset = array(subset, dtype=float32)
        self._project(subset)
        self._warp()
        self._mask()
        result = self._resample()
        if out_file:
            self.save_raster(result, self.target_profile, output_filename=out_file)
        return result

    def _project(self, subset):

        proj_path = os.path.join(self.temp_dir, 'tiled_proj.tif')
        setattr(self, 'projection', proj_path)

        profile = copy.deepcopy(self.target_profile)
        profile['dtype'] = float32
        bb = self.bbox.as_tuple()

        if self.src_bounds_wsen:
            bounds = self.src_bounds_wsen
        else:
            bounds = (bb[0], bb[1],
                      bb[2], bb[3])

        dst_affine, dst_width, dst_height = cdt(CRS({'init': 'epsg:4326'}),
                                                CRS({'init': 'epsg:4326'}),
                                                subset.shape[1],
                                                subset.shape[2],
                                                *bounds,
                                                )

        profile.update({'crs': CRS({'init': 'epsg:4326'}),
                        'transform': dst_affine,
                        'width': dst_width,
                        'height': dst_height,
                        'count': subset.shape[0]})

        with rasopen(proj_path, 'w', **profile) as dst:
            dst.write(subset)

    def _warp(self):

        reproj_path = os.path.join(self.temp_dir, 'reproj.tif')
        setattr(self, 'reprojection', reproj_path)

        with rasopen(self.projection, 'r') as src:
            src_profile = src.profile
            src_bounds = src.bounds
            src_array = src.read()

        dst_profile = copy.deepcopy(self.target_profile)
        dst_profile['dtype'] = float32
        bounds = src_bounds
        dst_affine, dst_width, dst_height = cdt(src_profile['crs'],
                                                dst_profile['crs'],
                                                src_profile['width'],
                                                src_profile['height'],
                                                *bounds)

        dst_profile.update({'crs': dst_profile['crs'],
                            'transform': dst_affine,
                            'width': dst_width,
                            'height': dst_height,
                            'count': src_array.shape[0]})

        with rasopen(reproj_path, 'w', **dst_profile) as dst:
            dst_array = empty((src_array.shape[0], dst_height, dst_width), dtype=float32)

            reproject(src_array, dst_array, src_transform=src_profile['transform'],
                      src_crs=src_profile['crs'], dst_crs=self.target_profile['crs'],
                      dst_transform=dst_affine, resampling=Resampling.bilinear,
                      num_threads=2)

            dst.write(dst_array)

    def _mask(self):

        mask_path = os.path.join(self.temp_dir, 'masked.tif')
        with rasopen(self.reprojection) as src:
            out_arr, out_trans = mask(src, self.clip_feature, crop=True,
                                      all_touched=True)
            out_meta = src.meta.copy()
            out_meta.update({'driver': 'GTiff',
                             'height': out_arr.shape[1],
                             'width': out_arr.shape[2],
                             'transform': out_trans,
                             'count': out_arr.shape[0]})

        with rasopen(mask_path, 'w', **out_meta) as dst:
            dst.write(out_arr)

        self._is_masked = True

        setattr(self, 'mask', mask_path)

    def _resample(self):

        # home = os.path.expanduser('~')
        # resample_path = os.path.join(home, 'images', 'sandbox', 'thredds', 'resamp_twx_{}.tif'.format(var))

        resample_path = os.path.join(self.temp_dir, 'resample.tif')

        if self._is_masked:
            ras_obj = self.mask
        else:
            ras_obj = self.reprojection

        with rasopen(ras_obj, 'r') as src:
            array = src.read()
            profile = src.profile
            res = src.res
            try:
                target_affine = self.target_profile['affine']
            except KeyError:
                target_affine = self.target_profile['transform']
            target_res = target_affine.a
            res_coeff = res[0] / target_res

            new_array = empty(shape=(array.shape[0], round(array.shape[1] * res_coeff),
                                     round(array.shape[2] * res_coeff)), dtype=float32)
            aff = src.transform
            new_affine = Affine(aff.a / res_coeff, aff.b, aff.c, aff.r, aff.e / res_coeff, aff.f)

            profile.update({'transform': self.target_profile['transform'],
                            'width': self.target_profile['width'],
                            'height': self.target_profile['height'],
                            'dtype': str(new_array.dtype),
                            'count': new_array.shape[0]})

            try:
                delattr(self, 'mask')
            except AttributeError:
                pass
            delattr(self, 'reprojection')

            with rasopen(resample_path, 'w', **profile) as dst:
                reproject(array, new_array, src_transform=aff, dst_transform=new_affine, src_crs=src.crs,
                          dst_crs=src.crs, resampling=Resampling.nearest)

                dst.write(new_array)

            with rasopen(resample_path, 'r') as src:
                arr = src.read()

            return arr

    def _date_index(self):
        date_ind = date_range(self.start, self.end, freq='d')

        return date_ind

    @staticmethod
    def _dtime_to_dtime64(dtime):
        dtnumpy = datetime64(dtime).astype(datetime64)
        return dtnumpy

    @staticmethod
    def save_raster(arr, geometry, output_filename):
        try:
            arr = arr.reshape(1, arr.shape[1], arr.shape[2])
        except IndexError:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        geometry['dtype'] = str(arr.dtype)

        with rasopen(output_filename, 'w', **geometry) as dst:
            dst.write(arr)
        return None

class GridMet(Thredds):
    """ U of I Gridmet
    
    Return as numpy array per met variable in daily stack unless modified.

    Available variables: ['bi', 'elev', 'erc', 'fm100', fm1000', 'pdsi', 'pet', 'pr', 'rmax', 'rmin', 'sph', 'srad',
                          'th', 'tmmn', 'tmmx', 'vs']
        ----------
        Observation elements to access. Currently available elements:
        - 'bi' : burning index [-]
        - 'elev' : elevation above sea level [m]
        - 'erc' : energy release component [-]
        - 'fm100' : 100-hour dead fuel moisture [%]
        - 'fm1000' : 1000-hour dead fuel moisture [%]
        - 'pdsi' : Palmer Drought Severity Index [-]
        - 'pet' : daily reference potential evapotranspiration [mm]
        - 'pr' : daily accumulated precipitation [mm]
        - 'rmax' : daily maximum relative humidity [%]
        - 'rmin' : daily minimum relative humidity [%]
        - 'sph' : daily mean specific humidity [kg/kg]
        - 'prcp' : daily total precipitation [mm]
        - 'srad' : daily mean downward shortwave radiation at surface [W m-2]
        - 'th' : daily mean wind direction clockwise from North [degrees]
        - 'tmmn' : daily minimum air temperature [K]
        - 'tmmx' : daily maximum air temperature [K]
        - 'vs' : daily mean wind speed [m -s]

    :param start: start of period of data, datetime.datetime object or string format 'YYY-MM-DD'
    :param end: end of period of data, datetime.datetime object or string format 'YYY-MM-DD'
    :param variables: List  of available variables. At lease one.
    :param date: date of data, datetime.datetime object or string format 'YYY-MM-DD'
    :param bbox: bounds.GeoBounds object representing spatial bounds
    :return: numpy.ndarray

    Must have either start and end, or date.
    Must have at least one valid variable. Invalid variables will be excluded gracefully.

    note: NetCDF dates are in xl '1900' format, i.e., number of days since 1899-12-31 23:59
          xlrd.xldate handles this for the time being

    """

    def __init__(self, variable=None, date=None, start=None, end=None, lat=None, lon=None):
        Thredds.__init__(self)

        self.date = date
        self.start = start
        self.end = end

        if isinstance(start, str):
            self.start = datetime.strptime(start, '%Y-%m-%d')
        if isinstance(end, str):
            self.end = datetime.strptime(end, '%Y-%m-%d')
        if isinstance(date, str):
            self.date = datetime.strptime(date, '%Y-%m-%d')

        self.variable = variable

        if variable != 'elev':
            if self.start and self.end is None:
                raise AttributeError('Must set both start and end date')

        self.lat = lat
        self.lon = lon

        self.service = 'thredds.northwestknowledge.net:8080'
        self.scheme = 'http'

        self.temp_dir = mkdtemp()

        self.available = ['elev', 'pr', 'rmax', 'rmin', 'sph', 'srad',
                          'th', 'tmmn', 'tmmx', 'pet', 'vs', 'erc', 'bi',
                          'fm100', 'pdsi']

        if self.variable not in self.available:
            Warning('Variable {} is not available'.
                    format(self.variable))

        self.kwords = {'bi': 'daily_mean_burning_index_g',
                       'elev': '',
                       'erc': 'energy_release_component-g',
                       'fm100': 'dead_fuel_moisture_100hr',
                       'fm1000': 'dead_fuel_moisture_1000hr',
                       'pdsi': 'daily_mean_palmer_drought_severity_index',
                       'etr': 'daily_mean_reference_evapotranspiration_alfalfa',
                       'pet': 'daily_mean_reference_evapotranspiration_grass',
                       'pr': 'precipitation_amount',
                       'rmax': 'daily_maximum_relative_humidity',
                       'rmin': 'daily_minimum_relative_humidity',
                       'sph': 'daily_mean_specific_humidity',
                       'srad': 'daily_mean_shortwave_radiation_at_surface',
                       'th': 'daily_mean_wind_direction',
                       'tmmn': 'daily_minimum_temperature',
                       'tmmx': 'daily_maximum_temperature',
                       'vs': 'daily_mean_wind_speed',
                       'vpd': 'daily_mean_vapor_pressure_deficit'}

        if variable != 'elev':
            if self.date:
                self.start = self.date
                self.end = self.date

            if self.start.year < self.end.year:
                self.single_year = False

            if self.start > self.end:
                raise ValueError('start date is after end date')

        if not self.bbox and not self.lat:
            raise AttributeError('No bbox or coordinates given')

    def get_point_timeseries(self):

        url = self._build_url()
        url = url + '#fillmismatch'
        xray = open_dataset(url)
        subset = xray.sel(lon=self.lon, lat=self.lat, method='nearest')
        subset = subset.loc[dict(day=slice(self.start, self.end))]
        subset = subset.rename({'day': 'time'})
        date_ind = self._date_index()
        subset['time'] = date_ind
        time = subset['time'].values
        series = subset[self.kwords[self.variable]].values
        df = DataFrame(data=series, index=time)
        df.columns = [self.variable]
        return df

    def _build_url(self):

        # ParseResult('scheme', 'netloc', 'path', 'params', 'query', 'fragment')
        if self.variable == 'elev':
            url = urlunparse([self.scheme, self.service,
                              '/thredds/dodsC/MET/{0}/metdata_elevationdata.nc'.format(self.variable),
                              '', '', ''])
        else:
            url = urlunparse([self.scheme, self.service,
                              '/thredds/dodsC/agg_met_{}_1979_CurrentYear_CONUS.nc'.format(self.variable),
                              '', '', ''])

        return url



# from CGMorton's RefET (github.com/WSWUP/RefET)
def air_pressure(elev, method='asce'):
    """Mean atmospheric pressure at station elevation (Eqs. 3 & 34)

    Parameters
    ----------
    elev : scalar or array_like of shape(M, )
        Elevation [m].
    method : {'asce' (default), 'refet'}, optional
        Calculation method:
        * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
        * 'refet' -- Calculations will follow RefET software.

    Returns
    -------
    ndarray
        Air pressure [kPa].

    Notes
    -----
    The current calculation in Ref-ET:
        101.3 * (((293 - 0.0065 * elev) / 293) ** (9.8 / (0.0065 * 286.9)))
    Equation 3 in ASCE-EWRI 2005:
        101.3 * (((293 - 0.0065 * elev) / 293) ** 5.26)
    Per Dr. Allen, the calculation with full precision:
        101.3 * (((293.15 - 0.0065 * elev) / 293.15) ** (9.80665 / (0.0065 * 286.9)))

    """
    pair = np.array(elev, copy=True, ndmin=1).astype(np.float64)
    pair *= -0.0065
    if method == 'asce':
        pair += 293
        pair /= 293
        np.power(pair, 5.26, out=pair)
    elif method == 'refet':
        pair += 293
        pair /= 293
        np.power(pair, 9.8 / (0.0065 * 286.9), out=pair)
    # np.power(pair, 5.26, out=pair)
    pair *= 101.3

    return pair


# from CGMorton's RefET (github.com/WSWUP/RefET)
def actual_vapor_pressure(q, pair):
    """"Actual vapor pressure from specific humidity

    Parameters
    ----------
    q : scalar or array_like of shape(M, )
        Specific humidity [kg/kg].
    pair : scalar or array_like of shape(M, )
        Air pressure [kPa].

    Returns
    -------
    ndarray
        Actual vapor pressure [kPa].

    Notes
    -----
    ea = q * pair / (0.622 + 0.378 * q)

    """
    ea = np.array(q, copy=True, ndmin=1).astype(np.float64)
    ea *= 0.378
    ea += 0.622
    np.reciprocal(ea, out=ea)
    ea *= pair
    ea *= q

    return ea


# from CGMorton's RefET (github.com/WSWUP/RefET)
def wind_height_adjust(uz, zw):
    """Wind speed at 2 m height based on full logarithmic profile (Eq. 33)

    Parameters
    ----------
    uz : scalar or array_like of shape(M, )
        Wind speed at measurement height [m s-1].
    zw : scalar or array_like of shape(M, )
        Wind measurement height [m].

    Returns
    -------
    ndarray
        Wind speed at 2 m height [m s-1].

    """
    return uz * 4.87 / np.log(67.8 * zw - 5.42)


def gridmet_elevation(lat, lon):
    g = GridMet('elev', lat=lat, lon=lon)
    elev = g.get_point_elevation()
    return elev


# ========================= EOF ====================================================================
