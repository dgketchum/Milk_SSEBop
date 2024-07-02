import os
import sys
from calendar import monthrange

import ee

from ee_api.landcover import get_landcover

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

ET_COLLECTION = 'projects/dri-milkriver/assets/ssebop/nldas/monthly/v0_0'
SMM_DISSOLVE = 'projects/ee-dgketchum/assets/milk/smm_dissolve'


def export_geo():
    geo = ee.Geometry.Polygon([
        [-113.75846106339121, 48.428983885591904],
        [-111.41837317276621, 48.428983885591904],
        [-111.41837317276621, 48.983607830957894],
        [-113.75846106339121, 48.983607830957894],
        [-113.75846106339121, 48.428983885591904],
    ])
    return geo

def get_test_fc():
    ag = ee.Geometry.Polygon([
        [-112.6565977459894, 48.68663591289309],
        [-112.6232954388605, 48.68663591289309],
        [-112.6232954388605, 48.70261246685249],
        [-112.6565977459894, 48.70261246685249],
        [-112.6565977459894, 48.68663591289309],
    ])

    grass = ee.Geometry.Polygon([
        [-112.86539981088687, 48.49227169041395],
        [-112.74695346078921, 48.49227169041395],
        [-112.74695346078921, 48.52457026274042],
        [-112.86539981088687, 48.52457026274042],
        [-112.86539981088687, 48.49227169041395],
    ])

    forest = ee.Geometry.Polygon([
        [-113.69254309464121, 48.89612942802007],
        [-113.44947058487558, 48.89612942802007],
        [-113.44947058487558, 48.96605036549195],
        [-113.69254309464121, 48.96605036549195],
        [-113.69254309464121, 48.89612942802007],
    ])

    features = [ee.Feature(g, {'FID': c}) for g, c in zip([ag, grass, forest], [1, 2, 3])]
    fc = ee.FeatureCollection(features)

    return fc


def export_gridded_data(tables, bucket, years, description, debug=False, clip_fc=None, join_col='STAID', **kwargs):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    """
    ee.Initialize()
    fc = ee.FeatureCollection(tables)

    if clip_fc:
        clip = ee.FeatureCollection(clip_fc)
    else:
        clip = None

    if not kwargs:
        target_classes = [1]
    else:
        target_classes = kwargs['target_classes']

    area = ee.Image.pixelArea()

    for yr in years:
        for month in range(1, 13):

            bands = None

            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            if clip:
                et = ee.ImageCollection(ET_COLLECTION).filterDate(s, e).select('et').mosaic().clip(clip)
            else:
                et = ee.ImageCollection(ET_COLLECTION).filterDate(s, e).select('et').mosaic()

            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

            if clip:
                lc_band = get_landcover(yr).clip(clip)
            else:
                lc_band = get_landcover(yr)

            lc_band = lc_band.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

            first, select_ = True, None

            for target_class in target_classes:

                lc_mask = lc_band.eq(target_class)
                lc_name = 'lc_{}'.format(target_class)
                lc_area = lc_band.mask(lc_mask).multiply(area).rename(lc_name).divide(1e6)

                et_band = et.mask(lc_mask)
                et_name = 'et_{}'.format(target_class)
                et_area = et_band.multiply(area).rename(et_name).divide(1e9)

                if first:
                    bands = et_area.addBands([lc_area])
                    select_ = [join_col, lc_name, et_name]
                    first = False
                else:
                    bands = bands.addBands([et_area, lc_area])
                    [select_.append(b) for b in [lc_name, et_name]]

            if debug:
                fc = get_test_fc()
                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.sum(),
                                           scale=30)
                info = data.getInfo()

            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.sum(),
                                       scale=30)

            out_desc = '{}_{}_{}'.format(description, yr, month)
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket=bucket,
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=select_)

            task.start()
            print(out_desc)


if __name__ == '__main__':
    ee.Authenticate()
    ee.Initialize(project='ssebop-montana')

    export_gridded_data(SMM_DISSOLVE, 'wudr', years=[i for i in range(1985, 2024)],
                        description='smm_lc_et', debug=False, join_col='FID', **{'target_classes': [1, 2, 3]})

# ========================= EOF ================================================================================
