import os
import sys
from calendar import monthrange

import ee
import pandas as pd

from ee_api.landcover import get_landcover

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

ET_COLLECTION = 'projects/dri-milkriver/assets/ssebop/nldas/monthly/v0_0'
SMM_DISSOLVE = 'projects/ee-dgketchum/assets/milk/smm_dissolve'
SMM_MULTI = 'projects/ee-dgketchum/assets/milk/smm_multi_aea'

CLASS_MAP = {0: 'all', 1: 'agriculture', 2: 'grass', 3: 'forest'}


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


def get_rs_study_fc():
    study = ee.Geometry.Polygon([
        [-119.92571095037613, 39.7557953359934],
        [-102.12785938787613, 39.6881938941324],
        [-102.12785938787613, 49.36630045599855],
        [-105.20403126287613, 49.194291951927646],
        [-105.24797657537613, 50.887463420761975],
        [-114.52043751287613, 50.887463420761975],
        [-114.52043751287613, 49.16556552862816],
        [-120.10149220037613, 49.28037119664759],
        [-119.92571095037613, 39.7557953359934],
    ])

    features = [ee.Feature(g, {'FID': c}) for g, c in zip([study], [1])]
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
        target_classes = None
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
            et_bands, lc_bands = [], []

            for target_class in target_classes:

                if target_class > 0:

                    lc_mask = lc_band.eq(target_class)
                    lc_name = 'lc_{}'.format(target_class)
                    lc_area = lc_mask.multiply(area).rename(lc_name)

                    et_band = et.mask(lc_mask)
                    et_name = 'et_{}'.format(target_class)
                    et_area = et_band.divide(1000.0).rename(et_name)  # to meters
                    et_bands.append(et_name), lc_bands.append(lc_name)

                else:
                    lc_mask = lc_band.gt(-1)
                    lc_name = 'lc_all'
                    lc_area = lc_mask.multiply(area).rename(lc_name)
                    et_name = 'et_all'
                    et_area = et.mask(lc_mask).divide(1000.0).rename(et_name)  # to meters
                    et_bands.append(et_name), lc_bands.append(lc_name)

                if first:
                    bands = et_area.addBands([lc_area])
                    select_ = [join_col, lc_name, et_name]
                    first = False
                else:
                    bands = bands.addBands([et_area, lc_area])
                    [select_.append(b) for b in [lc_name, et_name]]

            et_data = bands.select(et_bands).reduceRegions(collection=fc,
                                                           reducer=ee.Reducer.mean(),
                                                           scale=30)
            lc_data = bands.select(lc_bands).reduceRegions(collection=fc,
                                                           reducer=ee.Reducer.sum(),
                                                           scale=30)
            if debug:
                et_data = et_data.getInfo()
                lc_data = lc_data.getInfo()

            out_desc = '{}_et_{}_{}'.format(description, yr, month)
            task = ee.batch.Export.table.toCloudStorage(
                et_data,
                description=out_desc,
                bucket=bucket,
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=[join_col] + et_bands)

            task.start()

            out_desc = '{}_lc_{}_{}'.format(description, yr, month)
            task = ee.batch.Export.table.toCloudStorage(
                lc_data,
                description=out_desc,
                bucket=bucket,
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=[join_col] + lc_bands)

            task.start()
            print(out_desc)


def export_mean_annual_raster(bucket, years, description, mask=None, clip_fc=None, resolution=30):
    ee.Initialize()

    if clip_fc:
        clip = ee.FeatureCollection(clip_fc)
    else:
        clip = None

    if clip:
        lc_band = get_landcover(2018).clip(clip)
    else:
        lc_band = get_landcover(2018)

    s = '{}-01-01'.format(years[0])
    e = '{}-12-31'.format(years[-1])

    if clip:
        et = ee.ImageCollection(ET_COLLECTION).filterDate(s, e).select('count').sum().divide(len(years)).clip(
            clip).float()
    else:
        et = ee.ImageCollection(ET_COLLECTION).filterDate(s, e).select('count').sum().divide(len(years)).float()

    lc_band = lc_band.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
    et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

    if mask:
        lc_mask = lc_band.eq(mask)
        et = et.mask(lc_mask)

    if mask:
        desc = '{}_{}_{}_lc_{}'.format(description, years[0], years[-1], mask)
    else:
        desc = '{}_{}_{}'.format(description, years[0], years[-1])

    region = clip.geometry() if clip else None

    task = ee.batch.Export.image.toCloudStorage(
        image=et,
        bucket=bucket,
        description=desc,
        region=region,
        scale=resolution,
        maxPixels=1e13,
    )

    task.start()
    print(desc)


def count_images_used(csv):
    ct = 0
    feature_coll = get_rs_study_fc()
    ETF = 'projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02'
    OPENET_ETF = 'projects/openet/ssebop/landsat/c02'
    df = pd.read_csv(csv)
    prs = ['{}{}'.format(str(r['PATH']).rjust(3, '0'), str(r['ROW']).rjust(3, '0')) for i, r in df.iterrows()]

    for year in range(2023, 2024):

        if year > 2022:
            coll = ee.ImageCollection(OPENET_ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
        else:
            coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))

        coll = coll.filterBounds(feature_coll)
        scenes = coll.aggregate_histogram('system:index').getInfo()
        instruments = list(set([k.split('_')[0] for k in scenes.keys()]))
        scenes = [k.split('_')[1] for k in scenes.keys()]
        scenes = [s for s in scenes if s in prs]
        ct += len(scenes)
        print('{} in {} from {}'.format(len(scenes), year, ' ,'.join(instruments)))

    print('{} images'.format(ct))


if __name__ == '__main__':
    ee.Authenticate()
    ee.Initialize(project='ssebop-montana')

    # export_gridded_data(SMM_MULTI, 'wudr', years=[i for i in range(1985, 2024)],
    #                     description='smm', debug=False, join_col='OBJECTID', **{'target_classes': [0, 1, 2, 3]})

    # masks = [None, 1, 2, 3]
    # for mask_ in masks:
    #     export_mean_annual_raster('wudr', years=[i for i in range(1985, 1986)], clip_fc=None,
    #                               description='smm', mask=mask_, resolution=30)
    #     break

    c = '/media/research/IrrigationGIS/milk/ancillary/wrs2_study_area.csv'
    count_images_used(c)
# ========================= EOF ================================================================================
