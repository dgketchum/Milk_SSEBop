import os
import sys
from calendar import monthrange

import ee
import numpy as np

from ee_api.landcover import get_crop_cover

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

ET_COLLECTION = 'projects/dri-milkriver/assets/ssebop/nldas/monthly/v0_0'
SMM_DISSOLVE = 'projects/ee-dgketchum/assets/milk/smm_dissolve'


def export_gridded_data(tables, bucket, years, description, debug=False, join_col='STAID', volumes=False):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    """
    ee.Initialize()
    fc = ee.FeatureCollection(tables)

    clip = ee.FeatureCollection(SMM_DISSOLVE)

    for yr in years:
        for month in range(1, 13):

            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            et = ee.ImageCollection(ET_COLLECTION).filterDate(s, e).select('et').mosaic().clip(clip)
            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('et')

            crops = get_crop_cover(yr).clip(clip)
            crops = crops.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('crops')

            crop_mask = crops.lt(1)
            et = et.mask(crop_mask)

            area = ee.Image.pixelArea()

            if volumes:
                crop_area = crops.multiply(area)
                et = et.multiply(area)

            bands = et.addBands([crop_area])
            select_ = [join_col, 'et', 'crops']

            if debug:
                samp = fc.filterMetadata('FID', 'equals', 0).geometry()
                field = bands.reduceRegions(collection=samp,
                                            reducer=ee.Reducer.sum(),
                                            scale=30)
                p = field.first().getInfo()['properties']
                print('{} propeteries {}'.format(yr, p))

            if volumes:
                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.sum(),
                                           scale=30)
            else:
                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.mean(),
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

    export_gridded_data(SMM_DISSOLVE, 'wudr', years=[i for i in range(2022, 2024)],
                        description='smm_crop_et', debug=False, join_col='FID', volumes=True)

# ========================= EOF ================================================================================
