import os
import sys
from calendar import monthrange

import ee
import numpy as np

from ee_api.landcover import get_cdl

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

ET_COLLECTION = 'projects/dri-milkriver/assets/ssebop/nldas/monthly/v0_0'
SMM_DISSOLVE = 'projects/ee-dgketchum/assets/milk/smm_dissolve'


def export_gridded_data(tables, bucket, years, description, features=None, min_years=0, debug=False,
                        join_col='STAID', extra_cols=None, volumes=False, buffer=None):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    """
    ee.Initialize()
    fc = ee.FeatureCollection(tables)
    if features:
        fc = fc.filter(ee.Filter.inList('STAID', features))
    if buffer:
        fc = fc.map(lambda x: x.buffer(buffer))

    clip = ee.FeatureCollection(SMM_DISSOLVE)

    for yr in years:
        for month in range(1, 13):

            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            et_sum = ee.ImageCollection(ET_COLLECTION).filterDate(s, e).mosaic().clip(clip)
            area = ee.Image.pixelArea()

            crops = get_cdl(yr)

            crop_mask = crops.lt(1)
            et = et_sum.mask(crop_mask)
            eff_ppt = eff_ppt.mask(irr_mask).rename('eff_ppt')
            ietr = ietr.mask(irr_mask)
            irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
            irr = irr_mask.multiply(area).rename('irr')

            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('et')


            iwu = et.subtract(eff_ppt).rename('iwu')

            if volumes:
                et = et.multiply(area)
                eff_ppt = eff_ppt.multiply(area)
                iwu = iwu.multiply(area)
                ppt = ppt.multiply(area)
                etr = etr.multiply(area)
                ietr = ietr.multiply(area)

            if yr > 1986 and month in range(4, 11):
                # bands = irr.addBands([irr, et, iwu, ppt, etr, eff_ppt, ietr])
                # select_ = [join_col, 'irr', 'et', 'iwu', 'ppt', 'etr', 'eff_ppt', 'ietr']
                bands = irr.addBands([ppt])
                select_ = [join_col, 'irr']

            else:
                bands = ppt.addBands([etr])
                select_ = [join_col, 'ppt', 'etr']

            if extra_cols:
                select_ += extra_cols

            if debug:
                samp = fc.filterMetadata('gwicid', 'equals', 77225).geometry()
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
    is_authorized()

    buffers = [100, 250, 500, 1000, 5000]
    for buf in buffers:
        export_gridded_data(GWIC_MT_WEST, 'wudr', years=[i for i in range(1987, 2022)],
                            description='gwic_{}_7FEB2024'.format(buf), min_years=5, features=None,
                            join_col='gwicid', debug=True, volumes=True, buffer=buf)

# ========================= EOF ================================================================================
