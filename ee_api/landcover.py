import ee


def aafc_binary():
    return {
        10: ("Cloud", 0),
        20: ("Water", 0),
        30: ("Exposed Land and Barren", 0),
        34: ("Urban and Developed", 0),
        35: ("Greenhouses", 0),
        80: ("Wetland", 0),
        85: ("Peatland", 0),
        120: ("Agriculture (undifferentiated)", 1),
        122: ("Pasture and Forages", 1),
        130: ("Too Wet to be Seeded", 1),
        131: ("Fallow", 1),
        132: ("Cereals", 1),
        133: ("Barley", 1),
        134: ("Other Grains", 1),
        135: ("Millet", 1),
        136: ("Oats", 1),
        137: ("Rye", 1),
        138: ("Spelt", 1),
        139: ("Triticale", 1),
        140: ("Wheat", 1),
        141: ("Switchgrass", 1),
        142: ("Sorghum", 1),
        143: ("Quinoa", 1),
        145: ("Winter Wheat", 1),
        146: ("Spring Wheat", 1),
        147: ("Corn", 1),
        148: ("Tobacco", 1),
        149: ("Ginseng", 1),
        150: ("Oilseeds", 1),
        151: ("Borage", 1),
        152: ("Camelina", 1),
        153: ("Canola and Rapeseed", 1),
        154: ("Flaxseed", 1),
        155: ("Mustard", 1),
        156: ("Safflower", 1),
        157: ("Sunflower", 1),
        158: ("Soybeans", 1),
        160: ("Pulses", 1),
        161: ("Other Pulses", 1),
        162: ("Peas", 1),
        163: ("Chickpeas", 1),
        167: ("Beans", 1),
        168: ("Fababeans", 1),
        174: ("Lentils", 1),
        175: ("Vegetables", 1),
        176: ("Tomatoes", 1),
        177: ("Potatoes", 1),
        178: ("Sugarbeets", 1),
        179: ("Other Vegetables", 1),
        180: ("Fruits", 1),
        181: ("Berries", 1),
        182: ("Blueberry", 1),
        183: ("Cranberry", 1),
        185: ("Other Berry", 1),
        188: ("Orchards", 1),
        189: ("Other Fruits", 1),
        190: ("Vineyards", 1),
        191: ("Hops", 1),
        192: ("Sod", 1),
        193: ("Herbs", 1),
        194: ("Nursery", 1),
        195: ("Buckwheat", 1),
        196: ("Canaryseed", 1),
        197: ("Hemp", 1),
        198: ("Vetch", 1),
        199: ("Other Crops", 1),

        110: ("Grassland", 2),
        50: ("Shrubland", 2),

        200: ("Forest (undifferentiated)", 3),
        210: ("Coniferous", 3),
        220: ("Broadleaf", 3),
        230: ("Mixedwood", 3),
    }


def cdl_key():
    key = {
        1: ('Corn', 1),
        2: ('Cotton', 1),
        3: ('Rice', 1),
        4: ('Sorghum', 1),
        5: ('Soybeans', 1),
        6: ('Sunflower', 1),
        10: ('Peanuts', 1),
        11: ('Tobacco', 1),
        12: ('Sweet Corn', 1),
        13: ('Pop or Orn Corn', 1),
        14: ('Mint', 1),
        21: ('Barley', 1),
        22: ('Durum Wheat', 1),
        23: ('Spring Wheat', 1),
        24: ('Winter Wheat', 1),
        25: ('Other Small Grains', 1),
        26: ('Dbl Crop WinWht/Soybeans', 1),
        27: ('Rye', 1),
        28: ('Oats', 1),
        29: ('Millet', 1),
        30: ('Speltz', 1),
        31: ('Canola', 1),
        32: ('Flaxseed', 1),
        33: ('Safflower', 1),
        34: ('Rape Seed', 1),
        35: ('Mustard', 1),
        36: ('Alfalfa', 1),
        37: ('Other Hay/Non Alfalfa', 1),
        38: ('Camelina', 1),
        39: ('Buckwheat', 1),
        41: ('Sugarbeets', 1),
        42: ('Dry Beans', 1),
        43: ('Potatoes', 1),
        44: ('Other Crops', 1),
        45: ('Sugarcane', 1),
        46: ('Sweet Potatoes', 1),
        47: ('Misc Vegs & Fruits', 1),
        48: ('Watermelons', 1),
        49: ('Onions', 1),
        50: ('Cucumbers', 1),
        51: ('Chick Peas', 1),
        52: ('Lentils', 1),
        53: ('Peas', 1),
        54: ('Tomatoes', 1),
        55: ('Caneberries', 1),
        56: ('Hops', 1),
        57: ('Herbs', 1),
        58: ('Clover/Wildflowers', 1),
        59: ('Sod/Grass Seed', 1),
        60: ('Switchgrass', 1),
        61: ('Fallow/Idle Cropland', 1),
        65: ('Barren', 1),
        66: ('Cherries', 1),
        67: ('Peaches', 1),
        68: ('Apples', 1),
        69: ('Grapes', 1),
        70: ('Christmas Trees', 1),
        71: ('Other Tree Crops', 1),
        72: ('Citrus', 1),
        74: ('Pecans', 1),
        75: ('Almonds', 1),
        76: ('Walnuts', 1),
        77: ('Pears', 1),
        92: ('Aquaculture', 1),
        204: ('Pistachios', 1),
        205: ('Triticale', 1),
        206: ('Carrots', 1),
        207: ('Asparagus', 1),
        208: ('Garlic', 1),
        209: ('Cantaloupes', 1),
        210: ('Prunes', 1),
        211: ('Olives', 1),
        212: ('Oranges', 1),
        213: ('Honeydew Melons', 1),
        214: ('Broccoli', 1),
        215: ('Avocados', 1),
        216: ('Peppers', 1),
        217: ('Pomegranates', 1),
        218: ('Nectarines', 1),
        219: ('Greens', 1),
        220: ('Plums', 1),
        221: ('Strawberries', 1),
        222: ('Squash', 1),
        223: ('Apricots', 1),
        224: ('Vetch', 1),
        225: ('Dbl Crop WinWht/Corn', 1),
        226: ('Dbl Crop Oats/Corn', 1),
        227: ('Lettuce', 1),
        229: ('Pumpkins', 1),
        230: ('Dbl Crop Lettuce/Durum Wht', 1),
        231: ('Dbl Crop Lettuce/Cantaloupe', 1),
        232: ('Dbl Crop Lettuce/Cotton', 1),
        233: ('Dbl Crop Lettuce/Barley', 1),
        234: ('Dbl Crop Durum Wht/Sorghum', 1),
        235: ('Dbl Crop Barley/Sorghum', 1),
        236: ('Dbl Crop WinWht/Sorghum', 1),
        237: ('Dbl Crop Barley/Corn', 1),
        238: ('Dbl Crop WinWht/Cotton', 1),
        239: ('Dbl Crop Soybeans/Cotton', 1),
        240: ('Dbl Crop Soybeans/Oats', 1),
        241: ('Dbl Crop Corn/Soybeans', 1),
        242: ('Blueberries', 1),
        243: ('Cabbage', 1),
        244: ('Cauliflower', 1),
        245: ('Celery', 1),
        246: ('Radishes', 1),
        247: ('Turnips', 1),
        248: ('Eggplants', 1),
        249: ('Gourds', 1),
        250: ('Cranberries', 1),
        254: ('Dbl Crop Barley/Soybeans', 1),

        64: ('Shrubland', 2),
        62: ('Pasture/Grass', 2),
        152: ('Shrubland', 2),
        176: ('Grassland/Pasture', 2),

        63: ('Forest', 3),
        141: ('Deciduous Forest', 3),
        142: ('Evergreen Forest', 3),
        143: ('Mixed Forest', 3),
    }

    return key


def remap_from_dct(key_):
    map_ = list(key_.keys())
    remap = [v[1] for k, v in key_.items()]
    return map_, remap


def get_landcover(yr):
    if yr < 2009:
        yr = 2009
    if yr > 2022:
        yr = 2022
    image = ee.Image('USDA/NASS/CDL/{}'.format(yr))
    crop = image.select('cropland')
    key = cdl_key()
    cdl_keys, our_keys = remap_from_dct(key)
    cdl_cult = crop.remap(cdl_keys, our_keys).rename('cultivated').resample('bilinear').int8()

    image = ee.ImageCollection('AAFC/ACI').filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr))
    crop = image.select('landcover').first()
    key = aafc_binary()
    aafc_keys, our_keys = remap_from_dct(key)
    aafc_cult = crop.remap(aafc_keys, our_keys).rename('cultivated').resample('bilinear').int8()

    cultivated = ee.ImageCollection(cdl_cult).merge(ee.ImageCollection(aafc_cult)).max()

    return cultivated


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
