import os
import sys
import io
import json
import numpy as np
import uuid
from affine import Affine as aff
import cv2
import math
import multiprocessing
import urllib.request as ur
from math import floor, pi, log, tan, atan, exp
from threading import Thread
import PIL.Image as pil
import rasterio as rsio
import rasterio.mask
from osgeo import gdal, osr
from rasterio import features
from geojson import Feature, FeatureCollection, dumps
import folium
from utils.configs import Downloader
import warnings

warnings.filterwarnings("ignore")


MAP_URL = "http://mts0.googleapis.com/vt?lyrs={style}&x={x}&y={y}&z={z}"


# generate img from tif file
def generate_img_from_tif(image_path):
    tif_img = rsio.open(image_path)
    img = np.dstack((tif_img.read(i+1) for i in range(3)))
    return img, tif_img.meta

# generate geotiff file from img
def generate_geotiff(image_array, file_name, meta):
    out_meta = meta.copy()
    out_meta.update(count=3, compress='lzw')
    with rsio.open(f"/kaggle/working/{file_name}_r2.tif", "w+", **out_meta) as new_tif:
        for i in range(3):
            new_tif.write(image_array[:, :, i], i+1)

# ------------------Interchange between WGS-84 and Web Mercator-------------------------
# WGS-84 to Web Mercator
def wgs_to_mercator(x, y):
    y = 85.0511287798 if y > 85.0511287798 else y
    y = -85.0511287798 if y < -85.0511287798 else y

    x2 = x * 20037508.34 / 180
    y2 = log(tan((90 + y) * pi / 360)) / (pi / 180)
    y2 = y2 * 20037508.34 / 180
    return x2, y2


# Web Mercator to WGS-84
def mercator_to_wgs(x, y):
    x2 = x / 20037508.34 * 180
    y2 = y / 20037508.34 * 180
    y2 = 180 / pi * (2 * atan(exp(y2 * pi / 180)) - pi / 2)
    return x2, y2

# Get tile coordinates in Google Maps based on latitude and longitude of WGS-84
def wgs_to_tile(j, w, z):
    '''
    Get google-style tile cooridinate from geographical coordinate
    j : Longittude
    w : Latitude
    z : zoom
    '''
    isnum = lambda x: isinstance(x, int) or isinstance(x, float)
    if not (isnum(j) and isnum(w)):
        raise TypeError("j and w must be int or float!")

    if not isinstance(z, int) or z < 0 or z > 22:
        raise TypeError("z must be int and between 0 to 22.")

    if j < 0:
        j = 180 + j
    else:
        j += 180
    j /= 360  # make j to (0,1)

    w = 85.0511287798 if w > 85.0511287798 else w
    w = -85.0511287798 if w < -85.0511287798 else w
    w = log(tan((90 + w) * pi / 360)) / (pi / 180)
    w /= 180  # make w to (-1,1)
    w = 1 - (w + 1) / 2  # make w to (0,1) and left top is 0-point

    num = 2 ** z
    x = floor(j * num)
    y = floor(w * num)
    return x, y

def pixls_to_mercator(zb):
    # Get the web Mercator projection coordinates of the four corners of the area according to the four corner coordinates of the tile
    inx, iny = zb["LT"]  # left top
    inx2, iny2 = zb["RB"]  # right bottom
    length = 20037508.3427892
    sum = 2 ** zb["z"]
    LTx = inx / sum * length * 2 - length
    LTy = -(iny / sum * length * 2) + length

    RBx = (inx2 + 1) / sum * length * 2 - length
    RBy = -((iny2 + 1) / sum * length * 2) + length

    # LT=left top,RB=right buttom
    # Returns the projected coordinates of the four corners
    res = {'LT': (LTx, LTy), 'RB': (RBx, RBy),
           'LB': (LTx, RBy), 'RT': (RBx, LTy)}
    return res

def tile_to_pixls(zb):
    # Tile coordinates are converted to pixel coordinates of the four corners
    out = {}
    width = (zb["RT"][0] - zb["LT"][0] + 1) * 256
    height = (zb["LB"][1] - zb["LT"][1] + 1) * 256
    out["LT"] = (0, 0)
    out["RT"] = (width, 0)
    out["LB"] = (0, -height)
    out["RB"] = (width, -height)
    return out

# Get coordinates
def getExtent(x1, y1, x2, y2, z, source="Google"):
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    Xframe = pixls_to_mercator(
        {"LT": (pos1x, pos1y), "RT": (pos2x, pos1y), "LB": (pos1x, pos2y), "RB": (pos2x, pos2y), "z": z})
    for i in ["LT", "LB", "RT", "RB"]:
        Xframe[i] = mercator_to_wgs(*Xframe[i])
    if source == "Google":
        pass
#     elif source == "Google China":
#         for i in ["LT", "LB", "RT", "RB"]:
#             Xframe[i] = gcj_to_wgs(*Xframe[i])
    else:
        raise Exception("Invalid argument: source.")
    return Xframe

# Save into geotiff file
def saveTiff(r, g, b, gt, filePath):
    fname_out = filePath
    driver = gdal.GetDriverByName('GTiff')
    # Create a 3-band dataset
    dset_output = driver.Create(fname_out, r.shape[1], r.shape[0], 3, gdal.GDT_Byte)
    dset_output.SetGeoTransform(gt)
    try:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326)
#         dset_output.SetSpatialRef(proj)
        dset_output.SetProjection(proj.ExportToWkt())
    except:
        print("Error: Coordinate system setting failed")
    dset_output.GetRasterBand(1).WriteArray(r)
    dset_output.GetRasterBand(2).WriteArray(g)
    dset_output.GetRasterBand(3).WriteArray(b)
    dset_output.FlushCache()
    dset_output = None
    print("Image Saved")

# Get tiles from urls functions
def get_url(source, x, y, z, style):  #
    if source == 'Google':
        url = MAP_URL.format(x=x, y=y, z=z, style=style)
    else:
        raise Exception("Unknown Map Source ! ")
    return url

def get_urls(x1, y1, x2, y2, z, source, style):
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    print("Total tiles number：{x} X {y}".format(x=lenx, y=leny))
    urls = [get_url(source, i, j, z, style) for j in range(pos1y, pos1y + leny) for i in range(pos1x, pos1x + lenx)]
    return urls

# Downloading and merging tiles functions
def merge_tiles(datas, x1, y1, x2, y2, z):
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    outpic = pil.new('RGBA', (lenx * 256, leny * 256))
    for i, data in enumerate(datas):
        picio = io.BytesIO(data)
        small_pic = pil.open(picio)
        y, x = i // lenx, i % lenx
        outpic.paste(small_pic, (x * 256, y * 256))
    print('Tiles merge completed')
    return outpic

def download_tiles(urls, multi=10):
    url_len = len(urls)
    datas = [None] * url_len
    if multi < 1 or multi > 20 or not isinstance(multi, int):
        raise Exception("multi of Downloader shuold be int and between 1 to 20.")
    tasks = [Downloader(i, multi, urls, datas) for i in range(multi)]
    for i in tasks:
        i.start()
    for i in tasks:
        i.join()
    return datas

# Generate input img function
def generate_input_img(left, top, right, bottom, zoom, filePath=None, style='s', server="Google"):
    """
    Download images based on spatial extent.
    East longitude is positive and west longitude is negative.
    North latitude is positive, south latitude is negative.
    Parameters
    ----------
    left, top : left-top coordinate, for example (100.361,38.866)
        
    right, bottom : right-bottom coordinate
        
    z : zoom
    filePath : File path for storing results, TIFF format
        
    style : 
        m for map; 
        s for satellite; 
        y for satellite with label; 
        t for terrain; 
        p for terrain with label; 
        h for label;
    
    source : Google
    """
    # ---------------------------------------------------------
    # Get the urls of all tiles in the extent
    urls = get_urls(left, top, right, bottom, zoom, server, style)

    # Group URLs based on the number of CPU cores to achieve roughly equal amounts of tasks
    urls_group = [urls[i:i + math.ceil(len(urls) / multiprocessing.cpu_count())] for i in
                  range(0, len(urls), math.ceil(len(urls) / multiprocessing.cpu_count()))]

    # Each set of URLs corresponds to a process for downloading tile maps
    print('Tiles downloading......')
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(download_tiles, urls_group)
    pool.close()
    pool.join()
    result = [x for j in results for x in j]
    print('Tiles download complete')

    # Combine downloaded tile maps into one map
    outpic = merge_tiles(result, left, top, right, bottom, zoom)
    outpic = outpic.convert('RGB')
#     print(outpic)
    r, g, b = cv2.split(np.array(outpic))

    # Get the spatial information of the four corners of the merged map and use it for outputting
    extent = getExtent(left, top, right, bottom, zoom, server)
    gt = (extent['LT'][0], (extent['RB'][0] - extent['LT'][0]) / r.shape[1], 0, extent['LT'][1], 0,
          (extent['RB'][1] - extent['LT'][1]) / r.shape[0])
#     saveTiff(r, g, b, gt, filePath)
    return np.array(outpic)

# Show coordinates
def print_coords(left, top, right, bottom, zoom):
    print(f"Coordinates: Left={left}, Top={top}, Right={right}, Bottom={bottom}, Zoom={zoom}")

# Get the geo ref of the given image array
def get_georeff(img_array, left, top, right, bottom, zoom):
    server = "Google"
    
    # Get the spatial information of the four corners of the merged map and use it for outputting
    extent = getExtent(left, top, right, bottom, zoom, server)
    gt = (extent['LT'][0], (extent['RB'][0] - extent['LT'][0]) / img_array.shape[1], 0, extent['LT'][1], 0,
          (extent['RB'][1] - extent['LT'][1]) / img_array.shape[0])
    return gt

def generate_coordinates(zoom, long_list, lat_list):
    # print(long_list, type(long_list))
    # print(lat_list, type(lat_list))
    long_list = long_list.tolist()
    lat_list = lat_list.tolist()
    coordinates = {
        "left": min(long_list),
        "top": max(lat_list),
        "right": max(long_list),
        "bottom": min(lat_list),
        "zoom": zoom
    }
    return coordinates

def generate_geometry(type, coordinates):
    roi = {
        "type": type,
        "coordinates": coordinates
    }

    features_collection = Feature(geometry=roi)
    roi_shape = [json.loads(dumps(features_collection))["geometry"]]
    return roi_shape

def clip_mask(final_mask, gt_mask, roi_shape, filePath):
    r, g, b = cv2.split(final_mask)
    tmp_name = str(uuid.uuid4()) + ".tif"
    saveTiff(r, g, b, gt_mask, filePath+tmp_name)
    with rsio.open(filePath+tmp_name, "r") as src:
        roi_clip, roi_trans = rasterio.mask.mask(src, roi_shape, filled=True, crop=True)
        roi_mask = np.transpose(roi_clip, (1, 2, 0))
    os.remove(filePath+tmp_name)
    return roi_mask, roi_trans


# Generate GeoJson file
def generate_geojson(red, gt):
    mask = red == 255
    gt_aff = aff.from_gdal(*gt)
    shapes = list(features.shapes(red, mask=mask, transform=gt_aff))
    features_list = []
    for polygon in shapes:
        features_list.append(Feature(geometry=polygon[0]))
    feature_collection = FeatureCollection(features_list)
    # with open(json_path, "w") as f:
    return dumps(feature_collection)

