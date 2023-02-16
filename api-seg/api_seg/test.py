import cv2
import json
import numpy as np
from affine import Affine as aff
import rasterio as rsio
import rasterio.mask
from rasterio import features
from geojson import Feature, FeatureCollection, dumps
from utils.configs import CropConfig, CropConfigInf, CropDataset
from utils.segmentation import pl_inference, generate_inf_model
from utils.geo_utility import saveTiff, generate_coordinates, get_georeff, generate_geometry, clip_mask
import warnings

warnings.filterwarnings("ignore")

# Generate GeoJson file
def generate_geojson_1(red, gt, roi):
    red_roi = rsio.mask.mask(red, roi, crop=True)
    mask = red_roi == 255
    gt_aff = aff.from_gdal(*gt)
    shapes = list(features.shapes(red, mask=mask, transform=gt_aff))
    features_list = []
    for polygon in shapes:
        features_list.append(Feature(geometry=polygon[0]))
    feature_collection = FeatureCollection(features_list)
    # with open(json_path, "w") as f:
    return dumps(feature_collection)

def main():
    # coordinates = {
    #     "left": 46.626621,
    #     "top": -16.073295,
    #     "right": 46.632174,
    #     "bottom": -16.077231,
    #     "zoom": 17
    # }
    roi = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            46.597688,
                            -16.073599
                        ],
                        [
                            46.602706,
                            -16.073434
                        ],
                        [
                            46.602663,
                            -16.076278
                        ],
                        [
                            46.597216,
                            -16.076443
                        ],
                        [
                            46.597688,
                            -16.073599
                        ]
                    ]
                ]
            }
    # features_collection = Feature(geometry=roi)
    # print(type(features_collection))
    # print(dumps(features_collection))
    # roi_shape = dumps(features_collection)
    # print([json.loads(roi_shape)["geometry"]])
    print(f"Generate ROI SHAPE {'#' * 10}")
    roi_shape = generate_geometry(**roi)
    # print(roi_shape)
    coordinates = generate_coordinates(17, *np.array(roi['coordinates'][0]).T)
    print(coordinates)
    model_dir = "/usr/src/app/api_seg/model/logs"
    inference_weights_path = "/usr/src/app/api_seg/model/mask_rcnn_crop_0057.h5"
    filePath = "/usr/src/app/api_seg/output/"

    model = generate_inf_model(model_dir, inference_weights_path)
    final_mask, contours_mask, gt = pl_inference(model, **coordinates)
    # print(final_mask.shape)
    # r, g, b = cv2.split(final_mask)
    # saveTiff(r, g, b, gt, filePath+"test_auto_1.tif")
    # with rsio.open(filePath+"test_auto_1.tif", "r") as src:
    #     roi_final_mask, roi_f_trans = rasterio.mask.mask(src, roi_shape, crop=True)
    #     print(roi_final_mask.shape)
    #     print(np.transpose(roi_final_mask, (1, 2, 0)).shape)
    #     roi_img_mask = np.transpose(roi_final_mask, (1, 2, 0))
    #     r, g, b = cv2.split(roi_img_mask)
    #     gt_roi = get_georeff(roi_img_mask, **coordinates)
    #     saveTiff(r, g, b, gt_roi, filePath+f"test_auto_1_crop.tif")
    #     print(f"File saved at {filePath}test_auto_1_crop.tif")
    roi_mask, roi_trans = clip_mask(final_mask, gt, roi_shape, filePath)
    r, g, b = cv2.split(roi_mask)
    gt_roi = get_georeff(roi_mask, **coordinates)
    saveTiff(r, g, b, roi_trans.to_gdal(), filePath+f"test_auto_tmp_crop.tif")
    print(f"File saved at {filePath}test_auto_tmp_crop.tif")


if __name__ == "__main__":
    main()