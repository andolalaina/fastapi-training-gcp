from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import requests
import json
from utils.segmentation import Model_MRCNN, pl_inference, generate_inf_model
from utils.geo_utility import generate_coordinates, generate_geojson, get_georeff, generate_geometry, clip_mask
import warnings
from pydantic import BaseModel

warnings.filterwarnings("ignore")

model_dir = "/usr/src/app/api_seg/model/logs"
inference_weights_path = "/usr/src/app/api_seg/model/mask_rcnn_crop_0057.h5"
filePath = "/usr/src/app/api_seg/output/"
model = generate_inf_model(model_dir, inference_weights_path)
app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
)


class Coords(BaseModel):
    type: str
    coordinates: list

@app.post("/get_crop_seg")
def get_crop_seg(coords: Coords):
    # generate roi shape
    print(f"Generate ROI SHAPE {'#' * 60}")
    roi_shape = generate_geometry(coords.type, coords.coordinates)
    print(f"Generate ROI SHAPE Done! {'#' * 60}")
    # get coordinates list and generate left, top, right and bottom coords
    print(f"Generate COORDINATES {'#' * 60}")
    coordinates_list = coords.coordinates[0]
    coord_arr = np.array(coordinates_list).T
    coordinates = generate_coordinates(17, *coord_arr)
    print(f"Generate COORDINATES Done! {'#' * 60}")
    # get the inference mask and transforms
    print(f"Generate INFERENCE MASK {'#' * 60}")
    final_mask, contours_mask, gt = pl_inference(model, **coordinates)
    print(f"Generate INFERENCE MASK Done! {'#' * 60}")
    print(f"Generate ROI MASK {'#' * 60}")
    roi_mask, roi_trans = clip_mask(final_mask, gt, roi_shape, filePath)
    r, g, b = cv2.split(roi_mask)
    json_str = generate_geojson(r, roi_trans.to_gdal())
    print(f"Generate ROI MASK Done! {'#' * 60}")
    return json.loads(json_str)

@app.post("/get_classification")
def get_class(coords: Coords):
    data = {
        "type": coords.type,
        "coordinates": coords.coordinates
    }
    result = requests.post("http://10.2.54.130:8000/classifications", json.dumps(data)).json()
    print(result)
    return result
