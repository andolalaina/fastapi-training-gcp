import os
import sys
import warnings
import numpy as np
import cv2
import skimage
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import PIL
from utils.configs import CropDataset, CropConfig, CropConfigInf
from utils.geo_utility import generate_input_img, get_georeff

sys.path.append("/usr/src/app/api_seg/Mask_RCNN")
from mrcnn import utils
from mrcnn.utils import compute_ap, compute_recall
from mrcnn.model import load_image_gt, mold_image
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

warnings.filterwarnings("ignore")

########################################################## TRAINING ##########################################################


# Training function
def train(model, config):
    """Train the model"""
    # Training dataset
    dataset_train = CropDataset()
    dataset_train.load_crop("/kaggle/input/mada-cropsegmentation", "train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = CropDataset()
    dataset_val.load_crop("/kaggle/input/mada-cropsegmentation", "val")
    dataset_val.prepare()
    
#     augmentations = iaa.Sequential([
#                     iaa.Fliplr(0.5), # horizontally flip 50% of the images
#                     iaa.Flipud(0.5),
#                     iaa.Affine(rotate=(-45, 45)) # rotate images between -45 and +45 degrees
#                 ], random_order=True)

    augmentations = iaa.Sometimes(3/4, iaa.OneOf([
                    iaa.Fliplr(0.5), # horizontally flip 50% of the images
                    iaa.Flipud(0.5),
                    iaa.Affine(rotate=(-45, 45)) # rotate images between -45 and +45 degrees
                ]))
    
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long.
    # Also, no need to train all layers, just the heads should do it
    print("Training network heads")
    model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE,
               epochs=100,
               layers='heads',
               augmentation=augmentations)

    
# Training Step function
def training_step(model_dir, weights_path):
    # By default: model_dir = DEFAULT_LOGS_DIR and weights_path = COCO_WEIGHTS_PATH
    config = CropConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_dir)
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"
    ])
    return model

# Show model train history
def show_history(model, epoch_num):
    history = model.keras_model.history.history
    epochs = range(epoch_num)

    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.plot(epochs, history['loss'], label="train loss")
    plt.plot(epochs, history['val_loss'], label="valid loss")
    plt.legend()
    plt.subplot(132)
    plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
    plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
    plt.legend()
    plt.subplot(133)
    plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
    plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
    plt.legend()
    plt.show()


########################################################## EVALUATION ##########################################################

# Evaluation function
def evaluate_model(dataset, model, config):
    APs = []
    ARs = []
    F1_scores = []
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, config, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, config)
        sample = np.expand_dims(scaled_image, 0)
        img_pred = model.detect(sample, verbose=0)
        r = img_pred[0]
        AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
        AR, positive_ids = compute_recall(r['rois'], gt_bbox, iou=0.2)
        ARs.append(AR)
        APs.append(AP)
        F1_scores.append((2* (np.mean(precisions) * np.mean(recalls)))/(np.mean(precisions) + np.mean(recalls)))
    mAP = np.mean(APs)
    mAR = np.mean(ARs)
    return mAP, mAR, F1_scores

# Evaluation steps function
def evaluation_step(inference_weights_path, model_dir):
    # By default model_dir = DEFAULT_LOGS_DIR
    config = CropConfigInf()
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
    model.load_weights(inference_weights_path, by_name=True)
    dataset_val = CropDataset()
    dataset_val.load_crop("/kaggle/input/mada-cropsegmentation", "val")
    dataset_val.prepare()
    mAP, mAR, F1_scores = evaluate_model(dataset_val, model, config)
    print("mAP: %.3f" % mAP)
    print("mAR: %.3f" % mAR)
    print("Mean F1_score: %.3f" % np.mean(F1_scores))

########################################################## INFERENCE ##########################################################

# Mask application function
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * 0 + color[c], #(1 - alpha) + alpha * color[c] * 255
                                  image[:, :, c])
    return image

# Generate Mask image
def get_masked_image(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=False, show_bbox=True,
                      colors=[255, 255, 255], captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    masked_image_1 = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)
    masked_image_2 = image.astype(np.uint8).copy()
#     final_mask = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     final_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB)
    result = None
    for i in range(N):

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image_1 = apply_mask(masked_image_1, mask, colors)
            result = masked_image_1
        else:
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = skimage.measure.find_contours(padded_mask, 0)
            points = np.fliplr(contours[0]) - 1
            points = points.reshape((-1, 1, 2))
            # drawing polygons into the image
            masked_image_2 = cv2.drawContours(masked_image_2, [np.int32(points)], -1, (255, 255, 255), 1)
            result = masked_image_2
    return result

# Get ploting axes settings
def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Generate inference model
def generate_inf_model(model_dir, inference_weights_path):
    # By default model_dir = DEFAULT_LOGS_DIR
    config = CropConfigInf()
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
    model.load_weights(inference_weights_path, by_name=True)
    model.keras_model._make_predict_function()
    return model

# Inference using JPG file
def infer_jpg(jpg_path, model):
    # with tf.device('/device:GPU:0'):
    img = mpimg.imread(jpg_path)
    result = model.detect([img], verbose=1)
    ax = get_ax(1)
    r1 = result[0]
    visualize.display_instances(img, r1['rois'], r1['masks'], r1['class_ids'],
    ['BG', 'crop'], r1['scores'], ax=ax, title="Predictions1")

# Inference pipeline
def pl_inference(model, left, top, right, bottom, zoom=17):
    server = "Google"
    style = "s"
    input_img = generate_input_img(left, top, right, bottom, zoom)
    print(f"Image Shape : {input_img.shape[0]}x{input_img.shape[1]}")
    # print(device_lib.list_local_devices())
    # with tf.device("/device:GPU:0"):
    result = model.detect([input_img], verbose=1)
    r1 = result[0]
    contours_mask = get_masked_image(input_img, r1['rois'], r1['masks'], r1['class_ids'], ['BG', 'crop'])
    final_mask = get_masked_image(input_img, r1['rois'], r1['masks'], r1['class_ids'], ['BG', 'crop'], show_mask=True)
    gt = get_georeff(final_mask, left, top, right, bottom, zoom)
    return final_mask, contours_mask, gt

# Model Class
class Model_MRCNN():
    def __init__(self):
        self.model_dir = "/usr/src/app/api_seg/model/logs"
        self.weights = "/usr/src/app/api_seg/model/mask_rcnn_crop_0057.h5"
        self.model = generate_inf_model(self.model_dir, self.weights)