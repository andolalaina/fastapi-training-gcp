import os
import sys
import urllib.request as ur
import numpy as np
import json
import skimage
from threading import Thread
import multiprocessing

sys.path.append("/usr/src/app/api_seg/Mask_RCNN")
from mrcnn import utils
from mrcnn.config import Config




# Training step Configurations
class CropConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "crop"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + crop

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50
    
    # Validation steps
    VALIDATION_STEPS = 50
    
    # Input image resizing
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # Diseable using mini mask
#     USE_MINI_MASK = False
    # Length of square anchor side in pixels
#     RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

# Inference step Configuration
class CropConfigInf(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "crop"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + crop

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    
#     IMAGE_MIN_DIM = 512
#     IMAGE_MAX_DIM = 2048

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.70
    
    DETECTION_MAX_INSTANCES = 1000

# Dataset Configuration
class CropDataset(utils.Dataset):
    
    def load_crop(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("crop", 1, "crop")
        
        # Train or validation dataset
        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Load annotations
        # change the via_xxxxxx.json to the name of the annotation file from VIA
        # NB: In VIA 2.x, regions was changed from a dict to a list
        annotations = json.load(open(os.path.join(dataset_dir, f"via_crop_{subset}.json")))
        annotations = list(annotations.values())
        
        # Skip unannotated images
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. Test are stores in the
            # shape_attributes
            # The if condition is needed to support VIA both versions 1.x and 2.x
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must reat
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            self.add_image(
                "crop", 
                image_id=a['filename'],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons)
            
        return None
    
    def load_mask(self, image_id):
        """ Generate instance masks for an image
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        
        image_info = self.image_info[image_id]
        if image_info["source"] != "crop":
            return super(self.__class__, self).load_mask(image_id)
        
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                       dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr[rr > mask.shape[0]-1] = mask.shape[0]-1
            cc[cc > mask.shape[1]-1] = mask.shape[1]-1
            mask[rr, cc, i] = 1
        
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    
    def image_reference(self, image_id):
        """Return the path of the image"""
        info = self.image_info[image_id]
        if info["source"] == "crop":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
                            

# ---------------------------------------------------------
class Downloader(Thread):
    # multiple threads downloader
    def __init__(self, index, count, urls, datas):
        # index represents the number of threads
        # count represents the total number of threads
        # urls represents the list of URLs nedd to be downloaded
        # datas represents the list of data need to be returned.
        super().__init__()
        self.urls = urls
        self.datas = datas
        self.index = index
        self.count = count

    def download(self, url):
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 Edg/88.0.705.68'}
        header = ur.Request(url, headers=HEADERS)
        err = 0
        while (err < 3):
            try:
                data = ur.urlopen(header).read()
            except:
                err += 1
            else:
                return data
        raise Exception("Bad network link.")

    def run(self):
        for i, url in enumerate(self.urls):
            if i % self.count != self.index:
                continue
            self.datas[i] = self.download(url)