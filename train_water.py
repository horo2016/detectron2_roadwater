import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer

from detectron2.engine import DefaultTrainer 
from detectron2.config import get_cfg 
from detectron2 import model_zoo 
from detectron2.engine import DefaultPredictor
#print(torch.__version__, torch.cuda.is_available()) # 1.5.0+cu101 True
#setup_logger()


def get_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_export_json.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for anno in annos:
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
                }
            objs.append(obj)
            #print(objs)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
    
    
path = "roadWater_datasets" # path to your image folder
for d in ["train", "val"]:
    DatasetCatalog.register("BLOCK_" + d, lambda d=d: get_dicts(path + "/" +  d))
    MetadataCatalog.get("BLOCK_" + d).set(thing_classes=["water"])
    



if __name__ == '__main__':
    #load config
    cfg = get_cfg() 
    #config output path 
    cfg.OUTPUT_DIR = "logs" 
    #load MASK RCNN MODEL
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    

    cfg.DATASETS.TRAIN = ("BLOCK_train",)     # our training dataset
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2     # number of parallel data loading workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")     # use pretrained weights
    cfg.SOLVER.IMS_PER_BATCH = 2     # in 1 iteration the model sees 2 images
    cfg.SOLVER.BASE_LR = 0.00025     # learning rate
     
    cfg.SOLVER.MAX_ITER = 500        # number of iteration
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128     # number of proposals to sample for training
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (BLOCK)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()