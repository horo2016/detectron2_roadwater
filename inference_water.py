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

from detectron2.utils.visualizer import ColorMode

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
#for d in ["train", "val"]:
    #DatasetCatalog.register("BLOCK_" + d, lambda d=d: get_dicts(path + "/" +  d))
    #MetadataCatalog.get("BLOCK_" + d).set(thing_classes=["water"])
    



if __name__ == '__main__':
    #load config
    cfg = get_cfg() 
    #config output path 
    cfg.OUTPUT_DIR = "logs" 
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.merge_from_file(
        "../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    )
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (BLOCK)
    cfg.DATASETS.TEST = ("BLOCK_val", )
    predictor = DefaultPredictor(cfg)
    dataset_dicts = get_dicts(path + "/" + "val")
    for d in random.sample(dataset_dicts, 1):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("BLOCK_train"), scale=0.8,instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        res = v.get_image()[:, :, ::-1]
        cv2.imshow("res",res)
        cv2.waitKey(0)
        cv2.imwrite("3.jpg",res)