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
print(torch.__version__, torch.cuda.is_available()) # 1.5.0+cu101 True
setup_logger()


def get_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_export_json.json")
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
            print(objs)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
    
    

path = "roadWater_datasets" # path to your image folder
for d in ["train", "val"]:
    DatasetCatalog.register("BLOCK_" + d, lambda d=d: get_dicts(path + "/" +  d))
    MetadataCatalog.get("BLOCK_" + d).set(thing_classes=["block"])
    

dataset_dicts = get_dicts(path + "/" + "train")
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("BLOCK_train"), scale=1.0)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image()[:, :, ::-1])
    plt.show()
    cv2.imwrite("1.jpg",vis.get_image()[:, :, ::-1])