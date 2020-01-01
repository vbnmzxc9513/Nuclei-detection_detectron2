import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

register_coco_instances("my_dataset", {}, "nucleus_cocoformat.json", "./train")
metadata = MetadataCatalog.get("my_dataset")
dataset_dicts = DatasetCatalog.get("my_dataset")

cfg = get_cfg()
cfg.merge_from_file("./model/mask_rcnn_R_50_FPN_3x.yaml")
cfg.OUTPUT_DIR = "./output"
#cfg.MODEL.WEIGHTS = os.path.join('model', "model_final_Cascade.pkl")  # if you have pre-trained weight.
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (data, fig, hazelnut)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) #  build output folder
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
