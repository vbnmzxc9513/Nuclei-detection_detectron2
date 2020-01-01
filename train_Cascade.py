import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "train_augmentation2_annotation.json", "./train_augmentation2")

metadata = MetadataCatalog.get("my_dataset")
dataset_dicts = DatasetCatalog.get("my_dataset")


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file("./model/cascade_mask_rcnn_R_50_FPN_3x.yaml")
cfg.OUTPUT_DIR = "./output_augment_Cascade"
cfg.MODEL.WEIGHTS = os.path.join('output_Cascade', "/model_0009999.pth")
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 32, 64, 128, 256, 512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]
cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
cfg.TEST.DETECTIONS_PER_IMAGE = 500
cfg.SOLVER.WARMUP_FACTOR = 0.01
cfg.SOLVER.WARMUP_ITERS = 10000
cfg.INPUT.CROP.SIZE = [1.0, 1.0]

#cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 20

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
