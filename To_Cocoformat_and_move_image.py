#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from os import listdir
from os.path import isfile, isdir, join
import shutil
import cv2
import numpy as np
import json
inpath = "./train_folder/"  # the train folder download from kaggle
outpath = "./train/"  # the folder putting all nuclei image
images_name = listdir(inpath)
cocoformat = {"licenses": [], "info": [], "images": [],
              "annotations": [], "categories": []}
# categories

# In[ ]:
cat = {"id": 1,
       "name": "nucleus",
       "supercategory": "nucleus",
       }
cocoformat["categories"].append(cat)
# images + annotations
# In[ ]:
mask_id = 1
for i, im_name in enumerate(images_name):
    t_image = cv2.imread(inpath + im_name + "/images/" + im_name + ".png")
    mask_folder = listdir(inpath + im_name + "/masks/")
    im = {"id": int(i+1),
          "width": int(t_image.shape[1]),
          "height": int(t_image.shape[0]),
          "file_name": im_name + ".png",
          }
    cocoformat["images"].append(im)
    for mask in mask_folder:
        t_image = cv2.imread(inpath + im_name + "/masks/" + mask, 0)
        ret, binary = cv2.threshold(
            t_image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        t_seg = np.where(t_image[:, :] == 255)

        all_seg_in_mask = []
        for s in range(len(contours)):
            seg = []
            for x in range(len(contours[s])):
                seg.append(int(contours[s][x][0][0]))
                seg.append(int(contours[s][x][0][1]))
            all_seg_in_mask.append(seg)
        ann = {"id": int(mask_id),
               "image_id": int(i) + 1,
               "category_id": int(1),
               "segmentation": all_seg_in_mask,
               "area": float(len(t_seg[0])),
               "bbox": [int(np.min(t_seg[1])), int(np.min(t_seg[0])),
                        int(np.max(t_seg[1]) - np.min(t_seg[1])),
                        int(np.max(t_seg[0]) - np.min(t_seg[0]))],
               "iscrowd": 0,
               }
        mask_id = mask_i + 1
        cocoformat["annotations"].append(ann)

# In[ ]:
with open("nucleus_cocoformat.json", "w") as f:
    json.dump(cocoformat, f)
# copy image to one folder

# In[ ]:
# use loop to save images
for f in files:
    image = listdir(inpath + f + "/images/")
    shutil.copyfile(inpath + f + "/images/" + image[0], outpath + image[0])
