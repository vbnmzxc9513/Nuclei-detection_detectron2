## train_Cascade.py :  
use ```python train_Cascade.py``` In begining, choose your path for .yaml to load model config and modify some parameter for train, then load pretrain weight. After loading configuration, starting trainging. If finish, you can get model weight in your output directory.  

## To_Cocoformat_and_move_image.ipynb :  
Demo notebook for convert cocoformat, use it to convert your mask data label. It transform the binary mask image to ploy point like (x y x y x y ......) by contour edges in the image. And move the kaggle dataset images to one selected folder.  

## data_augmentation.ipynb	:  
Used for data augmentation. It can produce additional data for model training. This file is mainly using the tool named CLoDSA which devloped by Joheras.  

## submission_and_visualize.ipynb :  
If finish training, you can use it for predict your test data, please check your model config path with .yaml file and load model weight with XXX.pth file loading correctly. You can visualize your model predict data for checking your predict weather correct, then run all to predict test data for the submission.


