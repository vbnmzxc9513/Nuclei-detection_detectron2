#### train_Cascade.py :  
use ```python train_Cascade.py``` In begining, choose your path for .yaml to load model config and modify some parameter for train, then load pretrain weight. After loading configuration, starting trainging. If finish, you can get model weight in your output directory.  

##### To_Cocoformat_and_move_image.ipynb : Demo notebook for convert cocoformat, use it to convert your mask data label.  
##### data_augmentation.ipynb	: Used for data augmentation. It can produce additional data for model training.  
##### submission_and_visualize.ipynb : If finish training, you can use it for predict your test data, just check your model frame and model weight loading correctly, and run all.
