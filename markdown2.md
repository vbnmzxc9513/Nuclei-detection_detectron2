## Modification guide


train_Cascade.py is the file we use for training. In this file, you are going to 
1. (20) Store the images_folder and annotations to the register. You can use the different folder name if you named the folder yourself. 
2. (24~38) Setup configurarion. There are a lot of base configurarion import from Config.Defaultconfig. If you want to set detail you can go to see the config folder and the .py file which named defaults.py.
* If you do not set it up, it will loadtheconfiguration from the backbone you set.
3. (43) Start training!
* After training, the weight will store in output file.

To_Cocoformat_and_move_image.ipynb. 
1. (1 cell) inpath set the folder name which download from kaggle. outpath set the folder name you want to store image. You should set the name to train folder and run all folder. Next set the variable to test folder name and run all whithout 2, 3, 4 cells.
2. (4 cell) you can set your JSON name yourself.

data_augmentation.ipynb
1. (3 cell) build the folder to save your augmentation images.
2. (8 cell) set the input folder path including the images which we want to augmentation.
3. (10 cell) set output name which you have built in third cell.
4. (above) choice the item you want to augmentation and run augmentor.applyAugmentation() to start.
* After augmentor.applyAugmentation() the augmentation will finish.

submission_and_visualize.ipynb
1. (3 cell) load data set and register it for detectron2 dataset. RLE function for decode. Then you should adjust the config data for model configuration. You should load the correct model weight and model config with .yaml file. 
2. (1 cell) You can select one image file for prediction your testdata.
3. (2 cell) Predict the test data and write for submission with all predict weight, the weight should in the out direction. Due to Kaggle competition evaluation, we resolve the predict overlap problem by the postprocess_masks function.  
