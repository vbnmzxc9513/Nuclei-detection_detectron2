## File explanation

train_Cascade.py is the file we use for training. In this file, you are going to 
1. (21) Store the images_folder and annotations to the register.
2. (25~36) Setup configurarion. There are a lot of base configurarion import from Config.Defaultconfig.
* If you do not set it up, it will loadtheconfiguration from the backbone you set.
3. (41) Start training!
* After training, the weight will store in output file.

detectron_test_mod_mult.py is the file we use for generating multiple submission files by check point at a time. In this file, you are going to
1. Same step as step 1-3 as above, expect that this time we only need to load testing data.
* Becuase we do not have annotation file for test data, we just send in toy information, since we are not going to use it; However if you want to do validation, you can follow thwe guide in train part of how to use it.
2. (95) Start testing in loop! Each loop would load onecheckpoint and generate one output.

## Modification guide

Since you may want to customize the training and testing, heres some quick notes:
* (26) You can load your annotatiobn bythisfunction
* (46) May be crucial to understand your own annotation's format
* (49) If your category start from 0, remove -1 (alsochange in test)
* (62) Set up class name in thing_classes corresponmd to id
* (80) Load the config file you like. Config files can be found at config directory.
* (85) Load the weight. Can found the pretrain weight from the config file you select, or change it to the located you put if you had prepared by yourself.
* (92) Need to modify your class count
* Same guidline can be apply to test process, and remember to syncronize them if you are going to do them in a sequence.
