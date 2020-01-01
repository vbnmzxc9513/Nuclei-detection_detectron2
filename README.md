# NVRDL2019FALL Data Science Bowl 2018

This is a project of implement instance segementation on data science bowl challenge with with 1349 labeled image and 100 tested image.
We had implement the task base on following backbone modules:
1. Detectron2, from https://github.com/facebookresearch/detectron2 or https://github.com/conansherry/detectron2 for windows build

The code We  had written were 
1. To_Cocoformat_and_moveImg.ipynb https://github.com/vbnmzxc9513/Nuclei-detection/blob/master/To_Cocoformat_and_moveImg.ipynb
and 
2. train_Cascade.py https://github.com/vbnmzxc9513/Nuclei-detection/blob/master/train_Cascade.py

3. submission_and_visualize.ipynb https://github.com/vbnmzxc9513/Nuclei-detection/blob/master/submission_and_visualize.ipynb
Both of these file were start modify from detectron2's colab notebook https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04
- Intel® Xeon® Gold 6136 Processor @ 3.0GHHz
- 1x NVIDIA NVIDIA TESLA V100 

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Train models](#train-models)
4. [Make Submission](#make-submission)

## Installation
### Windows Version ###
Using Anaconda is strongly recommended.
```
conda create -n torchdet python=3.6
activate torchdet
```
#### Note: following part from detectron2's repository ####

### Requirements
- Python >= 3.6(Conda)
- PyTorch 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, needed by demo and visualization
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install git+https://github.com/facebookresearch/fvcore`
- pycocotools: `pip install cython; pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`
- VS2019(no test in older version)/CUDA10.1(no test in older version)
* Self Note: The installation of pycocotools may be influence by some library, so the installation order may be importance. Make sure to open a new environment for minimum the risk of crash the environment. 

### Several files must be changed by manually
```
file1: 
  {your evn path}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h(190)
    static constexpr size_t DEPTH_LIMIT = 128;
      change to -->
    static const size_t DEPTH_LIMIT = 128;
file2: 
  {your evn path}\Lib\site-packages\torch\include\pybind11\cast.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\pybind11\cast.h(1449)
    explicit operator type&() { return *(this->value); }
      change to -->
    explicit operator type&() { return *((type*)this->value); }
```

### Build detectron2

After having the above dependencies, run:
```
conda activate {your env}

"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

git clone https://github.com/conansherry/detectron2

cd detectron2

python setup.py build develop
```
Note: you may need to rebuild detectron2 after reinstalling a different build of PyTorch.

#### Note: above part from detectron2's repository ####

### Linux Version ###

#### Note: below part from detectron2's repository ####


The [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has step-by-step instructions that install detectron2.
The [Dockerfile](https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile)
also installs detectron2 with a few simple commands.

### Requirements
- Linux or macOS
- Python ≥ 3.6
- PyTorch ≥ 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, needed by demo and visualization
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- GCC ≥ 4.9


### Build and Install Detectron2

After having the above dependencies, run:
```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install -e .

# or, as an alternative to `pip install`, use
# python setup.py build develop
```
Note: you often need to rebuild detectron2 after reinstalling PyTorch.

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
Undefined torch/aten/caffe2 symbols, or segmentation fault immediately when running the library.
</summary>

This can happen if detectron2 or torchvision is not
compiled with the version of PyTorch you're running.

If you use a pre-built torchvision, uninstall torchvision & pytorch, and reinstall them
following [pytorch.org](http://pytorch.org).
If you manually build detectron2 or torchvision, remove the files you built (`build/`, `**/*.so`)
and rebuild them.

If you cannot resolve the problem, please include the output of `gdb -ex "r" -ex "bt" -ex "quit" --args python -m detectron2.utils.collect_env`
in your issue.
</details>

<details>
<summary>
Undefined C++ symbols in `detectron2/_C*.so`.
</summary>
Usually it's because the library is compiled with a newer C++ compiler but run with an old C++ run time.
This can happen with old anaconda.

Try `conda update libgcc`. Then remove the files you built (`build/`, `**/*.so`) and rebuild them.
</details>

<details>
<summary>
"Not compiled with GPU support" or "Detectron2 CUDA Compiler: not available".
</summary>
CUDA is not found when building detectron2.
You should make sure

```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```

print valid outputs at the time you build detectron2.
</details>

<details>
<summary>
"invalid device function" or "no kernel image is available for execution".
</summary>

Two possibilities:

* You build detectron2 with one version of CUDA but run it with a different version.

  To check whether it is the case,
  use `python -m detectron2.utils.collect_env` to find out inconsistent CUDA versions.
	In the output of this command, you should expect "Detectron2 CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA"
	to contain cuda libraries of the same version.

	When they are inconsistent,
	you need to either install a different build of PyTorch (or build by yourself)
	to match your local CUDA installation, or install a different version of CUDA to match PyTorch.

* Detectron2 or PyTorch/torchvision is not built with the correct compute compatibility for the GPU model.

	The compute compatibility for PyTorch is available in `python -m detectron2.utils.collect_env`.

	The compute compatibility of detectron2/torchvision defaults to match the GPU found on the machine
	during building, and can be controlled by `TORCH_CUDA_ARCH_LIST` environment variable during building.

	Visit [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) to find out
	the correct compute compatibility for your device.

</details>

<details>
<summary>
Undefined CUDA symbols or cannot open libcudart.so.
</summary>

The version of NVCC you use to build detectron2 or torchvision does
not match the version of CUDA you are running with.
This often happens when using anaconda's CUDA runtime.

Use `python -m detectron2.utils.collect_env` to find out inconsistent CUDA versions.
In the output of this command, you should expect "Detectron2 CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA"
to contain cuda libraries of the same version.

When they are inconsistent,
you need to either install a different build of PyTorch (or build by yourself)
to match your local CUDA installation, or install a different version of CUDA to match PyTorch.
</details>


<details>
<summary>
"ImportError: cannot import name '_C'".
</summary>
Please build and install detectron2 following the instructions above.
</details>

#### Note: above part from detectron2's repository ####

## Dataset Preparation

### Download files
From https://www.kaggle.com/c/data-science-bowl-2018/data download all the file.
After unzip renamed stage1_train folder named train_folder, and renamed stage2_test folder named test-folder.
You may need to put the training file in root directory of your detectron2's installation path:
```
Detectron2
+- ...
+- train_folder
+- test_folder
```
From https://github.com/vbnmzxc9513/Nuclei-detection download To_Cocoformat_and_moveImg.ipynb, submission_and_visualize.ipynb	 and train_Cascade.py, and located them in root directory as well:
```
Detectron2
+- ...
+- train_folder
+- test_folder
+- To_Cocoformat_and_moveImg.ipynb	
+- submission_and_visualize.ipynb
+- train_Cascade.py
```
### Dataset Preprocessing ###
To let the file can be used by our code, a prior dataset preprocessing may be requied.
First, running following command:
```
jupyter notebook
```
The browser would open a web page where you can execute the ipynb files.
Find the "To_Cocoformat_and_moveImg.ipynb" file and execute it.

To prepare the training data, simply execute all the cells.
After preparing training data, change the input path and output path into:
```
inpath = "./test_folder/"
outpath = "./test/"
```
Then, run through all the cell except 2nd, 3rd and 4th cell to finish the data preprocessing.
Here we provide a structure visualization to explain what change on folder strcture would be made:

## Folder before Processing
![image](https://github.com/vbnmzxc9513/Nuclei-detection/blob/master/demo/trainfolder_before.png)
## Folder after Processing
![image](https://github.com/vbnmzxc9513/Nuclei-detection/blob/master/demo/trainfolder_after.png)


## Train models
To train models, run following commands.
```
python train_Cascade.py
```
After training, it may generate a folder named "output", with weight file named (iteration_count).pth

## Make Submission
After training, from jupyter notebook, execute "submission_and_visualize.ipynb"
Simply run through all the cell and the result would be generate.

## File Explanation
### train_Cascade.py :  
In begining, choose your path for .yaml to load model config and modify some parameter for train, then load pretrain weight. After loading configuration, starting trainging. Use ```python train_Cascade.py```.   If finish, you can get model weight in your output directory.  

### To_Cocoformat_and_move_image.ipynb :  
Demo notebook for convert cocoformat, use it to convert your mask data label. It transform the binary mask image to ploy point like (x y x y x y ......) by contour edges in the image. And move the kaggle dataset images to one selected folder.  

### data_augmentation.ipynb	:  
Used for data augmentation. It can produce additional data for model training. This file is mainly using the tool named CLoDSA which devloped by Joheras.  

### submission_and_visualize.ipynb :  
If finish training, you can use it for predict your test data, please check your model config path with .yaml file and load model weight with XXX.pth file loading correctly. You can visualize your model predict data for checking your predict weather correct, then run all to predict test data for the submission.

## Demo Prediction Result
![image](https://github.com/vbnmzxc9513/Nuclei-detection/blob/master/demo/demo.png)

## Reference  
Refrence : https://github.com/facebookresearch/detectron2   
