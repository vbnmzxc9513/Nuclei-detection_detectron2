#!/usr/bin/env python
# coding: utf-8

# # Augmenting a dataset for instance segmentation
# 
# In this notebook, we illustrate how CLODSA can be employed to augment a dataset of images devoted to instance segmentation that was annotated using the [COCO format](http://cocodataset.org/#home). 

# We will use a small dataset of shapes. 

# In[25]:


get_ipython().system('wget https://www.dropbox.com/s/eus5u6ug1s7ftzc/shapes.zip?dl=0 -O shapes.zip')
get_ipython().system('unzip shapes.zip')


# We can check the elements of the shapes folder that are a json file with the coco annotation and two images.

# In[ ]:


get_ipython().system('ls shapes/')


# We create the output folder.

# In[1]:


get_ipython().system('mkdir train_augmentation8')


# ## Augmentation techniques
# 
# For this example, we consider three augmentation techniques. 
# 
# The augmentation techniques applied in this example are:
# - Rotation.
# - Flip.

# ## Installing the necessary libraries
# 
# In case that CLODSA is not installed in your system, the first task consists in installing it using ``pip``.

# In[ ]:


get_ipython().system('pip install clodsa')


# ## Loading the necessary libraries
# 
# The first step in the pipeline consists in loading the necessary libraries to apply the data augmentation techniques in CLODSA.

# In[2]:


from matplotlib import pyplot as plt
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Creating the augmentor object
# 
# As explained in the documentation of CLODSA, we need to specify some parameters for the augmentation process, and use them to create an augmentor object.  
# 
# _The kind of problem_. In this case, we are working in an instance segmentation problem.

# In[3]:


PROBLEM = "instance_segmentation"


# _The annotation mode_. The annotation is provided using the coco format in a file called annotations.json. 

# In[4]:


ANNOTATION_MODE = "coco"


# _The input path_. The input path containing the images. 

# In[5]:


INPUT_PATH = "./train"


# _The generation mode_. In this case, linear, that is, all the augmentation techniques are applied to all the images of the original dataset. 

# In[6]:


GENERATION_MODE = "linear"


# _The output mode_. The generated images will be stored in a new folder called output.  

# In[7]:


OUTPUT_MODE = "coco"
OUTPUT_PATH= "train_augmentation8/"


# Using the above information, we can create our augmentor object. 

# In[8]:


augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH})


# ## Adding the augmentation techniques
# 
# Now, we define the techniques that will be applied in our augmentation process and add them to our augmentor object. To illustrate the transformations, we will use the following image of the dataset. 
# 
# 

# First of all, we must define a transformer generator.

# In[10]:


transformer = transformerGenerator(PROBLEM)


# _Rotations:_

# In[9]:


for angle in [30,60,210,270,90]:
    rotate = createTechnique("rotate", {"angle" : angle})
    augmentor.addTransformer(transformer(rotate))


# In[11]:


rotate = createTechnique("rotate", {"angle" : 0})
augmentor.addTransformer(transformer(rotate))


# _Flips:_

# In[12]:


flip1 = createTechnique("flip",{"flip":1})
augmentor.addTransformer(transformer(flip1))
flip0 = createTechnique("flip",{"flip":0})
augmentor.addTransformer(transformer(flip0))


# In[13]:


flip_1 = createTechnique("flip",{"flip":-1})
augmentor.addTransformer(transformer(flip_1))


# Invert 

# In[11]:


invert = createTechnique("invert",{})
augmentor.addTransformer(transformer(invert))


# Bilateral blurring

# In[11]:


Bblurring = createTechnique("bilateral_blurring", {"diameter" : 11, "sigmaColor": 21, "sigmaSpace":7})
augmentor.addTransformer(transformer(Bblurring))


# Gaussian blurring

# In[12]:


Gaussian = createTechnique("gaussian_blur", {"kernel" : 3})
augmentor.addTransformer(transformer(Gaussian))


# Gaussian noise

# In[13]:


gaussian_noise = createTechnique("gaussian_noise", {"mean" : 0,"sigma":10})
augmentor.addTransformer(transformer(gaussian_noise))


# Average blurring

# In[14]:


average_blurring = createTechnique("average_blurring",{"kernel":3})
augmentor.addTransformer(transformer(average_blurring))


# Blurring

# In[14]:


blurring = createTechnique("blurring", {"ksize" : 3})
augmentor.addTransformer(transformer(blurring))


# Dropout

# In[15]:


dropout = createTechnique("dropout",{"percentage":0.02})
augmentor.addTransformer(transformer(dropout))


# Dropout2

# In[15]:


dropout2 = createTechnique("dropout",{"percentage":0.03})
augmentor.addTransformer(transformer(dropout2))


# Dropout3

# In[16]:


dropout2 = createTechnique("dropout",{"percentage":0.08})
augmentor.addTransformer(transformer(dropout2))


# Elastic deformation

# In[18]:


Elastic = createTechnique("elastic",{"alpha":5,"sigma":0.05})
augmentor.addTransformer(transformer(Elastic))


# Gamma correction

# In[11]:


Gamma = createTechnique("gamma",{"gamma":1.5})
augmentor.addTransformer(transformer(Gamma))


# Salt and Pepper

# In[16]:


salt_and_pepper = createTechnique("salt_and_pepper", {"low" : 0,"up":10})
augmentor.addTransformer(transformer(salt_and_pepper))


# Translation

# In[18]:


translation = createTechnique("translation", {"x":10,"y":10})
augmentor.addTransformer(transformer(translation))


# In[22]:


translation2 = createTechnique("translation", {"x":6,"y":8})
augmentor.addTransformer(transformer(translation2))


# Crop

# In[19]:


crop = createTechnique("crop",{"percentage":0.6,"startFrom": "CENTER"})
augmentor.addTransformer(transformer(crop))
crop2 = createTechnique("crop",{"percentage":0.6,"startFrom": "TOPLEFT"})
augmentor.addTransformer(transformer(crop2))
crop3 = createTechnique("crop",{"percentage":0.6,"startFrom": "BOTTOMRIGHT"})
augmentor.addTransformer(transformer(crop3))


# Histgram

# In[12]:


t = createTechnique("equalize_histogram",{})
augmentor.addTransformer(transformer(t))


# In[13]:


t = createTechnique("raise_blue", {"power" : 0.8})
augmentor.addTransformer(transformer(t))


# In[15]:


t = createTechnique("raise_hue", {"power" : 0.8})
augmentor.addTransformer(transformer(t))


# In[16]:


t = createTechnique("raise_value", {"power" : 0.8})
augmentor.addTransformer(transformer(t))


# ## Applying the augmentation process
# 
# Finally, we apply the augmentation process (this might take some time depending on the number of images of the original dataset and the number of transformations that will be applied). 

# In[18]:


augmentor.applyAugmentation()


# We can now check the elements of the output folder. 

# In[ ]:


get_ipython().system('ls output/')


# Finally, we can visualize the results using some of the tools provided by [the COCO API](https://github.com/cocodataset/cocoapi).

# In[ ]:


get_ipython().system('pip install pycocotools')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


# In[ ]:


image_directory = 'output/'
annotation_file = 'output/annotation.json'


# In[ ]:


example_coco = COCO(annotation_file)


# In[ ]:


categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))


# Each execution of the following cells show a different image of the output dataset.

# In[ ]:


category_ids = example_coco.getCatIds(catNms=['square'])
image_ids = example_coco.getImgIds(catIds=category_ids)
image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]


# In[ ]:


image_data


# In[ ]:


image = io.imread(image_directory + image_data['file_name'])
plt.imshow(image); plt.axis('off')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)
example_coco.showAnns(annotations)


# If you are executing this notebook in Colaboratory, you need to download the generated files. To that aim, you can create a zip folder and download it using the following commands. 

# In[ ]:


get_ipython().system('zip -r shapes.zip output')
from google.colab import files
files.download('shapes.zip')

