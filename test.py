import nibabel as nib
from model import *
from data import *
import skimage.io as io
from keras.models import load_model
import numpy as np

img = nib.load('Case196.nii.gz')
img = img.get_data()
img = np.array(img)
size = img.shape[2]
print(size)
