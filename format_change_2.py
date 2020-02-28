import nibabel as nib
import os
import numpy as np
import scipy
from scipy.misc import imsave
import glob
os.mkdir(r'data/test/image_jpg/')
img_path = r'data/test/image/'
saveimg_path = r'data/test/image_jpg/'
img_names = os.listdir(img_path)
num = len([name for name in os.listdir(img_path)])
print(num)
k = 0
for i,img_name in zip(range(1,num),img_names):
    print(img_name)
    img = nib.load(img_path + img_name).get_data()
    img = np.array(img)
    size = img.shape[2]
    for j in range(size):
        scipy.misc.imsave(saveimg_path+str(j+k)+ '.jpg', img[:,:,j])
    k+=j+1
    int(i)


