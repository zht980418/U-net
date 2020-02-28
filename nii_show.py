import nibabel as nib
import skimage.io as io
import numpy as np
img=nib.load(r'data\test\image\Case196.nii.gz')
img_arr=img.get_fdata()
img_arr=np.squeeze(img_arr)
io.imshow(img_arr)
