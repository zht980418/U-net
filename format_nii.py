import nibabel as nib
from model import *
from data import *
import skimage.io as io
from keras.models import load_model
import numpy as np
model=unet("unet_membrane_loss_DSC.hdf5")
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
testGene = testGenerator("data/test/image/")
path =r'data/nii'
img_test,test_num,test_size,sample,img_size,affine,name,test_shape= test_load_img(r'data/test/image/')
data = []
print(type(sample))
for img,i in zip(img_test,range(test_num)):
    img_predict = model.predict(img,batch_size=1)
    img_predict = img_predict / np.max(img_predict)
    img_predict[img_predict>0.5]=1 
    img_predict[img_predict<=0.5]=0
    img_predict = np.squeeze(img_predict)
    data.append(img_predict)
    print(i)
m = 0
for i,k in zip(test_size,range(196,211)):
    print(i)
    print(type(k))
    nii = np.zeros(shape=test_shape[k-196])
    for j in range(i):
        print(data[j+m].shape)
        nii[:,:,j]=data[j+m]
    m += i 
    nii = nib.Nifti1Image(nii,affine[k-196])
    nib.save(nii,os.path.join(path,name[k-196]))