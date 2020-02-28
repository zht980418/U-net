from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np 
import os
from scipy import ndimage
import pylab
import scipy
import glob
import skimage.io as io
import skimage.transform as trans
import nibabel as nib
import scipy
from scipy.misc import imsave
import nibabel as nib
import scipy
from PIL import Image
from scipy.misc import imsave
from keras import backend as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import skimage.transform as trans
img_path=r'data/train/image/'
mask_path=r'data/train/groundtruth/'
test_path=r'data/test/image/'
def load_img(path):
    data = []
    img_names = os.listdir(path)
    num = len([name for name in os.listdir(path)])
    for img_name,i in zip(img_names,range(1,174)):
        img = nib.load(path + img_name).get_data()
        img = np.array(img)
        size = img.shape[2]
        for j in range(1,size-2):
            page = img[:,:,j]
            page = np.expand_dims(page,axis=0)
            page = np.expand_dims(page,axis=3)
            data.append(page)
    return data

def load_mask(path):
    data = []
    img_names = os.listdir(path)
    num = len([name for name in os.listdir(path)])
    for img_name,i in zip(img_names,range(1,174)):
        img = nib.load(path + img_name).get_data()
        img = np.array(img)
        size = img.shape[2]

        for j in range(1,size-2):
            page = img[:,:,j]
            page = np.expand_dims(page,axis=0)
            page = np.expand_dims(page,axis=3)
            data.append(page)
    for mask in data:
        mask[mask != 0] = 1
    return data

def val_load_img(path):
    data = []
    img_names = os.listdir(path)
    num = len([name for name in os.listdir(path)])
    for img_name,i in zip(img_names,range(175,194)):
        img = nib.load(path + img_name).get_data()
        img = np.array(img)
        size = img.shape[2]
        for j in range(1,size-2):
            page = img[:,:,j]
            page = np.expand_dims(page,axis=0)
            page = np.expand_dims(page,axis=3)
            data.append(page)
    return data

def val_load_mask(path):
    data = []
    img_names = os.listdir(path)
    num = len([name for name in os.listdir(path)])
    for img_name,i in zip(img_names,range(175,194)):
        img = nib.load(path + img_name).get_data()
        img = np.array(img)
        size = img.shape[2]

        for j in range(1,size-2):
            page = img[:,:,j]
            page = np.expand_dims(page,axis=0)
            page = np.expand_dims(page,axis=3)
            data.append(page)
    for mask in data:
        mask[mask != 0] = 1
    return data

def adjustData(img,mask):
    img = img / np.max(img)
    return (img,mask)

def test_load_img(path):
    data = []
    test_size = []
    img_size = []
    affine =[]
    name = []
    img_names = os.listdir(path)
    num = 0
    shape = []
    for img_name in img_names:
        name.append(img_name)
        img = nib.load(path + img_name)
        affine.append(img.affine)
        img = img.get_data()
        sample = img
        img = np.array(img)
        size = img.shape[2]
        shape.append(img.shape)
        test_size.append(size)
        for j in range(size):
            img_size.append(img.shape[0:1])
            num = num+1
            page = img[:,:,j]
            page = np.expand_dims(page,axis=0)
            page = np.expand_dims(page,axis=3)
            data.append(page)
    print(num)
    return (data,num,test_size,sample,img_size,affine,name,shape)

def trainGenerator(img_path,mask_path):
    img_train = tuple(load_img(img_path))
    mask_train = tuple(load_mask(mask_path))
    for (img,mask) in zip(img_train,mask_train):
        img,mask = adjustData(img,mask)
        yield (img,mask)

def validationGenerator(img_path,mask_path):
    img_train = tuple(val_load_img(img_path))
    mask_train = tuple(val_load_mask(mask_path))
    for (img,mask) in zip(img_train,mask_train):
        img,mask = adjustData(img,mask)
        yield (img,mask)

def testGenerator(path):
    img_test,num = test_load_img(path)
    for img in img_test:
            img = img / np.max(img)
            yield img

def saveResult(save_path,npyfile,flag_multi_class = False):
    for i,item in enumerate(npyfile):
        item[item>0.5]=1 
        item[item<=0.5]=0
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),item)

def DSC(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    P = K.sum(y_pred_f)
    TP = K.sum(y_true_f*y_pred_f)
    print(TP)
    FP = P -TP
    print(FP)
    N = K.sum(1-y_true_f)
    FN = K.sum(y_true_f)-TP
    print(FN)
    TN = N-FN
    print(TN)
    DSC = 2*TP/(2*TP+FP+FN)
    return DSC
def PPV(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    P = K.sum(y_pred_f)
    TP = K.sum(y_true_f*y_pred_f)
    print(TP)
    FP = P -TP
    print(FP)
    N = K.sum(1-y_true_f)
    FN = K.sum(y_true_f)-TP
    print(FN)
    TN = N-FN
    print(TN)
    PPV = TP/(TP+FP)
    return PPV
def Sensitivity(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    P = K.sum(y_pred_f)
    TP = K.sum(y_true_f*y_pred_f)
    print(TP)
    FP = P -TP
    print(FP)
    N = K.sum(1-y_true_f)
    FN = K.sum(y_true_f)-TP
    print(FN)
    TN = N-FN
    print(TN)
    Sensitivity = TP/(TP+FN)
    return Sensitivity