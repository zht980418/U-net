from model import *
from data import *
import skimage.io as io
from keras.models import load_model
import numpy as np
model=unet("unet_membrane_loss_DSC.hdf5")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
testGene = testGenerator("data/test/image/")
img_test,test_num,test_size= test_load_img(r'data/test/image/')
for img,i in zip(img_test,range(test_num)):
    img_predict = model.predict(img,batch_size=1)
    img_predict = img_predict / np.max(img_predict)
    img_predict[img_predict>0.5]=1 
    img_predict[img_predict<=0.5]=0
    img_predict = np.squeeze(img_predict)
    io.imsave(os.path.join("data/result","%d_predict.png"%i),img_predict)