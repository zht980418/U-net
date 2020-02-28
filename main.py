from model import *
from data_2 import *
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.utils import Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


myGene = trainGenerator('data/train/image/','data/train/groundtruth/')

model = unet(pretrained_weights="unet_membrane_loss_DSC.hdf5")
model_checkpoint = ModelCheckpoint('/home/wukeyu/zhang/BME/unet_membrane_loss_DSC.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=400,epochs=4,validation_data=validationGenerator('data/train/image/','data/train/groundtruth/'), validation_steps=40,callbacks=[model_checkpoint])

testGene = testGenerator("data/test/image/")
img_test,test_num = test_load_img(r'data/test/image/')
results = model.predict_generator(testGene,test_num-1,verbose=1)
saveResult("data/result",results)