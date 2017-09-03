#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from nets import unet_model
import numpy as np
import argparse

LOAD_PATH='address to load'

parser = argparse.ArgumentParser()
parser.add_argument(
    '-r', '--retrain', help='argument to turn on retrain mode', action="store_true")
args = parser.parse_args()

# load data
load = np.load(LOAD_PATH)
#batch normalization
img_clip, mask_clip = load['img_clip'], load['mask_clip'] // 255
img_clip = img_clip / 255
mean = img_clip.mean(axis=0)
img_clip -= mean
print('data loaded')

datagen = ImageDataGenerator()
# dataflow
model = unet_model()
if args.retrain:
    print('retrain mode')
    model.load_weights('weight_unet.hdf5', by_name=True)
else:
    print('train mode')
# fits the model on batches with real-time data augmentation:
epochs = 100
checkpoint = ModelCheckpoint(
    'weight_unet.hdf5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
model.fit_generator(datagen.flow(img_clip, mask_clip, batch_size=32),
                    steps_per_epoch=len(img_clip) / 32, epochs=epochs, callbacks=[checkpoint])
