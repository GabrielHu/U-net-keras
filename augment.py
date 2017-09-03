#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

LOAD_PATH = 'address to load'
SAVE_PATH = 'address to save'

load = np.load(LOAD_PATH)
img_clip, mask_clip = load['img_clip'], load['mask_clip']
print('data loaded')
img_clip_ = []
for img in range(len(img_clip)):
    img_clip_.append(np.concatenate(
        [img_clip[img], mask_clip[img], mask_clip[img]], axis=-1))
datagen = ImageDataGenerator(rotation_range=45, width_shift_range=0.05, height_shift_range=0.05,
                             horizontal_flip=True, vertical_flip=True, fill_mode='reflect')

batches = 0
img_clip_aug = []
mask_clip_aug = []
for x_batch, _ in datagen.flow(np.array(img_clip_), mask_clip, batch_size=5):
    img_clip_aug.append(np.expand_dims(x_batch[:, :, :, 0], axis=-1))
    mask_clip_aug.append(np.expand_dims(x_batch[:, :, :, 1], axis=-1))
    batches += 1
    if batches >= 2000:
        break
print('data augmented')
img_clip = np.concatenate(img_clip_aug, axis=0)
mask_clip = np.concatenate(mask_clip_aug, axis=0)
print(np.shape(img_clip))
np.savez_compressed(SAVE_PATH,
                    img_clip=img_clip, mask_clip=mask_clip)
print('augment done')
