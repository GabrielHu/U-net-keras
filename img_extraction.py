import os
import numpy as np
from PIL import Image
#from skimage.measure import label, regionprops

PATH = 'address to import'
SAVE_PATH = 'address to save'


def list_img(path):
    '''find all the images in the folder'''
    mask = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if root.split('/')[-1] == 'Tuned_Mask':
                mask.append(root + os.sep + filename)

    return mask


def clip_img(mask):
    '''fill each side of images to enlarge their size to 512x512'''
    img_clip = []
    mask_clip = []
    for img in mask:
        img_mask = np.asarray(Image.open(img))
        img_img = np.asarray(Image.open(
            '/'.join(img.split('/')[:-2]) + os.sep + '_'.join(img.split('/')[-1].split('_')[:2]) + '_i.png'))
        
        new_img, new_mask = np.zeros((512, 512)).astype(
            np.uint8), np.zeros((512, 512)).astype(np.uint8)
        '''
        # find centroid
        props = regionprops(label(img_mask))
        max_area = 0
        index = 0
        for i in range(len(props)):
            if props[i].area > max_area:
                max_area = props[i].area
                index = i

        # unpack coord
        coord = props[index].centroid
        X, Y = int(np.round(coord[0])), int(np.round(coord[1]))
        
        
        if X - 32 < 0:
            if Y - 32 < 0:
                new_img[:, :] = img_img[0:64, 0:64]
                new_mask[:, :] = img_mask[0:64, 0:64]
            if Y + 32 >= img_img.shape[1]:
                new_img[:, :] = img_img[0:64,
                                        img_img.shape[1] - 65:img_img.shape[1] - 1]
                new_mask[:, :] = img_mask[0:64,
                                          img_img.shape[1] - 65:img_img.shape[1] - 1]
            else:
                new_img[:, :] = img_img[0:64, Y - 32:Y + 32]
                new_mask[:, :] = img_mask[0:64, Y - 32:Y + 32]
        elif X + 32 >= img_img.shape[0]:
            if Y - 32 < 0:
                new_img[:, :] = img_img[img_img.shape[0] -
                                        65:img_img.shape[0] - 1, 0:64]
                new_mask[:, :] = img_mask[img_img.shape[0] -
                                          65:img_img.shape[0] - 1, 0:64]
            if Y + 32 >= img_img.shape[1]:
                new_img[:, :] = img_img[img_img.shape[0] - 65:img_img.shape[0] - 1,
                                        img_img.shape[1] - 65:img_img.shape[1] - 1]
                new_mask[:, :] = img_mask[img_img.shape[0] - 65:img_img.shape[0] - 1,
                                          img_img.shape[1] - 65:img_img.shape[1] - 1]
            else:
                new_img[:, :] = img_img[img_img.shape[0] -
                                        65:img_img.shape[0] - 1, Y - 32:Y + 32]
                new_mask[:, :] = img_mask[img_img.shape[0] -
                                          65:img_img.shape[0] - 1, Y - 32:Y + 32]
        elif Y - 32 < 0:
            new_img[:, :] = img_img[X - 32:X + 32, 0:64]
            new_mask[:, :] = img_mask[X - 32:X + 32, 0:64]
        elif Y + 32 >= img_img.shape[1]:
            new_img[:, :] = img_img[X - 32:X + 32,
                                    img_img.shape[1] - 65:img_img.shape[1] - 1]
            new_mask[:, :] = img_mask[X - 32:X + 32,
                                      img_img.shape[1] - 65:img_img.shape[1] - 1]
        else:
            new_img[:, :] = img_img[X - 32:X + 32, Y - 32:Y + 32]
            new_mask[:, :] = img_mask[X - 32:X + 32, Y - 32:Y + 32]
        '''
        offsetX = 512 - img_img.shape[0]
        upper_offsetX = int(np.round(offsetX / 2))
        lower_offsetX = offsetX - upper_offsetX
        offsetY = 512 - img_img.shape[1]
        upper_offsetY = int(np.round(offsetY / 2))
        lower_offsetY = offsetY - upper_offsetY
        new_img[upper_offsetX:-lower_offsetX,
                upper_offsetY:-lower_offsetY] = img_img[:, :]
        new_mask[upper_offsetX:-lower_offsetX,
                 upper_offsetY:-lower_offsetY] = img_mask[:, :]

        # save
        img_clip.append(np.expand_dims(new_img, axis=0))
        mask_clip.append(np.expand_dims(new_mask, axis=0))

    print('done')
    img_clip = np.concatenate(img_clip, axis=0)
    mask_clip = np.concatenate(mask_clip, axis=0)
    print(np.shape(img_clip))
    print(np.shape(mask_clip))
    return np.expand_dims(img_clip, axis=-1), np.expand_dims(mask_clip, axis=-1)


def main():
    mask = list_img(PATH)
    print(len(mask))
    img_clip, mask_clip = clip_img(mask)
    np.savez_compressed(SAVE_PATH,
                        img_clip=img_clip, mask_clip=mask_clip)


if __name__ == '__main__':
    main()
