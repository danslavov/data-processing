import os
import time
import warnings
import glob
import random
import shutil

import PIL
import torch
import cv2

warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
import skimage
from skimage.transform import rotate, AffineTransform, warp, resize, rescale
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

IMG = 'C:/Users/Admin/Desktop/crop/img/all_captured'
MASK = 'C:/Users/Admin/Desktop/crop/mask/all_captured'

IMG_ROT = 'C:/Users/Admin/Desktop/crop/img/augmented/1_rotated'
MASK_ROT = 'C:/Users/Admin/Desktop/crop/mask/augmented/1_rotated'

IMG_FLIP = 'C:/Users/Admin/Desktop/crop/img/augmented/2_flipped'
MASK_FLIP = 'C:/Users/Admin/Desktop/crop/mask/augmented/2_flipped'

IMG_FLIP_ROT = 'C:/Users/Admin/Desktop/crop/img/augmented/3_flipped_rotated'
MASK_FLIP_ROT = 'C:/Users/Admin/Desktop/crop/mask/augmented/3_flipped_rotated'

IMG_SHIFT = 'C:/Users/Admin/Desktop/crop/img/augmented/4_shifted'
MASK_SHIFT = 'C:/Users/Admin/Desktop/crop/mask/augmented/4_shifted'

IMG_SHIFT_ROT = 'C:/Users/Admin/Desktop/crop/img/augmented/5_shifted_rotated'
MASK_SHIFT_ROT = 'C:/Users/Admin/Desktop/crop/mask/augmented/5_shifted_rotated'

IMG_SHIFT_FLIP = 'C:/Users/Admin/Desktop/crop/img/augmented/6_shifted_flipped'
MASK_SHIFT_FLIP = 'C:/Users/Admin/Desktop/crop/mask/augmented/6_shifted_flipped'

IMG_SHIFT_FLIP_ROT = 'C:/Users/Admin/Desktop/crop/img/augmented/7_shifted_flipped_rotated'
MASK_SHIFT_FLIP_ROT = 'C:/Users/Admin/Desktop/crop/mask/augmented/7_shifted_flipped_rotated'

ALL_IMG = 'D:/crop/img/all'
ALL_MASK = 'D:/crop/mask/all'

DATASET_ROOT = 'D:/dataset'
MODEL_TEST = '{}/test'.format(DATASET_ROOT)
MODEL_TEST_LABELS = '{}/test_labels'.format(DATASET_ROOT)
MODEL_TRAIN = '{}/train'.format(DATASET_ROOT)
MODEL_TRAIN_LABELS = '{}/train_labels'.format(DATASET_ROOT)
MODEL_VAL = '{}/val'.format(DATASET_ROOT)
MODEL_VAL_LABELS = '{}/val_labels'.format(DATASET_ROOT)

SHIFT_RATIO = 0.2
TRAIN_PORTION = 0.7
TEST_PORTION = VALIDATION_PORTION = (1 - TRAIN_PORTION) / 2


def main():
    start = time.time()

    # 1. rotate captured
    # paths = (IMG, MASK, IMG_ROT, MASK_ROT)
    # rotate_and_save(*paths)

    # 2. flip captured
    # paths = (IMG, MASK, IMG_FLIP, MASK_FLIP)
    # flip_and_save(*paths)

    # 3. flip rotated
    # paths = (IMG_ROT, MASK_ROT, IMG_FLIP_ROT, MASK_FLIP_ROT)
    # flip_and_save(*paths)

    # 4. shift captured
    # paths = (IMG, MASK, IMG_SHIFT, MASK_SHIFT)
    # shift_and_save(*paths, SHIFT_RATIO)

    # 5. shift rotated
    # paths = (IMG_ROT, MASK_ROT, IMG_SHIFT_ROT, MASK_SHIFT_ROT)
    # shift_and_save(*paths, SHIFT_RATIO)

    # 7. shift flipped
    # paths = (IMG_FLIP, MASK_FLIP, IMG_SHIFT_FLIP, MASK_SHIFT_FLIP)
    # shift_and_save(*paths, SHIFT_RATIO)

    # 4. shift flipped-rotated
    # paths = (IMG_FLIP_ROT, MASK_FLIP_ROT, IMG_SHIFT_FLIP_ROT, MASK_SHIFT_FLIP_ROT)
    # shift_and_save(*paths, SHIFT_RATIO)

    # ... (additional) change brightness and add nose to all above:
    # c, r, f, f-r, s, s-r, s-f, s-f-r
    # NB: but only images, not masks!

    # Read all images and masks, shuffle, split into train, test and validation.
    # Copy to corresponding directories of the ENet project.
    feed_data_to_project(ALL_IMG, ALL_MASK)

    print(time.time() - start)


def feed_data_to_project(img_dir, mask_dir):
    file_names = [f for f in os.listdir(img_dir)]
    random.shuffle(file_names)
    train_end_idx = int(round(len(file_names) * TRAIN_PORTION))
    test_end_idx = train_end_idx + int(round(len(file_names) * TEST_PORTION))
    # val_end_idx = int(round(len(file_names) * VALIDATION_PORTION))
    train_names = file_names[:train_end_idx]
    test_names = file_names[train_end_idx:test_end_idx]
    val_names = file_names[test_end_idx:]
    for name in train_names:
        img_full_name = os.path.join(img_dir, name)
        shutil.copy(img_full_name, MODEL_TRAIN)
        mask_full_name = os.path.join(mask_dir, name)
        shutil.copy(mask_full_name, MODEL_TRAIN_LABELS)
    for name in test_names:
        img_full_name = os.path.join(img_dir, name)
        shutil.copy(img_full_name, MODEL_TEST)
        mask_full_name = os.path.join(mask_dir, name)
        shutil.copy(mask_full_name, MODEL_TEST_LABELS)
    for name in val_names:
        img_full_name = os.path.join(img_dir, name)
        shutil.copy(img_full_name, MODEL_VAL)
        mask_full_name = os.path.join(mask_dir, name)
        shutil.copy(mask_full_name, MODEL_VAL_LABELS)


def rotate_and_save(img_source, mask_source, img_dest, mask_dest):
    for file_name in os.listdir(img_source):
        img = cv2.imread(os.path.join(img_source, file_name))
        mask = cv2.imread(os.path.join(mask_source, file_name))
        rotated = rotate_90_180_270(img, mask)
        name_rot = '{}_r_{{}}.png'.format(file_name.split('.')[0])
        for angle in (90, 180, 270):
            for pair_index in range(len(rotated)):
                cv2.imwrite(os.path.join(img_dest, name_rot.format(angle)), rotated[pair_index][0])
                cv2.imwrite(os.path.join(mask_dest, name_rot.format(angle)), rotated[pair_index][1])


def flip_and_save(img_source, mask_source, img_dest, mask_dest):
    for file_name in os.listdir(img_source):
        img = cv2.imread(os.path.join(img_source, file_name))
        mask = cv2.imread(os.path.join(mask_source, file_name))
        flipped = flip_horizontally(img, mask)
        name_flip = '{}_f.png'.format(file_name.split('.')[0])
        cv2.imwrite(os.path.join(img_dest, name_flip), flipped[0])
        cv2.imwrite(os.path.join(mask_dest, name_flip), flipped[1])


def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def horizontal_shift(img, mask, ratio=0.0):

    # Currently, don't want random ratio:
    # if ratio > 1 or ratio < 0:
    #     print('Value should be less than 1 and greater than 0')
    #     return img, mask
    # ratio = random.uniform(-ratio, ratio)

    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
        mask = mask[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
        mask = mask[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    mask = fill(mask, h, w)
    return img, mask


def vertical_shift(img, mask, ratio=0.0):

    # Currently, don't want random ratio:
    # if ratio > 1 or ratio < 0:
    #     print('Value should be less than 1 and greater than 0')
    #     return img, mask
    # ratio = random.uniform(-ratio, ratio)  # currently, don't want random shift

    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
        mask = mask[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
        mask = mask[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    mask = fill(mask, h, w)
    return img, mask


def change_brightness_multiplier(img, low, high=0.0):
    # NB: Don't apply it to masks!
    if high == 0.0:
        high = low
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def zoom(img, mask, lower, upper=0.0):
    if upper == 0.0:
        upper = lower
    value = random.uniform(lower, upper)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    mask = mask[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    mask = fill(mask, h, w)
    return img, mask


def change_brightness_value(img, value):
    # NB: Don't apply it to masks!
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img


def rotate_by_angle(img, angle):
    # leaves black parts
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def add_noise(noise_type, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    noise_type : str
        One of the following strings, selecting the type of noise to add:
        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    -----------
    Author: Shubham Pachori
    Source: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """
    if noise_type == "gauss":
       row,col,ch= image.shape
       mean = 0
       var = 0.1
       sigma = var**0.5
       gauss = np.random.normal(mean,sigma,(row,col,ch))
       gauss = gauss.reshape(row,col,ch)
       noisy = image + gauss
       return noisy
    elif noise_type == "s&p":
       row,col,ch = image.shape
       s_vs_p = 0.5
       amount = 0.004
       out = np.copy(image)
       # Salt mode
       num_salt = np.ceil(amount * image.size * s_vs_p)
       coords = [np.random.randint(0, i - 1, int(num_salt))
               for i in image.shape]
       out[coords] = 1

       # Pepper mode
       num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
       coords = [np.random.randint(0, i - 1, int(num_pepper))
               for i in image.shape]
       out[coords] = 0
       return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy


def transformations_from_other_libraries():
    # NOT USED. PIL, skimage, etc

    # image = Image.open(os.path.join(SOURCE_DIR, SOURCE_IMAGE))#.convert('RGBA')
    # image = io.imread(os.path.join(SOURCE_DIR, SOURCE_IMAGE))

    # rotated = cv2.rotate()
    # rotated_mask =
    # s(rotated, TARGET_DIR, name='r')
    # s(rotated_mask, TARGET_DIR, name='r-m')

    # transform = AffineTransform(translation=(100, 200))
    # wrap_shift = warp(image, transform, mode='wrap')

    # flip_left_right = np.fliplr(image)
    # flip_up_down = np.flipud(image)

    # std_dev = 0.05
    # noisy_random = random_noise(image, var=std_dev ** 2)
    # blurred = gaussian(image, sigma=std_dev, multichannel=True)

    # h = image.shape[0]
    # w = image.shape[0]
    # resized = resize(image, (int(h * 0.3), int(w * 0.3)))

    # rescaled = rescale(image, 0.3, multichannel=True)
    pass


def rotate_90_180_270(image, mask):
    # Rotate at 90, 180, 270 degrees;
    # Produces 3 new pairs image-mask.
    image_rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    mask_rotated_90 = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    image_rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    mask_rotated_180 = cv2.rotate(mask, cv2.ROTATE_180)
    image_rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    mask_rotated_270 = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return list(((image_rotated_90, mask_rotated_90),
                 (image_rotated_180, mask_rotated_180),
                 (image_rotated_270, mask_rotated_270)))


def flip_horizontally(image, mask):
    # No need to flip vertically, as it will duplicate with
    # horizontally-flipped image rotated at angle+180.
    # Produces 1 new pair image-mask.
    image_flipped = cv2.flip(image, 1)
    mask_flipped = cv2.flip(mask, 1)
    return image_flipped, mask_flipped


def shift(image, mask, ratio):
    image_L, mask_L = horizontal_shift(image, mask, -ratio)
    image_R, mask_R = horizontal_shift(image, mask, ratio)
    image_U, mask_U = vertical_shift(image, mask, -ratio)
    image_D, mask_D = vertical_shift(image, mask, ratio)
    return list(((image_L, mask_L), (image_R, mask_R), (image_U, mask_U), (image_D, mask_D)))


def shift_and_save(source_img_dir, source_mask_dir, target_img_dir, target_mask_dir, shift_ratio):
    # Sift left, right, up, down by SHIFT_RATIO. New images: 4
    directions = {0: 'L', 1: 'R', 2: 'U', 3: 'D'}
    for file_name in os.listdir(source_img_dir):
        img = cv2.imread(os.path.join(source_img_dir, file_name))
        mask = cv2.imread(os.path.join(source_mask_dir, file_name))
        shifted = shift(img, mask, shift_ratio)
        for i in range(len(shifted)):
            img_shift = shifted[i][0]
            mask_shift = shifted[i][1]
            name_shift = '{}_s_{}_{}.png'.format(file_name.split('.')[0], shift_ratio, directions[i])
            cv2.imwrite(os.path.join(target_img_dir, name_shift), img_shift)
            cv2.imwrite(os.path.join(target_mask_dir, name_shift), mask_shift)


def v(image):
    cv2.imshow('0', image)
    cv2.waitKey(0)


def s(image, name):
    cv2.imwrite(os.path.join(TARGET_DIR, name) + '.png', image)


if __name__ == '__main__':
    main()
