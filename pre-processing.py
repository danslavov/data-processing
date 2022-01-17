import shutil
from shutil import copyfile

import numpy
from natsort import natsorted
import json
import os
import time

from cv2 import cv2
import numpy as np

CLASS_TYPE = 'mixed'
STAGE = 'train'
SOURCE_DIR_STRUCTURE_IMG = 'D:/dataset/{}'
SOURCE_DIR_STRUCTURE_MASK = 'D:/dataset/{}_labels'
DEST_DIR_STRUCTURE_IMG = 'D:/dataset_small/{}'
DEST_DIR_STRUCTURE_MASK = 'D:/dataset_small/{}_labels'
SOURCE_DIR_IMG = SOURCE_DIR_STRUCTURE_IMG.format(STAGE)
SOURCE_DIR_MASK = SOURCE_DIR_STRUCTURE_MASK.format(STAGE)
DEST_DIR_IMG = DEST_DIR_STRUCTURE_IMG.format(STAGE)
DEST_DIR_MASK = DEST_DIR_STRUCTURE_MASK.format(STAGE)
DIM_RESIZED = (525, 525)
# background_img_begin_name = 'background_begin_1.png'
# background_img_middle_name = 'background_middle_1.png'
# background_img_end_name = 'background_end_1.png'

# NB: InteractLabeler doesn't work with gray images, only RGB.
# Therefore, pre-processing should not output gray images.

def main():
    start = time.time()
    # refine_masks(SOURCE_DIR, DEST_DIR)
    # overlay(SOURCE_DIR, SOURCE_DIR_MASK, DEST_DIR)  # only for representation and experiments
    # rename_and_accumulate(SOURCE_DIR, DEST_DIR, CLASS_TYPE)  # gather all images/masks in one folder with unique names
    # path = (SOURCE_DIR_IMG, SOURCE_DIR_MASK, DEST_DIR_IMG, DEST_DIR_MASK)
    # resize(*path, DIM_RESIZED)
    count_files_by_class('C:/Users/Admin/PycharmProjects/ENet-PyTorch-davidtvs/data/Elements/train')

    # dir = 'C:/Users/Admin/Desktop'
    # refine_masks(dir, dir)

    # subtract_background_one_image()
    # remove_shadows_one_image()

    print(time.time() - start)


def s(image):
    cv2.imshow('0', image)
    cv2.waitKey(0)


# *****************   CROP TO 700 X 700   *****************
#
# DIR_ORIG = 'D:/captured_images/orig/mixed'
# DIR_CROP = 'C:/Users/Admin/Desktop/crop/img/mixed'
# start = time.time()
# for filename in os.listdir(DIR_ORIG):
#     img = cv2.imread(os.path.join(DIR_ORIG, filename))
#     cropped = img[180:880, 650:1350]
#     cv2.imwrite(os.path.join(DIR_CROP, filename), cropped)
# end = time.time()
# print(end - start)
# exit()
# *********************************************************


# cv2.namedWindow('output', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('output', 1000, 1000)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# *****************   REMOVE BACKGROUND  --  not for InteractLabeler   *****************
# # load image
# img = cv2.imread(os.path.join(DIR_ORIG, filename))
#
# # convert to gray
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # threshold input image as mask:
# # all values below the threshold (second argument) become 0
# # all values above the threshold become (third argument)
# for i in range(10, 150, 10):
#     mask = cv2.threshold(gray, i, 255, cv2.THRESH_BINARY)[1]
#     cv2.imwrite(os.path.join(DIR_ORIG, 'mask{}.jpg'.format(i)), mask)
#
# # negate mask
# mask = 255 - mask
#
# # apply morphology (open and close) to remove isolated extraneous noise
# # use borderconstant of black since foreground touches the edges
# kernel = np.ones((3, 3), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
# # anti-alias the mask -- blur then stretch
# # blur alpha channel
# mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
# # linear stretch so that 127.5 goes to 0, but 255 stays 255
# mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
#
# # put mask into alpha channel
# result = img.copy()
# result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
# result[:, :, 3] = mask
#
# # save resulting masked image
# cv2.imwrite('C:/Users/User/Desktop/tmp/crop/tmp/1_result.png', result)  # .jpg not working !!!
# *********************************************************
#
# *****************   SUBTRACT BACKGROUND  ****************
# CLASS_TYPE = 'mixed'
# SOURCE_DIR = 'C:/Users/Admin/Desktop/crop/img/{}'.format(CLASS_TYPE)
# DEST_DIR = 'C:/Users/Admin/Desktop/crop/no_back/{}'.format(CLASS_TYPE)
# background_img_begin_name = 'background_begin.png'
# background_img_middle_name = 'background_middle.png'
# background_img_end_name = 'background_end.png'
#
#
# def subtract_in_range(images, background_img_name, start, stop):
#     background = cv2.imread(os.path.join(SOURCE_DIR, background_img_name))
#     for img_number in range(start, stop):
#         img_name = images[img_number]
#         img = cv2.imread(os.path.join(SOURCE_DIR, img_name))
#         result = cv2.subtract(background, img)
#         cv2.imwrite(os.path.join(DEST_DIR, img_name), result)
#
#
# all_images = os.listdir(SOURCE_DIR)
# all_images = natsorted(all_images)
# subtract_in_range(all_images, background_img_begin_name, 0, 76)
# subtract_in_range(all_images, background_img_end_name, 76, 152)
# exit()
# *********************************************************
#
# *****************   CHANGE BLACK TO CLASS-SPECIFIC AND COPY  ***********************
# #from black to class-specific
# CLASS_TYPE = 'resistor'
# # SOURCE_DIR = 'C:/InteractLabeler1_2_1/Labels'
# SOURCE_DIR = 'C:/InteractLabeler1_2_1/Labels/green'
# # DEST_DIR = 'C:/Users/Admin/Desktop/crop/mask_from_tool/{}'.format(CLASS_TYPE)
# DEST_DIR = 'C:/InteractLabeler1_2_1/Labels'
#
# for mask_name in os.listdir(SOURCE_DIR):
#     mask = cv2.imread(os.path.join(SOURCE_DIR, mask_name))
#     black = [0, 0, 0]
#     blue = [255, 0, 0]  # capacitor
#     red = [0, 0, 255]  # capacitor-flat
#     yellow = [0, 255, 255]  # diode
#     green = [0, 255, 0]  # resistor
#     grey = [170, 170, 170]  # background
#     cond_black = (mask == black).all(axis=2)
#     cond_blue = (mask == blue).all(axis=2)
#     cond_red = (mask == red).all(axis=2)
#     cond_yellow = (mask == yellow).all(axis=2)
#     cond_green = (mask == green).all(axis=2)
#
#     # If pixels are black, make them class-specific
#     mask[np.where(cond_green)] = yellow
#     cv2.imwrite(os.path.join(DEST_DIR, mask_name), mask)
# exit()
# *********************************************************
#
# *****************   CHANGE COLOR TO BACKGROUND AND COPY  ****************
# # Unlabeled pixels remain black (0, 0, 0),
# # or another color which doesn't correspond to the class.
# # So all colors which are not class-specific need to be changed
# # to background color (170, 170, 170).
# CLASS_TYPE = 'mixed'
# SOURCE_DIR = 'C:/InteractLabeler1_2_1/Labels'
# DEST_DIR = 'C:/Users/Admin/Desktop/crop/mask_from_tool/{}'.format(CLASS_TYPE)
#
# for mask_name in os.listdir(SOURCE_DIR):
#     mask = cv2.imread(os.path.join(SOURCE_DIR, mask_name))
#     blue = [255, 0, 0]  # capacitor
#     red = [0, 0, 255]  # capacitor-flat
#     yellow = [0, 255, 255]  # diode
#     green = [0, 255, 0]  # resistor
#     grey = [170, 170, 170]  # background
#     cond_blue = (mask == blue).all(axis=2)
#     cond_red = (mask == red).all(axis=2)
#     cond_yellow = (mask == yellow).all(axis=2)
#     cond_green = (mask == green).all(axis=2)
#
#     # If pixels are not blue, red, yellow or green, turn them to grey:
#     condition = np.invert((np.logical_or(np.logical_or(np.logical_or(cond_blue, cond_red), cond_yellow), cond_green)))
#     mask[np.where(condition)] = grey
#     cv2.imwrite(os.path.join(DEST_DIR, mask_name), mask)
# exit()
# ************************************************************************************
#
# *****************   RENAME AND COPY   **********************************************
# CLASS_TYPE = 'mixed'
# SOURCE_DIR = 'C:/Users/Admin/Desktop/crop/mask_from_tool/{}'.format(CLASS_TYPE)
# DEST_DIR = 'C:/Users/Admin/Desktop/crop/mask/{}'.format(CLASS_TYPE)
# for source_file_name in os.listdir(SOURCE_DIR):
#     source_file_path = os.path.join(SOURCE_DIR, source_file_name)
#     dest_file_name = source_file_name.replace('_L', '')
#     dest_file_path = os.path.join(DEST_DIR, dest_file_name)
#     copyfile(source_file_path, dest_file_path)
# exit()
# ************************************************************************************
#
# *****************   REMOVE SHADOWS   ***********************************************
# CLASS_TYPE = 'mixed'
# SOURCE_DIR = 'C:/Users/Admin/Desktop/crop/no_back/{}'.format(CLASS_TYPE)
# DEST_DIR = 'C:/Users/Admin/Desktop/crop/no_shadow_only_remaining/{}'.format(CLASS_TYPE)
# THRESHOLD = 35
# for file_name in os.listdir(SOURCE_DIR):
#     img = cv2.imread(os.path.join(SOURCE_DIR, file_name))
#     lower = (0, 0, 0)
#     upper = (THRESHOLD, THRESHOLD, THRESHOLD)
#     mask = cv2.inRange(img, lower, upper)
#     img[mask != 0] = [0, 0, 0]
#     result = cv2.imwrite(os.path.join(DEST_DIR, file_name), img)
#     if not result:
#         print("Can't write")
#         break
# exit()
#
# ************************************************************************************
# *****************   F U N C T I O N S   ********************************************

# TODO: purpose and usage?
def refine_output(dir, file):
    path = os.path.join(dir, file)
    img = cv2.imread(path)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(dir, 'grey.png'), img_grey)
    thresh = 76
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(dir, 'tresh_grey.png'), thresh_img)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create an empty image for contours
    img_contours = np.zeros(img.shape)
    # draw the contours on the empty image
    cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 1)
    # save image
    cv2.imwrite(os.path.join(dir, 'cont.png'), img_contours)


def count_files_by_class(location):
    capacitor = capacitor_flat = resistor = mixed = unknown = 0
    for file_name in os.listdir(location):
        if file_name.startswith('capacitor_'):
            capacitor += 1
        elif file_name.startswith('capacitor-flat_'):
            capacitor_flat += 1
        elif file_name.startswith('resistor_'):
            resistor += 1
        elif file_name.startswith('mixed_'):
            mixed += 1
        else:
            unknown += 1
    total = capacitor + capacitor_flat + resistor + mixed
    with(open('log.txt', 'a')) as results:
        results.write('{}\ncapacitor: {}\ncapacitor-flat: {}\nresistor: {}\nmixed: {}\nunknown: {}\ntotal: {}\n\n'
                      .format(location.split('/')[-1].upper(), capacitor, capacitor_flat, resistor, mixed, unknown, total))


def refine_masks(source_dir, destination_dir):

    file_name = 'mask.png'

    # for file_name in os.listdir(source_dir):
    mask = cv2.imread(os.path.join(source_dir, file_name))

    # Apply OPEN-CLOSE (introduces some non-class-specific pixel values)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # All pixels values which don't belong to a class (incl. background),
    # are changed to background
    blue = [255, 0, 0]  # capacitor
    red = [0, 0, 255]  # capacitor-flat
    yellow = [0, 255, 255]  # resistor
    green = [0, 255, 0]  # not used
    grey = [170, 170, 170]  # background
    cond_blue = (mask == blue).all(axis=2)
    cond_red = (mask == red).all(axis=2)
    cond_yellow = (mask == yellow).all(axis=2)
    cond_green = (mask == green).all(axis=2)

    # If pixels are not blue, red, yellow or green, turn them to grey:
    condition = np.invert((np.logical_or(np.logical_or(np.logical_or(cond_blue, cond_red), cond_yellow), cond_green)))
    mask[np.where(condition)] = grey
    file_name = 'res.png'
    cv2.imwrite(os.path.join(destination_dir, file_name), mask)


def overlay(img_dir, mask_dir, destination_dir):
    for file_name in os.listdir(mask_dir):
        img = cv2.imread(os.path.join(img_dir, file_name))
        mask = cv2.imread(os.path.join(mask_dir, file_name))
        dst = cv2.addWeighted(img, 1, mask, 0.4, 5)
        cv2.imwrite(os.path.join(destination_dir, file_name), dst)


def rename_and_accumulate(source_dir, destination_dir, class_name):
    for file_name in os.listdir(source_dir):
        original = r'{}/{}'.format(source_dir, file_name)
        target = r'{}/{}_{}'.format(destination_dir, class_name, file_name)
        shutil.copyfile(original, target)


def resize(source_dir_img, source_dir_mask, dest_dir_img, dest_dir_mask, target_dimension):
    for file_name in os.listdir(source_dir_img):
        img = cv2.imread(os.path.join(source_dir_img, file_name))
        mask = cv2.imread(os.path.join(source_dir_mask, file_name))
        resized_img = cv2.resize(img, target_dimension)
        resized_mask = cv2.resize(mask, target_dimension)
        cv2.imwrite(os.path.join(dest_dir_img, file_name), resized_img)
        cv2.imwrite(os.path.join(dest_dir_mask, file_name), resized_mask)


def subtract_background_one_image():
    dir = 'C:/Users/Admin/Desktop'
    cropped = 'mixed_75.png'
    back = 'background_middle_2.png'

    img = cv2.imread(os.path.join(dir, cropped))
    background = cv2.imread(os.path.join(dir, back))
    result = cv2.subtract(background, img)
    cv2.imwrite(os.path.join(dir, 'result.png'), result)


def remove_shadows_one_image():
    threshold = 75
    dir = 'C:/Users/Admin/Desktop'
    file_name = 'result.png'
    img = cv2.imread(os.path.join(dir, file_name))
    lower = (0, 0, 0)
    upper = (threshold, threshold, threshold)
    mask = cv2.inRange(img, lower, upper)
    img[mask != 0] = [0, 0, 0]
    result = cv2.imwrite(os.path.join(dir, 'result.png'), img)
    if not result:
        print("Can't write")


# # Apply several operations with the mask wile each time overlaying it over the image.
# # For representation purposes.
# Not used in practice, since blur introduces pixels with different values (not class-specific).
def smoothen_and_overlay_mask_over_image():
    img_orig = cv2.imread('C:/Users/Admin/Desktop/mixed_13_r_90_f.png')
    mask = cv2.imread('C:/Users/Admin/Desktop/mask.png')
    # dst = cv2.addWeighted(img_orig, 1, mask, 0.4, 5)
    # cv2.imwrite('C:/Users/Admin/Desktop/res/over/mask_L.png', dst)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('C:/Users/Admin/Desktop/res/mask_oc.png', mask)
    dst = cv2.addWeighted(img_orig, 1, mask, 0.4, 5)
    cv2.imwrite('C:/Users/Admin/Desktop/res/over/mask_oc.png', dst)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite('C:/Users/Admin/Desktop/res/mask_blur.png', mask)
    dst = cv2.addWeighted(img_orig, 1, mask, 0.4, 5)
    cv2.imwrite('C:/Users/Admin/Desktop/res/over/mask_blur.png', dst)
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite('C:/Users/Admin/Desktop/res/mask_stretch.png', mask)
    dst = cv2.addWeighted(img_orig, 1, mask, 0.4, 5)  # this is the best
    cv2.imwrite('C:/Users/Admin/Desktop/res/over/mask_stretch.png', dst)


if __name__ == '__main__':
    main()
