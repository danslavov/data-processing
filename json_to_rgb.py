"""
Reads JSON file produced by VGG Image Annotator (VIA)
and converts it to mask where each class is in different color.
"""

import json
import os

from cv2 import cv2
import numpy as np

DIR_ORIG = 'C:/Users/User/Desktop/tmp/!removed_background'
DIR_CROP = 'C:/Users/User/Desktop/tmp/crop/diode/100-150'
DIR_VIA = 'C:/Users/User/Desktop/tmp/via'
FILE_EXTENSION = 'jpg'
FILENAME = '1.jpg'

# Dimensions should match those of ground truth image
MASK_WIDTH = 700
MASK_HEIGHT = 700
CHANNEL_COUNT = 3

# TODO: organize grey and color levels for all classes so that can be easily iterated
# TODO: need to get each class name from the JSON somehow !!!
CLASS_GREY_LEVEL = 250
CLASS_COLOR = (100, 20, 200)

# Dictionary containing polygon coordinates for each mask:
# key = polygon name; value = list of point coordinates in the shape [y, x]
polygons = {}


def main():
    json_path = os.path.join(DIR_VIA, '1.json')
    count = 0  # Count of total images saved

    # Read JSON file
    with open(json_path) as f:
        data = json.load(f)

    # Extract X and Y coordinates if available and update dictionary
    # Each entry contains all data for one annotated image.
    for entry in data:
        file_name_json = data[entry]["filename"]
        sub_count = 0  # Contains count of masks for a single ground truth image
        if len(data[entry]["regions"]) > 1:
            for _ in range(len(data[entry]["regions"])):
                key = file_name_json[:-4] + "*" + str(sub_count + 1)
                add_to_dict(data, entry, key, sub_count)
                sub_count += 1
        else:
            add_to_dict(data, entry, file_name_json[:-4], 0)
    print("\nDict size: ", len(polygons))

    all_masks = []  # to store each mask of a single object (instance)
    # For each entry in dictionary, generate 1-channel mask
    for entry in polygons:
        num_masks = entry.split("*")
        mask_folder = DIR_VIA
        mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
        try:
            arr = np.array(polygons[entry])
        except:
            print("Not found:", entry)
            continue
        count += 1
        cv2.fillPoly(mask, [arr],
                     color=CLASS_GREY_LEVEL)  # fill the polygon area with grey level according to the class
        all_masks.append(mask)

        # Combine all masks of single objects into one image
        # TODO: For ENet compatibility, all instances of particular class can be combined into one greyscale image
        #  and afterwards, all these images to be placed in a tensor as channels.
        combined = np.zeros((MASK_WIDTH, MASK_HEIGHT))
        for single_mask in all_masks:
            combined += single_mask

        # Create 3-channel mask
        color_mask = np.float32(mask)
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB)

        # TODO: Replace each grey level with the corresponding color for this class
        low = np.array([CLASS_GREY_LEVEL, CLASS_GREY_LEVEL, CLASS_GREY_LEVEL])
        high = np.array([CLASS_GREY_LEVEL, CLASS_GREY_LEVEL, CLASS_GREY_LEVEL])
        whites = cv2.inRange(color_mask, low, high)
        color_mask[whites > 0] = CLASS_COLOR
        mask_name = entry.split('*')[1]
        cv2.imwrite('{}/{}.png'.format(DIR_VIA, mask_name), color_mask)


def add_to_dict(data, itr, key, count):
    try:
        x_points = data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
    except:
        print("No polygon for ", key)
        return
    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])
    polygons[key] = all_points


if __name__ == '__main__':
    main()
