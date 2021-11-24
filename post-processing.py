import math
import os.path
import time

import numpy as np
from cv2 import cv2


DIR = 'C:/Users/Admin/Desktop/tmp'
FILE = '0.png'


def main():
    find_mass_center(DIR, FILE)

def find_orientation(dir, file):
    img = cv2.imread(os.path.join(dir, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[img == 170] = 0  # make background black
    img[img == 29] = 220  # make masks bright

    # convert the grayscale image to binary image
    # ret, thresh = cv2.threshold(img, 75, 255, 0)  # not needed yet

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        # contoured = cv2.drawContours(empty_img, [c], -1, 100, thickness=-1)

        center = find_mass_center(contour)
        point = find_closest_contour_point(contour, center)

        # TODO: calculate the slope of the line between center and point


        exit()


def find_closest_contour_point(contour, center):
    # Find the distance between closest contour point and center
    shortest_distance = cv2.pointPolygonTest(contour, center, True)
    # Compare shortest distance with distance between each contour point and center
    for entry in contour:
        point = entry[0]
        distance = find_distance(point, center)
        if distance == shortest_distance:
            return point


# A bit slower
def find_closest_contour_point_1(contour, center):
    shortest_distance = 9999
    closest_point = 0
    for entry in contour:
        point = entry[0]
        distance = find_distance(point, center)
        if distance < shortest_distance:
            shortest_distance = distance
            closest_point = point
    return closest_point


def find_mass_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX, cY]


def find_distance(point_1, point_2):
    y1, x1 = point_1
    y2, x2 = point_2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Not needed:
def skeletonize(img):
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        # Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img) == 0:
            break

    return skel


def s(file, name='result'):
    cv2.imwrite(os.path.join(DIR, '{}.png'.format(name)), file)


def v(image):
    cv2.imshow('0', image)
    cv2.waitKey(0)


def misc():
    # https://stackoverflow.com/questions/42203898/python-opencv-blob-detection-or-circle-detection
    # im = cv2.imread('extract_blue.jpg')
    # imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im_gauss = cv2.GaussianBlur(imgray, (5, 5), 0)
    # ret, thresh = cv2.threshold(im_gauss, 127, 255, 0)
    # # get contours
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # contours_area = []
    # # calculate area and filter into new array
    # for con in contours:
    #     area = cv2.contourArea(con)
    #     if 1000 < area < 10000:
    #         contours_area.append(con)

    pass


if __name__ == '__main__':
    main()
