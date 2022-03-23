from __future__ import print_function
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt


def get_rect(rectangle_params):
    w, h = rectangle_params[0], rectangle_params[1]
    p1, p2, p3, p4 = [w / 2, h / 2], [- w / 2,  h / 2], [ - w / 2, -h / 2], [w / 2, -h / 2]
    return p1, p2, p3, p4


def get_circle(radius, vertex_num):
    x = [radius * np.cos(2 * np.pi / vertex_num * i) for i in range(vertex_num)]
    y = [radius * np.sin(2 * np.pi / vertex_num * i) for i in range(vertex_num)]

    result = [[x[i], y[i]] for i in range(len(x))]
    return result


predefined_convex_hulls = [
    [get_circle(18, 40), "крышка"],
    [get_circle(18, 40), "крышка"],
    [get_circle(18, 40), "крышка"],
    [[[20, 123], [24, 103], [80, 71], [95, 71], [132, 39], [140, 50], [34, 123]], "нож"],
    [[[20, 123], [24, 103], [80, 71], [95, 71], [132, 39], [140, 50], [34, 123]], "нож"],
    [[[20, 123], [24, 103], [80, 71], [95, 71], [132, 39], [140, 50], [34, 123]], "нож"],
    [[[20, 123], [24, 103], [80, 71], [95, 71], [132, 39], [140, 50], [34, 123]], "нож"],
    [get_rect([63, 40]), "футляр для наушников"],
    [get_rect([63, 40]), "футляр для наушников"],
    [get_rect([63, 40]), "футляр для наушников"],
    [[[42, 75], [21, 68], [7, 55], [4, 42], [4, 28], [13, 12], [23, 4], [45, 6], [56, 2], [65, 10], [85, 31], [76, 40], [60, 60]], "рулетка"],
    [[[42, 75], [21, 68], [7, 55], [4, 42], [4, 28], [13, 12], [23, 4], [45, 6], [56, 2], [65, 10], [85, 31], [76, 40], [60, 60]], "рулетка"],
    [[[42, 75], [21, 68], [7, 55], [4, 42], [4, 28], [13, 12], [23, 4], [45, 6], [56, 2], [65, 10], [85, 31], [76, 40], [60, 60]], "рулетка"],
    [[[42, 75], [21, 68], [7, 55], [4, 42], [4, 28], [13, 12], [23, 4], [45, 6], [56, 2], [65, 10], [85, 31], [76, 40], [60, 60]], "рулетка"],
    [[[42, 75], [21, 68], [7, 55], [4, 42], [4, 28], [13, 12], [23, 4], [45, 6], [56, 2], [65, 10], [85, 31], [76, 40], [60, 60]], "рулетка"],
    [[[42, 75], [21, 68], [7, 55], [4, 42], [4, 28], [13, 12], [23, 4], [45, 6], [56, 2], [65, 10], [85, 31], [76, 40], [60, 60]], "рулетка"],
    [get_rect([10, 120]), "ручка"],
    [get_rect([10, 120]), "ручка"],
    [get_rect([10, 120]), "ручка"],
    [get_rect([10, 120]), "ручка"],

]


def thresh_callback(src, threshold):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))
    height = src.shape[0]
    width = src.shape[1]

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(canny_output, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_cnt = 0
    for i, c in enumerate(contours):
        if (cv2.arcLength(c, False) < 500):
            continue
        num_cnt += 1

    contours_poly = [None] * num_cnt
    boundRect = [None] * num_cnt
    centers = [None] * num_cnt
    radius = [None] * num_cnt
    index = -1
    for i, c in enumerate(contours):
        if (cv2.arcLength(c, False) < 500):
            continue
        index += 1
        contours_poly[index] = cv2.approxPolyDP(c, 3, True)
        boundRect[index] = cv2.boundingRect(contours_poly[index])
        centers[index], radius[index] = cv2.minEnclosingCircle(contours_poly[index])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    hierarchy = hierarchy[0]

    color = [(255, 44, 0),
             (254, 178, 0),
             (161, 238, 0),
             (129, 6, 168),
             (0, 165, 124),
             (18, 62, 170),
             ]
    cropped_objects = []
    counter = 0
    for i, c in enumerate(contours):
        if (cv2.arcLength(c, False) < 500):
            continue
        if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
            cv2.drawContours(drawing, contours, i, color[counter % 5], 2)
        else:
            cv2.drawContours(drawing, contours, i, (0, 255, 0), 2)

        y, x, h, w, = boundRect[counter]
        if (w < 20 or h < 20):
            continue
        cropped_objects.append(
            src[(int)(get_min(x, 0)):(int)(get_max(x + w, height)), (int)(get_min(y, 0)):(int)(get_max(y + h, width))])
        counter += 1

    return drawing, cropped_objects


def get_min(x, bound):
    if (x > bound + 10):
        return x - 10
    return bound


def get_max(x, bound):
    if (x < bound - 10):
        return x + 10
    return bound


def read_image(filename, directory):
    path = os.path.join(directory, filename)
    original_image = imread(path)
    return original_image


def match_points(src, dst, verbose=False):
    sift = cv2.SIFT_create(sigma=3.5)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(src, None)
    kp2, des2 = sift.detectAndCompute(dst, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    if verbose and len(good_matches) > 0:
        img_matches = cv2.drawMatches(src, kp1, dst, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_matches, interpolation='nearest')
        plt.show()

    return good_matches


def find_objects_on_img(path):
    thresh = 100
    image = read_image(path, "")

    result, cropped_objects = thresh_callback(image, thresh)

    single_item_dataset_directory = os.path.join(os.path.curdir, "Dataset")
    single_item_files = [f for f in listdir(single_item_dataset_directory) if
                         isfile(join(single_item_dataset_directory, f))]

    single_item_files.sort(key=lambda s: int(s.rstrip('.jpg')))
    best_matches_len = []
    best_matches_names = []
    best_matches_polygons = []
    for fragment in cropped_objects:
        src = cv2.cvtColor(fragment, cv2.COLOR_BGR2GRAY)
        best_match_len = 0
        best_i = 1
        for i, single_item_file in enumerate(single_item_files, 0):
            dst = cv2.imread(join(single_item_dataset_directory, single_item_file))
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            matches = match_points(src, dst)
            len_matches = len(matches)
            if len_matches > best_match_len:
                best_match_len = len_matches
                best_i = i
        best_matches_len.append(best_match_len)
        best_matches_polygons.append(predefined_convex_hulls[best_i][0])
        best_matches_names.append(predefined_convex_hulls[best_i][1])

    return best_matches_names, best_matches_polygons
