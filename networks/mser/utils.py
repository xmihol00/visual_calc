import cv2
import numpy as np
import imutils
import math

# Checks if a rectangle is inside the test rectangle
def is_inside_box(check_box, test_box):
    cx, cy, *_ = check_box
    tx, ty, tw, th = test_box
    min_x = tx
    max_x = tx + tw
    min_y = ty
    max_y = ty + th
    if (min_x <= cx <= max_x) and (min_y <= cy <= max_y):
        return True
    return False


def filter_boxes(boxes, search_area):
    # Discard boxes which are too large or have extreme aspect ratios
    max_area_threshold = 0.15
    while True:
        reset = False
        for index, box in enumerate(boxes):
            x, y, w, h = box
            box_area = w * h
            if box_area > int(search_area * max_area_threshold):
                reset = True
                boxes = np.delete(boxes, index, 0)
                break
            if (w / 28) > h or (h / 28) > w:
                reset = True
                boxes = np.delete(boxes, index, 0)
                break
        if not reset:
            break

    # Some regions of interest may overlap, so we need to remove those
    while True:
        reset = False
        for index, box in enumerate(boxes):
            for test_box in boxes:
                if np.array_equal(box, test_box):
                    continue
                if is_inside_box(box, test_box):
                    reset = True
                    boxes = np.delete(boxes, index, 0)
                    break
            if reset:
                break
        if not reset:
            break

    return boxes


# Resizes a grayscale image with arbitrary size to (28, 28) while preserving the original aspect ratio
def resize_image(x, w, h, fill_value=255):
    square_img = np.full((28, 28), fill_value, x.dtype)
    interpolation_mode = cv2.INTER_CUBIC
    if w > h:
        resized = imutils.resize(x, width=28, inter=interpolation_mode)
        new_height = resized.shape[0]
        top_padding = math.floor((28 - new_height) / 2)
        square_img[top_padding:top_padding+new_height, 0:28] = resized
    elif h > w:
        resized = imutils.resize(x, height=28, inter=interpolation_mode)
        new_width = resized.shape[1]
        left_padding = math.floor((28 - new_width) / 2)
        square_img[0:28, left_padding:left_padding+new_width] = resized
    else:
        resized = imutils.resize(x, width=28, height=28, inter=interpolation_mode)
        square_img = resized
    return square_img


# Resizes a grayscale image with arbitrary size to have at most (28, 28) size while preserving the aspect ratio
def resize_image_no_padding(x, w, h):
    interpolation_mode = cv2.INTER_CUBIC
    if w > h:
        resized = imutils.resize(x, width=28, inter=interpolation_mode)
    elif h > w:
        resized = imutils.resize(x, height=28, inter=interpolation_mode)
    else:
        resized = imutils.resize(x, width=28, height=28, inter=interpolation_mode)
    return resized


# Pads a binary image with size (x, 28) or (28, x) to (28, 28)
def pad_to_size(x, fill_value):
    square_img = np.full((28, 28), fill_value, x.dtype)
    h = x.shape[0]
    w = x.shape[1]
    if w > h:
        top_padding = math.floor((28 - h) / 2)
        square_img[top_padding:top_padding + h, 0:28] = x
    elif h > w:
        left_padding = math.floor((28 - w) / 2)
        square_img[0:28, left_padding:left_padding + w] = x
    else:
        square_img = x
    return square_img


def try_eval_equation(eq):
    try:
        res = eval(eq)
    except:
        res = "ERROR"
    return res
