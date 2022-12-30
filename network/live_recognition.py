import math
import cv2
import numpy as np
import imutils
from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing


def configGPU():
    # Ensure we only use a fixed amount of memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Set memory limit to something lower than total GPU memory
                tf.config.set_logical_device_configuration(gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


# Evaluates multiple images and returns the predicted labels with the highest probability
def evaluate_all(imgs):
    prediction = model(imgs, training=False)
    highest_probability = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(highest_probability)


# Checks if a rectangle is inside the test rectangle
def is_inside_box(check_box, test_box):
    cx, cy, cw, ch = check_box
    tx, ty, tw, th = test_box
    min_x = tx
    max_x = tx + tw
    min_y = ty
    max_y = ty + th
    if (min_x <= cx <= max_x) and (min_y <= cy <= max_y):
        return True
    return False


# Resizes a grayscale image with arbitrary size to (28, 28) while preserving the original aspect ratio
def resize_image(x, w, h):
    square_img = np.full((28, 28), 255, x.dtype)
    if w > h:
        resized = imutils.resize(x, width=28)
        new_height = resized.shape[0]
        top_padding = math.floor((28 - new_height) / 2)
        square_img[top_padding:top_padding+new_height, 0:28] = resized
    elif h > w:
        resized = imutils.resize(x, height=28)
        new_width = resized.shape[1]
        left_padding = math.floor((28 - new_width) / 2)
        square_img[0:28, left_padding:left_padding+new_width] = resized
    else:
        resized = imutils.resize(x, width=28, height=28)
        square_img = resized
    return square_img


def compute_equation(boxes, labels):
    # TODO: Sort boxes and labels by coordinate and compute result
    return "= ?"


def evaluate_frame(img, use_adaptive_treshold=False):
    vis_img = img.copy()

    search_area = search_width * search_height
    x_crop_start = int((frame_width - search_width) / 2)
    x_crop_end = x_crop_start + search_width
    y_crop_start = int((frame_height - search_height) / 2)
    y_crop_end = y_crop_start + search_height
    img = img[y_crop_start:y_crop_end, x_crop_start:x_crop_end]

    # mark crop region
    cv2.rectangle(vis_img, (x_crop_start, y_crop_start), (x_crop_end, y_crop_end), (255, 0, 255), 1)

    # TODO: Find optimal thresholding method
    # Apply a threshold to get a binary image for region detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_adaptive_treshold is True:
        #blurred = cv2.medianBlur(gray, 3)
        #blurred = cv2.bilateralFilter(gray, 5, 11, 7)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        black_white = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 11)
    else:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        (thresh, black_white) = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow("Threshold result", black_white)

    # Find regions of interest with MSER
    regions, potentially_valid_boxes = mser.detectRegions(black_white)

    # Discard boxes which are too large or have extreme aspect ratios
    max_area_threshold = 0.15
    while True:
        reset = False
        for index, box in enumerate(potentially_valid_boxes):
            x, y, w, h = box
            box_area = w * h
            if box_area > int(search_area * max_area_threshold):
                reset = True
                potentially_valid_boxes = np.delete(potentially_valid_boxes, index, 0)
                break
            if (w / 28) > h or (h / 28) > w:
                reset = True
                potentially_valid_boxes = np.delete(potentially_valid_boxes, index, 0)
                break
        if not reset:
            break

    # Some regions of interest may overlap, so we need to remove those
    while True:
        reset = False
        for index, box in enumerate(potentially_valid_boxes):
            for test_box in potentially_valid_boxes:
                if np.array_equal(box, test_box):
                    continue
                if is_inside_box(box, test_box):
                    reset = True
                    potentially_valid_boxes = np.delete(potentially_valid_boxes, index, 0)
                    break
            if reset:
                break
        if not reset:
            break

    reshaped_boxes = None

    # For every valid region of interest we transform it to the desired shape (28, 28)
    # and format (binary black & white). Then we evaluate the region using our trained model
    for box in potentially_valid_boxes:
        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]
        scaled = resize_image(roi, w, h)
        # TODO: Improve thresholding to deal with shadows
        (thresh, black_white) = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY_INV)
        int_bw = (black_white / 255).astype(int)
        reshaped = np.reshape(int_bw, (1, 28, 28))
        if reshaped_boxes is None:
            reshaped_boxes = reshaped
        else:
            reshaped_boxes = np.concatenate((reshaped_boxes, reshaped))

    if reshaped_boxes is None:
        return vis_img

    labels = evaluate_all(reshaped_boxes)

    # Draw bounding box and result
    for index, box in enumerate(potentially_valid_boxes):
        x, y, w, h = box
        color = (255, 125, 0)
        cv2.rectangle(vis_img, (x + x_crop_start, y + y_crop_start), (x + w + x_crop_start, y + h + y_crop_start), color, 1)
        cv2.putText(vis_img, str(labels[index]), (x + 10 + x_crop_start, y - 10 + y_crop_start), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    eq_result = compute_equation(potentially_valid_boxes, labels)
    cv2.putText(vis_img, str(eq_result), (int(frame_width / 2), y_crop_end + 16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    return vis_img


frame_width = 640
frame_height = 480
search_width = 320
search_height = 160

# MSER returns the areas of interest
mser = cv2.MSER_create(delta=25, min_area=20, max_variation=0.4)

model = keras.models.load_model('../models/test_model')
label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = np.load('../models/test_model/classes.npy')

# TODO: Set index to work for own camera setup
vid = cv2.VideoCapture(2)

if not vid.isOpened():
    raise IOError("Camera not found!")

configGPU()

while True:
    ret, frame = vid.read()
    if ret is False:
        continue

    frame = np.resize(frame, (frame_height, frame_width, 3))
    vis = evaluate_frame(frame, True)
    cv2.imshow('Result', vis)

    keycode = cv2.waitKey(5)
    if keycode == 27: # ESC
        break

vid.release()
cv2.destroyAllWindows()
