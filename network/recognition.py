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


# Evaluates a single image using the pretrained model and returns the label with the highest probability
def evaluate(x):
    configGPU()
    model = keras.models.load_model('../models/test_model')
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.classes_ = np.load('../models/test_model/classes.npy')
    prediction = model.predict(x)
    highest_probability = np.argmax(prediction)
    label = labelEncoder.inverse_transform([highest_probability])[0]
    return label


# Evaluates multiple images and returns the predicted labels with the highest probability
def evaluate_all(imgs):
    configGPU()
    model = keras.models.load_model('../models/test_model')
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.classes_ = np.load('../models/test_model/classes.npy')
    prediction = model.predict(imgs)
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
    square_img = np.full((28, 28), 255, roi.dtype)
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


# MSER returns the areas of interest
mser = cv2.MSER_create(delta=25, min_area=20, max_variation=0.4)

# Load an equation image resize it if necessary
img = cv2.imread("../testing/eq4.png")
if img.shape[1] > 400:
    img = imutils.resize(img, 400)
orig = img.copy()
# Apply a threshold to get a binary image for region detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, black_white) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find regions of interest with MSER
regions, potentially_valid_boxes = mser.detectRegions(black_white)

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

print("#Valid boxes: " + str(potentially_valid_boxes.shape[0]))

eval_boxes = None

# For every valid region of interest we transform it to the desired shape (28, 28)
# and format (binary black & white). Then we evaluate the region using our trained model
for box in potentially_valid_boxes:
    x, y, w, h = box
    roi = gray[y:y+h, x:x+w]
    scaled = resize_image(roi, w, h)
    (thresh, black_white) = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY_INV)
    int_bw = (black_white / 255).astype(int)
    reshaped = np.reshape(int_bw, (1, 28, 28))
    if eval_boxes is None:
        eval_boxes = reshaped
    else:
        eval_boxes = np.concatenate((eval_boxes, reshaped))

labels = evaluate_all(eval_boxes)

# Draw bounding box and result
for index, box in enumerate(potentially_valid_boxes):
    x, y, w, h = box
    color = (255, 125, 0)
    cv2.rectangle(orig, (x, y), (x + w, y + h), color, 1)
    cv2.putText(orig, str(labels[index]), (x + 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)


cv2.imshow('img', orig)
cv2.waitKey(0)
