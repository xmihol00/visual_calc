import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing
import utils


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


# Evaluates multiple images (up to 32) and returns the predicted labels with the highest probability
def evaluate_all(imgs):
    prediction = model(imgs, training=False)
    highest_probability = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(highest_probability)


def compute_equation(boxes, labels):
    pairs = []
    for index, box in enumerate(boxes):
        x, y, w, h = box
        pair = [labels[index], x]
        pairs.append(pair)

    sorted_pairs = sorted(pairs, key=lambda l: l[1], reverse=False)

    equation = ""
    for pair in sorted_pairs:
        label, x = pair
        equation = equation + label

    try:
        result = str(eval(equation))
    except:
        result = "?"

    return "= " + result


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

    valid_boxes = utils.filter_boxes(potentially_valid_boxes, search_area)

    reshaped_boxes = None

    # For every valid region of interest we transform it to the desired shape (28, 28)
    # and format (binary black & white). Then we evaluate the region using our trained model
    for box in valid_boxes:
        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]
        scaled = utils.resize_image(roi, w, h)
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
    for index, box in enumerate(valid_boxes):
        x, y, w, h = box
        color = (255, 125, 0)
        cv2.rectangle(vis_img, (x + x_crop_start, y + y_crop_start), (x + w + x_crop_start, y + h + y_crop_start), color, 1)
        cv2.putText(vis_img, str(labels[index]), (x + 10 + x_crop_start, y - 10 + y_crop_start), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    eq_result = compute_equation(valid_boxes, labels)
    cv2.putText(vis_img, str(eq_result), (int(frame_width / 2), y_crop_end + 16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    return vis_img


frame_width = 640
frame_height = 480
search_width = 320
search_height = 160

# MSER returns the areas of interest
mser = cv2.MSER_create(delta=25, min_area=20, max_variation=0.4)

model = keras.models.load_model('../../models/test_model')
label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = np.load('../../models/test_model/classes.npy')

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
