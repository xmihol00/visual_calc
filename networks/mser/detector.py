import os

import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing
from networks.mser import utils

class Detector:

    __model = None
    __label_encoder = None
    __mser = None

    def __configGPU(self, use_gpu=True):
        if use_gpu:
            # Ensure we only use a fixed amount of memory
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        # Set memory limit to something lower than total GPU memory
                        tf.config.set_logical_device_configuration(gpu, [
                            tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
        else:
            tf.config.set_visible_devices([], 'GPU')

    # Evaluates multiple images (up to 32) and returns the predicted labels and probabilities sorted by probabilities
    def __evaluate_all(self, imgs, multi_prediction=True):
        predictions = self.__model(imgs, training=False)
        kernel = np.ones((3, 3), np.uint8)
        eroded_imgs = None
        dilated_imgs = None
        if multi_prediction:
            for img in imgs:
                eroded = cv2.erode(img.astype(np.uint8), kernel, iterations=1)
                dilated = cv2.dilate(img.astype(np.uint8), kernel, iterations=1)
                eroded = np.reshape(eroded, (1, 28, 28))
                dilated = np.reshape(dilated, (1, 28, 28))
                if eroded_imgs is None:
                    eroded_imgs = eroded
                else:
                    eroded_imgs = np.concatenate((eroded_imgs, eroded))
                if dilated_imgs is None:
                    dilated_imgs = dilated
                else:
                    dilated_imgs = np.concatenate((dilated_imgs, dilated))
            eroded_predictions = self.__model(eroded_imgs, training=False)
            dilated_predictions = self.__model(dilated_imgs, training=False)
            predictions = predictions + eroded_predictions * 0.25 + dilated_predictions * 0.25
        predicted_labels = []
        probabilities = []
        for digit_prediction in predictions:
            sorted_indices = np.argsort(digit_prediction)
            digit_probability = np.sort(digit_prediction)
            labels = self.__label_encoder.inverse_transform(sorted_indices)
            predicted_labels.append(labels[::-1])
            probabilities.append(digit_probability[::-1])
        return predicted_labels, probabilities

    # Detects and recognizes the digits in an image and returns the boxes of the digits and the labels sorted
    # by probability
    def detect_digits_in_img(self, img, use_adaptive_treshold=False, show_intermediary_result=False):
        # Apply a threshold to get a binary image for region detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if use_adaptive_treshold is True:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            black_white = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11)
        else:
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, black_white = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        if show_intermediary_result:
            cv2.imshow("Threshold result", black_white)

        # Find regions of interest with MSER
        _, potentially_valid_boxes = self.__mser.detectRegions(black_white)

        img_area = img.shape[0] * img.shape[1]
        valid_boxes = utils.filter_boxes(potentially_valid_boxes, img_area)

        reshaped_boxes = None

        # For every valid region of interest we transform it to the desired shape (28, 28)
        # and format (binary black & white). Then we evaluate the region using our trained model
        for box in valid_boxes:
            x, y, w, h = box
            roi = gray[y:y + h, x:x + w]
            scaled = utils.resize_image_no_padding(roi, w, h)
            kernel = np.ones((3, 3), np.uint8)
            cv2.erode(scaled, kernel, iterations=1)
            _, black_white = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            padded = utils.pad_to_size(black_white, 0)
            int_bw = (padded / 255).astype(int)
            reshaped = np.reshape(int_bw, (1, 28, 28))
            if reshaped_boxes is None:
                reshaped_boxes = reshaped
            else:
                reshaped_boxes = np.concatenate((reshaped_boxes, reshaped))

        if reshaped_boxes is None:
            return None, None, None

        labels, probabilities = self.__evaluate_all(reshaped_boxes)
        return valid_boxes, labels, probabilities

    # Returns a number of equations as strings according to the probability
    def compute_equation(self, boxes, labels, probabilities, num_equations=1):
        tuples = []
        for index, box in enumerate(boxes):
            x, *_ = box
            label_prob_index = [labels[index], probabilities[index], x]
            tuples.append(label_prob_index)

        sorted_tuples = sorted(tuples, key=lambda l: l[2], reverse=False)

        equations = []

        for index in range(0, num_equations):
            lowest_probability = 999.9
            lowest_probability_digit_index = -1
            equation = ""
            for digit_index, digit_tuple in enumerate(sorted_tuples):
                labels, probabilities, x_coord = digit_tuple
                if probabilities[0] < lowest_probability :
                    lowest_probability = probabilities[0]
                    lowest_probability_digit_index = digit_index
                label = labels[0]
                if label == "%":
                    label = "/"
                equation = equation + label
            equations.append(equation)
            sorted_tuples[lowest_probability_digit_index][0] = sorted_tuples[lowest_probability_digit_index][0][1::]
            sorted_tuples[lowest_probability_digit_index][1] = sorted_tuples[lowest_probability_digit_index][1][1::]

        return equations

    def __init__(self, use_gpu=True):
        self.__configGPU(use_gpu)
        dirname = os.path.dirname(__file__)
        model_filename = os.path.join(dirname, '../../models/test_model')
        self.__model = keras.models.load_model(model_filename)
        self.__label_encoder = preprocessing.LabelEncoder()
        classes_filename = os.path.join(dirname, '../../models/test_model/classes.npy')
        self.__label_encoder.classes_ = np.load(classes_filename)
        # MSER returns the areas of interest
        self.__mser = cv2.MSER_create(delta=25, min_area=20, max_variation=0.4)
