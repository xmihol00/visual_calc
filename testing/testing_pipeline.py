import os
import cv2
import matplotlib.pyplot as plt
import sys
import glob
import imutils
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from networks.custom_recursive_CNN import CustomRecursiveCNN
from networks.mser.detector import Detector
import data_preprocessing.handwritten_equtions as hwe
from const_config import WRITERS_PATH
from const_config import PREDICTION_SAMPLES

augmented_model = CustomRecursiveCNN("cpu", True, PREDICTION_SAMPLES)
augmented_model.load()
augmented_model = augmented_model.eval()

mser_detector = Detector(use_gpu=False)

for file_name in sorted(glob.glob(f"{WRITERS_PATH}*.jpg")):
    image, areas = hwe.equation_areas(file_name)
    for sample, (row1, row2, col1, col2) in zip(hwe.samples_from_area(image, areas), areas):
        predictions = augmented_model(sample)
        string_labels = hwe.extract_string_labels(predictions)

        area = image[row1:row2, col1:col2]
        gray = (area * 255).astype(np.uint8)
        gray = 255 - gray
        padded_gray = cv2.copyMakeBorder(gray, 80, 80, 120, 120, cv2.BORDER_CONSTANT, value=255)
        img = cv2.cvtColor(padded_gray, cv2.COLOR_GRAY2BGR)
        img = imutils.resize(img, width=320, inter=cv2.INTER_AREA)
        valid_boxes, labels, probabilities = mser_detector.detect_digits_in_img(img, False, False)
        eq_results = mser_detector.compute_equation(valid_boxes, labels, probabilities, 3)
        weight = 6
        for equation_result in eq_results:
            for _ in range(0, weight):
                string_labels.append(equation_result)
            weight = weight - 2
    
        final_prediction = hwe.parse_string_labels(string_labels)

        plt.imshow(area > 0, cmap='gray')
        plt.title(final_prediction)
        plt.show()
                