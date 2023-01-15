import cv2
import imutils
import numpy as np
from utils import try_eval_equation

from Detector import Detector

if __name__ == '__main__':
    frame_width = 320
    frame_height = 160

    detector = Detector()

    img = cv2.imread("../../testing/med2.JPG")
    img = cv2.resize(img, (frame_width, frame_height))
    vis_img = img.copy()

    valid_boxes, labels, probabilities = detector.detect_digits_in_img(img, True, False)

    # Draw bounding box and result
    for index, box in enumerate(valid_boxes):
        x, y, w, h = box
        color = (255, 125, 0)
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(vis_img, str(labels[index][0]), (x + 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    eq_results = detector.compute_equation(valid_boxes, labels, probabilities, 3)
    evaluated_res = try_eval_equation(eq_results[0])
    cv2.putText(vis_img, str(eq_results[0]) + " = " + str(evaluated_res), (int(frame_width / 2) - 32, 16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cv2.imshow('Result', vis_img)
    cv2.waitKey(0)