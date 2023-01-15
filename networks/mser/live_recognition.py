import cv2
import numpy as np
from networks.mser.detector import Detector
from utils import try_eval_equation

def evaluate_frame(detector, img, use_adaptive_treshold=False):
    vis_img = img.copy()

    search_width = 320
    search_height = 160
    x_crop_start = int((frame_width - search_width) / 2)
    x_crop_end = x_crop_start + search_width
    y_crop_start = int((frame_height - search_height) / 2)
    y_crop_end = y_crop_start + search_height
    img = img[y_crop_start:y_crop_end, x_crop_start:x_crop_end]

    # mark crop region
    cv2.rectangle(vis_img, (x_crop_start, y_crop_start), (x_crop_end, y_crop_end), (255, 0, 255), 1)

    valid_boxes, labels, probabilities = detector.detect_digits_in_img(img, use_adaptive_treshold, True)

    if valid_boxes is None:
        return vis_img

    # Draw bounding box and result
    for index, box in enumerate(valid_boxes):
        x, y, w, h = box
        color = (255, 125, 0)
        cv2.rectangle(vis_img, (x + x_crop_start, y + y_crop_start), (x + w + x_crop_start, y + h + y_crop_start), color, 1)
        cv2.putText(vis_img, str(labels[index][0]), (x + 10 + x_crop_start, y - 10 + y_crop_start), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    eq_results = detector.compute_equation(valid_boxes, labels, probabilities, 1)
    evaluated_res = try_eval_equation(eq_results[0])
    cv2.putText(vis_img,  str(eq_results[0]) + " = " + str(evaluated_res), (x_crop_start, y_crop_end + 16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    return vis_img


if __name__ == '__main__':
    frame_width = 640
    frame_height = 480

    # TODO: Set index to work for own camera setup
    vid = cv2.VideoCapture(2)

    if not vid.isOpened():
        raise IOError("Camera not found!")

    detector = Detector()

    while True:
        ret, frame = vid.read()
        if ret is False:
            continue

        frame = np.resize(frame, (frame_height, frame_width, 3))
        vis = evaluate_frame(detector, frame, False)
        cv2.imshow('Result', vis)

        keycode = cv2.waitKey(5)
        if keycode == 27:  # ESC
            break

    vid.release()
    cv2.destroyAllWindows()
