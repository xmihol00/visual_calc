import os
import cv2
import cv2 as cv
from matplotlib import pyplot as plt

from networks.mser.utils import filter_boxes, pad_to_size
from networks.mser.utils import resize_image, resize_image_no_padding

img_size = (320, 160)

mser = cv2.MSER_create(delta=25, min_area=20, max_variation=0.4)

dirname = os.path.dirname(__file__)
img1 = cv2.resize(cv.imread(f'{dirname}/../../testing/med1.JPG', cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_LINEAR)
img2 = cv2.resize(cv.imread(f'{dirname}/../../testing/hard2.JPG', cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_LINEAR)

blurred1 = cv2.GaussianBlur(img1, (5, 5), 0)
blurred2 = cv2.GaussianBlur(img2, (5, 5), 0)

images = [blurred1, blurred2]

block_size = 15
constant = 11

recog_block_size = 3
recog_constant = 5

for img in images:
    threshold = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, constant)

    regions, potentially_valid_boxes = mser.detectRegions(threshold)
    valid_boxes = filter_boxes(potentially_valid_boxes, 320 * 160)
    for index, box in enumerate(valid_boxes):
        x, y, w, h = box
        roi = img[y:y + h, x:x + w]
        scaled = resize_image_no_padding(roi, w, h)
        ret1, res1 = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY_INV)
        ret2, res2 = cv.threshold(scaled, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        ret3, res3 = cv.threshold(scaled, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_TRIANGLE)
        res4 = cv.adaptiveThreshold(scaled, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, recog_block_size, recog_constant)
        res5 = cv.adaptiveThreshold(scaled, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, recog_block_size, recog_constant)

        titles = ["Original image", 'Scaled Image', 'Global v=127', 'Global + otsu', 'Global + triangle',
                  'Adaptive Mean (' + str(recog_block_size) + "," + str(recog_constant) + ")",
                  'Adaptive Gaussian (' + str(recog_block_size) + "," + str(recog_constant) + ")"]
        images = [roi, scaled, pad_to_size(res1, 0), pad_to_size(res2, 0), pad_to_size(res3, 0), pad_to_size(res4, 0), pad_to_size(res5, 0)]
        for i in range(7):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])
        plt.show()

