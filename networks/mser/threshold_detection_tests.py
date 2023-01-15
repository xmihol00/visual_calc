import os
import cv2
import cv2 as cv
from matplotlib import pyplot as plt

from networks.mser.utils import filter_boxes

img_size = (320, 160)

mser = cv2.MSER_create(delta=25, min_area=20, max_variation=0.4)

dirname = os.path.dirname(__file__)
img1 = cv2.resize(cv.imread(f'{dirname}/../../testing/easy1.JPG', cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_LINEAR)
img2 = cv2.resize(cv.imread(f'{dirname}/../../testing/easy2.JPG', cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_LINEAR)
img3 = cv2.resize(cv.imread(f'{dirname}/../../testing/med1.JPG', cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_LINEAR)
img4 = cv2.resize(cv.imread(f'{dirname}/../../testing/med2.JPG', cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_LINEAR)
img5 = cv2.resize(cv.imread(f'{dirname}/../../testing/hard1.JPG', cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_LINEAR)
img6 = cv2.resize(cv.imread(f'{dirname}/../../testing/hard2.JPG', cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_LINEAR)

orig_images = [img1, img2, img3, img4, img5, img6]
images = [img1, img2, img3, img4, img5, img6]

for img in orig_images:
    blurred = cv2.medianBlur(img, 3)
    images.append(blurred)

for img in orig_images:
    blurred = cv2.bilateralFilter(img, 5, 11, 7)
    images.append(blurred)

for img in orig_images:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    images.append(blurred)

block_size = 15
constant = 11

for img in images:
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret2, th2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    ret3, th3 = cv.threshold(img, 127, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
    th4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, constant)
    th5 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, constant)

    thresholds = [th1, th2, th3, th4, th5]
    visualized = []

    for threshold in thresholds:
        regions, potentially_valid_boxes = mser.detectRegions(threshold)
        valid_boxes = filter_boxes(potentially_valid_boxes, 320 * 160)
        vis = cv2.cvtColor(threshold.copy(), cv2.COLOR_GRAY2RGB)
        for index, box in enumerate(valid_boxes):
            x, y, w, h = box
            vis = cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 125, 0), 1)
        visualized.append(vis)

    titles = ['Original Image', 'Global v=127', 'Global + otsu', 'Global + triangle',
              'Adaptive Mean (' + str(block_size) + "," + str(constant) + ")",
              'Adaptive Gaussian (' + str(block_size) + "," + str(constant) + ")"]
    images = [img] + visualized
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
