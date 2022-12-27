from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import re

IMAGE_PATH = "./data/equation_images/"
PREDICTION_SAMPLES = 16 * 4

sys.path.append(os.path.join(os.path.dirname(__file__), "network"))
from network.custom_CNN_v1 import CustomCNNv1
from network.custom_CNN_v3 import CustomCNNv3
import label_extractors
from const_config import EQUATION_IMAGE_WIDTH
from const_config import EQUATION_IMAGE_HEIGHT
from const_config import YOLO_LABELS_PER_IMAGE

model = CustomCNNv3("cpu", PREDICTION_SAMPLES)
model.load()
model = model.eval()

for file_name in os.listdir(IMAGE_PATH):
    image = Image.open(f"{IMAGE_PATH}{file_name}").convert('L')
    image = np.asarray(image)
    if image.sum() * 2 > image.shape[0] * image.shape[1] * 255:
        image = np.vectorize(lambda x: 0 if x >= 128 else 1)(image)
    else:
        image = np.vectorize(lambda x: 1 if x >= 128 else 0)(image)

    interresting_rows = image.sum(axis=1) > image.shape[1] * 0.009
    row_areas = []
    min_row_separator = 50
    area_start = 0
    area_end = 0
    ongoing_area = False
    separation_count = 0
    for i, row in enumerate(interresting_rows):
        if row:
            separation_count = 0
            area_end = i
            if not ongoing_area:
                area_start = i
                ongoing_area = True
        elif ongoing_area:
            separation_count += 1
            if separation_count == min_row_separator:
                ongoing_area = False
                row_areas.append((area_start, area_end + 1))
    
    if ongoing_area:
        row_areas.append((area_start, area_end + 1))
    
    #for row1, row2 in row_areas:
    #    area = image[row1:row2, :]
    #    plt.imshow(area, cmap='gray')
    #    plt.show()

    areas = []
    min_col_separator = image.shape[0] * 0.15
    area_start = 0
    area_end = 0
    ongoing_area = False
    separation_count = 0
    for row1, row2 in row_areas:
        for i, col in enumerate(image[row1:row2].sum(axis=0) > (row2 - row1) * 0.025):
            if col:
                area_end = i
                separation_count = 0
                if not ongoing_area:
                    area_start = i
                    ongoing_area = True
            elif ongoing_area:
                separation_count += 1
                if separation_count == min_col_separator:
                    ongoing_area = False
                    areas.append((row1, row2, area_start, area_end + 1))
        
        if ongoing_area:
            areas.append((row1, row2, area_start, area_end + 1))

    for row1, row2, col1, col2 in areas:
        area = image[row1:row2, col1:col2]
        #plt.imshow(area, cmap='gray')
        #plt.show()
        area_sum = area.sum()
        area_max = area.shape[0] * area.shape[1]
        if area_sum > area_max * 0.015 and area_sum < area_max * 0.2:
            resized_image = Image.fromarray((area * 255).astype(np.uint8), 'L')
            resized_image.thumbnail((resized_image.width, 36), Image.NEAREST)
            if resized_image.width > EQUATION_IMAGE_WIDTH:
                continue
            final_images = np.zeros((PREDICTION_SAMPLES, 38, 288))

            for i, (y1, y2) in enumerate([(0, 38), (1, 37), (2, 36), (3, 35)]):
                resized_image = Image.fromarray((area * 255).astype(np.uint8), 'L')
                resized_image.thumbnail((resized_image.width, y2 - y1), Image.NEAREST)
                if resized_image.width > EQUATION_IMAGE_WIDTH:
                    continue
                width_shift = (EQUATION_IMAGE_WIDTH - resized_image.width) // 2
                resized_image = np.asarray(resized_image)
                for j, shift in enumerate(range(-8, 8)):
                    augmented_width_shift = width_shift + shift
                    if augmented_width_shift < 0:
                        augmented_width_shift = 0
                    elif augmented_width_shift + resized_image.shape[1] > EQUATION_IMAGE_WIDTH:
                        augmented_width_shift = EQUATION_IMAGE_WIDTH - resized_image.shape[1]
                    final_images[i*16 + j, y1:y2, augmented_width_shift:resized_image.shape[1] + augmented_width_shift] = resized_image

            samples = torch.tensor((final_images > 0).astype(np.float32))
            samples = samples.unsqueeze(1)
            predictions = model(samples)

            classifications = [None] * PREDICTION_SAMPLES
            for i, sample in enumerate(samples):
                j = i * YOLO_LABELS_PER_IMAGE
                classifications[i] = label_extractors.yolo_prediction_only_class(predictions[j:j + YOLO_LABELS_PER_IMAGE], sep='')

            filtered_classifications = []
            for classified in classifications:
                if re.match(r"^(\d+[\+\-\*/])+\d+$", classified):
                    filtered_classifications.append(classified)

            try:
                classified = max(filtered_classifications, key=lambda x: sum([x == y for y in filtered_classifications]))
            except:
                continue


            #sample = sample[0].numpy()
            #for i in range(1, 18):
            #    sample[:, i * 16] = 0.5
            plt.imshow(area, cmap='gray')
            plt.title(f"{classified}")
            plt.show()
                