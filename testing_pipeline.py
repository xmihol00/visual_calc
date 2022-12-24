from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

IMAGE_PATH = "./data/equation_images/"
sys.path.append(os.path.join(os.path.dirname(__file__), "network"))
from network.custom_CNN_v1 import CustomCNNv1
import label_extractors

model = CustomCNNv1()
model.load()
model = model.eval()

for file_name in os.listdir(IMAGE_PATH):
    image = Image.open(f"{IMAGE_PATH}{file_name}").convert('L')
    image = np.asarray(image)
    if image.sum() * 2 > image.shape[0] * image.shape[1] * 255:
        image = np.vectorize(lambda x: 0 if x >= 128 else 1)(image)
        
    else:
        image = np.vectorize(lambda x: 1 if x >= 128 else 0)(image)

    interresting_rows = image.sum(axis=1) > image.shape[0] * 0.005
    row_areas = []
    min_row_separator = image.shape[0] * 0.025
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

    areas = []
    min_col_separator = image.shape[1] * 0.15
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
        if area_sum > area_max * 0.025 and area_sum < area_max * 0.2:
            resized_image = Image.fromarray((area * 255).astype(np.uint8), 'L')
            resized_image.thumbnail((resized_image.width, 30), Image.NEAREST)
            final_image = np.zeros((38, 288))
            width_shift = (288 - resized_image.width) // 2
            try:
                final_image[4:34, width_shift:resized_image.width + width_shift] = np.asarray(resized_image)
            except:
                continue

            sample = torch.tensor((final_image > 0).astype(np.float32))
            sample = sample.unsqueeze(0).unsqueeze(0)
            prediction = model(sample)
        
            classified = label_extractors.yolo_prediction_only_class(prediction)

            plt.imshow(sample[0][0].numpy(), cmap='gray')
            plt.title(f"{torch.argmax(prediction, 1).numpy()}\n{classified}.")
            plt.show()
                