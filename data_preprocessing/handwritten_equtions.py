from PIL import Image
import os
import numpy as np
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from const_config import EQUATION_IMAGE_WIDTH
from const_config import LABELS_PER_IMAGE
from const_config import PREDICTION_SAMPLES
import label_extractors

def equation_areas(file_name):
    image = Image.open(file_name).convert('L')
    image = np.asarray(image)
    if image.sum() * 2 > image.shape[0] * image.shape[1] * 255:
        image = image < 128
    else:
        image = image >= 128

    interresting_rows = image.sum(axis=1) > image.shape[1] * 0.01
    row_areas = []
    min_row_separator = 25
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
    area_start = 0
    area_end = 0
    ongoing_area = False
    for row1, row2 in row_areas:
        for i, col in enumerate(image[row1:row2].sum(axis=0) > (row2 - row1) * 0.04):
            if col:
                area_end = i
                if not ongoing_area:
                    area_start = i
                    ongoing_area = True
        
        if ongoing_area:
            areas.append((row1, row2, area_start, area_end + 1))
        ongoing_area = False

    return image, areas

def samples_from_area(image, areas):
    for row1, row2, col1, col2 in areas:
        area = image[row1:row2, col1:col2]
        if area.shape[0] < 38:
            continue
        
        final_images = np.zeros((PREDICTION_SAMPLES, 38, 288))
        for i, (y1, y2) in enumerate([(0, 38), (1, 37), (2, 36), (3, 35)]):
            resized_image = Image.fromarray((area * 255).astype(np.uint8), 'L')
            resized_image.thumbnail((resized_image.width, y2 - y1), Image.NEAREST)
            resized_image = np.asarray(resized_image)
            resized_image = resized_image[:38, :288]
            width_shift = (EQUATION_IMAGE_WIDTH - resized_image.shape[1]) // 2
            for j, shift in enumerate(range(-8, 8)):
                augmented_width_shift = width_shift + shift
                if augmented_width_shift < 0:
                    augmented_width_shift = 0
                elif augmented_width_shift + resized_image.shape[1] > EQUATION_IMAGE_WIDTH:
                    augmented_width_shift = EQUATION_IMAGE_WIDTH - resized_image.shape[1]
                final_images[i*16 + j, y1:y2, augmented_width_shift:resized_image.shape[1] + augmented_width_shift] = resized_image

            samples = torch.tensor((final_images > 0).astype(np.float32))
            samples = samples.unsqueeze(1)
            
        yield samples


def parse_perdictions(predictions):
    string_labels = [None] * PREDICTION_SAMPLES
    for i in range(PREDICTION_SAMPLES):
        j = i * LABELS_PER_IMAGE
        string_labels[i] = label_extractors.prediction_only_class(predictions[j:j + LABELS_PER_IMAGE], sep='')

    try:
        final_prediction = max(string_labels, key=lambda x: sum([x == y for y in string_labels]))
    except:
        final_prediction = ""

    return final_prediction

def extract_string_labels(predictions):
    string_labels = [None] * PREDICTION_SAMPLES
    for i in range(PREDICTION_SAMPLES):
        j = i * LABELS_PER_IMAGE
        string_labels[i] = label_extractors.prediction_only_class(predictions[j:j + LABELS_PER_IMAGE], sep='')
    
    return string_labels

def parse_string_labels(string_labels):
    try:
        final_prediction = max(string_labels, key=lambda x: sum([x == y for y in string_labels]))
    except:
        final_prediction = ""

    return final_prediction