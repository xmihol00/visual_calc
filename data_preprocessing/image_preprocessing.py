from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import sys

IMAGE_PATH = "./data/equation_images/"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import MODEL_PATH
from const_config import YOLO_V6_MODEL_FILENAME
from const_config import YOLO_LABELS_PER_IMAGE
from const_config import YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES
import label_extractors

class YoloInspiredCNNv6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_part = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, (3, 3), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 1), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, (2, 2), padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, (2, 2), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(0.1),
        )

        self.fully_conn_part = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES),
        )

    def forward(self, x):
        results = [None] * 13
        x = self.conv_part(x)
        for i in range(YOLO_LABELS_PER_IMAGE):
            j = 2 * i
            results[i] = self.fully_conn_part(x[:, :, :, j:j+2])
        return torch.cat(results, 1).reshape(-1, YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES)

model = YoloInspiredCNNv6()
with open(f"{MODEL_PATH}{YOLO_V6_MODEL_FILENAME}", "rb") as file:
            model.load_state_dict(torch.load(file))
model = model.eval()

for file_name in os.listdir(IMAGE_PATH):
    image = Image.open(f"{IMAGE_PATH}{file_name}").convert('L')
    image = np.asarray(image)
    if image.sum() * 2 > image.shape[0] * image.shape[1] * 255:
        image = np.vectorize(lambda x: 0 if x >= 128 else 1)(image)
        
    else:
        image = np.vectorize(lambda x: 1 if x >= 128 else 0)(image)

    interresting_rows = image.sum(axis=1) > image.shape[0] * 0.01
    row_areas = []
    min_row_separator = image.shape[0] * 0.025
    area_start = 0
    area_end = 0
    ongoing_area = False
    separation_count = 0
    for i, row in enumerate(interresting_rows):
        if row:
            if not ongoing_area:
                area_start = i
                ongoing_area = True
            separation_count = 0
            area_end = i
        elif ongoing_area:
            separation_count += 1
            if separation_count == min_row_separator:
                ongoing_area = False
                row_areas.append((area_start, area_end + 1))
    
    if ongoing_area:
        row_areas.append((area_start, area_end + 1))

    col_areas = []
    min_col_separator = image.shape[1] * 0.2
    area_start = 0
    area_end = 0
    ongoing_area = False
    separation_count = 0
    for row1, row2 in row_areas:
        for i, col in enumerate(image[row1:row2].sum(axis=0) > (row2 - row1) * 0.005):
            if col:
                area_end = i
                if not ongoing_area:
                    area_start = i
                    ongoing_area = True
            elif ongoing_area:
                separation_count += 1
                if separation_count == min_col_separator:
                    ongoing_area = False
                    col_areas.append((area_start, area_end + 1))
    
    if ongoing_area:
        col_areas.append((area_start, area_end + 1))

    for row1, row2 in row_areas:
        for col1, col2 in col_areas:
            area = image[row1:row2, col1:col2]
            area_sum = area.sum()
            area_max = area.shape[0] * area.shape[1]
            if area_sum > area_max * 0.025 and area_sum < area_max * 0.2:
                resized_image = Image.fromarray((area * 255).astype(np.uint8), 'L')
                resized_image.thumbnail((resized_image.width, 28), Image.NEAREST)
                final_image = np.zeros((38, 228))
                try:
                    final_image[5:33, 2:resized_image.width + 2] = np.asarray(resized_image)
                except:
                    continue
                sample = torch.tensor((final_image > 0).astype(np.float32))
                sample = sample.unsqueeze(0).unsqueeze(0)
                prediction = model(sample)
            
                classified = label_extractors.yolo_prediction_only_class(prediction)

                plt.imshow(sample[0][0].numpy(), cmap='gray')
                plt.title(f"Image classified as {classified}.")
                plt.show()
                