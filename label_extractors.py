import torch

from const_config import YOLO_V1_LABELS_PER_IMAGE
from const_config import NUMBER_OF_DIGITS

OPERATOR_DICT = { 10.0: "+", 11.0: "-", 12.0: "*", 13.0: "/"}

def dod_90x30(labels, idx):
        label = labels[idx]
        return f"{int(label[0])} {OPERATOR_DICT[label[1]]} {int(label[2])}"
    
def yolo_v1(labels, idx):
    label_str = ""
    for label in labels[idx * YOLO_V1_LABELS_PER_IMAGE : YOLO_V1_LABELS_PER_IMAGE + idx * YOLO_V1_LABELS_PER_IMAGE]:
        if label[0] == 1: # label contains digit or operator
            if label[1] >= NUMBER_OF_DIGITS: # label contains operator
                label_str += f" {OPERATOR_DICT[label[1]]} "
            else: # label contains digit
                label_str += f"{label[1]}"

    return label_str

def yolo_v1_prediction(predictions):
    label = ""
    for prediction in predictions:
        if prediction[0] > 0:
            idx = torch.argmax(prediction[1:])
            if idx >= NUMBER_OF_DIGITS:
                label += f" {['+', '-', '*', '/'][idx - NUMBER_OF_DIGITS]} "
            else:
                label += f"{idx}"
    
    return label
