import torch

from const_config import YOLO_LABELS_PER_IMAGE
from const_config import NUMBER_OF_DIGITS
from const_config import NUMBER_OF_OPERATORS

OPERATOR_DICT = { 10.0: "+", 11.0: "-", 12.0: "*", 13.0: "/"}

def dod_90x30(labels, idx):
        label = labels[idx]
        return f"{int(label[0])} {OPERATOR_DICT[label[1]]} {int(label[2])}"
    
def yolo(labels, idx):
    label_str = ""
    for label in labels[idx]:
        if label[0] == 1: # label contains digit or operator
            if label[1] >= NUMBER_OF_DIGITS: # label contains operator
                label_str += f" {OPERATOR_DICT[label[1]]} "
            else: # label contains digit
                label_str += f"{label[1]}"

    return label_str

def yolo(labels, idx):
    label_str = ""
    for label in labels[idx]:
        if label[0] == 1: # label contains digit or operator
            if label[1] >= NUMBER_OF_DIGITS: # label contains operator
                label_str += f" {OPERATOR_DICT[label[1]]} "
            else: # label contains digit
                label_str += f"{label[1]}"

    return label_str

def yolo_prediction(predictions):
    label = ""
    for prediction in predictions:
        if torch.sigmoid(prediction[0]) > 0.5:
            idx = torch.argmax(prediction[1:])
            if idx >= NUMBER_OF_DIGITS:
                label += f"{[' + ', ' - ', ' * ', ' / ', ''][idx - NUMBER_OF_DIGITS]}"
            else:
                label += f"{idx}"
    
    return label

def yolo_only_class(labels, idx):
    label_str = ""
    #print(idx, labels[idx * YOLO_LABELS_PER_IMAGE:idx * YOLO_LABELS_PER_IMAGE + YOLO_LABELS_PER_IMAGE, 1])
    for label in labels[idx * YOLO_LABELS_PER_IMAGE:idx * YOLO_LABELS_PER_IMAGE + YOLO_LABELS_PER_IMAGE, 1]:
        if label < NUMBER_OF_DIGITS:
            label_str += f"{label}"
        elif label < NUMBER_OF_DIGITS + NUMBER_OF_OPERATORS: # label contains operator
            label_str += f" {OPERATOR_DICT[label]} "

    return label_str

def yolo_prediction_only_class(predictions, sep=' '):
    label = ""
    for prediction in predictions:
        idx = torch.argmax(prediction)
        if idx < NUMBER_OF_DIGITS:
            label += f"{idx}"
        elif idx < NUMBER_OF_DIGITS + NUMBER_OF_OPERATORS:
            label += f"{sep}{['+', '-', '*', '/', ''][idx - NUMBER_OF_DIGITS]}{sep}"
    
    return label
