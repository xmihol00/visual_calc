import os
import sys
import torch
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from const_config import YOLO_LOSS_BIAS

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()
        self.cel = nn.CrossEntropyLoss()
    
    def forward(self, predictions, labels):
        indices_with_class = (labels[:, 0] == 1).nonzero(as_tuple=True)[0] # find the indices of labels, which represent characters
        return (self.bcel(predictions[:, 0:1], labels[:, 0:1].to(torch.float32)) + # the error if the network is correct that a part of an image contains character or not
                self.cel(predictions[indices_with_class, 1:], labels[indices_with_class, 1])) # the Cross Entropy error only on those parts, which contain character

class YoloLossBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()
        self.cel = nn.CrossEntropyLoss()
    
    def forward(self, predictions, labels):
        indices_with_class = (labels[:, 0] == 1).nonzero(as_tuple=True)[0] # find the indices of labels, which represent characters
        return (YOLO_LOSS_BIAS * self.bcel(predictions[:, 0:1], labels[:, 0:1].to(torch.float32)) + # the error if the network is correct that a part of an image contains character or not
                self.cel(predictions[indices_with_class, 1:], labels[indices_with_class, 1])) # the Cross Entropy error only on those parts, which contain character

class YoloLossNoClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()
        self.cel = nn.CrossEntropyLoss()
    
    def forward(self, predictions, labels):
        return (self.bcel(predictions[:, 0:1], labels[:, 0:1].to(torch.float32)) + # the error if the network is correct that a part of an image contains character or not
                self.cel(predictions[:, 1:], labels[:, 1])) # the Cross Entropy error on all predictions

class YoloLossNoClassBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()
        self.cel = nn.CrossEntropyLoss()
    
    def forward(self, predictions, labels):
        return (YOLO_LOSS_BIAS * self.bcel(predictions[:, 0:1], labels[:, 0:1].to(torch.float32)) + # the error if the network is correct that a part of an image contains character or not
                self.cel(predictions[:, 1:], labels[:, 1])) # the Cross Entropy error on all predictions

class YoloLossOnlyClasses(nn.Module):
    def __init__(self):
        super().__init__()
        self.cel = nn.CrossEntropyLoss()
    
    def forward(self, predictions, labels):
        return self.cel(predictions, labels[:, 1]) # the Cross Entropy error on all predictions
