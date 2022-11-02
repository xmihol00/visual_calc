import torch
from torch import nn

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()
        self.cel = nn.CrossEntropyLoss()
    
    def forward(self, predictions, labels):
        indices_with_class = (labels[:, 0] == 1).nonzero(as_tuple=True)[0]
        return self.bcel(predictions[:, 0:1], labels[:, 0:1].to(torch.float32)) + self.cel(predictions[indices_with_class, 1:], labels[indices_with_class, 1])