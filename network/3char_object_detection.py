import numpy as np
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

EQUATIONS_PATH = "../data/equations/"
TRAINING_IMAGES_FILENAME = "equations_training_images.npy"
TRAINING_LABELS_FILENAME = "equations_training_labels.npy"

IMAGE_WIDTH = 158
IMAGE_HEIGHT = 48
CHAR_COUNT = 14
POSITION_COUNT = 3

class TrainingDataset():
    def __init__(self):
        self.training_data = torch.from_numpy(np.expand_dims(np.load(f"{EQUATIONS_PATH}{TRAINING_IMAGES_FILENAME}", allow_pickle=True), axis=1))
        self.training_labels = torch.from_numpy(np.load(f"{EQUATIONS_PATH}{TRAINING_LABELS_FILENAME}", allow_pickle=True))

    def __getitem__(self, idx):
        return self.training_data[idx], self.training_labels[idx]

    def __len__(self):
        return self.training_labels.shape[0]

class OutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        output = torch.empty_like(input)
        output[:-3, :] = torch.exp(input[:-POSITION_COUNT, :])
        output[-3:, :] = input[-POSITION_COUNT:, :] / IMAGE_WIDTH
        output[:14, :] /= torch.sum(output[0:CHAR_COUNT, :], 0)
        output[14:28, :] /= torch.sum(output[CHAR_COUNT:2*CHAR_COUNT, :], 0)
        output[28:42, :] /= torch.sum(output[2*CHAR_COUNT:3*CHAR_COUNT, :], 0)
        return output

class ThreeCharDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(4, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * (IMAGE_WIDTH - 6) * (IMAGE_HEIGHT - 6), 1000),
            nn.ReLU(),
            nn.Linear(1000, 45),
            OutputLayer()
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    classifier = ThreeCharDetector()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    for _ in range(1):
        for images, labels in DataLoader(TrainingDataset(), 1):
            output = classifier(images)
            loss = loss_function(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Accuracy: {1 - loss.item()}")
