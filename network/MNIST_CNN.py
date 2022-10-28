import sys
import numpy as np
import torch
import idx2numpy
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

MNIST_PATH = "../data/mnist/"
MODEL_PATH = "../models/"

TRAINING_IMAGES_FILENAME = "train-images.idx3-ubyte"
TRAINING_LABELS_FILENAME = "train-labels.idx3-ubyte"
TESTING_IMAGES_FILENAME = "t10k-images.idx3-ubyte"
TESTING_LABELS_FILENAME = "t10k-labels.idx3-ubyte"

MODEL_FILE_NAME = "mnist_CNN.pt"

class TrainingDataset():
    def __init__(self):
        self.training_data = torch.from_numpy(np.expand_dims(idx2numpy.convert_from_file(f"{MNIST_PATH}{TRAINING_IMAGES_FILENAME}") / 255.0, axis=1)).to(torch.float32)
        self.training_labels = torch.from_numpy(np.array(idx2numpy.convert_from_file(f"{MNIST_PATH}{TRAINING_IMAGES_FILENAME}")))

    def __getitem__(self, idx):
        return self.training_data[idx], self.training_labels[idx]

    def __len__(self):
        return self.training_labels.shape[0]

class TestingDataset():
    def __init__(self):
        self.testing_data = torch.from_numpy(np.expand_dims(idx2numpy.convert_from_file(f"{MNIST_PATH}{TESTING_IMAGES_FILENAME}") / 255.0, axis=1)).to(torch.float32)
        self.testing_labels = torch.from_numpy(np.array(idx2numpy.convert_from_file(f"{MNIST_PATH}{TESTING_LABELS_FILENAME}")))

    def __getitem__(self, idx):
        return self.testing_data[idx], self.testing_labels[idx]

    def __len__(self):
        return self.testing_labels.shape[0]

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 96, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(96 * 24 * 24, 10)
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    classifier = MNIST_CNN()
    loss_function = nn.CrossEntropyLoss()

    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.005)

        for i in range(1, 6):
            for images, labels in DataLoader(TrainingDataset(), 32):
                output = classifier(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Loss in epoch {i}: {loss.item()}")

        with open(f"{MODEL_PATH}{MODEL_FILE_NAME}", "wb") as file:
            torch.save(classifier.state_dict(), file)
    else:
        with open(f"{MODEL_PATH}{MODEL_FILE_NAME}", "rb") as file:
            classifier.load_state_dict(torch.load(file))
        
        for image, label in DataLoader(TestingDataset(), 1):
            output = classifier(image)
            classified = torch.argmax(output).item()
            labeled = label.item()
            
            plt.imshow(image[0][0].numpy(), cmap='gray')
            plt.title(f"Image classified as {classified} and labeled as {labeled}.")
            plt.show()
            
