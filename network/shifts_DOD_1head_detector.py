import numpy as np
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

EQUATIONS_PATH = "../data/equations/"
TRAINING_IMAGES_FILENAME = "equations_132x40_training_images.npy"
TRAINING_LABELS_FILENAME = "equations_132x40_training_labels.npy"
MODEL_PATH = "../models/"

IMAGE_WIDTH = 132
IMAGE_HEIGHT = 40

BATCH_SIZE = 4

class TrainingDataset():
    def __init__(self):
        self.training_data = torch.from_numpy(np.expand_dims(np.load(f"{EQUATIONS_PATH}{TRAINING_IMAGES_FILENAME}", allow_pickle=True), axis=1))
        self.training_labels = torch.from_numpy(np.load(f"{EQUATIONS_PATH}{TRAINING_LABELS_FILENAME}", allow_pickle=True))

    def __getitem__(self, idx):
        return self.training_data[idx], self.training_labels[idx]

    def __len__(self):
        return self.training_labels.shape[0]

class ThreeCharLoss():
    def __init__(self):
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def __call__(self, output, labels):
        return (self.CrossEntropyLoss(output[:,  0:10], labels[:, 0]) + 
                self.CrossEntropyLoss(output[:, 10:14], labels[:, 1]) + 
                self.CrossEntropyLoss(output[:, 14:24], labels[:, 2]))

class ThreeCharDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, (3, 3)),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, (3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (3, 3)),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(int(64 * (((IMAGE_WIDTH - 4) / 2 - 4) / 2 - 2) * (((IMAGE_HEIGHT - 4) / 2 - 4) / 2 - 2)), 1000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 24)
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    classifier = ThreeCharDetector()
    loss_function = ThreeCharLoss()
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        training_set = TrainingDataset()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)
        with open(f"{MODEL_PATH}shifts_3char_1head_detector.pt", "rb") as file:
            classifier.load_state_dict(torch.load(file))

        for i in range(1, 6):
            j = 0
            for images, labels in DataLoader(training_set, BATCH_SIZE):
                output = classifier(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                j += 1
                if j % 100 == 0:
                    print(f"Loss in epoch {i}, iteration {j}: {loss.item()}")
            
            
            with open(f"{MODEL_PATH}shifts_3char_1head_detector_{i}.pt", "wb") as file:
                    torch.save(classifier.state_dict(), file)

    else:
        with open(f"{MODEL_PATH}shifts_3char_1head_detector.pt", "rb") as file:
            classifier.load_state_dict(torch.load(file))
        
        operators = ["+", "-", "*", "/"]
        classifier = classifier.eval()
        for image, label in DataLoader(TrainingDataset(), 1):
            output = classifier(image)
            classified = [torch.argmax(output[0, 0:10]).item(), operators[torch.argmax(output[0, 10:14]).item()], torch.argmax(output[0, 14:24]).item()]
            labeled = [label[0][0].item(), operators[label[0][1].item()], label[0][2].item()]
            
            plt.imshow(image[0][0].numpy(), cmap='gray')
            plt.title(f"Image classified as {classified} and labeled as {labeled}.")
            plt.show()
                
          
