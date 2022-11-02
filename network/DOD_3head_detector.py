import numpy as np
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

EQUATIONS_PATH = "../data/equations/"
TRAINING_IMAGES_FILENAME = "equations_90x30_training_images.npy"
TRAINING_LABELS_FILENAME = "equations_90x30_training_labels.npy"
MODEL_PATH = "../models/"

IMAGE_WIDTH = 90
IMAGE_HEIGHT = 30

BATCH_SIZE = 2

# cite: https://towardsdatascience.com/multilabel-classification-with-pytorch-in-5-minutes-a4fa8993cbc7
# cite: https://towardsdatascience.com/object-detection-with-neural-networks-a4e2c46b4491
# cite: https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/

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
        label_1 = labels[:, 0]
        label_2 = labels[:, 1]
        label_3 = labels[:, 2]
        return self.CrossEntropyLoss(output[0], label_1) + self.CrossEntropyLoss(output[1], label_2) + self.CrossEntropyLoss(output[2], label_3)

class ThreeCharDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = nn.Sequential(
            nn.Conv2d(1, 4, (3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, (3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.first_digit = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32 * (IMAGE_WIDTH - 8) * (IMAGE_HEIGHT - 8), 1000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 10)
        )

        self.operator = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32 * (IMAGE_WIDTH - 8) * (IMAGE_HEIGHT - 8), 1000),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 4)
        )

        self.second_digit = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32 * (IMAGE_WIDTH - 8) * (IMAGE_HEIGHT - 8), 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 10)
        )
    
    def forward(self, x):
        x = self.back_bone(x)
        return [self.first_digit(x), self.operator(x), self.second_digit(x)]

if __name__ == "__main__":
    classifier = ThreeCharDetector()
    loss_function = ThreeCharLoss()
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        training_set = TrainingDataset()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)

        for i in range(1, 8):
            for images, labels in DataLoader(training_set, BATCH_SIZE):
                output = classifier(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Loss in epoch {i}: {loss.item()}")
            
            with open(f"{MODEL_PATH}3char_3head_detector_{i}.pt", "wb") as file:
                    torch.save(classifier.state_dict(), file)

    else:
        with open(f"{MODEL_PATH}3char_3head_detector.pt", "rb") as file:
            classifier.load_state_dict(torch.load(file))
        
        operators = ["+", "-", "*", "/"]
        classifier = classifier.eval()
        for image, label in DataLoader(TrainingDataset(), 1):
            output = classifier(image)
            classified = [torch.argmax(output[0]).item(), operators[torch.argmax(output[1]).item()], torch.argmax(output[2]).item()]
            labeled = [label[0][0].item(), operators[label[0][1].item()], label[0][2].item()]
            
            plt.imshow(image[0][0].numpy(), cmap='gray')
            plt.title(f"Image classified as {classified} and labeled as {labeled}.")
            plt.show()
                
          
