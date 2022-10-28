import numpy as np
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import itertools as itt
import matplotlib.pyplot as plt

EQUATIONS_PATH = "../data/equations/"
TRAINING_IMAGES_FILENAME = "equations_training_images.npy"
TRAINING_LABELS_FILENAME = "equations_training_labels.npy"
MODEL_PATH = "../models/"

IMAGE_WIDTH = 30
IMAGE_HEIGHT = 30
CHAR_COUNT = 14
CHARACTER_COUNT = 3

BATCH_SIZE = 32

# cite: https://towardsdatascience.com/multilabel-classification-with-pytorch-in-5-minutes-a4fa8993cbc7
# cite: https://towardsdatascience.com/object-detection-with-neural-networks-a4e2c46b4491

class TrainingDataset():
    def __init__(self, batch_size, loss_function):
        self.training_data = torch.from_numpy(np.expand_dims(np.load(f"{EQUATIONS_PATH}{TRAINING_IMAGES_FILENAME}", allow_pickle=True), axis=1))
        self.training_labels = torch.from_numpy(np.load(f"{EQUATIONS_PATH}{TRAINING_LABELS_FILENAME}", allow_pickle=True))
        self.batch_size = batch_size
        self.indices = [None] * batch_size
        self.index = 0
        self.loss_function = loss_function

    def __getitem__(self, idx):
        self.indices[self.index] = idx
        self.index = (self.index + 1) % self.batch_size
        return self.training_data[idx], self.training_labels[idx]

    def __len__(self):
        return self.training_labels.shape[0]

    def flip_labels(self, output, labels):
        with torch.no_grad():
            for i in range(labels.shape[0]):
                permutations = itt.permutations(range(3))
                best = [self.loss_function(output[i:i+1], labels[i:i+1]), next(permutations)]
                
                for indices in permutations:
                    flipped_label = torch.tensor([[labels[i][indices[0]], labels[i][indices[1]], labels[i][indices[2]],
                                                   labels[i][indices[0] + CHARACTER_COUNT], labels[i][indices[1] + CHARACTER_COUNT], labels[i][indices[2] + CHARACTER_COUNT]]], 
                                                   dtype=torch.float32)
                    loss = self.loss_function(output[i:i+1], flipped_label)
                    if loss < best[0]:
                        best = [loss, indices]

                if best[1] != (0, 1, 2):
                    indices = best[1]
                    self.training_labels[self.indices[i]] = torch.tensor([[labels[i][indices[0]], labels[i][indices[1]], labels[i][indices[2]],
                                                                           labels[i][indices[0] + CHARACTER_COUNT], labels[i][indices[1] + CHARACTER_COUNT], labels[i][indices[2] + CHARACTER_COUNT]]], 
                                                                           dtype=torch.float32)


class ThreeCharLoss():
    def __init__(self):
        self.MSEloss = nn.MSELoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def __call__(self, output, labels):
        return self.CrossEntropyLoss(output, labels)
        #return (self.CrossEntropyLoss(output[0], labels[:, 0]) + 
        #        self.CrossEntropyLoss(output[1], labels[:, 1]) + 
        #        self.CrossEntropyLoss(output[2], labels[:, 2]))
        #return (self.CrossEntropyLoss(output[:, 0:CHAR_COUNT], labels[:, 0].to(torch.long)) + 
        #        self.CrossEntropyLoss(output[:, CHAR_COUNT:2*CHAR_COUNT], labels[:, 1].to(torch.long)) +
        #        self.CrossEntropyLoss(output[:, 2*CHAR_COUNT:3*CHAR_COUNT], labels[:, 2].to(torch.long)))# +
        #        #self.MSEloss(output[:, -CHARACTER_COUNT:], labels[:, 3:]))

class ThreeCharDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = nn.Sequential(
            nn.Conv2d(1, 4, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(4, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten()
        )

        self.first_digit = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32 * (IMAGE_WIDTH - 8) * (IMAGE_HEIGHT - 8), 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 10)
        )

        self.first_operator = nn.Sequential(
            nn.Linear(32 * (IMAGE_WIDTH - 8) * (IMAGE_HEIGHT - 8), 1000),
            nn.ReLU(),
            nn.Linear(1000, 4)
        )

        self.second_digit = nn.Sequential(
            nn.Linear(32 * (IMAGE_WIDTH - 8) * (IMAGE_HEIGHT - 8), 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )
    
    def forward(self, x):
        x = self.back_bone(x)
        return self.first_digit(x) #, self.first_operator(x), self.second_digit(x)]

if __name__ == "__main__":
    classifier = ThreeCharDetector()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    loss_function = ThreeCharLoss()
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        training_set = TrainingDataset(BATCH_SIZE, loss_function)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)

        for i in range(1, 6):
            for images, labels in DataLoader(training_set, BATCH_SIZE):
                images = images.to(device)
                labels = labels.to(device)
                output = classifier(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Loss in epoch {i}: {loss.item()}")
                #training_set.flip_labels(output, labels)
                #print("labels:", labels[0])
                #print("prediction:", [torch.argmax(output[0, :CHAR_COUNT]).item(), torch.argmax(output[0, CHAR_COUNT:2*CHAR_COUNT]).item(), torch.argmax(output[0, 2*CHAR_COUNT:3*CHAR_COUNT]).item()])


        with open(f"{MODEL_PATH}three_char_detector.pt", "wb") as file:
                torch.save(classifier.state_dict(), file)
    else:
        with open(f"{MODEL_PATH}three_char_detector.pt", "rb") as file:
                classifier.load_state_dict(torch.load(file))
        
        classifier = classifier.eval()
        for image, label in DataLoader(TrainingDataset(1, None), 1):
            image = image.to(device)
            label = label.to(device)
            output = classifier(image)
            classified = torch.argmax(output).item()
            labeled = label.item()
            
            plt.imshow(image[0][0].numpy(), cmap='gray')
            plt.title(f"Image classified as {classified} and labeled as {labeled}.")
            plt.show()
                
            
