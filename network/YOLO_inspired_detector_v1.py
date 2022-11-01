import sys
import numpy as np
import torch
import idx2numpy
import matplotlib.pyplot as plt
from torch import nn

EQUATIONS_PATH = "./data/equations/"
TRAINING_IMAGES_FILENAME = "equations_RND_230x38_training_images_%s.npy"
TRAINING_LABELS_FILENAME = "equations_RND_230x38_training_labels_%s.npy"
MODEL_PATH = "./models/"

IMAGE_WIDTH = 132
IMAGE_HEIGHT = 40

BATCH_SIZE = 4

class DataLoader():
    def __init__(self, batch_size, batches_per_file, number_of_files):
        self.file_idx = -1
        self.batch_idx = batches_per_file - 1 # point to the last index of a batch
        self.BATCH_SIZE = batch_size
        self.BATCHES_PER_FILE = batches_per_file
        self.NUMBER_OF_FILES = number_of_files
        self.image_file = None
        self.label_file = None
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.batch_idx == self.BATCHES_PER_FILE - 1: # iterated through all batches in file, new must be opened
            self.file_idx += 1
            if self.file_idx == self.NUMBER_OF_FILES: # iterated through all files, reset to initial state and stop iteration
                self.__init__(self.BATCH_SIZE, self.BATCHES_PER_FILE, self.NUMBER_OF_FILES)
                raise StopIteration

            self.batch_idx = -1 # new batch is loaded
            self.image_file = np.load(f"{EQUATIONS_PATH}{TRAINING_IMAGES_FILENAME % self.file_idx}", allow_pickle=True)
            self.label_file = np.load(f"{EQUATIONS_PATH}{TRAINING_LABELS_FILENAME % self.file_idx}", allow_pickle=True)
        
        self.batch_idx += 1 # move to the next index in a batch
        return torch.from_numpy(self.image_file[self.batch_idx]), torch.from_numpy(self.label_file[self.batch_idx])

class YoloInspiredDetectorV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, (3, 3), stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 64, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 15, (1, 1), stride=1, padding=0),
        )
    
    def forward(self, x):
        return self.model(x).reshape(x.shape[0] * 25, 15)

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()
        self.cel = nn.CrossEntropyLoss()
    
    def forward(self, predictions, labels):
        indices_with_class = (labels[:, 0] == 1).nonzero(as_tuple=True)[0]
        return self.bcel(predictions[:, 0:1], labels[:, 0:1].to(torch.float32)) + self.cel(predictions[indices_with_class, 1:], labels[indices_with_class, 1])

if __name__ == "__main__":
    #t1 = torch.tensor([[[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]]], [[[13, 14, 15, 16]], [[17, 18, 19, 20]], [[21, 22, 23, 24]]]])
    #print(t1.shape)
    #print(t1)
    #t2 = t1.reshape(t1.shape[0] * 4, 3)
    #print(t2.shape)
    #print(t2)
    #t3 = t2.reshape(t1.shape)
    #print(t3)
    #exit()

    model = YoloInspiredDetectorV1()
    loss_function = YoloLoss()
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #with open(f"{MODEL_PATH}shifts_3char_1head_detector.pt", "rb") as file:
        #    classifier.load_state_dict(torch.load(file))

        for i in range(1, 101):
            j = 0
            for images, labels in DataLoader(8, 100, 4):
                output = model(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                j += 1
                if j % 100 == 0:
                    print(f"Loss in epoch {i}, iteration {j}: {loss.item()}")
            
            
            #with open(f"{MODEL_PATH}shifts_3char_1head_detector_{i}.pt", "wb") as file:
            #        torch.save(classifier.state_dict(), file)

    else:
        #with open(f"{MODEL_PATH}shifts_3char_1head_detector.pt", "rb") as file:
        #    classifier.load_state_dict(torch.load(file))
        
        operators = ["+", "-", "*", "/"]
        model = model.eval()
        for batch in DataLoader(8, 100, 4):
            for image, label in batch:
                output = model(image)
                #classified = [torch.argmax(output[0, 0:10]).item(), operators[torch.argmax(output[0, 10:14]).item()], torch.argmax(output[0, 14:24]).item()]
                #labeled = [label[0][0].item(), operators[label[0][1].item()], label[0][2].item()]

                plt.imshow(image[0][0].numpy(), cmap='gray')
                #plt.title(f"Image classified as {classified} and labeled as {labeled}.")
                plt.show()
