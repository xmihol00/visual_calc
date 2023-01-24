import os
import random
import sys
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import lr_scheduler as sdl
import Levenshtein as lv

from utils.data_loaders import DataLoader
from utils.loss_functions import CustomCrossEntropyLoss
from utils.evaluation import EarlyStopping
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import label_extractors
from const_config import BATCH_SIZE_TRAINING
from const_config import BATCHES_PER_FILE_TRAINING
from const_config import NUMBER_OF_FILES_TRAINING
from const_config import BATCH_SIZE_VALIDATION
from const_config import BATCHES_PER_FILE_VALIDATION
from const_config import NUMBER_OF_FILES_VALIDATION
from const_config import CUSTOM_CNN_FILENAME
from const_config import LABELS_PER_IMAGE
from const_config import OUTPUTS_PER_LABEL
from const_config import BATCH_SIZE_TESTING
from const_config import BATCHES_PER_FILE_TESTING
from const_config import NUMBER_OF_FILES_TESTING
from const_config import NUMBER_OF_FILES_TESTING
from const_config import AUGMENTED_MODELS_PATH
from const_config import NOT_AUGMENTED_MODELS_PATH
from const_config import AUGMENTED_EQUATIONS_PATH
from const_config import EQUATIONS_PATH
from const_config import SEED
from const_config import RESULTS_PATH

class CustomCNN(nn.Module):
    def __init__(self, augmentation):
        super().__init__()
        self.model_path = AUGMENTED_MODELS_PATH if augmentation else NOT_AUGMENTED_MODELS_PATH
        self.results = [None] * LABELS_PER_IMAGE

        self.whole_image = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, (5, 1), padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )

        self.part_of_image = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, OUTPUTS_PER_LABEL),
        )

    def forward(self, x):
        x = self.whole_image(x)
        for i in range(LABELS_PER_IMAGE):
            j = 4 * i
            self.results[i] = self.part_of_image(x[:, :, :, j:j+4]) # slice the input input from part 1 of the network
        return torch.cat(self.results, 1).reshape(-1, OUTPUTS_PER_LABEL)
    
    def load(self):
        with open(f"{self.model_path}{CUSTOM_CNN_FILENAME}", "rb") as file:
            self.load_state_dict(torch.load(file))
    
    def save(self):
        with open(f"{self.model_path}{CUSTOM_CNN_FILENAME}", "wb") as file:
            torch.save(self.state_dict(), file)

if __name__ == "__main__":
    # fix randomness
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="Train the neural network.")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate the neural network.")
    parser.add_argument("-a", "--augmentation", action="store_true", help="Use augmented data set.")
    args = parser.parse_args()

    equations_path = AUGMENTED_EQUATIONS_PATH if args.augmentation else EQUATIONS_PATH
    device = torch.device("cpu")
    if torch.cuda.is_available() and args.train: # move to GPU, if available
        device = torch.device("cuda")
        print("Running on GPU")

    model = CustomCNN(args.augmentation)
    model.to(device)
    loss_function = CustomCrossEntropyLoss() # just basic CE loss inside

    if args.train:
        # load data with custom data loader to save memory
        training_loader = DataLoader("training/", equations_path, BATCH_SIZE_TRAINING, BATCHES_PER_FILE_TRAINING, NUMBER_OF_FILES_TRAINING, device)
        validation_loader = DataLoader("validation/", equations_path, BATCH_SIZE_VALIDATION, BATCHES_PER_FILE_VALIDATION, NUMBER_OF_FILES_VALIDATION, device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = sdl.StepLR(optimizer, 5, 0.25) # decay learning rate each 5 epochs by 0.25
        early_stopper = EarlyStopping() # custom early stopping with patience of 3

        for i in range(1, 125):
            model.train() # trainig mode
            total_loss = 0
            for images, labels in training_loader:
                output = model(images) # predict
                loss = loss_function(output, labels) # compute loss

                optimizer.zero_grad()
                loss.backward()     # compute gradients
                optimizer.step()    # update weights 
                total_loss += loss.item()
            
            scheduler.step()
            print(f"Training loss in epoch {i}: {total_loss / (BATCHES_PER_FILE_TRAINING * NUMBER_OF_FILES_TRAINING)}")
            
            model.eval() # evaluation mode
            total_loss = 0
            for images, labels in validation_loader:
                output = model(images)
                loss = loss_function(output, labels)
                total_loss += loss.item()
            
            print(f"  Validation loss in epoch {i}: {total_loss / (BATCHES_PER_FILE_VALIDATION * NUMBER_OF_FILES_VALIDATION)}")
            if early_stopper(model, total_loss): # no improvement in multiple successive epochs
                break
        model.save()

    elif args.evaluate:
        model.load()
        model.eval() # evaluation mode
        test_dataloader = DataLoader("testing/", equations_path, BATCH_SIZE_TESTING, BATCHES_PER_FILE_TESTING, NUMBER_OF_FILES_TESTING, device)

        distances = [0] * 9
        for images, labels in test_dataloader:
            predictions = model(images)
            labels = labels.reshape(-1, LABELS_PER_IMAGE, 2).numpy()
            
            for i in range(BATCH_SIZE_TESTING):
                j = i * LABELS_PER_IMAGE
                labeled = label_extractors.labels_only_class(labels, i, sep="")
                classified = label_extractors.prediction_only_class(predictions[j:j+LABELS_PER_IMAGE], sep="")
                distances[lv.distance(labeled, classified, score_cutoff=7)] += 1
        
        print(f"distances: {distances}")

        bins = [i * 10 for i in range(10)]
        annotations_x = [i * 10 + 5 for i in range(10)]

        # bar plot
        figure, axis = plt.subplots(1, 1)
        figure.set_size_inches(10, 8.6)
        plt.subplots_adjust(left=-0.03, bottom=0.07, right=1.05, top=0.96, hspace=0.1, wspace=0.02)
        axis.set_xticks(annotations_x[:-1], distances)
        axis.set_yticks([], [])
        axis.set_frame_on(False)
        *_, patches = axis.hist(bins[:-1], bins, weights=distances)
        patches[0].set_facecolor("green")
        patches[1].set_facecolor("mediumseagreen")
        patches[2].set_facecolor("orange")
        for patch in patches[3:]:
            patch.set_facecolor("red")
        for count, x_pos in zip(distances, annotations_x):
            axis.annotate(str(count), (x_pos, count), ha="center", va="bottom", fontweight="bold")
        plt.savefig(f"{RESULTS_PATH}custom_CNN_evaluation")
        plt.show()
            
    else:
        model.load()
        model = model.eval()
        test_dataloader =  DataLoader("testing/", equations_path, BATCH_SIZE_TESTING, BATCHES_PER_FILE_TESTING, NUMBER_OF_FILES_TESTING, device)

        for images, labels in test_dataloader:
            labels = labels.numpy()
            for i in range(BATCH_SIZE_TESTING):
                prediction = model(images[i : i + 1])
                
                labeled = label_extractors.labels_only_class(labels, i) # get the string label
                classified = label_extractors.prediction_only_class(prediction) # get the string prediction

                # plot of test samples with prediction and ground truth
                plt.imshow(images[i][0].numpy(), cmap='gray')
                plt.title(f"Image classified as {classified} and labeled as {labeled}.")
                plt.show()

