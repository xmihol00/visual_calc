import os
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import lr_scheduler as sdl
import Levenshtein as lv
import numpy as np
import argparse
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
try:
    from utils.data_loaders import DataLoader
    from utils.loss_functions import CustomCrossEntropyLoss
    from utils.evaluation import EarlyStopping
except:
    from data_loaders import DataLoader
    from loss_functions import CustomCrossEntropyLoss
    from evaluation import EarlyStopping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import label_extractors
from const_config import BATCH_SIZE_TRAINING
from const_config import BATCHES_PER_FILE_TRAINING
from const_config import NUMBER_OF_FILES_TRAINING
from const_config import BATCH_SIZE_VALIDATION
from const_config import BATCHES_PER_FILE_VALIDATION
from const_config import NUMBER_OF_FILES_VALIDATION
from const_config import CUSTOM_RECURSIVE_CNN_FILENAME
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

class CustomRecursiveCNN(nn.Module):
    def __init__(self, device, augmentation, batch_size=0):
        super().__init__()
        self.model_path = AUGMENTED_MODELS_PATH if augmentation else NOT_AUGMENTED_MODELS_PATH
        self.results = [None] * (LABELS_PER_IMAGE + 1)
        self.device = device
        if batch_size:
            self.results[0] = torch.zeros((batch_size, OUTPUTS_PER_LABEL)).to(device)
            self.results[0][:, OUTPUTS_PER_LABEL - 1] = 1

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

        self.image_part_conv = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

        self.image_part_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 + OUTPUTS_PER_LABEL, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, OUTPUTS_PER_LABEL),
        )

    def forward(self, x):
        x = self.whole_image(x)
        for i in range(1, LABELS_PER_IMAGE + 1):
            j = 4 * (i - 1)
            intermidiate = self.image_part_conv(x[:, :, :, j:j+4])
            self.results[i] = self.image_part_fc(torch.cat([self.results[i - 1], intermidiate], 1))

        return torch.cat(self.results[1:], 1).reshape(-1, OUTPUTS_PER_LABEL)
    
    def load(self):
        with open(f"{self.model_path}{CUSTOM_RECURSIVE_CNN_FILENAME}", "rb") as file:
            self.load_state_dict(torch.load(file))
    
    def save(self):
        with open(f"{self.model_path}{CUSTOM_RECURSIVE_CNN_FILENAME}", "wb") as file:
            torch.save(self.state_dict(), file)
    
    def change_batch_size(self, batch_size):
        self.results[0] = torch.zeros((batch_size, OUTPUTS_PER_LABEL)).to(self.device)
        self.results[0][:, OUTPUTS_PER_LABEL - 1] = 1

if __name__ == "__main__":
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

    model = CustomRecursiveCNN(device, args.augmentation)
    model.to(device)
    loss_function = CustomCrossEntropyLoss()
    
    if args.train:
        model.change_batch_size(BATCH_SIZE_TRAINING)
        training_loader = DataLoader("training/", equations_path, BATCH_SIZE_TRAINING, BATCHES_PER_FILE_TRAINING, NUMBER_OF_FILES_TRAINING, device)
        validation_loader = DataLoader("validation/", equations_path, BATCH_SIZE_VALIDATION, BATCHES_PER_FILE_VALIDATION, NUMBER_OF_FILES_VALIDATION, device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = sdl.StepLR(optimizer, 5, 0.25)
        early_stopper = EarlyStopping()

        for i in range(1, 125):
            model.train()
            total_loss = 0
            for images, labels in training_loader:
                output = model(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            print(f"Training loss in epoch {i}: {total_loss / (BATCHES_PER_FILE_TRAINING * NUMBER_OF_FILES_TRAINING)}")
            
            model.eval()
            total_loss = 0
            for images, labels in validation_loader:
                output = model(images)
                loss = loss_function(output, labels)
                total_loss += loss.item()
            
            print(f"  Validation loss in epoch {i}: {total_loss / (BATCHES_PER_FILE_VALIDATION * NUMBER_OF_FILES_VALIDATION)}")
            if early_stopper(model, total_loss):
                break

        model.save()
    elif args.evaluate:
        model.load()
        model.change_batch_size(BATCH_SIZE_TESTING)
        model = model.eval()
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

        figure, axis = plt.subplots(1, 1)
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
        plt.savefig(f"{RESULTS_PATH}custom_recursive_CNN_evaluation")
        plt.show()
            
    else:
        model.load()
        model.change_batch_size(BATCH_SIZE_TESTING)
        model = model.eval()
        test_dataloader = DataLoader("testing/", equations_path, BATCH_SIZE_TESTING, BATCHES_PER_FILE_TESTING, NUMBER_OF_FILES_TESTING, device)

        for images, labels in test_dataloader:
            labels = labels.numpy()
            for i in range(BATCH_SIZE_TESTING):
                prediction = model(images[i : i + 1])
                
                labeled = label_extractors.labels_only_class(labels, i)
                classified = label_extractors.prediction_only_class(prediction)

                plt.imshow(images[i][0].numpy(), cmap='gray')
                plt.title(f"Image classified as {classified} and labeled as {labeled}.")
                plt.show()

