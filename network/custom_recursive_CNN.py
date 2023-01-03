import os
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import lr_scheduler as sdl
import Levenshtein as lv

from utils.data_loaders import DataLoader
from utils.loss_functions import YoloLossOnlyClasses
from utils.evaluation import EarlyStopping
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import BATCH_SIZE_TRAINING
from const_config import BATCHES_PER_FILE_TRAINING
from const_config import NUMBER_OF_FILES_TRAINING
from const_config import BATCH_SIZE_VALIDATION
from const_config import BATCHES_PER_FILE_VALIDATION
from const_config import NUMBER_OF_FILES_VALIDATION
from const_config import CUDA
from const_config import MODEL_PATH
from const_config import CUSTOM_RECURSIVE_CNN
from const_config import YOLO_LABELS_PER_IMAGE
from const_config import YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES
from const_config import BATCH_SIZE_TESTING
from const_config import BATCHES_PER_FILE_TESTING
from const_config import NUMBER_OF_FILES_TESTING
from const_config import NUMBER_OF_FILES_TESTING
import label_extractors

class CustomCNNv3(nn.Module):
    def __init__(self, device, batch_size=BATCH_SIZE_TRAINING):
        super().__init__()
        self.results = [None] * (YOLO_LABELS_PER_IMAGE + 1)
        self.results[0] = torch.zeros((batch_size, YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES)).to(device)
        self.results[0][:, YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES - 1] = 1

        self.whole_image = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 1), padding=0),
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
            nn.Linear(256 * 4 + YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES),
        )

    def forward(self, x):
        x = self.whole_image(x)
        for i in range(1, YOLO_LABELS_PER_IMAGE + 1):
            j = 4 * (i - 1)
            intermidiate = self.image_part_conv(x[:, :, :, j:j+4])
            self.results[i] = self.image_part_fc(torch.cat([self.results[i - 1], intermidiate], 1))

        return torch.cat(self.results[1:], 1).reshape(-1, YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES)
    
    def load(self):
        with open(f"{MODEL_PATH}{CUSTOM_RECURSIVE_CNN}", "rb") as file:
            self.load_state_dict(torch.load(file))
    
    def save(self):
        with open(f"{MODEL_PATH}{CUSTOM_RECURSIVE_CNN}", "wb") as file:
            torch.save(self.state_dict(), file)

if __name__ == "__main__":
    exe_type = sys.argv[1].lower() if len(sys.argv) > 1 else ""

    device = torch.device("cpu")
    if CUDA and exe_type == "train": # move to GPU, if available
        device = torch.device("cuda")
        print("Running on GPU")

    model = CustomCNNv3(device)
    model.to(device)
    loss_function = YoloLossOnlyClasses()
    
    if exe_type == "train":
        training_loader = DataLoader("training/", BATCH_SIZE_TRAINING, BATCHES_PER_FILE_TRAINING, NUMBER_OF_FILES_TRAINING, device, "288x38")
        validation_loader = DataLoader("validation/", BATCH_SIZE_VALIDATION, BATCHES_PER_FILE_VALIDATION, NUMBER_OF_FILES_VALIDATION, device, "288x38")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = sdl.StepLR(optimizer, 10, 0.25)
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
    elif exe_type == "eval":
        model.load()
        model = model.eval()
        test_dataloader = validation_loader = DataLoader("testing/", BATCH_SIZE_TESTING, BATCHES_PER_FILE_TESTING, NUMBER_OF_FILES_TESTING, device, "288x38")

        distances = [0] * 9
        for images, labels in validation_loader:
            labels = labels.to("cpu").numpy()
            predictions = model(images).to("cpu")
            
            for i in range(BATCH_SIZE_TESTING):
                j = i * YOLO_LABELS_PER_IMAGE
                labeled = label_extractors.yolo_only_class(labels, i).replace(' ', '')
                classified = label_extractors.yolo_prediction_only_class(predictions[j:j+YOLO_LABELS_PER_IMAGE]).replace(' ', '')
                print(labeled, "x" , classified)
                distances[lv.distance(labeled, classified, score_cutoff=7)] += 1
        
        print(distances)
        plt.hist([i for i in range(9)], [i for i in range(10)], weights=distances)
        plt.show()
            
    else:
        model.load()
        model = model.eval()
        test_dataloader = validation_loader = DataLoader("testing/", BATCH_SIZE_TESTING, BATCHES_PER_FILE_TESTING, NUMBER_OF_FILES_TESTING, device, "288x38")

        for images, labels in validation_loader:
            labels = labels.numpy()
            for i in range(BATCH_SIZE_TESTING):
                prediction = model(images[i : i + 1])
                
                labeled = label_extractors.yolo_only_class(labels, i)
                classified = label_extractors.yolo_prediction_only_class(prediction)

                plt.imshow(images[i][0].numpy(), cmap='gray')
                plt.title(f"Image classified as {classified} and labeled as {labeled}.")
                plt.show()

