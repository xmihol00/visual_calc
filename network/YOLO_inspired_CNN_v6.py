import os
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import lr_scheduler as sdl

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import BATCH_SIZE_TRAINING
from const_config import BATCHES_PER_FILE_TRAINING
from const_config import NUMBER_OF_FILES_TRAINING
from const_config import BATCH_SIZE_VALIDATION
from const_config import BATCHES_PER_FILE_VALIDATION
from const_config import NUMBER_OF_FILES_VALIDATION
from const_config import CUDA
from const_config import YOLO_TRAINING_IMAGES_FILENAME
from const_config import YOLO_TRAINING_LABELS_FILENAME
from const_config import MODEL_PATH
from const_config import YOLO_V6_MODEL_FILENAME
from const_config import YOLO_LABELS_PER_IMAGE
from const_config import YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES
import label_extractors
from utils.data_loaders import DataLoader
from utils.loss_functions import YoloLossOnlyClasses
from utils.evaluation import EarlyStopping

class YoloInspiredCNNv6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_part = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, (3, 3), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 1), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, (2, 2), padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, (2, 2), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(0.1),
        )

        self.fully_conn_part = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES),
        )

    def forward(self, x):
        results = [None] * 13
        x = self.conv_part(x)
        for i in range(YOLO_LABELS_PER_IMAGE):
            j = 2 * i
            results[i] = self.fully_conn_part(x[:, :, :, j:j+2])
        return torch.cat(results, 1).reshape(-1, YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES)

if __name__ == "__main__":
    model = YoloInspiredCNNv6()
    loss_function = YoloLossOnlyClasses()
    
    device = torch.device("cpu")
    if CUDA and len(sys.argv) > 1 and sys.argv[1].lower() == "train": # move to GPU, if available
        device = torch.device("cuda")
        model.to(device)
        print(f"Running on GPU")
    
    training_loader = DataLoader("training/", BATCH_SIZE_TRAINING, BATCHES_PER_FILE_TRAINING, NUMBER_OF_FILES_TRAINING, device, YOLO_TRAINING_IMAGES_FILENAME, YOLO_TRAINING_LABELS_FILENAME)
    validation_loader = DataLoader("validation/", BATCH_SIZE_VALIDATION, BATCHES_PER_FILE_VALIDATION, NUMBER_OF_FILES_VALIDATION, device, YOLO_TRAINING_IMAGES_FILENAME, YOLO_TRAINING_LABELS_FILENAME)

    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = sdl.StepLR(optimizer, 40, 0.5)
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

        with open(f"{MODEL_PATH}{YOLO_V6_MODEL_FILENAME}", "wb") as file:
            torch.save(model.state_dict(), file)
    else:
        with open(f"{MODEL_PATH}{YOLO_V6_MODEL_FILENAME}", "rb") as file:
            model.load_state_dict(torch.load(file))
        
        operators = ["+", "-", "*", "/"]
        model = model.eval()
        for images, labels in validation_loader:
            labels = labels.numpy()
            for i in range(BATCH_SIZE_VALIDATION):
                prediction = model(images[i : i + 1])
                
                labeled = label_extractors.yolo_only_class(labels, i)
                classified = label_extractors.yolo_prediction_only_class(prediction)

                plt.imshow(images[i][0].numpy(), cmap='gray')
                plt.title(f"Image classified as {classified} and labeled as {labeled}.")
                plt.show()
