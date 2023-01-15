import argparse
import os
import random
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import lr_scheduler as sdl

from utils.data_loaders import DataLoader
from utils.loss_functions import BCEBiasedFollowedByCELoss
import utils.NN_blocks as blocks
from utils.evaluation import EarlyStopping
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import label_extractors
from const_config import BATCH_SIZE_TESTING
from const_config import BATCH_SIZE_TRAINING
from const_config import BATCHES_PER_FILE_TESTING
from const_config import EQUATIONS_PATH
from const_config import NUMBER_OF_FILES_TESTING
from const_config import BATCHES_PER_FILE_TRAINING
from const_config import NUMBER_OF_FILES_TRAINING
from const_config import BATCH_SIZE_VALIDATION
from const_config import BATCHES_PER_FILE_VALIDATION
from const_config import NUMBER_OF_FILES_VALIDATION
from const_config import CUDA
from const_config import MODELS_PATH
from const_config import YOLO_INSPIRED_MODEL_FILENAME
from const_config import LABELS_PER_IMAGE
from const_config import OUTPUTS_PER_LABEL
from const_config import SEED

class YOLOInspiredCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample_blocks = nn.ModuleList([blocks.CNN_downsampling(1, 32), blocks.CNN_downsampling(32, 64), blocks.CNN_downsampling(64, 128)])
        self.residual_blocks = nn.ModuleList([blocks.CNN_residual(32), blocks.CNN_residual(64), blocks.CNN_residual(128)])
        self.YOLO_block = blocks.CNN_head(128, OUTPUTS_PER_LABEL)

    def forward(self, x):
        for i in range(len(self.residual_blocks)):
            x = self.downsample_blocks[i](x)
            x = x + self.residual_blocks[i](x)
        
        x = self.YOLO_block(x)
        return x.reshape(x.shape[0] * LABELS_PER_IMAGE, OUTPUTS_PER_LABEL)
    
    def load(self):
        with open(f"{MODELS_PATH}{YOLO_INSPIRED_MODEL_FILENAME}", "rb") as file:
            self.load_state_dict(torch.load(file))
    
    def save(self):
        with open(f"{MODELS_PATH}{YOLO_INSPIRED_MODEL_FILENAME}", "wb") as file:
            torch.save(self.state_dict(), file)

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

    equations_path = EQUATIONS_PATH
    device = torch.device("cpu")
    if CUDA and args.train: # move to GPU, if available
        device = torch.device("cuda")
        print("Running on GPU")

    model = YOLOInspiredCNN()
    loss_function = BCEBiasedFollowedByCELoss()
    model.to(device)
    
    if args.train:
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
                
                del images, labels
            
            scheduler.step()
            print(f"Training loss in epoch {i}: {total_loss / (BATCHES_PER_FILE_TRAINING * NUMBER_OF_FILES_TRAINING)}")

            model.eval()
            total_loss = 0
            for images, labels in validation_loader:
                output = model(images)
                loss = loss_function(output, labels)
                total_loss += loss.item()

                del images, labels
            
            print(f"  Validation loss in epoch {i}: {total_loss / (BATCHES_PER_FILE_VALIDATION * NUMBER_OF_FILES_VALIDATION)}")
            if early_stopper(model, total_loss):
                break
        
        model.save()

    elif args.evaluate:
        model.load()
        model.eval()
        for images, labels in DataLoader("testing/", equations_path, BATCH_SIZE_TESTING, BATCHES_PER_FILE_TESTING, NUMBER_OF_FILES_TESTING, device):
            labels = labels.numpy()
            for i in range(BATCH_SIZE_TRAINING):
                prediction = model(images[i : i + 1])
                
                labeled = label_extractors.labels(labels, i)
                classified = label_extractors.prediction(prediction)

                plt.imshow(images[i][0].numpy(), cmap='gray')
                plt.title(f"Image classified as {classified} and labeled as {labeled}.")
                plt.show()
