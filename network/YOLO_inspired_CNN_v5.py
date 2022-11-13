import os
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import lr_scheduler as sdl

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import BATCH_SIZE
from const_config import BATCHES_PER_FILE
from const_config import NUMBER_OF_FILES
from const_config import CUDA
from const_config import YOLO_TRAINING_IMAGES_FILENAME
from const_config import YOLO_TRAINING_LABELS_FILENAME
from const_config import MODEL_PATH
from const_config import YOLO_V5_MODEL_FILENAME
from const_config import YOLO_LABELS_PER_IMAGE
from const_config import YOLO_OUTPUTS_PER_LABEL_NO_CLASS
import label_extractors
from utils.data_loaders import DataLoader
from utils.loss_functions import YoloLossNoClassBias
import utils.NN_blocks as blocks

class YoloInspiredCNNv5(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample_blocks = nn.ModuleList([blocks.CNN_downsampling(1, 8, 1), blocks.CNN_downsampling(8, 16, 1), 
                                                blocks.CNN_downsampling(16, 32, 1), blocks.CNN_downsampling(32, 64, 1)])
        self.blocks = nn.ModuleList([blocks.CNN_residual(8), blocks.CNN_residual(16), blocks.CNN_residual(32), blocks.CNN_residual(64)])
        self.YOLO_block = blocks.YOLO(64, YOLO_OUTPUTS_PER_LABEL_NO_CLASS)

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.downsample_blocks[i](x)
            x = self.blocks[i](x) + x
        
        x = self.YOLO_block(x)
        return x.reshape(x.shape[0] * YOLO_LABELS_PER_IMAGE, YOLO_OUTPUTS_PER_LABEL_NO_CLASS)

if __name__ == "__main__":
    model = YoloInspiredCNNv5()
    loss_function = YoloLossNoClassBias()
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = sdl.StepLR(optimizer, 25, 0.5)
        
        device = torch.device("cpu")
        if CUDA: # move to GPU, if available
            device = torch.device("cuda")
            model.to(device)
            print(f"Running on GPU")

        try: # loading already pre-trained model
            with open(f"{MODEL_PATH}{YOLO_V5_MODEL_FILENAME}", "rb") as file:
                model.load_state_dict(torch.load(file))
        except:
            pass

        for i in range(1, 125):
            for images, labels in DataLoader("training/", BATCH_SIZE, BATCHES_PER_FILE, NUMBER_OF_FILES, device, YOLO_TRAINING_IMAGES_FILENAME, YOLO_TRAINING_LABELS_FILENAME):
                output = model(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            print(f"Loss in epoch {i}: {loss.item()}")
            with open(f"{MODEL_PATH}{YOLO_V5_MODEL_FILENAME}", "wb") as file:
                    torch.save(model.state_dict(), file)

    else:
        with open(f"{MODEL_PATH}{YOLO_V5_MODEL_FILENAME}", "rb") as file:
            model.load_state_dict(torch.load(file))
        
        operators = ["+", "-", "*", "/"]
        model = model.eval()
        for images, labels in DataLoader("training/", BATCH_SIZE, BATCHES_PER_FILE, NUMBER_OF_FILES, torch.device("cpu"), YOLO_TRAINING_IMAGES_FILENAME, YOLO_TRAINING_LABELS_FILENAME):
            labels = labels.numpy()
            for i in range(BATCH_SIZE):
                prediction = model(images[i : i + 1])
                
                labeled = label_extractors.yolo(labels, i)
                classified = label_extractors.yolo_prediction(prediction)

                plt.imshow(images[i][0].numpy(), cmap='gray')
                plt.title(f"Image classified as {classified} and labeled as {labeled}.")
                plt.show()
