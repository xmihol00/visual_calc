import os
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import BATCH_SIZE
from const_config import BATCHES_PER_FILE
from const_config import NUMBER_OF_FILES
from const_config import CUDA
from const_config import YOLO_TRAINING_IMAGES_FILENAME
from const_config import YOLO_TRAINING_LABELS_FILENAME
from const_config import MODEL_PATH
from const_config import YOLO_V3_MODEL_FILENAME
from const_config import YOLO_LABELS_PER_IMAGE
from const_config import YOLO_OUTPUTS_PER_LABEL
import label_extractors
from utils.data_loaders import DataLoader
from utils.loss_functions import YoloLoss

def CNN_downsampling_block(in_channels, out_channels):
    intermidiate_channels = out_channels * 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(out_channels, intermidiate_channels, (3, 3), stride=2, padding=1),
        nn.BatchNorm2d(intermidiate_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(intermidiate_channels, out_channels, (1, 1), stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )

def CNN_block(channels):
    intermidiate_channels = channels * 2
    return nn.Sequential(
        nn.Conv2d(channels, intermidiate_channels, (3, 3), stride=1, padding=1),
        nn.BatchNorm2d(intermidiate_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(intermidiate_channels, channels, (1, 1), stride=1, padding=0),
        nn.BatchNorm2d(channels),
        nn.LeakyReLU(0.1),
    )

def YOLO_block(in_channels, out_channels):
    intermidiate_channels = in_channels * 2
    return nn.Sequential(
        nn.Conv2d(in_channels, intermidiate_channels, (3, 3), stride=1, padding=0),
        nn.BatchNorm2d(intermidiate_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(intermidiate_channels, out_channels, (1, 1), stride=1, padding=0),
    )

class YoloInspiredCNNv3(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample_blocks = nn.ModuleList([CNN_downsampling_block(1, 32), CNN_downsampling_block(32, 64), CNN_downsampling_block(64, 128)])
        self.blocks = nn.ModuleList([CNN_block(32), CNN_block(64), CNN_block(128)])
        self.YOLO_block = YOLO_block(128, YOLO_OUTPUTS_PER_LABEL)

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.downsample_blocks[i](x)
            x = self.blocks[i](x) + x
        
        x = self.YOLO_block(x)
        return x.reshape(x.shape[0] * YOLO_LABELS_PER_IMAGE, YOLO_OUTPUTS_PER_LABEL)

if __name__ == "__main__":
    model = YoloInspiredCNNv3()
    loss_function = YoloLoss()
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        
        device = torch.device("cpu")
        if CUDA:
            device = torch.device("cuda")
            model.to(device)
            print(f"Running on GPU")

        try: # loading already pre-trained model
            with open(f"{MODEL_PATH}{YOLO_V3_MODEL_FILENAME}", "rb") as file:
                model.load_state_dict(torch.load(file))
        except:
            pass

        for i in range(1, 16):
            j = 0
            for images, labels in DataLoader(BATCH_SIZE, BATCHES_PER_FILE, NUMBER_OF_FILES, device, YOLO_TRAINING_IMAGES_FILENAME, YOLO_TRAINING_LABELS_FILENAME):
                output = model(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                j += 1
                if j % 100 == 0:
                    print(f"Loss in epoch {i}, iteration {j}: {loss.item()}")
            
            with open(f"{MODEL_PATH}{YOLO_V3_MODEL_FILENAME}", "wb") as file:
                    torch.save(model.state_dict(), file)

    else:
        with open(f"{MODEL_PATH}{YOLO_V3_MODEL_FILENAME}", "rb") as file:
            model.load_state_dict(torch.load(file))
        
        operators = ["+", "-", "*", "/"]
        model = model.eval()
        for images, labels in DataLoader(BATCH_SIZE, BATCHES_PER_FILE, NUMBER_OF_FILES, torch.device("cpu"), YOLO_TRAINING_IMAGES_FILENAME, YOLO_TRAINING_LABELS_FILENAME):
            labels = labels.numpy()
            for i in range(BATCH_SIZE):
                prediction = model(images[i : i + 1])
                
                labeled = label_extractors.yolo(labels, i)
                classified = label_extractors.yolo_prediction(prediction)

                plt.imshow(images[i][0].numpy(), cmap='gray')
                plt.title(f"Image classified as {classified} and labeled as {labeled}.")
                plt.show()
