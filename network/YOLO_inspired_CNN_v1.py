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
from const_config import YOLO_V1_TRAINING_IMAGES_FILENAME
from const_config import YOLO_V1_TRAINING_LABELS_FILENAME
from const_config import MODEL_PATH
from const_config import YOLO_V1_MODEL_FILENAME
import label_extractors
from utils.data_loaders import DataLoader
from utils.loss_functions import YoloLoss

class YoloInspiredCNNv1(nn.Module):
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

            nn.Conv2d(256, 128, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 32, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 15, (1, 1), stride=1, padding=0),
        )
    
    def forward(self, x):
        return self.model(x).reshape(x.shape[0] * 25, 15)

if __name__ == "__main__":
    device = None
    if CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Running on device: {device}")

    model = YoloInspiredCNNv1()
    model.to(device)
    loss_function = YoloLoss()
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        try: # loading already pre-trained model
            with open(f"{MODEL_PATH}{YOLO_V1_MODEL_FILENAME}", "rb") as file:
                model.load_state_dict(torch.load(file))
        except:
            pass

        for i in range(1, 101):
            j = 0
            for images, labels in DataLoader(BATCH_SIZE, BATCHES_PER_FILE, NUMBER_OF_FILES, device, YOLO_V1_TRAINING_IMAGES_FILENAME, YOLO_V1_TRAINING_LABELS_FILENAME):
                output = model(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                j += 1
                if j % 100 == 0:
                    print(f"Loss in epoch {i}, iteration {j}: {loss.item()}")
            
            with open(f"{MODEL_PATH}{YOLO_V1_MODEL_FILENAME}", "wb") as file:
                    torch.save(model.state_dict(), file)

    else:
        with open(f"{MODEL_PATH}{YOLO_V1_MODEL_FILENAME}", "rb") as file:
            model.load_state_dict(torch.load(file))
        
        operators = ["+", "-", "*", "/"]
        model = model.eval()
        for images, labels in DataLoader(BATCH_SIZE, BATCHES_PER_FILE, NUMBER_OF_FILES, device, YOLO_V1_TRAINING_IMAGES_FILENAME, YOLO_V1_TRAINING_LABELS_FILENAME):
            labels = labels.to("cpu").numpy()
            for i in range(BATCH_SIZE):
                prediction = model(images[i : i + 1])
                
                labeled = label_extractors.yolo_v1(labels, i)
                classified = label_extractors.yolo_v1_prediction(prediction)

                plt.imshow(images[i][0].to("cpu").numpy(), cmap='gray')
                plt.title(f"Image classified as {classified} and labeled as {labeled}.")
                plt.show()
