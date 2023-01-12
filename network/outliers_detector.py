import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import ALL_MERGED_PREPROCESSED_PATH
from const_config import IMAGES_FILENAME
from const_config import LABELS_FILENAME
from const_config import IMAGE_HEIGHT
from const_config import IMAGE_WIDTH
from const_config import MODELS_PATH
from const_config import OUTLIERS_DETECTOR_FILENAME
from const_config import CLEANED_PREPROCESSED_PATH
from const_config import IMAGES_FILENAME
from const_config import LABELS_FILENAME
from const_config import SEED

np.random.seed(SEED)
torch.manual_seed(SEED)

class MergedDataset():
    def __init__(self):
        self.images = torch.from_numpy(np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{IMAGES_FILENAME}", allow_pickle=True))
        self.labels = torch.from_numpy(np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{LABELS_FILENAME}", allow_pickle=True))
        self.len = self.labels.shape[0]
        self.index = 0

    def __getitem__(self, idx):
        return self.images[idx:idx + 1], self.labels[idx]

    def __len__(self):
        return self.len
    
    def get_batch(self, size):
        if self.index < self.len:
            old_index = self.index
            self.index += size
            return old_index, self.images[old_index:self.index].unsqueeze(1), self.labels[old_index:self.index]
        else:
            self.index = 0
            return None

class OutliersDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, (3, 3)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * (IMAGE_WIDTH - 4) * (IMAGE_HEIGHT - 4), 14)
        )
    
    def forward(self, x):
        return self.model(x)

def model_accuracy(classifier, data_set, batch_size):
    classifier = classifier.eval()
    correct = 0
    for images, labels in DataLoader(data_set, batch_size):
        output = classifier(images)
        correct += (torch.argmax(output, dim=1) == labels.to(torch.int64)).sum()
        
    return correct / data_set.__len__()

if __name__ == "__main__":
    classifier = OutliersDetector()
    loss_function = nn.CrossEntropyLoss()
    data_set = MergedDataset()
    batch_size = 128

    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.005)
        nmber_of_batches = data_set.__len__() / batch_size
        average_loss = 0

        for i in range(1, 16):
            j = 0
            classifier = classifier.train()
            for images, labels in DataLoader(data_set, batch_size):
                output = classifier(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                average_loss += loss.item() / nmber_of_batches
                j += 1
                if j % 100 == 0:
                    print(f"Loss in epoch {i}, batch {j}: {loss.item()}")        

            print(f"Average loss in epoch {i}: {average_loss}")
            with open(f"{MODELS_PATH}{OUTLIERS_DETECTOR_FILENAME}", "wb") as file:
                torch.save(classifier.state_dict(), file)

            accuracy = model_accuracy(classifier, data_set, batch_size)
            print(f"Model accuracy: {accuracy * 100}%")
            if accuracy > 0.9:
                break

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "accuracy":
        accuracy = model_accuracy(classifier, data_set, batch_size)
        print(f"Model accuracy: {accuracy * 100}%")

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "clean":
        with open(f"{MODELS_PATH}{OUTLIERS_DETECTOR_FILENAME}", "rb") as file:
            classifier.load_state_dict(torch.load(file))

        correct_indices = torch.empty((0))
        classifier = classifier.eval()
        softmax = nn.Softmax(dim=1)

        while True:
            batch = data_set.get_batch(batch_size)
            if batch == None:
                break
                
            index_shift, images, labels = batch
            output = classifier(images)
            
            indices = torch.add((torch.argmax(output, dim=1) == labels * (softmax(output) > 0.85).sum(dim=1)).nonzero(), index_shift)
            correct_indices = torch.cat((correct_indices, indices))
        
        print(f"Original number of samples: {data_set.__len__()}, number of samples after cleaning: {correct_indices.shape[0]}.")
        print("%.2f%s samples removed." % ((1 - correct_indices.shape[0] / data_set.__len__()) * 100, "%"))
        
        del data_set
        correct_indices = correct_indices.to(torch.int32).numpy()

        os.makedirs(CLEANED_PREPROCESSED_PATH, exist_ok=True)
        np.save(f"{CLEANED_PREPROCESSED_PATH}{IMAGES_FILENAME}", np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{IMAGES_FILENAME}", allow_pickle=True)[correct_indices].squeeze(1))
        np.save(f"{CLEANED_PREPROCESSED_PATH}{LABELS_FILENAME}", np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{LABELS_FILENAME}", allow_pickle=True)[correct_indices].squeeze(1))

        print("Images file size: %.3f MB" % (os.stat(f'{CLEANED_PREPROCESSED_PATH}{IMAGES_FILENAME}').st_size / (1024 * 1024)))
        print("Labels file size: %.3f MB" % (os.stat(f'{CLEANED_PREPROCESSED_PATH}{LABELS_FILENAME}').st_size / (1024 * 1024)))

    else:
        classifier = classifier.eval()
        for image, label in DataLoader(data_set, 1):
            output = classifier(image)
            classified = torch.argmax(output).item()
            labeled = label.item()
            
            plt.imshow(image[0][0].numpy(), cmap='gray')
            plt.title(f"Image classified as {classified} and labeled as {labeled}.")
            plt.show()
            
