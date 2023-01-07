import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from const_config import EQUATIONS_PATH
from const_config import LABEL_DIMENSIONS
from const_config import IMAGES_FILENAME_TEMPLATE
from const_config import LABELS_FILENAME_TEMPLATE

class DataLoader():
    def __init__(self, directory, batch_size, batches_per_file, number_of_files, device):
        self.BATCH_SIZE = batch_size
        self.SAMPLES_PER_FILE = batches_per_file * batch_size
        self.NUMBER_OF_FILES = number_of_files

        self.device = device
        self.directory = directory

        self.file_idx = -1
        self.sample_idx = self.SAMPLES_PER_FILE # point behind the last index of a file
        self.indices = np.random.choice(self.SAMPLES_PER_FILE, self.SAMPLES_PER_FILE, replace=False) # randomly place images and labels in a batch from a file
        
        self.images_file_template = IMAGES_FILENAME_TEMPLATE
        self.labels_file_template = LABELS_FILENAME_TEMPLATE
        self.image_file = None
        self.label_file = None
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.sample_idx == self.SAMPLES_PER_FILE: # iterated all samples in a file, new must be opened
            self.file_idx += 1
            if self.file_idx == self.NUMBER_OF_FILES: # iterated through all files, reset to initial state and stop iteration
                self.file_idx = -1
                self.sample_idx = self.SAMPLES_PER_FILE # point behind the last index of a file
                self.indices = np.random.choice(self.SAMPLES_PER_FILE, self.SAMPLES_PER_FILE, replace=False) # randomly place images and labels in a batch from a file
                raise StopIteration

            self.sample_idx = 0 # new file is loaded
            self.image_file = np.load(f"{EQUATIONS_PATH}{self.directory}{self.images_file_template % self.file_idx}", allow_pickle=True)
            self.label_file = np.load(f"{EQUATIONS_PATH}{self.directory}{self.labels_file_template % self.file_idx}", allow_pickle=True)
        
        old_idx = self.sample_idx
        self.sample_idx += self.BATCH_SIZE # move to the next batch index
        return (torch.from_numpy(self.image_file[self.indices[old_idx:self.sample_idx]]).to(self.device), 
                torch.from_numpy(self.label_file[self.indices[old_idx:self.sample_idx]].reshape(-1, LABEL_DIMENSIONS)).to(self.device))
