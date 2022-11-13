import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from const_config import EQUATIONS_PATH

class DataLoader():
    def __init__(self, directory, batch_size, batches_per_file, number_of_files, device, images_file_template, labels_file_template):
        self.directory = directory
        self.file_idx = -1
        self.batch_idx = batches_per_file - 1 # point to the last index of a batch
        self.BATCH_SIZE = batch_size
        self.BATCHES_PER_FILE = batches_per_file
        self.NUMBER_OF_FILES = number_of_files
        self.device = device
        self.images_file_template = images_file_template 
        self.labels_file_template = labels_file_template 
        self.image_file = None
        self.label_file = None
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.batch_idx == self.BATCHES_PER_FILE - 1: # iterated through all batches in file, new must be opened
            self.file_idx += 1
            if self.file_idx == self.NUMBER_OF_FILES: # iterated through all files, reset to initial state and stop iteration
                self.__init__(self.directory, self.BATCH_SIZE, self.BATCHES_PER_FILE, self.NUMBER_OF_FILES, self.device, 
                              self.images_file_template, self.labels_file_template)
                raise StopIteration

            self.batch_idx = -1 # new batch is loaded
            self.image_file = np.load(f"{EQUATIONS_PATH}{self.directory}{self.images_file_template % self.file_idx}", allow_pickle=True)
            self.label_file = np.load(f"{EQUATIONS_PATH}{self.directory}{self.labels_file_template % self.file_idx}", allow_pickle=True)
        
        self.batch_idx += 1 # move to the next index in a batch
        return torch.from_numpy(self.image_file[self.batch_idx]).to(self.device), torch.from_numpy(self.label_file[self.batch_idx]).to(self.device)