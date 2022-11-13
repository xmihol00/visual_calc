import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import BATCH_SIZE_TRAINING
from const_config import BATCHES_PER_FILE_TRAINING
from const_config import DATA_DIRECTORIES_INFO
import label_extractors


EQUATIONS_PATH = "./data/equations/"
TRAINING_IMAGES_FILENAME = "equations_%s_training_images_%s.npy"
TRAINING_LABELS_FILENAME = "equations_%s_training_labels_%s.npy"

HELP_MSG = "Run as: python equation_plot.py ['image type'] ['batch size'] ['batches per file'] ['number of files']"

PLOTTED_WINDOWS_COUNT = 1
SUBPLOT_X_COUNT = 4
SUBPLOT_Y_COUNT = 2

class ImagePlotter():
    def __init__(self, subplot_x_cnt, subplot_y_cnt, file_type, figsize=(13, 12), directory=""):
        self.subplot_x_cnt = subplot_x_cnt
        self.subplot_y_cnt = subplot_y_cnt
        self.figsize = figsize
        self.row_idx = 0
        self.col_idx = 0
        self.directory = directory
        self.figure, self.axes = plt.subplots(self.subplot_x_cnt, self.subplot_y_cnt, figsize=self.figsize)
        self.figure.suptitle(self.directory[:-1] + " samples")
        self.operator_dict = { 10.0: "+", 11.0: "-", 12.0: "*", 13.0: "/"}

        if file_type == "90x30":
            self.label_extractor = label_extractors.dod_90x30
        elif file_type == "132x40":
            pass
        elif file_type == "230x38":
            self.label_extractor = label_extractors.yolo
        else:
            print("Unknown image type.", file=sys.stderr)
            print(HELP_MSG, file=sys.stderr)
            exit(1)
    
    def plot(self, images, labels, idx):
        image = images[idx, 0]
        self.axes[self.row_idx, self.col_idx].imshow(image, cmap="gray")
        self.axes[self.row_idx, self.col_idx].set_title(self.label_extractor(labels, idx))
        
        self.col_idx += 1
        if self.col_idx == self.subplot_y_cnt:
            self.col_idx = 0
            self.row_idx += 1
            if self.row_idx == self.subplot_x_cnt:
                self.row_idx = 0
                plt.show()
                self.figure, self.axes = plt.subplots(self.subplot_x_cnt, self.subplot_y_cnt, figsize=self.figsize)
                self.figure.suptitle(self.directory[:-1] + " samples")
    
    def reset(self, directory):
        self.row_idx = 0
        self.col_idx = 0
        self.directory = directory
        plt.close()
        self.figure, self.axes = plt.subplots(self.subplot_x_cnt, self.subplot_y_cnt, figsize=self.figsize)
        self.figure.suptitle(self.directory[:-1] + " samples")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Not enough arguments.", file=sys.stderr)
        print(HELP_MSG, file=sys.stderr)
        exit(1)
    
    if len(sys.argv) >= 5:
        BATCH_SIZE_TRAINING = int(sys.argv[2])
        BATCHES_PER_FILE_TRAINING = int(sys.argv[3])
        NUMBER_OF_FILES = int(sys.argv[4])
    
    TRAINING_IMAGES_FILENAME = TRAINING_IMAGES_FILENAME % (sys.argv[1], "%s")
    TRAINING_LABELS_FILENAME = TRAINING_LABELS_FILENAME % (sys.argv[1], "%s")
    
    plotter = ImagePlotter(SUBPLOT_X_COUNT, SUBPLOT_Y_COUNT, sys.argv[1])
    for directory, batch_size, batches_per_file, _ in DATA_DIRECTORIES_INFO:
        plotter.reset(directory)
        for i in range(2):
            images_file = np.load(f"{EQUATIONS_PATH}{directory}{TRAINING_IMAGES_FILENAME % i}", allow_pickle=True)
            labels_file = np.load(f"{EQUATIONS_PATH}{directory}{TRAINING_LABELS_FILENAME % i}", allow_pickle=True)
            for j in range(PLOTTED_WINDOWS_COUNT * SUBPLOT_X_COUNT * SUBPLOT_Y_COUNT):
                plotter.plot(images_file, labels_file, j)
