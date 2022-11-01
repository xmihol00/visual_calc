import numpy as np
import matplotlib.pyplot as plt
import sys

EQUATIONS_PATH = "./data/equations/"
TRAINING_IMAGES_FILENAME = "equations_%s_training_images_%s.npy"
TRAINING_LABELS_FILENAME = "equations_%s_training_labels_%s.npy"

HELP_MSG = "Run as: python equation_plot.py ['image type'] ['batch size'] ['batches per file'] ['number of files']"
BATCH_SIZE = 8
BATCHES_PER_FILE = 100
FILES = 4

PLOTTED_WINDOWS_COUNT = 10
SUBPLOT_X_COUNT = 4
SUBPLOT_Y_COUNT = 2

ONE_HOT_CHAR_CNT = 14  # number of true/false values to encode a character
NUMBER_OF_DIGITS = 10

class ImagePlotter():
    def __init__(self, subplot_x_cnt, subplot_y_cnt, file_type, figsize=(13, 12)):
        self.subplot_x_cnt = subplot_x_cnt
        self.subplot_y_cnt = subplot_y_cnt
        self.figsize = figsize
        self.row_idx = 0
        self.col_idx = 0
        _, self.axes = plt.subplots(self.subplot_x_cnt, self.subplot_y_cnt, figsize=self.figsize)
        self.operator_dict = { 10.0: "+", 11.0: "-", 12.0: "*", 13.0: "/"}

        if file_type == "DOD_90x30":
            self.label_extractor = self.extract_dod_90x30_label
        elif file_type == "DOD_132x40":
            pass
        elif file_type == "RND_230x38":
            self.label_extractor = self.extract_rnd_230x38_label
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
                _, self.axes = plt.subplots(self.subplot_x_cnt, self.subplot_y_cnt, figsize=self.figsize)

    def extract_dod_90x30_label(self, labels, idx):
        label = labels[idx]
        return f"{int(label[0])} {self.operator_dict[label[1]]} {int(label[2])}"
    
    def extract_rnd_230x38_label(self, labels, idx):
        LABELS_PER_IMAGE = 25
        label_str = ""
        for label in labels[idx * LABELS_PER_IMAGE : LABELS_PER_IMAGE + idx * LABELS_PER_IMAGE]:
            if label[0] == 1: # label contains digit or operator
                if label[1] > 9: # label contains operator
                    label_str += f" {self.operator_dict[label[1]]} "
                else: # label contains digit
                    label_str += f"{label[1]}"

        return label_str

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Not enough arguments.", file=sys.stderr)
        print(HELP_MSG, file=sys.stderr)
        exit(1)
    
    if len(sys.argv) >= 5:
        BATCH_SIZE = int(sys.argv[2])
        BATCHES_PER_FILE = int(sys.argv[3])
        FILES = int(sys.argv[4])
    
    TRAINING_IMAGES_FILENAME = TRAINING_IMAGES_FILENAME % (sys.argv[1], "%s")
    TRAINING_LABELS_FILENAME = TRAINING_LABELS_FILENAME % (sys.argv[1], "%s")

    plotter = ImagePlotter(SUBPLOT_X_COUNT, SUBPLOT_Y_COUNT, sys.argv[1])
    for i in range(PLOTTED_WINDOWS_COUNT):
        image_file = np.load(f"{EQUATIONS_PATH}{TRAINING_IMAGES_FILENAME % i}", allow_pickle=True)
        label_file = np.load(f"{EQUATIONS_PATH}{TRAINING_LABELS_FILENAME % i}", allow_pickle=True)
        for j in range(BATCHES_PER_FILE):
            image_batch = image_file[j]
            label_batch = label_file[j]
            for k in range(BATCH_SIZE):
                plotter.plot(image_batch, label_batch, k)
