import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import DATA_DIRECTORIES_INFO
from const_config import IMAGES_FILENAME_TEMPLATE
from const_config import LABELS_FILENAME_TEMPLATE
from const_config import AUGMENTED_EQUATIONS_PATH
from const_config import EQUATIONS_PATH
import label_extractors

PLOTTED_WINDOWS_COUNT = 1
SUBPLOT_X_COUNT = 4
SUBPLOT_Y_COUNT = 2

class ImagePlotter():
    def __init__(self, subplot_x_cnt, subplot_y_cnt, directory=""):
        self.subplot_x_cnt = subplot_x_cnt
        self.subplot_y_cnt = subplot_y_cnt
        self.row_idx = 0
        self.col_idx = 0
        self.directory = directory
        self.figure, self.axis = plt.subplots(self.subplot_x_cnt, self.subplot_y_cnt)
        self.figure.suptitle(self.directory)
        self.operator_dict = { 10.0: "+", 11.0: "-", 12.0: "*", 13.0: "/"}
        self.label_extractor = label_extractors.labels_only_class
    
    def plot(self, images, labels, idx):
        image = images[idx, 0]
        self.axis[self.row_idx, self.col_idx].imshow(image, cmap="gray")
        self.axis[self.row_idx, self.col_idx].set_title(self.label_extractor(labels, idx))
        self.axis[self.row_idx, self.col_idx].set_xticks([], [])
        self.axis[self.row_idx, self.col_idx].set_yticks([], [])
        self.axis[self.row_idx, self.col_idx].set_frame_on(False)

        self.col_idx += 1
        if self.col_idx == self.subplot_y_cnt:
            self.col_idx = 0
            self.row_idx += 1
            if self.row_idx == self.subplot_x_cnt:
                self.row_idx = 0
                plt.get_current_fig_manager().full_screen_toggle()
                plt.show(block=False)
                plt.pause(6)
                plt.close()
                self.figure, self.axis = plt.subplots(self.subplot_x_cnt, self.subplot_y_cnt)
                self.figure.suptitle(self.directory)
    
    def reset(self, directory):
        self.row_idx = 0
        self.col_idx = 0
        self.directory = directory
        plt.close()
        self.figure, self.axis = plt.subplots(self.subplot_x_cnt, self.subplot_y_cnt)
        self.figure.suptitle(self.directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--augment", action="store_true", help="Use augmented data set.")
    args = parser.parse_args()
    main_directory = AUGMENTED_EQUATIONS_PATH if args.augment else EQUATIONS_PATH

    plotter = ImagePlotter(SUBPLOT_X_COUNT, SUBPLOT_Y_COUNT)
    for directory, batch_size, batches_per_file, _ in DATA_DIRECTORIES_INFO:
        plotter.reset(f"{main_directory}{directory}")
        for i in range(1):
            images_file = np.load(f"{main_directory}{directory}{IMAGES_FILENAME_TEMPLATE % i}", allow_pickle=True)
            labels_file = np.load(f"{main_directory}{directory}{LABELS_FILENAME_TEMPLATE % i}", allow_pickle=True)
            for j in range(PLOTTED_WINDOWS_COUNT * SUBPLOT_X_COUNT * SUBPLOT_Y_COUNT):
                plotter.plot(images_file, labels_file, j)
