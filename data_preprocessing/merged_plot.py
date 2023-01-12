import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import CLEANED_PREPROCESSED_PATH
from const_config import IMAGES_FILENAME
from const_config import LABELS_FILENAME

PLOT_WIDTH_HEIGHT = 4
images = np.load(f"{CLEANED_PREPROCESSED_PATH}{IMAGES_FILENAME}", allow_pickle=True)
labels = np.load(f"{CLEANED_PREPROCESSED_PATH}{LABELS_FILENAME}", allow_pickle=True)

for _ in range(3):
    sample_count = images.shape[0]
    figure, axis = plt.subplots(PLOT_WIDTH_HEIGHT, PLOT_WIDTH_HEIGHT)
    figure.suptitle(f"{CLEANED_PREPROCESSED_PATH}")
    for i in range(PLOT_WIDTH_HEIGHT):
        for j in range(PLOT_WIDTH_HEIGHT):
            index = rnd.randint(0, sample_count - 1)
            axis[i, j].imshow(images[index], cmap="gray")
            axis[i, j].set_title(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"][labels[index]])
            axis[i, j].set_xticks([], [])
            axis[i, j].set_yticks([], [])
            axis[i, j].set_frame_on(False)
    
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show(block=False)
    plt.pause(1.5)
    plt.close()