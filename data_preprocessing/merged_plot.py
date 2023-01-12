import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import ALL_MERGED_PREPROCESSED_PATH
from const_config import IMAGES_FILENAME
from const_config import LABELS_FILENAME
PLOT_WIDTH_HEIGHT = 4

images = np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{IMAGES_FILENAME}", allow_pickle=True)
labels = np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{LABELS_FILENAME}", allow_pickle=True)

for _ in range(25):
    sample_count = images.shape[0]
    _, axes = plt.subplots(PLOT_WIDTH_HEIGHT, PLOT_WIDTH_HEIGHT)
    for i in range(PLOT_WIDTH_HEIGHT):
        for j in range(PLOT_WIDTH_HEIGHT):
            index = rnd.randint(0, sample_count - 1)
            axes[i, j].imshow(images[index], cmap="gray")
            axes[i, j].set_title(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"][labels[index]])
    
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show(block=False)
    plt.pause(1)
    plt.close()