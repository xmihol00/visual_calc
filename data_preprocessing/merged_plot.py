import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import ALL_MERGED_PREPROCESSED_PATH
from const_config import ALL_IMAGES_FILENAME
from const_config import ALL_LABELS_FILENAME
PLOT_WIDTH_HEIGHT = 4

images = np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{ALL_IMAGES_FILENAME}", allow_pickle=True)
labels = np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{ALL_LABELS_FILENAME}", allow_pickle=True)

for _ in range(10):
    sample_count = images.shape[0]
    _, axes = plt.subplots(PLOT_WIDTH_HEIGHT, PLOT_WIDTH_HEIGHT, figsize=(14, 14))
    for i in range(PLOT_WIDTH_HEIGHT):
        for j in range(PLOT_WIDTH_HEIGHT):
            index = rnd.randint(0, sample_count - 1)
            axes[i, j].imshow(images[index], cmap="gray")
            axes[i, j].set_title(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"][labels[index]])
    
    plt.show()