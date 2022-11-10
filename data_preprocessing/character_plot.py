import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import TRAINING_PREPROCESSED_PATH
PLOT_WIDTH_HEIGHT = 4

for file_name in os.listdir(f"{TRAINING_PREPROCESSED_PATH}"):
    samples = np.load(f"{TRAINING_PREPROCESSED_PATH}{file_name}", allow_pickle=True)
    sample_count = samples.shape[0]
    _, axes = plt.subplots(PLOT_WIDTH_HEIGHT, PLOT_WIDTH_HEIGHT, figsize=(14, 14))
    for i in range(PLOT_WIDTH_HEIGHT):
        for j in range(PLOT_WIDTH_HEIGHT):
            axes[i, j].imshow(samples[rnd.randint(0, sample_count - 1)], cmap="gray")
    
    plt.show()
