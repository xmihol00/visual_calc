import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import TRAINING_PREPROCESSED_PATH
from const_config import VALIDATION_PREPROCESSED_PATH
from const_config import TESTING_PREPROCESSED_PATH
PLOT_WIDTH_HEIGHT = 4

for directory in [ TRAINING_PREPROCESSED_PATH, VALIDATION_PREPROCESSED_PATH, TESTING_PREPROCESSED_PATH]:
    for file_name in os.listdir(directory):
        samples = np.load(f"{directory}{file_name}", allow_pickle=True)
        sample_count = samples.shape[0]
        figure, axis = plt.subplots(PLOT_WIDTH_HEIGHT, PLOT_WIDTH_HEIGHT)
        figure.suptitle(f"{directory}{file_name}")
        for i in range(PLOT_WIDTH_HEIGHT):
            for j in range(PLOT_WIDTH_HEIGHT):
                axis[i, j].imshow(samples[rnd.randint(0, sample_count - 1)], cmap="gray")
                axis[i, j].set_xticks([], [])
                axis[i, j].set_yticks([], [])
                axis[i, j].set_frame_on(False)

        plt.get_current_fig_manager().full_screen_toggle()
        plt.show(block=False)
        plt.pause(1.5)
        plt.close()
