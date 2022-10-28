import numpy as np
import matplotlib.pyplot as plt

EQUATIONS_PATH = "../data/equations/"
PLOT_COUNT = 10
SUBPLOT_X_COUNT = 4
SUBPLOT_Y_COUNT = 3

ONE_HOT_CHAR_CNT = 14  # number of true/false values to encode a character
DIGIT_CNT = 10
POS_1_IDX = 3
POS_2_IDX = POS_1_IDX + 1
POS_3_IDX = POS_1_IDX + 2

images = np.load(f"{EQUATIONS_PATH}equations_training_images.npy", allow_pickle=True)
labels = np.load(f"{EQUATIONS_PATH}equations_training_labels.npy", allow_pickle=True)

operator_dict = { 10.0: "+", 11.0: "-", 12.0: "*", 13.0: "/"}

for i in range(0, PLOT_COUNT*SUBPLOT_X_COUNT*SUBPLOT_Y_COUNT, SUBPLOT_X_COUNT*SUBPLOT_Y_COUNT):
    _, ax = plt.subplots(SUBPLOT_X_COUNT, SUBPLOT_Y_COUNT, figsize=(13, 12))
    for j in range(SUBPLOT_X_COUNT):
        for k in range(SUBPLOT_Y_COUNT):
            idx = i+j*SUBPLOT_Y_COUNT+k

            # get digit and operator labels
            digit_1 = int(labels[idx, 0])
            operator = operator_dict[labels[idx, 1]]
            digit_2 = int(labels[idx, 2])

            # denormalize positions
            labels[idx, 3:] *= 90
            labels[idx, 3:] += 0.1

            # plot
            ax[j, k].imshow(images[idx], cmap='gray')
            ax[j, k].set_title(f"{digit_1} {operator} {digit_2} -- x1: {int(labels[idx, POS_1_IDX])}, x2: {int(labels[idx, POS_2_IDX])}, x3: {int(labels[idx, POS_3_IDX])}")
    
    plt.show()
