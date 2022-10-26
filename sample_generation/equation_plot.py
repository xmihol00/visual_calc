import numpy as np
import matplotlib.pyplot as plt

EQUATIONS_PATH = "../data/equations/"
PLOT_COUNT = 10
SUBPLOT_X_COUNT = 4
SUBPLOT_Y_COUNT = 3

ONE_HOT_CHAR_CNT = 14  # number of true/false values to encode a character
DIGIT_CNT = 10
POS_1_IDX = 3 * ONE_HOT_CHAR_CNT
POS_2_IDX = POS_1_IDX + 1
POS_3_IDX = POS_1_IDX + 2

images = np.load(f"{EQUATIONS_PATH}equations_training_images.npy", allow_pickle=True)
labels = np.load(f"{EQUATIONS_PATH}equations_training_labels.npy", allow_pickle=True)

operator_dict = { 0: "+", 1: "-", 2: "*", 3: "/"}

for i in range(0, PLOT_COUNT*SUBPLOT_X_COUNT*SUBPLOT_Y_COUNT, SUBPLOT_X_COUNT*SUBPLOT_Y_COUNT):
    _, ax = plt.subplots(SUBPLOT_X_COUNT, SUBPLOT_Y_COUNT, figsize=(13, 12))
    for j in range(SUBPLOT_X_COUNT):
        for k in range(SUBPLOT_Y_COUNT):
            idx = i+j*SUBPLOT_Y_COUNT+k

            # get digits and operators from one-hot encoding
            digit_1 = np.argmax(labels[idx, :ONE_HOT_CHAR_CNT])
            operator = operator_dict[np.argmax(labels[idx, ONE_HOT_CHAR_CNT:2 * ONE_HOT_CHAR_CNT]) - DIGIT_CNT]
            digit_2 = np.argmax(labels[idx, 2 * ONE_HOT_CHAR_CNT:3 * ONE_HOT_CHAR_CNT])

            # denormalize positions
            labels[idx, POS_1_IDX:] *= 158
            labels[idx, POS_1_IDX:] += 0.1

            # plot
            ax[j, k].imshow(images[idx], cmap='gray')
            ax[j, k].set_title(f"{digit_1} {operator} {digit_2} -- x1: {int(labels[idx, POS_1_IDX])}, x2: {int(labels[idx, POS_2_IDX])}, x3: {int(labels[idx, POS_3_IDX])}")
    
    plt.show()
