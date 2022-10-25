from ast import operator
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

DATA_PATH = "../data/separated_characters/"

digits = []
for file_name in ["zeros.npy", "ones.npy", "twos.npy", "threes.npy", "fours.npy", 
                  "fives.npy", "sixes.npy", "sevens.npy", "eights.npy", "nines.npy"]:
    digits.append(np.load(f"{DATA_PATH}{file_name}", allow_pickle=True))

operators = []
for file_name in ["pluses.npy", "minuses.npy", "astrics.npy", "slashes.npy"]:
    operators.append(np.load(f"{DATA_PATH}{file_name}", allow_pickle=True))

equations = np.zeros((10, 48, 128))
labels = np.zeros((10, 2, 3))
for i in range(equations.shape[0]):
    digit_type_1 = rnd.randint(0, 9)
    digit_idx_1 = rnd.randint(0, digits[digit_type_1].shape[0] - 1)
    digit_1 = digits[digit_type_1][digit_idx_1][0]

    operator_type_1 = rnd.randint(0, 3)
    operator_idx_1 = rnd.randint(0, operators[operator_type_1].shape[0] - 1)
    operator_1 = operators[operator_type_1][operator_idx_1][0]

    digit_type_2 = rnd.randint(0, 9)
    digit_idx_2 = rnd.randint(0, digits[digit_type_2].shape[0] - 1)
    digit_2 = digits[digit_type_2][digit_idx_2][0]

    y1, y2, y3 = np.random.randint(0, equations.shape[1] - 29, 3)
    char_x_area = int(equations.shape[2]/3)
    x1 = rnd.randint(0, char_x_area - 26)
    x2 = rnd.randint(char_x_area - 3, 2 * char_x_area - 26)
    x3 = rnd.randint(2 * char_x_area - 3, equations.shape[2] - 29)
    equations[i, y1:y1+28, x1:x1+28] = digit_1
    equations[i, y2:y2+28, x2:x2+28] = operator_1
    equations[i, y3:y3+28, x3:x3+28] = digit_2
    labels[i, 0, :] = digit_type_1, operator_type_1, digit_type_2
    labels[i, 1, :] = x1 + 14, x2 + 14, x3 + 14

    _, ax = plt.subplots(1,1)
    ax.imshow(equations[i], cmap='gray')
    ax.set_title(f"{digit_type_1} {operator_type_1} {digit_type_2} || {x1 + 14} {x2 + 14} {x3 + 14}")
    plt.show()
