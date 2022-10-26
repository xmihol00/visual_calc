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

labels = np.zeros((10, 2, 3))
samples = np.empty((100, 2))
for i in range(samples.shape[0]):
    digit_type_1 = rnd.randint(0, 9)
    digit_idx_1 = rnd.randint(0, digits[digit_type_1].shape[0] - 1)
    digit_1 = digits[digit_type_1][digit_idx_1][0]

    operator_type_1 = rnd.randint(0, 3)
    operator_idx_1 = rnd.randint(0, operators[operator_type_1].shape[0] - 1)
    operator_1 = operators[operator_type_1][operator_idx_1][0]

    digit_type_2 = rnd.randint(0, 9)
    digit_idx_2 = rnd.randint(0, digits[digit_type_2].shape[0] - 1)
    digit_2 = digits[digit_type_2][digit_idx_2][0]

    equation = np.zeros((48, 158))
    y1, y2, y3 = np.random.randint(0, equation.shape[1] - 29, 3)
    char_x_area = int((equation.shape[2] - 30) / 3)
    x1 = rnd.randint(15, char_x_area - 10)
    x2 = rnd.randint(char_x_area + 10, 2 * char_x_area - 10)
    x3 = rnd.randint(2 * char_x_area + 10, equation.shape[2] - 14)
    x_shift = rnd.randint(-15, 15)
    x_shift_image = x_shift + 28
    
    equation[y1:y1 + 28, x1 + x_shift:x1 + x_shift_image] = digit_1
    equation[y2:y2 + 28, x2 + x_shift:x2 + x_shift_image] = operator_1
    equation[y3:y3 + 28, x3 + x_shift:x3 + x_shift_image] = digit_2
    samples[i, 0] = equation

    labels = np.zeros((2, 3))
    labels[0, :] = digit_type_1, operator_type_1, digit_type_2
    labels[1, :] = (x1 + 14, x2 + 14, x3 + 14) + x_shift
    samples[i, 1] = labels

    _, ax = plt.subplots(1,1)
    ax.imshow(samples[i][0], cmap='gray')
    ax.set_title(f"{digit_type_1} {operator_type_1} {digit_type_2} || {labels[i, 1, 0] } {labels[i, 1, 1] } {labels[i, 1, 0] }")
    plt.show()
