import numpy as np
import random as rnd

CHARACTERS_PATH = "../data/separated_characters/"
EQUATIONS_PATH = "../data/equations/"
TRAINING_IMAGES_FILENAME = "equations_training_images.npy"
TRAINING_LABELS_FILENAME = "equations_training_labels.npy"

NUMBER_OF_SAMPLES = 50000                                    # number of generated equations
CHAR_IMAGE_WIDTH = 28                                       # width of a character image in pixels
CHAR_IMAGE_HEIGHT = 28                                      # height of a character image in pixels
CHAR_IMAGE_WIDTH_HALF = int(CHAR_IMAGE_WIDTH / 2)           # half of the width of a character image in pixels
X_SHIFT = 0                                                 # number of pixels by which the constructed image can be shifted in the final image
X_SHIFT_HALF = int(X_SHIFT / 2)                             # half of the whole image shift in pixels
X_SHIFT_HALF_OVERLAP = max(X_SHIFT_HALF - 5, 0)             # decreased half of the shift, so images can overlap a bit
FINAL_IMAGE_WIDTH = 30                                      # width of the final image without shift in pixels
FINAL_IMAGE_WIDTH_WITH_SHIFT = FINAL_IMAGE_WIDTH + X_SHIFT  # width of the final image with shift in pixels
FINAL_IMAGE_HEIGHT = 30                                     # height of the final image in pixels
CHARACTER_IN_IMAGE_WIDTH = int((FINAL_IMAGE_WIDTH) / 3)     # area for one character across the x axes, character is randomly placed in this area
ONE_HOT_CHAR_CNT = 14                                       # number of true/false values to encode a character
POSITION_CNT = 3                                            # one position for each character in the final image

digits = []
for file_name in ["zeros.npy", "ones.npy", "twos.npy", "threes.npy", "fours.npy", 
                  "fives.npy", "sixes.npy", "sevens.npy", "eights.npy", "nines.npy"]:
    digits.append(np.load(f"{CHARACTERS_PATH}{file_name}", allow_pickle=True))

operators = []
for file_name in ["pluses.npy", "minuses.npy", "astrics.npy", "slashes.npy"]:
    operators.append(np.load(f"{CHARACTERS_PATH}{file_name}", allow_pickle=True))

DIGIT_CNT = len(digits)
OPERATOR_CNT = len(operators)

images = np.zeros((NUMBER_OF_SAMPLES, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH_WITH_SHIFT), dtype=np.float32)
labels = np.zeros((NUMBER_OF_SAMPLES), dtype=np.longlong)
for i in range(NUMBER_OF_SAMPLES):
    # random selection of the 1st digit
    digit_type_1 = rnd.randint(0, DIGIT_CNT - 1)
    digit_idx_1 = rnd.randint(0, digits[digit_type_1].shape[0] - 1)
    digit_1 = digits[digit_type_1][digit_idx_1][0]

    # random selection of the operator
    operator_type_1 = rnd.randint(0, OPERATOR_CNT - 1)
    operator_idx_1 = rnd.randint(0, operators[operator_type_1].shape[0] - 1)
    operator_1 = operators[operator_type_1][operator_idx_1][0]

    # random selection of the 2nd digit
    digit_type_2 = rnd.randint(0, 9)
    digit_idx_2 = rnd.randint(0, digits[digit_type_2].shape[0] - 1)
    digit_2 = digits[digit_type_2][digit_idx_2][0]

    # random position of the characters across the y axes
    y1, y2, y3 = np.random.randint(0, FINAL_IMAGE_HEIGHT - CHAR_IMAGE_WIDTH, 3)
    
    # random positions of the characters across the x axes
    x1 = 0 #rnd.randint(X_SHIFT_HALF, CHARACTER_IN_IMAGE_WIDTH - X_SHIFT_HALF_OVERLAP)
    x2 = 29 #rnd.randint(CHARACTER_IN_IMAGE_WIDTH + X_SHIFT_HALF_OVERLAP, 2 * CHARACTER_IN_IMAGE_WIDTH - X_SHIFT_HALF_OVERLAP)
    x3 = 59 #rnd.randint(2 * CHARACTER_IN_IMAGE_WIDTH + X_SHIFT_HALF_OVERLAP, FINAL_IMAGE_WIDTH + X_SHIFT_HALF - CHAR_IMAGE_WIDTH - 1)

    # shift of all the characters in the final image in the width direction
    x_shift = 0 #rnd.randint(-X_SHIFT_HALF, X_SHIFT_HALF)
    x_shift_image = x_shift + CHAR_IMAGE_WIDTH
    
    # composition of the final image from 2 randomly chosen digits and a randomly chosen character
    images[i, y1:y1 + CHAR_IMAGE_HEIGHT, x1 + x_shift:x1 + x_shift_image] += digit_1
    #images[i, y2:y2 + CHAR_IMAGE_HEIGHT, x2 + x_shift:x2 + x_shift_image] += operator_1
    #images[i, y3:y3 + CHAR_IMAGE_HEIGHT, x3 + x_shift:x3 + x_shift_image] += digit_2

    # one-hot encoding of the classes
    labels[i] = digit_type_1
    #labels[i, DIGIT_CNT + operator_type_1] = 1
    #labels[i, DIGIT_CNT + OPERATOR_CNT + digit_type_2] = 1

    # possition assignment and normalization to values between 0 and 2
    # labels[i, 3:] = x1 + CHAR_IMAGE_WIDTH_HALF, x2 + CHAR_IMAGE_WIDTH_HALF, x3 + CHAR_IMAGE_WIDTH_HALF
    # labels[i, 3:] += x_shift
    # labels[i, 3:] /= FINAL_IMAGE_WIDTH + X_SHIFT

np.save(f"{EQUATIONS_PATH}{TRAINING_IMAGES_FILENAME}", images)
np.save(f"{EQUATIONS_PATH}{TRAINING_LABELS_FILENAME}", labels)
