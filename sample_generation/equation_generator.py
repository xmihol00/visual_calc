import numpy as np
import random as rnd
import sys

CHARACTERS_PATH = "../data/separated_characters/"
EQUATIONS_PATH = "../data/equations/"
TRAINING_IMAGES_FILENAME = "equations_%s_training_images_%s.npy"
TRAINING_LABELS_FILENAME = "equations_%s_training_labels_%s.npy"

HELP_MSG = "Run as: python equation_generator.py ['image type'] ['batch size'] ['batches per file'] ['number of files']"
BATCH_SIZE = 4
BATCHES_PER_FILE = 100
FILES = 3
NUMBER_OF_DIGITS = 10
NUMBER_OF_OPERATORS = 4

NUMBER_OF_SAMPLES = 50000                                   # number of generated equations
CHARACTER_IMAGE_WIDTH = 28                                  # width of a character image in pixels
CHARACTER_IMAGE_HEIGHT = 28                                 # height of a character image in pixels
CHAR_IMAGE_WIDTH_HALF = int(CHARACTER_IMAGE_WIDTH / 2)      # half of the width of a character image in pixels
X_SHIFT = 30                                                # number of pixels by which the constructed image can be shifted in the final image
X_SHIFT_HALF = int(X_SHIFT / 2)                             # half of the whole image shift in pixels
X_SHIFT_HALF_OVERLAP = max(X_SHIFT_HALF - 5, 0)             # decreased half of the shift, so images can overlap a bit
FINAL_IMAGE_WIDTH = 102                                     # width of the final image without shift in pixels
FINAL_IMAGE_WIDTH_WITH_SHIFT = FINAL_IMAGE_WIDTH + X_SHIFT  # width of the final image with shift in pixels
FINAL_IMAGE_HEIGHT = 40                                     # height of the final image in pixels
CHARACTER_IN_IMAGE_WIDTH = int((FINAL_IMAGE_WIDTH) / 3)     # area for one character across the x axes, character is randomly placed in this area

def get_digit():
    pass

class DigitGenerator():
    def __init__(self):
        self.digits = []
        for file_name in ["zeros.npy", "ones.npy", "twos.npy", "threes.npy", "fours.npy", 
                          "fives.npy", "sixes.npy", "sevens.npy", "eights.npy", "nines.npy"]:
            self.digits.append(np.load(f"{CHARACTERS_PATH}{file_name}", allow_pickle=True))
    
    def get(self):
        digit_type = rnd.randint(0, NUMBER_OF_DIGITS - 1)
        digit_idx = rnd.randint(0, self.digits[digit_type].shape[0] - 1)
        digit = self.digits[digit_type][digit_idx][0]
        return digit, digit_type

class OperatorGenerator():
    def __init__(self):
        self.operators = []
        for file_name in ["pluses.npy", "minuses.npy", "astrics.npy", "slashes.npy"]:
            self.operators.append(np.load(f"{CHARACTERS_PATH}{file_name}", allow_pickle=True))
    
    def get(self):
        operator_type = rnd.randint(0, NUMBER_OF_OPERATORS - 1)
        operator_idx = rnd.randint(0, self.operators[operator_type].shape[0] - 1)
        operator = self.operators[operator_type][operator_idx][0]
        return operator, operator_type + NUMBER_OF_DIGITS

def dod_90x30(digits: DigitGenerator, operators: OperatorGenerator, batch_size, batches_per_file, files):
    IMAGE_WIDTH = 90
    IMAGE_HEIGHT = 30
    WIDTH_FOR_CHARCTER = IMAGE_WIDTH // 3

    for i in range(files):
        batches_of_images = np.empty((batches_per_file), dtype=object)
        batches_of_labels = np.empty((batches_per_file), dtype=object)

        for j in range(batches_per_file):
            image_batch = np.empty((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
            label_batch = np.empty((batch_size, 3), dtype=np.uint8)

            for k in range(batch_size):
                # randomly select 2 digits and 1 operator
                digit_1, label_1 = digits.get()
                operator_1, label_2 = operators.get()
                digit_2, label_3 = digits.get()

                # random position of the characters across the x axes
                x1, x2, x3 = np.random.randint(0, WIDTH_FOR_CHARCTER - CHARACTER_IMAGE_WIDTH, 3)
                x2 += CHARACTER_IMAGE_WIDTH
                x3 += 2 * CHARACTER_IMAGE_WIDTH

                # random position of the characters across the y axes
                y1, y2, y3 = np.random.randint(0, IMAGE_HEIGHT - CHARACTER_IMAGE_HEIGHT, 3)

                # composition of the final image from 2 randomly chosen digits and 1 randomly chosen character
                image_batch[k, y1:y1 + CHARACTER_IMAGE_HEIGHT, x1:x1 + CHARACTER_IMAGE_WIDTH] = digit_1
                image_batch[k, y2:y2 + CHARACTER_IMAGE_HEIGHT, x2:x2 + CHARACTER_IMAGE_WIDTH] = operator_1
                image_batch[k, y3:y3 + CHARACTER_IMAGE_HEIGHT, x3:x3 + CHARACTER_IMAGE_WIDTH] = digit_2

                # labels for digits 0-9 and for operators 0-3
                label_batch[k, 0] = label_1
                label_batch[k, 1] = label_2
                label_batch[k, 2] = label_3

            batches_of_images[j] = image_batch
            batches_of_labels[j] = label_batch
    
        # save file of chosen number of batches
        np.save(f"{EQUATIONS_PATH}{TRAINING_IMAGES_FILENAME % str(i)}", batches_of_images)
        np.save(f"{EQUATIONS_PATH}{TRAINING_LABELS_FILENAME % str(i)}", batches_of_labels)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Not enough arguments.", file=sys.stderr)
        print(HELP_MSG, file=sys.stderr)
        exit(1)
    
    if len(sys.argv) >= 5:
        BATCH_SIZE = int(sys.argv[2])
        BATCHES_PER_FILE = int(sys.argv[3])
        FILES = int(sys.argv[4])

    digits = DigitGenerator()
    operators = OperatorGenerator()

    argument = sys.argv[1]
    TRAINING_IMAGES_FILENAME = TRAINING_IMAGES_FILENAME % (argument, "%s")
    TRAINING_LABELS_FILENAME = TRAINING_LABELS_FILENAME % (argument, "%s")

    if argument == "DOD_90x30":
        dod_90x30(digits, operators, BATCH_SIZE, BATCHES_PER_FILE, FILES)
    elif argument == "DOD_132x40":
        pass
    elif argument == "RND_222x38":
        pass
    else:
        print("Unknown image type.", file=sys.stderr)
        print(HELP_MSG, file=sys.stderr)
        exit(1)

    exit(0)

digits = []
for file_name in ["zeros.npy", "ones.npy", "twos.npy", "threes.npy", "fours.npy", 
                  "fives.npy", "sixes.npy", "sevens.npy", "eights.npy", "nines.npy"]:
    digits.append(np.load(f"{CHARACTERS_PATH}{file_name}", allow_pickle=True))

operators = []
for file_name in ["pluses.npy", "minuses.npy", "astrics.npy", "slashes.npy"]:
    operators.append(np.load(f"{CHARACTERS_PATH}{file_name}", allow_pickle=True))

NUMBER_OF_DIGITS = len(digits)
NUMBER_OF_OPERATORS = len(operators)

images = np.zeros((NUMBER_OF_SAMPLES, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH_WITH_SHIFT), dtype=np.float32)
labels = np.zeros((NUMBER_OF_SAMPLES, 3), dtype=np.uint8)
for i in range(NUMBER_OF_SAMPLES):
    # random selection of the 1st digit
    digit_type_1 = rnd.randint(0, NUMBER_OF_DIGITS - 1)
    digit_idx_1 = rnd.randint(0, digits[digit_type_1].shape[0] - 1)
    digit_1 = digits[digit_type_1][digit_idx_1][0]

    # random selection of the operator
    operator_type_1 = rnd.randint(0, NUMBER_OF_OPERATORS - 1)
    operator_idx_1 = rnd.randint(0, operators[operator_type_1].shape[0] - 1)
    operator_1 = operators[operator_type_1][operator_idx_1][0]

    # random selection of the 2nd digit
    digit_type_2 = rnd.randint(0, 9)
    digit_idx_2 = rnd.randint(0, digits[digit_type_2].shape[0] - 1)
    digit_2 = digits[digit_type_2][digit_idx_2][0]

    # random position of the characters across the y axes
    y1, y2, y3 = np.random.randint(0, FINAL_IMAGE_HEIGHT - CHARACTER_IMAGE_WIDTH, 3)
    
    # random positions of the characters across the x axes
    x1 = rnd.randint(X_SHIFT_HALF, CHARACTER_IN_IMAGE_WIDTH - X_SHIFT_HALF_OVERLAP)
    x2 = rnd.randint(CHARACTER_IN_IMAGE_WIDTH + X_SHIFT_HALF_OVERLAP, 2 * CHARACTER_IN_IMAGE_WIDTH - X_SHIFT_HALF_OVERLAP)
    x3 = rnd.randint(2 * CHARACTER_IN_IMAGE_WIDTH + X_SHIFT_HALF_OVERLAP, FINAL_IMAGE_WIDTH + X_SHIFT_HALF - CHARACTER_IMAGE_WIDTH - 1)

    # shift of all the characters in the final image in the width direction
    x_shift = rnd.randint(-X_SHIFT_HALF, X_SHIFT_HALF)
    x_shift_image = x_shift + CHARACTER_IMAGE_WIDTH
    
    # composition of the final image from 2 randomly chosen digits and a randomly chosen character
    images[i, y1:y1 + CHARACTER_IMAGE_HEIGHT, x1 + x_shift:x1 + x_shift_image] += digit_1
    images[i, y2:y2 + CHARACTER_IMAGE_HEIGHT, x2 + x_shift:x2 + x_shift_image] += operator_1
    images[i, y3:y3 + CHARACTER_IMAGE_HEIGHT, x3 + x_shift:x3 + x_shift_image] += digit_2

    # labels for digits 0-9 and for operators 0-3
    labels[i, 0] = digit_type_1
    labels[i, 1] = operator_type_1
    labels[i, 2] = digit_type_2

    # possition assignment and normalization to values between 0 and 2
    # labels[i, 3:] = x1 + CHAR_IMAGE_WIDTH_HALF, x2 + CHAR_IMAGE_WIDTH_HALF, x3 + CHAR_IMAGE_WIDTH_HALF
    # labels[i, 3:] += x_shift
    # labels[i, 3:] /= FINAL_IMAGE_WIDTH + X_SHIFT

np.save(f"{EQUATIONS_PATH}{TRAINING_IMAGES_FILENAME}", images)
np.save(f"{EQUATIONS_PATH}{TRAINING_LABELS_FILENAME}", labels)
