import numpy as np
import random as rnd
import sys

CHARACTERS_PATH = "./data/separated_characters/"
EQUATIONS_PATH = "./data/equations/"
TRAINING_IMAGES_FILENAME = "equations_%s_training_images_%s.npy"
TRAINING_LABELS_FILENAME = "equations_%s_training_labels_%s.npy"

HELP_MSG = "Run as: python equation_generator.py ['image type'] ['batch size'] ['batches per file'] ['number of files']"
BATCH_SIZE = 8
BATCHES_PER_FILE = 100
FILES = 4
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
        batches_of_images = np.zeros((batches_per_file), dtype=object)
        batches_of_labels = np.zeros((batches_per_file), dtype=object)

        for j in range(batches_per_file):
            image_batch = np.zeros((batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
            label_batch = np.zeros((batch_size, 3), dtype=np.uint8)

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
                image_batch[k, 0, y1 : y1 + CHARACTER_IMAGE_HEIGHT, x1 : x1 + CHARACTER_IMAGE_WIDTH] = digit_1
                image_batch[k, 0, y2 : y2 + CHARACTER_IMAGE_HEIGHT, x2 : x2 + CHARACTER_IMAGE_WIDTH] = operator_1
                image_batch[k, 0, y3 : y3 + CHARACTER_IMAGE_HEIGHT, x3 : x3 + CHARACTER_IMAGE_WIDTH] = digit_2

                # labels for digits 0-9 and for operators 0-3
                label_batch[k, 0] = label_1
                label_batch[k, 1] = label_2
                label_batch[k, 2] = label_3

            batches_of_images[j] = image_batch
            batches_of_labels[j] = label_batch
    
        # save file of chosen number of batches
        np.save(f"{EQUATIONS_PATH}{TRAINING_IMAGES_FILENAME % str(i)}", batches_of_images)
        np.save(f"{EQUATIONS_PATH}{TRAINING_LABELS_FILENAME % str(i)}", batches_of_labels)

def rnd_230x38(digits: DigitGenerator, operators: OperatorGenerator, batch_size, batches_per_file, files):
    IMAGE_WIDTH = 230
    IMAGE_HEIGHT = 38
    MIN_CHARACTERS = 3
    MAX_CHARACTERS = 8
    LABELS_PER_IMAGE = 25

    for i in range(files):
        batches_of_images = np.zeros((batches_per_file), dtype=object)
        batches_of_labels = np.zeros((batches_per_file), dtype=object)

        for j in range(batches_per_file):
            image_batch = np.zeros((batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
            label_batch = np.zeros((batch_size * LABELS_PER_IMAGE, 2), dtype=np.uint8)

            for k in range(batch_size):
                number_of_characters = rnd.randint(MIN_CHARACTERS, MAX_CHARACTERS)
                character_middle_idxs = np.zeros(number_of_characters)
                labels = np.zeros(number_of_characters, dtype=np.uint8)
                digit_probability = 1.0 # first character must be a digit
                current_image_idx = 0
                for l in range(number_of_characters):
                    if l == number_of_characters - 1: # last character must be a digit
                        digit_probability = 1.0

                    character = None
                    label = None
                    if rnd.random() <= digit_probability: # next character is a digit
                        character, label = digits.get()
                        digit_probability /= 2
                    else: # next character is an operator
                        character, label = operators.get()
                        digit_probability = 1.0 # next character must be a digit
                    
                    character_height = character.shape[0]
                    character_width = character.shape[1]
                    y_idx = rnd.randint(0, IMAGE_HEIGHT - character_height) # randomly verticaly place the character
                    image_batch[k, 0, y_idx : y_idx + character_height, current_image_idx : current_image_idx + character_width] = character
                    character_middle_idxs[l] = current_image_idx + character_width // 2 # store the middle index of the character
                    current_image_idx += character_width
                    labels[l] = label # store the label for the character
            
                x_shift = rnd.randint(0, IMAGE_WIDTH - current_image_idx)
                image_batch[k] = np.roll(image_batch[k], shift=x_shift, axis=2) # shifting the image across x axis
                character_middle_idxs = (character_middle_idxs + x_shift) % IMAGE_WIDTH

                width_per_label_box = IMAGE_WIDTH / LABELS_PER_IMAGE
                current_label_box = 0.0
                character_idx = 0
                for l in range(LABELS_PER_IMAGE):
                    label_idx = k * LABELS_PER_IMAGE + l
                    if (character_idx < number_of_characters and character_middle_idxs[character_idx] >= current_label_box and 
                        character_middle_idxs[character_idx] <= current_label_box + width_per_label_box): # center pf a character is in a label box
                        label_batch[label_idx, 0] = 1
                        label_batch[label_idx, 1] = labels[character_idx]
                        character_idx += 1
                    else:
                        label_batch[label_idx, 0] = 0
                        label_batch[label_idx, 1] = 0

                    current_label_box += width_per_label_box # next label box

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
    elif argument == "RND_230x38":
        rnd_230x38(digits, operators, BATCH_SIZE, BATCHES_PER_FILE, FILES)
    else:
        print("Unknown image type.", file=sys.stderr)
        print(HELP_MSG, file=sys.stderr)
        exit(1)
