import os
import numpy as np
import random as rnd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import BATCH_SIZE
from const_config import BATCHES_PER_FILE
from const_config import NUMBER_OF_FILES
from const_config import NUMBER_OF_DIGITS 
from const_config import NUMBER_OF_OPERATORS
from const_config import CHARACTERS_PATH
from const_config import EQUATIONS_PATH
from const_config import YOLO_LABELS_PER_IMAGE
from const_config import TRAINING_IMAGES_FILENAME_TEMPLATE
from const_config import TRAINING_LABELS_FILENAME_TEMPLATE
from const_config import DATA_DIRECTORIES

HELP_MSG = "Run as: python equation_generator.py ['image type'] ['batch size'] ['batches per file'] ['number of files']"

class DigitGenerator():
    def __init__(self, directory = "training/"):
        self.digits = []
        for file_name in ["zeros.npy", "ones.npy", "twos.npy", "threes.npy", "fours.npy", 
                          "fives.npy", "sixes.npy", "sevens.npy", "eights.npy", "nines.npy"]:
            self.digits.append(np.load(f"{CHARACTERS_PATH}{directory}{file_name}", allow_pickle=True))
    
    def get(self):
        digit_type = rnd.randint(0, NUMBER_OF_DIGITS - 1) # randomly choose type of a digit
        digit_idx = rnd.randint(0, self.digits[digit_type].shape[0] - 1) # randomly choose a digit of the chosen type
        digit = self.digits[digit_type][digit_idx] # find the digit
        return digit, digit_type

class OperatorGenerator():
    def __init__(self, directory = "training/"):
        self.operators = []
        for file_name in ["pluses.npy", "minuses.npy", "asterisks.npy", "slashes.npy"]:
            self.operators.append(np.load(f"{CHARACTERS_PATH}{directory}{file_name}", allow_pickle=True))
    
    def get(self):
        operator_type = rnd.randint(0, NUMBER_OF_OPERATORS - 1) # randomly choose type of a digit
        operator_idx = rnd.randint(0, self.operators[operator_type].shape[0] - 1) # randomly choose a digit of the chosen type
        operator = self.operators[operator_type][operator_idx] # find the operator
        return operator, operator_type + NUMBER_OF_DIGITS      # change the label for operators from 0-3 to 10-13

def dod_90x30(digits: DigitGenerator, operators: OperatorGenerator, directory, batch_size, batches_per_file, files):
    CHARACTER_IMAGE_WIDTH = 28                    # width of a character image in pixels
    CHARACTER_IMAGE_HEIGHT = 28                   # height of a character image in pixels
    FINAL_IMAGE_WIDTH = 90                        # width of the generated image
    FINAL_IMAGE_HEIGHT = 30                       # height of the generated image
    WIDTH_FOR_CHARCTER = FINAL_IMAGE_WIDTH // 3   # space along the x axis for a single character

    for i in range(files):
        # allocate space for batches in a file
        batches_of_images = np.zeros((batches_per_file), dtype=object)
        batches_of_labels = np.zeros((batches_per_file), dtype=object)

        for j in range(batches_per_file):
            # allocate space for a batch
            image_batch = np.zeros((batch_size, 1, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH), dtype=np.float32)
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
                y1, y2, y3 = np.random.randint(0, FINAL_IMAGE_HEIGHT - CHARACTER_IMAGE_HEIGHT, 3)

                # composition of the final image from 2 randomly chosen digits and 1 randomly chosen character
                image_batch[k, 0, y1 : y1 + CHARACTER_IMAGE_HEIGHT, x1 : x1 + CHARACTER_IMAGE_WIDTH] = digit_1
                image_batch[k, 0, y2 : y2 + CHARACTER_IMAGE_HEIGHT, x2 : x2 + CHARACTER_IMAGE_WIDTH] = operator_1
                image_batch[k, 0, y3 : y3 + CHARACTER_IMAGE_HEIGHT, x3 : x3 + CHARACTER_IMAGE_WIDTH] = digit_2

                # labels for digits 0-9 and for operators 0-3
                label_batch[k, 0] = label_1
                label_batch[k, 1] = label_2
                label_batch[k, 2] = label_3

            # inser current batch in to the file
            batches_of_images[j] = image_batch
            batches_of_labels[j] = label_batch
    
        # save file of chosen number of batches
        np.save(f"{EQUATIONS_PATH}{directory}{TRAINING_IMAGES_FILENAME_TEMPLATE % str(i)}", batches_of_images)
        np.save(f"{EQUATIONS_PATH}{directory}{TRAINING_LABELS_FILENAME_TEMPLATE % str(i)}", batches_of_labels)

def yolo_230x38(digits: DigitGenerator, operators: OperatorGenerator, directory, batch_size, batches_per_file, files):
    FINAL_IMAGE_WIDTH = 230     # width of the generated image
    FINAL_IMAGE_HEIGHT = 38     # height of the generated image
    MIN_CHARACTERS = 3          # minimum characters in an image
    MAX_CHARACTERS = 8          # maximum characters in an image
    MIN_CHARACTER_WIDTH = FINAL_IMAGE_WIDTH // YOLO_LABELS_PER_IMAGE

    for i in range(files):
        # allocate space for batches in a file
        batches_of_images = np.zeros((batches_per_file), dtype=object)
        batches_of_labels = np.zeros((batches_per_file), dtype=object)

        for j in range(batches_per_file):
            # allocate space for a batch
            image_batch = np.zeros((batch_size, 1, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH), dtype=np.float32)
            # each label consist of a information, wheather there is ([:, 0]=1) or isn't ([:, 0]=0) a character and 
            # if there is, then it's label i.e. [2, 1] = 5 (second part of the image contains digit 5)
            label_batch = np.zeros((batch_size * YOLO_LABELS_PER_IMAGE, 2), dtype=np.uint8) 

            for k in range(batch_size):
                number_of_characters = rnd.randint(MIN_CHARACTERS, MAX_CHARACTERS) # randomly pick number of generated character for this image
                character_middle_idxs = np.zeros(number_of_characters) # allocate space to store indices where character were placed in to the image
                labels = np.zeros(number_of_characters, dtype=np.uint8) # allocate space for intermidiate labels, consisting just of class identification
                digit_probability = 1.0 # first character must be a digit
                current_image_idx = 0 # first character will be placed at the start of the image
                charactor_separator = 0
                for l in range(number_of_characters):
                    if l == number_of_characters - 1: # last character must be a digit
                        digit_probability = 1.0

                    character = None
                    label = None
                    if rnd.random() <= digit_probability: # next character is a digit
                        character, label = digits.get()
                        digit_probability /= 2 # decrease the probability of next character being a digit
                    else: # next character is an operator
                        character, label = operators.get()
                        digit_probability = 1.0 # next character must be a digit
                    
                    character_height = character.shape[0]
                    character_width = character.shape[1]
                    left_padding = 0
                    right_padding = 0
                    if character_width < MIN_CHARACTER_WIDTH: # character is not wide enough 
                        padding = MIN_CHARACTER_WIDTH - character_width
                        left_padding = rnd.randint(0, padding)  # padding before character
                        right_padding = padding - left_padding  # padding after character

                    current_image_idx += left_padding
                    y_idx = rnd.randint(0, FINAL_IMAGE_HEIGHT - character_height) # randomly verticaly place the character
                    image_batch[k, 0, y_idx : y_idx + character_height, current_image_idx : current_image_idx + character_width] = character # place the character just behind the previous one
                    character_middle_idxs[l] = current_image_idx + character_width // 2 # store the index of the middle of the character
                    current_image_idx += character_width + right_padding + rnd.randint(1, 2) # update the index, where next character will be place, add padding between characters
                    labels[l] = label # store the label for the character
            
                x_shift = rnd.randint(0, FINAL_IMAGE_WIDTH - current_image_idx)
                image_batch[k] = np.roll(image_batch[k], shift=x_shift, axis=2) # shifting the image to right across x axis
                character_middle_idxs = (character_middle_idxs + x_shift) % FINAL_IMAGE_WIDTH # the position of the midpoints of the characters must be shifted as well

                width_per_label_box = FINAL_IMAGE_WIDTH / YOLO_LABELS_PER_IMAGE # wdth of a part of an image, which is labeled
                current_label_box = 0.0
                character_idx = 0
                for l in range(YOLO_LABELS_PER_IMAGE):
                    label_idx = k * YOLO_LABELS_PER_IMAGE + l # get the index for labels for this image
                    if (character_idx < number_of_characters and character_middle_idxs[character_idx] >= current_label_box and 
                        character_middle_idxs[character_idx] <= current_label_box + width_per_label_box): # if the center of a character is in a label box
                        label_batch[label_idx, 0] = 1 # this part of an image contains a character
                        label_batch[label_idx, 1] = labels[character_idx] # class of the character
                        character_idx += 1
                    else:
                        label_batch[label_idx, 0] = 0 # this part of an image doesn't contain a character
                        label_batch[label_idx, 1] = NUMBER_OF_DIGITS + NUMBER_OF_OPERATORS # there is a special class also for none-character

                    current_label_box += width_per_label_box # next label box

            # insert batch to the file
            batches_of_images[j] = image_batch
            batches_of_labels[j] = label_batch
    
        # save file of chosen number of batches
        np.save(f"{EQUATIONS_PATH}{directory}{TRAINING_IMAGES_FILENAME_TEMPLATE % str(i)}", batches_of_images)
        np.save(f"{EQUATIONS_PATH}{directory}{TRAINING_LABELS_FILENAME_TEMPLATE % str(i)}", batches_of_labels)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Not enough arguments.", file=sys.stderr)
        print(HELP_MSG, file=sys.stderr)
        exit(1)

    type = sys.argv[1]
    TRAINING_IMAGES_FILENAME_TEMPLATE = TRAINING_IMAGES_FILENAME_TEMPLATE % (type, "%s")
    TRAINING_LABELS_FILENAME_TEMPLATE = TRAINING_LABELS_FILENAME_TEMPLATE % (type, "%s")

    for directory in DATA_DIRECTORIES:
        digits = DigitGenerator(directory)
        operators = OperatorGenerator(directory)

        if type == "90x30":
            dod_90x30(digits, operators, directory, BATCH_SIZE, BATCHES_PER_FILE, NUMBER_OF_FILES)
        elif type == "132x40":
            pass
        elif type == "230x38":
            yolo_230x38(digits, operators, directory, BATCH_SIZE, BATCHES_PER_FILE, NUMBER_OF_FILES)
        else:
            print("Unknown image type.", file=sys.stderr)
            print(HELP_MSG, file=sys.stderr)
            exit(1)
