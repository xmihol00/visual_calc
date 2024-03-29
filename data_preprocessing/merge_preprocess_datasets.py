from PIL import Image
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import DIGIT_AND_OPERATORS_1_TRAIN
from const_config import DIGIT_AND_OPERATORS_1_VALIDATION
from const_config import DIGIT_AND_OPERATORS_1_TEST
from const_config import DIGIT_AND_OPERATORS_2_PATH
from const_config import ALL_MERGED_PREPROCESSED_PATH
from const_config import IMAGE_WIDTH
from const_config import IMAGE_HEIGHT
from const_config import IMAGES_FILENAME
from const_config import LABELS_FILENAME
from const_config import SEED

np.random.seed(SEED)

MAX_IMAGE_PIXEL_SUM = IMAGE_WIDTH * IMAGE_HEIGHT * 255

os.makedirs(ALL_MERGED_PREPROCESSED_PATH, exist_ok=True)

label_dict = { "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, 
               "+": 10, "-": 11, "*": 12, "%": 13 }

data_set_1 = np.concatenate((np.load(DIGIT_AND_OPERATORS_1_TRAIN, allow_pickle=True),
                             np.load(DIGIT_AND_OPERATORS_1_VALIDATION, allow_pickle=True),
                             np.load(DIGIT_AND_OPERATORS_1_TEST, allow_pickle=True)))

# select only needed symbols from the data set
data_set_1 = data_set_1[((data_set_1[:, 1] >= "0") & (data_set_1[:, 1] <= "9")) | (data_set_1[:, 1] == "+") |
                         (data_set_1[:, 1] == "-") | (data_set_1[:, 1] == "*") | (data_set_1[:, 1] == "%")]

sample_count = data_set_1.shape[0]
for directory in os.listdir(f"{DIGIT_AND_OPERATORS_2_PATH}"): # count all of the samples in merged data set 2 and 3
    sample_count += len(os.listdir(f"{DIGIT_AND_OPERATORS_2_PATH}{directory}"))

images = np.zeros((sample_count, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
labels = np.zeros(sample_count, dtype=np.uint8)

indices = np.random.choice(sample_count, sample_count, replace=False) # randomly place images and labels in the final file
index = 0

for image_label in data_set_1:
    images[indices[index], :, :] = image_label[0]
    labels[indices[index]] = label_dict[image_label[1]]
    index += 1

for directory, label in [("0/", 0), ("1/", 1), ("2/", 2), ("3/", 3), ("4/", 4), ("5/", 5), ("6/", 6),
                         ("7/", 7), ("8/", 8), ("9/", 9),
                         ("+/", 10), ("-/", 11), ("x/", 12), (",/", 13)]:
    for file_name in os.listdir(f"{DIGIT_AND_OPERATORS_2_PATH}{directory}"):
        # open, resize and conver to numpy array
        image = np.array(Image.open(f"{DIGIT_AND_OPERATORS_2_PATH}{directory}{file_name}").resize((IMAGE_HEIGHT, IMAGE_WIDTH)))

        if image.sum() * 2 < MAX_IMAGE_PIXEL_SUM: # determins wheater the character is on black or white background
            image = np.array(image > 64, dtype=np.float32) # treshold
        else:
            image = np.array(image < 192, dtype=np.float32) # treshold
        
        images[indices[index], :, :] = image
        labels[indices[index]] = label
        index += 1
    
np.save(f"{ALL_MERGED_PREPROCESSED_PATH}{IMAGES_FILENAME}", images)
np.save(f"{ALL_MERGED_PREPROCESSED_PATH}{LABELS_FILENAME}", labels)

# print stats of the generated data set
print(f"Number of samples: {sample_count}")
print(f"Images file size: {os.stat(f'{ALL_MERGED_PREPROCESSED_PATH}{IMAGES_FILENAME}').st_size / (1024 * 1024)} MB")
print(f"Labels file size: {os.stat(f'{ALL_MERGED_PREPROCESSED_PATH}{LABELS_FILENAME}').st_size / (1024 * 1024)} MB")
