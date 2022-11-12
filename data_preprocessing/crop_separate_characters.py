from PIL import Image
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import DIGIT_AND_OPERATORS_1_PATH
from const_config import DIGIT_AND_OPERATORS_2_PATH
from const_config import PREPROCESSED_PATH
from const_config import TRAINING_PREPROCESSED_PATH
from const_config import VALIDATION_PREPROCESSED_PATH
from const_config import TESTING_PREPROCESSED_PATH
from const_config import IMAGE_WIDTH
from const_config import IMAGE_HEIGHT

from const_config import ALL_MERGED_PREPROCESSED_PATH
from const_config import ALL_IMAGES_FILENAME
from const_config import ALL_LABELS_FILENAME

all_images = np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{ALL_IMAGES_FILENAME}", allow_pickle=True)
all_labels = np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{ALL_LABELS_FILENAME}", allow_pickle=True)


for label, target_file_name in enumerate(["zeros", "ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", 
                                          "pluses", "minuses", "asterisks", "slashes"]):
    class_indices = np.where(all_labels == label)[0]
    sample_count = class_indices.shape[0]
    target_file = np.empty((sample_count), dtype=object)
    
    for i, sample_index in enumerate(class_indices):
        image = all_images[sample_index]
        coordinates = np.argwhere(image > 0)
        target_file[i] = image[:, coordinates.min(axis=0)[1]:coordinates.max(axis=0)[1] + 1]
    
    validation_idx = int(sample_count * 0.8)
    testing_idx = int(sample_count * 0.9)
    np.save(f"{TRAINING_PREPROCESSED_PATH}{target_file_name}.npy", target_file[:validation_idx])
    np.save(f"{VALIDATION_PREPROCESSED_PATH}{target_file_name}.npy", target_file[validation_idx:testing_idx])
    np.save(f"{TESTING_PREPROCESSED_PATH}{target_file_name}.npy", target_file[testing_idx:])
