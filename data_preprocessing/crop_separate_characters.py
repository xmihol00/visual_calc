import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import TRAINING_PREPROCESSED_PATH
from const_config import VALIDATION_PREPROCESSED_PATH
from const_config import TESTING_PREPROCESSED_PATH

from const_config import CLEANED_PREPROCESSED_PATH
from const_config import IMAGES_FILENAME
from const_config import LABELS_FILENAME
from const_config import SEED

np.random.seed(SEED)

os.makedirs(TRAINING_PREPROCESSED_PATH, exist_ok=True)
os.makedirs(VALIDATION_PREPROCESSED_PATH, exist_ok=True)
os.makedirs(TESTING_PREPROCESSED_PATH, exist_ok=True)

all_images = np.load(f"{CLEANED_PREPROCESSED_PATH}{IMAGES_FILENAME}", allow_pickle=True)
all_labels = np.load(f"{CLEANED_PREPROCESSED_PATH}{LABELS_FILENAME}", allow_pickle=True)

for label, target_file_name in enumerate(["zeros", "ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", 
                                          "pluses", "minuses", "asterisks", "slashes"]):
    class_indices = np.where(all_labels == label)[0]
    sample_count = class_indices.shape[0]
    target_file = np.empty((sample_count), dtype=object)
    
    for i, sample_index in enumerate(class_indices):
        image = all_images[sample_index]
        coordinates = np.argwhere(image > 0)
        # crop the width just to the symbol and keep the same height
        target_file[i] = image[:, coordinates.min(axis=0)[1]:coordinates.max(axis=0)[1] + 1]
    
    validation_idx = int(sample_count * 0.8)
    testing_idx = int(sample_count * 0.9)
    np.save(f"{TRAINING_PREPROCESSED_PATH}{target_file_name}.npy", target_file[:validation_idx]) # 80 %
    np.save(f"{VALIDATION_PREPROCESSED_PATH}{target_file_name}.npy", target_file[validation_idx:testing_idx]) # 10 %
    np.save(f"{TESTING_PREPROCESSED_PATH}{target_file_name}.npy", target_file[testing_idx:]) # 10 %