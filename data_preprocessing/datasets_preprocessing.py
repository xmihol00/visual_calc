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


for directory, target_file_name in [("0/", "zeros"), ("1/", "ones"), ("2/", "twos"), ("3/", "threes"), ("4/", "fours"), ("5/", "fives"), ("6/", "sixes"),
                                    ("7/", "sevens"), ("8/", "eights"), ("9/", "nines"),
                                    ("plus/", "pluses"), ("minus/", "minuses"), ("asterisk/", "asterisks"), ("slash/", "slashes")]:
    file_names = os.listdir(f"{DIGIT_AND_OPERATORS_2_PATH}{directory}")
    sample_count = len(file_names)
    target_file = np.empty((sample_count), dtype=object)
    for i, file_name in enumerate(file_names):
        image = np.array(Image.open(f"{DIGIT_AND_OPERATORS_2_PATH}{directory}{file_name}").resize((IMAGE_WIDTH, IMAGE_HEIGHT)))

        if target_file_name == "asterisks":
            image = np.array(image > 64, dtype=np.float32)
        else:
            image = np.array(image < 192, dtype=np.float32)
        
        coordinates = np.argwhere(image > 0)
        target_file[i] = image[:, coordinates.min(axis=0)[1]:coordinates.max(axis=0)[1] + 1]
    
    validation_idx = int(sample_count * 0.8)
    testing_idx = int(sample_count * 0.9)
    np.save(f"{TRAINING_PREPROCESSED_PATH}{target_file_name}.npy", target_file[:validation_idx])
    np.save(f"{VALIDATION_PREPROCESSED_PATH}{target_file_name}.npy", target_file[:validation_idx])
    np.save(f"{TESTING_PREPROCESSED_PATH}{target_file_name}.npy", target_file[:validation_idx])

for directory, source_file_name in [("training/", "CompleteDataSet_testing_tuples.npy"), ("validation/", "CompleteDataSet_validation_tuples.npy"),
                                    ("testing/", "CompleteDataSet_testing_tuples.npy")]:
    source_file = np.load(f"{DIGIT_AND_OPERATORS_1_PATH}{source_file_name}", allow_pickle=True)
    for label, target_file_name in [("0", "zeros"), ("1", "ones"), ("2", "twos"), ("3", "threes"), ("4", "fours"), ("5", "fives"), ("6", "sixes"),
                                    ("7", "sevens"), ("8", "eights"), ("9", "nines"), 
                                    ("+", "pluses"), ("-", "minuses"), ("*", "asterisks"), ("%", "slashes")]:
        sample_indices = np.where(source_file[:, 1] == label)[0]
        sample_count = sample_indices.shape[0]
        target_file = np.empty((sample_count), dtype=object)
        
        for i, sample_index in enumerate(sample_indices):
            image = source_file[sample_index][0]
            coordinates = np.argwhere(image > 0)
            target_file[i] = image[:, coordinates.min(axis=0)[1]:coordinates.max(axis=0)[1] + 1]
        
        target_file = np.append(target_file, np.load(f"{PREPROCESSED_PATH}{directory}{target_file_name}.npy", allow_pickle=True))
        np.save(f"{PREPROCESSED_PATH}{directory}{target_file_name}.npy", target_file)
