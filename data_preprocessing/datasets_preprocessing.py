from PIL import Image
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from const_config import DIGIT_AND_OPERATORS_1_PATH
from const_config import DIGIT_AND_OPERATORS_2_PATH
from const_config import NPY_IMAGE_PATH
from const_config import TRAINING_NPY_IMAGE_PATH
from const_config import VALIDATION_NPY_IMAGE_PATH
from const_config import TESTING_NPY_IMAGE_PATH
from const_config import IMAGE_WIDTH
from const_config import IMAGE_HEIGHT


for directory, target_file_name in [("0/", "zeros"), ("1/", "ones"), ("2/", "twos"), ("3/", "threes"), ("4/", "fours"), ("5/", "fives"), ("6/", "sixes"),
                                    ("7/", "sevens"), ("8/", "eights"), ("9/", "nines"),
                                    ("plus/", "pluses"), ("minus/", "minuses"), ("asterisk/", "asterisks")]:
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
    np.save(f"{TRAINING_NPY_IMAGE_PATH}{target_file_name}.npy", target_file[:validation_idx])
    np.save(f"{VALIDATION_NPY_IMAGE_PATH}{target_file_name}.npy", target_file[:validation_idx])
    np.save(f"{TESTING_NPY_IMAGE_PATH}{target_file_name}.npy", target_file[:validation_idx])

for label, target_file_name in [("0", "zeros"), ("1", "ones"), ("2", "twos"), ("3", "threes"), ("4", "fours"), ("5", "fives"), ("6", "sixes"),
                                ("7", "sevens"), ("8", "eights"), ("9", "nines"), 
                                ("+", "pluses"), ("-", "minuses"), ("*", "astrics"), ("%", "slashes")]:
    pass
