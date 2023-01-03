from PIL import Image
from PIL import ImageOps
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from const_config import TRAINING_PREPROCESSED_PATH
from const_config import VALIDATION_PREPROCESSED_PATH
from const_config import TESTING_PREPROCESSED_PATH

from const_config import ALL_MERGED_PREPROCESSED_PATH
from const_config import ALL_IMAGES_FILENAME
from const_config import ALL_LABELS_FILENAME

all_images = np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{ALL_IMAGES_FILENAME}", allow_pickle=True)
all_labels = np.load(f"{ALL_MERGED_PREPROCESSED_PATH}{ALL_LABELS_FILENAME}", allow_pickle=True)


for label, target_file_name in enumerate(["zeros", "ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", 
                                          "pluses", "minuses", "asterisks", "slashes"]):
    class_indices = np.where(all_labels == label)[0]
    sample_count = class_indices.shape[0]
    target_file = np.empty((sample_count * 3), dtype=object)
    
    for i, sample_index in enumerate(class_indices):
        j = i * 3
        image = all_images[sample_index]
        coordinates = np.argwhere(image > 0)
        target_file[j] = image[:, coordinates.min(axis=0)[1]:coordinates.max(axis=0)[1] + 1]
        pil_image = Image.fromarray((target_file[j] * 255).astype(np.uint8), 'L')
        pil_image.thumbnail((pil_image.width, pil_image.height - 2), Image.BOX)
        target_file[j + 1] = (np.asarray(pil_image) > 0).astype(np.float32)
        if target_file_name in ["pluses", "minuses", "asterisks", "slashes"]:
            pil_image = Image.fromarray((target_file[j] * 255).astype(np.uint8), 'L')
            pil_image.thumbnail((pil_image.width, pil_image.height - 6), Image.BOX)
            padded_image = np.zeros((pil_image.height + 4, pil_image.width))
            padded_image[2:pil_image.height + 2, :] = (np.asarray(pil_image) > 0).astype(np.float32)
            target_file[j + 2] = padded_image
        else:
            pil_image = Image.fromarray((target_file[j] * 255).astype(np.uint8), 'L')
            pil_image = ImageOps.contain(pil_image, (pil_image.width + 3, pil_image.height + 3))
            target_file[j + 2] = (np.asarray(pil_image) >= 128).astype(np.float32)

    validation_idx = int(sample_count * 0.8)
    testing_idx = int(sample_count * 0.9)
    np.save(f"{TRAINING_PREPROCESSED_PATH}{target_file_name}.npy", target_file[:validation_idx])
    np.save(f"{VALIDATION_PREPROCESSED_PATH}{target_file_name}.npy", target_file[validation_idx:testing_idx])
    np.save(f"{TESTING_PREPROCESSED_PATH}{target_file_name}.npy", target_file[testing_idx:])
