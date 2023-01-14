import os
import matplotlib.pyplot as plt
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "networks"))
from networks.custom_recursive_CNN import CustomRecursiveCNN
import data_preprocessing.handwritten_equtions as hwe
from const_config import WRITERS_PATH
from const_config import PREDICTION_SAMPLES

model = CustomRecursiveCNN("cpu", True, PREDICTION_SAMPLES)
model.load()
model = model.eval()

for file_name in sorted(glob.glob(f"{WRITERS_PATH}*.jpg")):
    image, areas = hwe.equation_areas(file_name)
    for sample, (row1, row2, col1, col2) in zip(hwe.samples_from_area(image, areas), areas):
        predictions = model(sample)
        final_prediction = hwe.parse_perdictions(predictions)

        plt.imshow(image[row1:row2, col1:col2] > 0, cmap='gray')
        plt.title(final_prediction)
        plt.show()
                