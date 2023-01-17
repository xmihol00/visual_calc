import os
import sys
import itertools as it
import argparse
import cv2
import matplotlib.pyplot as plt
import pickle
import Levenshtein as lv
import glob
import imutils
import numpy as np

from networks.custom_recursive_CNN import CustomRecursiveCNN
from networks.utils.data_loaders import DataLoader
import label_extractors
import data_preprocessing.handwritten_equtions as hwe
from networks.mser.detector import Detector

from const_config import DATA_DIR
from const_config import COMPRESSED_DATA_SET_1_PATH
from const_config import DATA_SET_1_EXTRACTED
from const_config import COMPRESSED_DATA_SET_2_PATH
from const_config import DATA_SET_2_EXTRACTED
from const_config import COMPRESSED_DATA_SET_3_PATH
from const_config import DIGIT_AND_OPERATORS_1_PATH
from const_config import DIGIT_AND_OPERATORS_2_PATH
from const_config import WRITERS_LABELS
from const_config import PREDICTION_SAMPLES
from const_config import BATCHES_PER_FILE_TESTING
from const_config import NUMBER_OF_FILES_TESTING
from const_config import BATCH_SIZE_TESTING
from const_config import LABELS_PER_IMAGE
from const_config import WRITERS_PATH
from const_config import AUGMENTED_EQUATIONS_PATH
from const_config import EQUATIONS_PATH
from const_config import RESULTS_PATH
from const_config import DATA_PREPROCESSING_PATH
from const_config import DATA_GENERATION_PATH
from const_config import NETWORKS_PATH

def color_patches(patches):
    patches[0].set_facecolor("green")
    patches[1].set_facecolor("mediumseagreen")
    patches[2].set_facecolor("orange")
    for patch in patches[3:]:
        patch.set_facecolor("red")

def annotate_bins(axis, counts, annotations_x):
    for count, x_pos in zip(counts, annotations_x):
        axis.annotate(str(count), (x_pos, count), ha="center", va="bottom", fontweight="bold")

def clean_axis(axis, annotations_x, distances):
    axis.set_xticks(annotations_x[:-1], distances)
    axis.set_yticks([], [])
    axis.set_frame_on(False)

def evaluate_model_on_test_set(model, augmented):
    model.load()
    model.change_batch_size(BATCH_SIZE_TESTING)
    model = model.eval()
    test_dataloader = DataLoader("testing/", AUGMENTED_EQUATIONS_PATH if augmented else EQUATIONS_PATH,
                                 BATCH_SIZE_TESTING, BATCHES_PER_FILE_TESTING, NUMBER_OF_FILES_TESTING, "cpu")

    distances = [0] * 9
    for images, labels in test_dataloader:
        predictions = model(images)
        labels = labels.reshape(-1, LABELS_PER_IMAGE, 2).numpy()
        
        for i in range(BATCH_SIZE_TESTING):
            j = i * LABELS_PER_IMAGE
            labeled = label_extractors.labels_only_class(labels, i, sep="")
            classified = label_extractors.prediction_only_class(predictions[j:j+LABELS_PER_IMAGE], sep="")
            distances[lv.distance(labeled, classified, score_cutoff=7)] += 1
    
    return distances

def evaluate_model_on_handwritten_set(model):
    model.load()
    model.change_batch_size(PREDICTION_SAMPLES)
    model = model.eval()

    predicted_equations = []
    for file_name in sorted(glob.glob(f"{WRITERS_PATH}*.jpg")):
        image, areas = hwe.equation_areas(file_name)
        for samples in hwe.samples_from_area(image, areas):
            predictions = model(samples)
            predicted_equations.append(hwe.parse_perdictions(predictions))
    
    with open(WRITERS_LABELS, "rb") as labels_file:
        labels = pickle.load(labels_file)
    
    distances = [0] * 9
    for label, prediction in zip(labels, predicted_equations):
        distances[lv.distance(label, prediction, score_cutoff=7)] += 1

    return distances


def evaluate_ensemble_on_handrwritten(model, mser_detector):
    model.load()
    model.change_batch_size(PREDICTION_SAMPLES)
    model = model.eval()

    predicted_equations = []
    for file_name in sorted(glob.glob(f"{WRITERS_PATH}*.jpg")):
        image, areas = hwe.equation_areas(file_name)
        for sample, (row1, row2, col1, col2) in zip(hwe.samples_from_area(image, areas), areas):
            predictions = model(sample)
            string_labels = hwe.extract_string_labels(predictions)

            area = image[row1:row2, col1:col2]
            string_labels += predict_MSER(mser_detector, area)

            final_prediction = hwe.parse_string_labels(string_labels)
            predicted_equations.append(final_prediction)

    with open(WRITERS_LABELS, "rb") as labels_file:
        labels = pickle.load(labels_file)
    
    distances = [0] * 9
    for label, prediction in zip(labels, predicted_equations):
        distances[lv.distance(label, prediction, score_cutoff=7)] += 1

    return distances

def evaluate_MSER_on_handrwritten(mser_detector):
    predicted_equations = []
    for file_name in sorted(glob.glob(f"{WRITERS_PATH}*.jpg")):
        image, areas = hwe.equation_areas(file_name)
        for row1, row2, col1, col2 in areas:
            area = image[row1:row2, col1:col2]
            string_labels = predict_MSER(mser_detector, area)

            final_prediction = hwe.parse_string_labels(string_labels)
            predicted_equations.append(final_prediction)

    with open(WRITERS_LABELS, "rb") as labels_file:
        labels = pickle.load(labels_file)
    
    distances = [0] * 9
    for label, prediction in zip(labels, predicted_equations):
        distances[lv.distance(label, prediction, score_cutoff=7)] += 1

    return distances

def predict_MSER(mser_detector, area):
    string_labels = []
    gray = (area * 255).astype(np.uint8)
    gray = 255 - gray
    padded_gray = cv2.copyMakeBorder(gray, 80, 80, 120, 120, cv2.BORDER_CONSTANT, value=255)
    img = cv2.cvtColor(padded_gray, cv2.COLOR_GRAY2BGR)
    img = imutils.resize(img, width=320, inter=cv2.INTER_AREA)
    valid_boxes, labels, probabilities = mser_detector.detect_digits_in_img(img, False, False)
    eq_results = mser_detector.compute_equation(valid_boxes, labels, probabilities, 2)
    weight = 6
    for equation_result in eq_results:
        for _ in range(0, weight):
            string_labels.append(equation_result)
        weight = weight - 2
    
    return string_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")
    parser.add_argument("-u", "--unzip", action="store_true", help="Perform extraction of relevant files from the downloaded data sets.")
    parser.add_argument("-p", "--preprocessing", action="store_true", help="Perform file preproccessing of extracted data sets.")
    parser.add_argument("-g", "--equation_generation", action="store_true", help="Perform equation generation from preproccesed data sets.")
    parser.add_argument("-d", "--dataset", action="store_true", help="Perform all data set related tasks.")
    parser.add_argument("-pd", "--plot_dataset", action="store_true", help="Plot separate symbols and whole equations.")
    parser.add_argument("-t", "--train", choices=["MSER_classifier", "custom_recursive_CNN", "custom_CNN", "YOLO_inspired_CNN"], help="Train specified model.")
    parser.add_argument("-e", "--evaluate", choices=["MSER_classifier", "custom_recursive_CNN", "custom_CNN", "YOLO_inspired_CNN"], help="Evaluate specified model.")
    parser.add_argument("-na", "--no_augmentation", action="store_true", help="Train or evaluate model on not augmented data sets, use with -t (--train) and -e (--evaluate) options.")
    parser.add_argument("-prMC", "--plot_results_MC", action="store_true", help="Plot results of the multi-classifier (custom_recursive_CNN).")
    parser.add_argument("-prMSER", "--plot_results_MSER", action="store_true", help="Plot results of the MSER based classifier.")
    parser.add_argument("-pr", "--plot_results", action="store_true", help="Plot results of an ensemble of the multi-classifier and the MSER based classifier.")
    args = parser.parse_args()

    bins = [i * 10 for i in range(10)]
    annotations_x = [i * 10 + 5 for i in range(10)]
    distances = [i for i in range(9)]

    if not (os.path.isfile(COMPRESSED_DATA_SET_1_PATH) and 
            os.path.isfile(COMPRESSED_DATA_SET_2_PATH) and 
            os.path.isfile(COMPRESSED_DATA_SET_3_PATH)):
        print(f"Files with data sets could not be found. Download them as described in the README.", file=sys.stderr)

    if args.unzip or args.dataset:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(DIGIT_AND_OPERATORS_1_PATH, exist_ok=True)
        os.makedirs(DIGIT_AND_OPERATORS_2_PATH, exist_ok=True)

        os.system(f"unzip -o {COMPRESSED_DATA_SET_1_PATH} {DATA_SET_1_EXTRACTED} -d {DIGIT_AND_OPERATORS_1_PATH}")
        os.system(f"unzip -o {COMPRESSED_DATA_SET_2_PATH} {DATA_SET_2_EXTRACTED} -d {DIGIT_AND_OPERATORS_2_PATH}")

        for folder_name in it.chain(range(10), ["+", "-", ","]):
            os.makedirs(f"{DIGIT_AND_OPERATORS_2_PATH}{folder_name}", exist_ok=True)
            os.system(f"unrar e -o+ {DIGIT_AND_OPERATORS_2_PATH}{DATA_SET_2_EXTRACTED} extracted_images/{folder_name} {DIGIT_AND_OPERATORS_2_PATH}{folder_name}")

        os.system(f"rm -f {DIGIT_AND_OPERATORS_2_PATH}{DATA_SET_2_EXTRACTED}")

        for input_folder_name, output_folder_name in zip(it.chain(range(48, 58), [43, 45, 42, 47]), it.chain(range(10), ["+", "-", "x", ","])):
            os.makedirs(f"{DIGIT_AND_OPERATORS_2_PATH}{output_folder_name}", exist_ok=True)
            os.system(f"tar -zxvf {COMPRESSED_DATA_SET_3_PATH} -C {DIGIT_AND_OPERATORS_2_PATH}{output_folder_name} --strip-components 2 curated/{input_folder_name} ")

    if args.preprocessing or args.dataset:
        os.system(f"python3 -u {DATA_PREPROCESSING_PATH}merge_preprocess_datasets.py")
        os.system(f"python3 -u {NETWORKS_PATH}outliers_detector.py clean")
        os.system(f"python3 -u {DATA_PREPROCESSING_PATH}crop_separate_augment_characters.py")
        os.system(f"python3 -u {DATA_PREPROCESSING_PATH}crop_separate_characters.py")

    if args.equation_generation or args.dataset:
        os.system(f"python3 -u {DATA_GENERATION_PATH}equation_generator.py --augment")
        os.system(f"python3 -u {DATA_GENERATION_PATH}equation_generator.py")

    if args.plot_dataset:
        os.system(f"python3 -u {DATA_PREPROCESSING_PATH}merged_plot.py")
        os.system(f"python3 -u {DATA_PREPROCESSING_PATH}separated_plot.py")
        os.system(f"python3 -u {DATA_GENERATION_PATH}equation_plot.py")

    augment_switch = "" if args.no_augmentation else "--augmentation"

    if args.train:
        if args.train == "MSER_classifier":
            args.train = "mser/classifier"
        os.system(f"python3 -u {NETWORKS_PATH}{args.train}.py --train {augment_switch}")

    if args.evaluate:
        if args.evaluate == "MSER_classifier":
            args.evaluate = "mser/classifier"
        os.system(f"python3 -u {NETWORKS_PATH}{args.evaluate}.py --evaluate {augment_switch}")

    if args.plot_results_MC:
        not_augmented_model = CustomRecursiveCNN(device="cpu", augmentation=False)
        augmented_model = CustomRecursiveCNN(device="cpu", augmentation=True)

        not_augmented_test_distances = evaluate_model_on_test_set(not_augmented_model, False)
        augmented_test_distances = evaluate_model_on_test_set(augmented_model, True)

        not_augmented_handwritten_distances = evaluate_model_on_handwritten_set(not_augmented_model)
        augmented_handwritten_distances = evaluate_model_on_handwritten_set(augmented_model)

        figure, axis = plt.subplots(2, 2)
        figure.set_size_inches(10, 7)
        plt.subplots_adjust(left=0.02, bottom=0.05, right=0.98, top=0.95, hspace=0.2, wspace=0.05)
        
        *_, patches = axis[0, 0].hist(bins[:-1], bins, weights=not_augmented_test_distances)
        color_patches(patches)
        annotate_bins(axis[0, 0], not_augmented_test_distances, annotations_x)
        clean_axis(axis[0, 0], annotations_x, distances)
        axis[0, 0].set_title("Results without augmentation on test data set")

        *_, patches = axis[0, 1].hist(bins[:-1], bins, weights=augmented_test_distances)
        color_patches(patches)
        annotate_bins(axis[0, 1], augmented_test_distances, annotations_x)
        clean_axis(axis[0, 1], annotations_x, distances)
        axis[0, 1].set_title("Results with augmentation on test data set")

        *_, patches = axis[1, 0].hist(bins[:-1], bins, weights=not_augmented_handwritten_distances)
        color_patches(patches)
        annotate_bins(axis[1, 0], not_augmented_handwritten_distances, annotations_x)
        clean_axis(axis[1, 0], annotations_x, distances)
        axis[1, 0].set_title("Results without augmentation on handwritten data set")

        *_, patches = axis[1, 1].hist(bins[:-1], bins, weights=augmented_handwritten_distances)
        color_patches(patches)
        annotate_bins(axis[1, 1], augmented_handwritten_distances, annotations_x)
        clean_axis(axis[1, 1], annotations_x, distances)
        axis[1, 1].set_title("Results with augmentation on handwritten data set")

        plt.savefig(f"{RESULTS_PATH}multi_classifier_results", dpi=400)
        plt.show()

    if args.plot_results_MSER:
        mser_detector = Detector(use_gpu=False)

        MSER_distances = evaluate_MSER_on_handrwritten(mser_detector)
        figure, axis = plt.subplots(1, 1)
        figure.set_size_inches(9, 6)
        plt.subplots_adjust(left=0.02, bottom=0.05, right=1.0, top=1.0, hspace=0.1, wspace=0.02)
        *_, patches = axis.hist(bins[:-1], bins, weights=MSER_distances)
        color_patches(patches)
        annotate_bins(axis, MSER_distances, annotations_x)
        clean_axis(axis, annotations_x, distances)

        plt.savefig(f"{RESULTS_PATH}MSER_results", dpi=400)
        plt.show()

    if args.plot_results:
        augmented_model = CustomRecursiveCNN(device="cpu", augmentation=True)
        mser_detector = Detector(use_gpu=False)

        ensemble_distances = evaluate_ensemble_on_handrwritten(augmented_model, mser_detector)
        figure, axis = plt.subplots(1, 1)
        figure.set_size_inches(9, 6)
        plt.subplots_adjust(left=0.02, bottom=0.05, right=1.0, top=1.0, hspace=0.1, wspace=0.02)
        *_, patches = axis.hist(bins[:-1], bins, weights=ensemble_distances)
        color_patches(patches)
        annotate_bins(axis, ensemble_distances, annotations_x)
        clean_axis(axis, annotations_x, distances)

        plt.savefig(f"{RESULTS_PATH}ensemble_results", dpi=400)
        plt.show()
        