import os
import sys
import itertools as it
import argparse
import matplotlib.pyplot as plt
import pickle
import Levenshtein as lv
import glob

from networks.custom_recursive_CNN import CustomRecursiveCNN
from networks.utils.data_loaders import DataLoader
import label_extractors
import data_preprocessing.handwritten_equtions as hwe

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

# https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
# https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators
# https://github.com/sueiras/handwritting_characters_database 

def color_patches(patches):
    patches[0].set_facecolor("green")
    patches[1].set_facecolor("mediumseagreen")
    patches[2].set_facecolor("orange")
    for patch in patches[3:]:
        patch.set_facecolor("red")

def annotate_bins(axis, counts, anotations_x):
    for count, x_pos in zip(counts, anotations_x):
        axis.annotate(str(count), (x_pos, count), ha="center", va="bottom", fontweight="bold")

def clean_axis(axis):
    axis.set_xticks([i * 10 + 5 for i in range(9)], [i for i in range(9)])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--unzip", action="store_true", help="Perform extraction of relevant files from the downloaded data sets.")
    parser.add_argument("-p", "--preprocessing", action="store_true", help="Perform file preproccessing of extracted data sets.")
    parser.add_argument("-g", "--equation_generation", action="store_true", help="Perform equation generation from preproccesed data sets.")
    parser.add_argument("-d", "--dataset", action="store_true", help="Perform all data set related tasks.")
    parser.add_argument("-pd", "--plot_dataset", action="store_true", help="Plot separate symbols and whole equations.")
    parser.add_argument("-t", "--train", choices=["custom_recursive_CNN", "custom_CNN", "YOLO_inspired_CNN"], help="Train specified model.")
    parser.add_argument("-e", "--evaluate", choices=["custom_recursive_CNN", "custom_CNN", "YOLO_inspired_CNN"], help="Evaluate specified model.")
    parser.add_argument("-prMC", "--plot_results_MC", action="store_true", help="Plot results of the multi-classifier (custom_recursive_CNN).")
    parser.add_argument("-prMSER", "--plot_results_MSER", action="store_true", help="Plot results of the MSER based classifier.")
    parser.add_argument("-pr", "--plot_results", action="store_true", help="Plot results of an ensemble of the multi-classifier and the MSER based classifier.")
    args = parser.parse_args()

    if args.unzip or args.dataset:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(DIGIT_AND_OPERATORS_1_PATH, exist_ok=True)
        os.makedirs(DIGIT_AND_OPERATORS_2_PATH, exist_ok=True)

        if not (os.path.isfile(COMPRESSED_DATA_SET_1_PATH) and 
                os.path.isfile(COMPRESSED_DATA_SET_2_PATH) and 
                os.path.isfile(COMPRESSED_DATA_SET_3_PATH)):
            print(f"Files with data sets could not be found.", file=sys.stderr)

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
        os.system("python3 data_preprocessing/merge_preprocess_datasets.py")
        os.system("python3 networks/outliers_detector.py clean")
        os.system("python3 data_preprocessing/crop_separate_augment_characters.py")
        os.system("python3 data_preprocessing/crop_separate_characters.py")

    if args.equation_generation or args.dataset:
        os.system("python3 data_generation/equation_generator.py --augment")
        os.system("python3 data_generation/equation_generator.py")

    if args.plot_dataset:
        os.system(f"python3 data_preprocessing/merged_plot.py")
        os.system(f"python3 data_preprocessing/separated_plot.py")
        os.system(f"python3 data_generation/equation_plot.py")

    if args.train:
        os.system(f"python3 networks/{args.train}.py --train --augmentation")

    if args.evaluate:
        os.system(f"python3 networks/{args.evaluate}.py --eval --augmentation")

    if args.plot_results_MC:
        bins = [i * 10 for i in range(10)]
        anotations_x = [i * 10 + 5 for i in range(10)]

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
        annotate_bins(axis[0, 0], not_augmented_test_distances, anotations_x)
        clean_axis(axis[0, 0])
        axis[0, 0].set_title("Results without augmentation on test data set")

        *_, patches = axis[0, 1].hist(bins[:-1], bins, weights=augmented_test_distances)
        color_patches(patches)
        annotate_bins(axis[0, 1], augmented_test_distances, anotations_x)
        clean_axis(axis[0, 1])
        axis[0, 1].set_title("Results with augmentation on test data set")

        *_, patches = axis[1, 0].hist(bins[:-1], bins, weights=not_augmented_handwritten_distances)
        color_patches(patches)
        annotate_bins(axis[1, 0], not_augmented_handwritten_distances, anotations_x)
        clean_axis(axis[1, 0])
        axis[1, 0].set_title("Results without augmentation on handwritten data set")

        *_, patches = axis[1, 1].hist(bins[:-1], bins, weights=augmented_handwritten_distances)
        color_patches(patches)
        annotate_bins(axis[1, 1], augmented_handwritten_distances, anotations_x)
        clean_axis(axis[1, 1])
        axis[1, 1].set_title("Results with augmentation on handwritten data set")

        plt.savefig("results/multi_classifier_results", dpi=400)
        plt.show()

    if args.plot_results_MC:
        pass # TODO

    if args.plot_results:
        pass # TODO
    