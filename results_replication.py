import os
import sys
import itertools as it
import argparse

from const_config import DATA_DIR
from const_config import COMPRESSED_DATA_SET_1_PATH
from const_config import DATA_SET_1_EXTRACTED
from const_config import COMPRESSED_DATA_SET_2_PATH
from const_config import DATA_SET_2_EXTRACTED
from const_config import COMPRESSED_DATA_SET_3_PATH
from const_config import DIGIT_AND_OPERATORS_1_PATH
from const_config import DIGIT_AND_OPERATORS_2_PATH

# https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
# https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators
# https://github.com/sueiras/handwritting_characters_database 

parser = argparse.ArgumentParser()
parser.add_argument("-se", "--skip_extraction", action="store_true", help="Skip extraction of relevant files from the downloaded data sets.")
parser.add_argument("-sp", "--skip_preprocessing", action="store_true", help="Skip file preproccessing of extracted data sets.")
parser.add_argument("-sg", "--skip_equation_generation", action="store_true", help="Skip equation generation from preproccesed data sets.")
parser.add_argument("-sd", "--skip_dataset", action="store_true", help="Skip data set related tasks entirely.")
parser.add_argument("-p", "--plot", action="store_true", help="Plot single digits/operators and whole equations.")
parser.add_argument("-t", "--train", choices=["custom_recursive_CNN", "custom_CNN", "YOLO_inspired_CNN"], help="Train specified model.")
parser.add_argument("-e", "--evaluate", choices=["custom_recursive_CNN", "custom_CNN", "YOLO_inspired_CNN"], help="Evaluate specified model.")
args = parser.parse_args()

if not args.skip_dataset:
    if not args.skip_extraction:
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

    if not args.skip_preprocessing:
        os.system("python3 data_preprocessing/merge_preprocess_datasets.py")
        os.system("python3 network/outliers_detector.py clean")
        os.system("python3 data_preprocessing/crop_separate_augment_characters.py")

    if not args.skip_equation_generation:
        os.system("python3 data_generation/equation_generator.py")

if args.plot:
    os.system(f"python3 data_preprocessing/merged_plot.py")
    os.system(f"python3 data_preprocessing/separated_plot.py")
    os.system(f"python3 data_generation/equation_plot.py")

if args.train:
    os.system(f"python3 network/{args.train}.py train")

if args.evaluate:
    os.system(f"python3 network/{args.evaluate}.py eval")