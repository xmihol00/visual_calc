import torch

CUDA = torch.cuda.is_available()

DIGIT_AND_OPERATORS_1_PATH = "./data/digits_and_operators_1/"
DIGIT_AND_OPERATORS_2_PATH = "./data/digits_and_operators_2/"
PREPROCESSED_PATH = "./data/prerocessed_separated_characters/"
TRAINING_PREPROCESSED_PATH = PREPROCESSED_PATH + "training/"
VALIDATION_PREPROCESSED_PATH = PREPROCESSED_PATH + "validation/"
TESTING_PREPROCESSED_PATH = PREPROCESSED_PATH + "testing/"
ALL_MERGED_PREPROCESSED_PATH = PREPROCESSED_PATH + "all/"
CLEANED_PREPROCESSED_PATH = PREPROCESSED_PATH + "cleaned/"
CHARACTERS_PATH = "./data/prerocessed_separated_characters/"
EQUATIONS_PATH = "./data/equations/"
MODEL_PATH = "./models/"

ALL_IMAGES_FILENAME = "images.npy"
ALL_LABELS_FILENAME = "labels.npy"

CLEANED_IMAGES_FILENAME = "cleaned_images.npy"
CLEANED_LABELS_FILENAME = "cleaned_labels.npy"

TRAINING_IMAGES_FILENAME_TEMPLATE = "equations_%s_training_images_%s.npy"
TRAINING_LABELS_FILENAME_TEMPLATE = "equations_%s_training_labels_%s.npy"

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

BATCH_SIZE = 32
BATCHES_PER_FILE = 40
NUMBER_OF_FILES = 10

NUMBER_OF_DIGITS = 10           # labels for digits are 0-9
NUMBER_OF_OPERATORS = 4         # labels for operators are 10-13 (+, -, *, /)

DATA_DIRECTORIES = ["training/", "validation/", "testing/"]

YOLO_TRAINING_IMAGES_FILENAME = "equations_230x38_training_images_%s.npy"
YOLO_TRAINING_LABELS_FILENAME = "equations_230x38_training_labels_%s.npy"
YOLO_LABELS_PER_IMAGE = 13      # image is separated into 25 parts of the same size along the x axis, each part has separate label, the YOLO networks are design such that the output is 25 predictions
YOLO_OUTPUTS_PER_LABEL = 15     # each label consist of an indicator if there is (1) a character or not (0) at index 0 and probabilities of a class at indices 1-14
YOLO_OUTPUTS_PER_LABEL_NO_CLASS = 16     # one more output than above indicating there is no digit on the image

OUTLIERS_DETECTOR_FILENAME = "outliers_detector.pt"
YOLO_V1_MODEL_FILENAME = "YOLO_inspired_CNN_v1.pt"
YOLO_V2_MODEL_FILENAME = "YOLO_inspired_CNN_v2.pt"
YOLO_V3_MODEL_FILENAME = "YOLO_inspired_CNN_v3.pt"
YOLO_V4_MODEL_FILENAME = "YOLO_inspired_CNN_v4.pt"
YOLO_V5_MODEL_FILENAME = "YOLO_inspired_CNN_v5.pt"

YOLO_LOSS_BIAS = 5
