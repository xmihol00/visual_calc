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

IMAGES_FILENAME_TEMPLATE = "equations_%s_images_%s.npy"
LABELS_FILENAME_TEMPLATE = "equations_%s_labels_%s.npy"

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

EQUATION_IMAGE_WIDTH = 288
EQUATION_IMAGE_HEIGHT = 38

BATCH_SIZE_TRAINING = 50
BATCHES_PER_FILE_TRAINING = 60
NUMBER_OF_FILES_TRAINING = 40

BATCH_SIZE_VALIDATION = BATCH_SIZE_TRAINING
BATCHES_PER_FILE_VALIDATION = BATCHES_PER_FILE_TRAINING
NUMBER_OF_FILES_VALIDATION = int(NUMBER_OF_FILES_TRAINING * 0.2)

BATCH_SIZE_TESTING = BATCH_SIZE_TRAINING
BATCHES_PER_FILE_TESTING = BATCHES_PER_FILE_TRAINING
NUMBER_OF_FILES_TESTING = int(NUMBER_OF_FILES_TRAINING * 0.2)

NUMBER_OF_DIGITS = 10           # labels for digits are 0-9
NUMBER_OF_OPERATORS = 4         # labels for operators are 10-13 (+, -, *, /)

DATA_DIRECTORIES_INFO = [("training/", BATCH_SIZE_TRAINING, BATCHES_PER_FILE_TRAINING, NUMBER_OF_FILES_TRAINING), 
                          ("validation/", BATCH_SIZE_VALIDATION, BATCHES_PER_FILE_VALIDATION, NUMBER_OF_FILES_VALIDATION), 
                          ("testing/", BATCH_SIZE_TESTING, BATCHES_PER_FILE_TESTING, NUMBER_OF_FILES_TESTING)]

YOLO_LABELS_PER_IMAGE = 18      # image is separated into 18 parts of the same size along the x axis, each part has separate label
YOLO_OUTPUTS_PER_LABEL = 15     # each label consist of an indicator if there is (1) a character or not (0) at index 0 and probabilities of a class at indices 1-14
YOLO_OUTPUTS_PER_LABEL_NO_CLASS = 16     # one more output than above indicating there is no digit on the image
YOLO_OUTPUTS_PER_LABEL_ONLY_CLASSES = 15 # only outputs for class classification including one non-character class
YOLO_LABEL_DIMENSIONS = 2

OUTLIERS_DETECTOR_FILENAME = "outliers_detector.pt"
YOLO_V1_MODEL_FILENAME = "YOLO_inspired_CNN_v1.pt"
YOLO_V2_MODEL_FILENAME = "YOLO_inspired_CNN_v2.pt"
YOLO_V3_MODEL_FILENAME = "YOLO_inspired_CNN_v3.pt"
YOLO_V4_MODEL_FILENAME = "YOLO_inspired_CNN_v4.pt"
YOLO_V5_MODEL_FILENAME = "YOLO_inspired_CNN_v5.pt"
YOLO_V6_MODEL_FILENAME = "YOLO_inspired_CNN_v6.pt"
YOLO_V7_MODEL_FILENAME = "YOLO_inspired_CNN_v7.pt"
CUSTOM_CNN_V1 = "CUSTOM_CNN_v1.pt"
CUSTOM_CNN_V2 = "CUSTOM_CNN_v2.pt"
CUSTOM_CNN_V3 = "CUSTOM_CNN_v3.pt"

YOLO_LOSS_BIAS = 5
