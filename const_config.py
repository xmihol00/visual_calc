import torch

SEED = 42

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
MODELS_PATH = "./models/"

IMAGES_FILENAME = "images.npy"
LABELS_FILENAME = "labels.npy"

IMAGES_FILENAME_TEMPLATE = "equations_images_%s.npy"
LABELS_FILENAME_TEMPLATE = "equations_labels_%s.npy"

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

LABELS_PER_IMAGE = 18      # image is separated into 18 parts of the same size along the x axis, each part has separate label
OUTPUTS_PER_LABEL = 15     # each label consist of an indicator if there is (1) a character or not (0) at index 0 and probabilities of a class at indices 1-14
LABEL_DIMENSIONS = 2

OUTLIERS_DETECTOR_FILENAME = "outliers_detector.pt"
YOLO_INSPIRED_MODEL_FILENAME = "YOLO_inspired_CNN.pt"
CUSTOM_CNN_FILENAME = "custom_CNN.pt"
CUSTOM_RECURSIVE_CNN_FILENAME = "custom_recursive_CNN.pt"

LOSS_BIAS = 5
