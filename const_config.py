import torch

SEED = 42

CUDA = torch.cuda.is_available()

DATA_DIR = "data"
DOWNLOADS_DIR = "downloads"

DIGIT_AND_OPERATORS_1_PATH = f"{DATA_DIR}/digits_and_operators_1/"
DIGIT_AND_OPERATORS_1_TRAIN = f"{DIGIT_AND_OPERATORS_1_PATH}CompleteDataSet_training_tuples.npy"
DIGIT_AND_OPERATORS_1_VALIDATION = f"{DIGIT_AND_OPERATORS_1_PATH}CompleteDataSet_testing_tuples.npy"
DIGIT_AND_OPERATORS_1_TEST = f"{DIGIT_AND_OPERATORS_1_PATH}CompleteDataSet_validation_tuples.npy"
DIGIT_AND_OPERATORS_2_PATH = f"{DATA_DIR}/digits_and_operators_2/"

PREPROCESSED_PATH = f"{DATA_DIR}/prerocessed_separated_characters/"
TRAINING_PREPROCESSED_PATH = PREPROCESSED_PATH + "training/"
VALIDATION_PREPROCESSED_PATH = PREPROCESSED_PATH + "validation/"
TESTING_PREPROCESSED_PATH = PREPROCESSED_PATH + "testing/"
ALL_MERGED_PREPROCESSED_PATH = PREPROCESSED_PATH + "all/"
CLEANED_PREPROCESSED_PATH = PREPROCESSED_PATH + "cleaned/"
CHARACTERS_PATH = f"{DATA_DIR}/prerocessed_separated_characters/"
AUGMENTED_EQUATIONS_PATH = f"{DATA_DIR}/augmented_equations/"
EQUATIONS_PATH = f"{DATA_DIR}/equations/"
MODELS_PATH = "models/"
AUGMENTED_MODELS_PATH = f"{MODELS_PATH}cleaned_dilatation_size_augmentation/"
NOT_AUGMENTED_MODELS_PATH = f"{MODELS_PATH}cleaned_no_augmentation/"

IMAGES_FILENAME = "images.npy"
LABELS_FILENAME = "labels.npy"

IMAGES_FILENAME_TEMPLATE = "equations_images_%s.npy"
LABELS_FILENAME_TEMPLATE = "equations_labels_%s.npy"

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

EQUATION_IMAGE_WIDTH = 288
EQUATION_IMAGE_HEIGHT = 38

BATCH_SIZE_TRAINING = 32
BATCHES_PER_FILE_TRAINING = 100
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

COMPRESSED_DATA_SET_1_PATH = f"{DOWNLOADS_DIR}/handwritten-digits-and-operators.zip"
DATA_SET_1_EXTRACTED = "CompleteDataSet_training_tuples.npy CompleteDataSet_testing_tuples.npy CompleteDataSet_validation_tuples.npy"

COMPRESSED_DATA_SET_2_PATH = f"{DOWNLOADS_DIR}/handwrittenmathsymbols.zip"
DATA_SET_2_EXTRACTED = "data.rar"

COMPRESSED_DATA_SET_3_PATH = f"{DOWNLOADS_DIR}/handwritting_characters_database.tar.gz"

WRITERS_PATH = "testing/writers/"
WRITERS_LABELS = f"{WRITERS_PATH}labels.pickle"

PREDICTION_SAMPLES = 16 * 4