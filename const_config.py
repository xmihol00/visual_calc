import torch

CUDA = torch.cuda.is_available()

CHARACTERS_PATH = "./data/separated_characters/"
EQUATIONS_PATH = "./data/equations/"
MODEL_PATH = "./models/"

TRAINING_IMAGES_FILENAME_TEMPLATE = "equations_%s_training_images_%s.npy"
TRAINING_LABELS_FILENAME_TEMPLATE = "equations_%s_training_labels_%s.npy"

BATCH_SIZE = 16
BATCHES_PER_FILE = 25
NUMBER_OF_FILES = 4

NUMBER_OF_DIGITS = 10           # labels for digits are 0-9
NUMBER_OF_OPERATORS = 4         # labels for operators are 10-13 (+, -, *, /)

YOLO_TRAINING_IMAGES_FILENAME = "equations_230x38_training_images_%s.npy"
YOLO_TRAINING_LABELS_FILENAME = "equations_230x38_training_labels_%s.npy"
YOLO_LABELS_PER_IMAGE = 25      # image is separated into 25 parts of the same size along the x axis, each part has separate label
YOLO_OUTPUTS_PER_LABEL = 15     # each label consist of an indicator if there is (1) a character or not (0) at index 0 and probabilities of a class at indices 1-14

YOLO_V1_MODEL_FILENAME = "YOLO_inspired_CNN_v1.pt"
YOLO_V2_MODEL_FILENAME = "YOLO_inspired_CNN_v2.pt"
YOLO_V3_MODEL_FILENAME = "YOLO_inspired_CNN_v3.pt"
