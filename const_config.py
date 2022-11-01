import torch

CUDA = torch.cuda.is_available()

CHARACTERS_PATH = "./data/separated_characters/"
EQUATIONS_PATH = "./data/equations/"
MODEL_PATH = "./models/"

TRAINING_IMAGES_FILENAME = "equations_%s_training_images_%s.npy"
TRAINING_LABELS_FILENAME = "equations_%s_training_labels_%s.npy"

BATCH_SIZE = 16
BATCHES_PER_FILE = 25
NUMBER_OF_FILES = 25

NUMBER_OF_DIGITS = 10
NUMBER_OF_OPERATORS = 4

YOLO_V1_TRAINING_IMAGES_FILENAME = "equations_YOLO_V1_training_images_%s.npy"
YOLO_V1_TRAINING_LABELS_FILENAME = "equations_YOLO_V1_training_labels_%s.npy"
YOLO_V1_MODEL_FILENAME = "YOLO_inspired_CNN_v1.pt"
YOLO_V1_LABELS_PER_IMAGE = 25

YOLO_V2_MODEL_FILENAME = "YOLO_inspired_CNN_v2.pt"
