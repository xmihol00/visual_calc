import numpy as np
import idx2numpy

DATA_PATH_COMPLETE = "../data/digits_and_characters_1/"
DATA_PATH_MNIST = "../data/mnist/"
SEPARATE_CHAR_PATH = "../data/separated_characters/"

training_data_complete = np.load(f"{DATA_PATH_COMPLETE}CompleteDataSet_testing_tuples.npy", allow_pickle=True)
training_data_mnist = idx2numpy.convert_from_file(f"{DATA_PATH_MNIST}train-images.idx3-ubyte")
training_labels_mnist = idx2numpy.convert_from_file(f"{DATA_PATH_MNIST}train-labels.idx1-ubyte")

for label, name in [("0", "zeros"), ("1", "ones"), ("2", "twos"), ("3", "threes"), ("4", "fours"), ("5", "fives"), ("6", "sixes"),
                    ("7", "sevens"), ("8", "eights"), ("9", "nines")]:
    complete_sample_indices = np.where(training_data_complete[:, 1] == label)[0]
    mnist_sample_indices = np.where(training_labels_mnist == int(label))[0]
    samples = np.empty((mnist_sample_indices.shape[0] + complete_sample_indices.shape[0], 2), dtype=object)

    for i in range(complete_sample_indices.shape[0]):
        samples[i][0] = training_data_complete[complete_sample_indices[i]][0].astype(np.float32)
        samples[i][1] = label
    
    for i in range(mnist_sample_indices.shape[0]):
        samples[i + complete_sample_indices.shape[0]][0] = (training_data_mnist[mnist_sample_indices[i]] / 255.0).astype(np.float32)
        samples[i + complete_sample_indices.shape[0]][1] = label
    
    np.save(f"{SEPARATE_CHAR_PATH}{name}", samples)


for label, name in [("+", "pluses"), ("-", "minuses"), ("*", "astrics"), ("%", "slashes")]:
    complete_sample_indices = training_data_complete[np.where(training_data_complete[:, 1] == label)]
    for i in range(complete_sample_indices.shape[0]):
        complete_sample_indices[i][0] = complete_sample_indices[i][0].astype(np.float32)

    np.save(f"{SEPARATE_CHAR_PATH}{name}", complete_sample_indices)
