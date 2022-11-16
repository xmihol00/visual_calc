import numpy as np
import cv2
import os

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "div", "mul"]


# Required to be inline with other datasets
def folder_name_to_label(name):
    if name == "mul":
        return "*"
    elif name == "div":
        return "%"
    else:
        return name


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a)).astype(int)
    return np.array(a)[p], np.array(b)[p]


if __name__ == '__main__':
    shuffle_data = True
    images = []
    labels = []
    for classFolder in classes:
        classDirectory = "./dataset/{0}".format(classFolder)
        for filepath in os.listdir(classDirectory):
            images.append(cv2.imread(classDirectory + "/" + filepath, 0))
            labels.append(folder_name_to_label(classFolder))
    if shuffle_data:
        shuffled_images, shuffled_labels = unison_shuffled_copies(images, labels)
        np.save("./dataset/handwritten_dataset_images.npy", shuffled_images)
        np.save("./dataset/handwritten_dataset_labels.npy", shuffled_labels)
    else:
        np.save("./dataset/handwritten_dataset_images.npy", images)
        np.save("./dataset/handwritten_dataset_labels.npy", labels)
