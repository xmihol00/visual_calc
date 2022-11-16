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


if __name__ == '__main__':
    images = []
    labels = []
    for classFolder in classes:
        classDirectory = "./dataset/{0}".format(classFolder)
        for filepath in os.listdir(classDirectory):
            images.append(cv2.imread(classDirectory + "/" + filepath, 0))
            labels.append(folder_name_to_label(classFolder))
    dataset = [images, labels]
    np.save("./dataset/handwritten_dataset.npy", dataset)
