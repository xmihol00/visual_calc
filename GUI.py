import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

import cv2
import imutils
from PIL import ImageTk, Image
import os
import numpy as np
import sys

from networks.mser.detector import Detector

sys.path.append(os.path.join(os.path.dirname(__file__), "networks"))
from networks.custom_recursive_CNN import CustomRecursiveCNN
import data_preprocessing.handwritten_equtions as hwe
from const_config import PREDICTION_SAMPLES

def extract_equations(model, image_filename, mser_detector=None):
    equations = []
    image, areas = hwe.equation_areas(image_filename)
    for sample, (row1, row2, col1, col2) in zip(hwe.samples_from_area(image, areas), areas):
        predictions = model(sample)
        string_labels = hwe.extract_string_labels(predictions)
        
        string_labels = []
        area = image[row1:row2, col1:col2]
        weight = 5
        if mser_detector is not None:
            gray = (area * 255).astype(np.uint8)
            gray = 255 - gray
            padded_gray = cv2.copyMakeBorder(gray, 80, 80, 120, 120, cv2.BORDER_CONSTANT, value=255)
            img = cv2.cvtColor(padded_gray, cv2.COLOR_GRAY2BGR)
            img = imutils.resize(img, width=320, inter=cv2.INTER_AREA)
            valid_boxes, labels, probabilities = mser_detector.detect_digits_in_img(img, False, False)
            eq_results = mser_detector.compute_equation(valid_boxes, labels, probabilities, 3)
            for equation_result in eq_results:
                for _ in range(0, weight):
                    string_labels.append(equation_result)
                weight = weight - 1

        final_prediction = hwe.parse_string_labels(string_labels)
        equations.append(final_prediction)

    return equations

def select_file(model, objects, mser_detector=None):
    for tk_object in objects[0]:
        tk_object.destroy()
    objects[0] = []

    filetypes = [("images", "*.jpg"), ("images", "*.png")]
    file_name = fd.askopenfilename(title="Choose an image", initialdir='~/', filetypes=filetypes)
    equations = extract_equations(model, file_name, mser_detector)

    image = Image.open(file_name)
    image.thumbnail((800, 500))
    image = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=image)
    image_label.image = image
    image_label.grid(row=2, columnspan=4)
    objects[0].append(image_label)

    counter = 3
    objects_idx = 1
    for equation in equations:
        if not len(equation):
            continue

        try:
            result = eval(equation)
        except:
            result = "error"
        
        equation_entry = tk.Entry(root)
        equation_entry.insert(0, equation)
        equal_label = tk.Label(root, text="=")
        result_label = tk.Label(root, text=result)
        button = tk.Button(root, text="recalculate", command=lambda equation_entry=equation_entry, result_label=result_label: recalculate_cell(equation_entry, result_label))

        equation_entry.grid(row=counter, column=0, pady=5, padx=(5, 0))
        equal_label.grid(row=counter, column=1, pady=5)
        result_label.grid(row=counter, column=2, pady=5)
        button.grid(row=counter, column=3, pady=5, padx=(0, 5))
        counter += 1

        objects[0].append(equation_entry)
        objects[0].append(result_label)
        objects[0].append(equal_label)
        objects[0].append(button)
        objects_idx += 4


def recalculate_cell(equation_entry, equation_label):
    equation = equation_entry.get()
    try:
        result = eval(equation)
    except:
        result = "error"
    
    equation_label.config(text=result)

if __name__ == "__main__":
    model = CustomRecursiveCNN(device="cpu", augmentation=True, batch_size=PREDICTION_SAMPLES)
    model.load()
    model = model.eval()

    use_gpu = False
    mser_detector = Detector(use_gpu)

    tk_objects = [[]]
    root = tk.Tk()
    root.title("Visual calculator")

    # make askopenfilename() not show hidden files
    try:
        try:
            root.tk.call("tk_getOpenFile", "-foobarbaz")
        except tk.TclError:
            pass
        root.tk.call("set", "::tk::dialog::file::showHiddenBtn", '1')
        root.tk.call("set", "::tk::dialog::file::showHiddenVar", '0')
    except:
        pass

    select_file_label = tk.Label(root, text="Select an image with an equation or equations.")
    open_button = ttk.Button(root, text="select", command=lambda:select_file(model, tk_objects, mser_detector))

    select_file_label.grid(row=0, columnspan=4, pady=5)
    open_button.grid(row=1, columnspan=4, pady=5)

    root.mainloop()
