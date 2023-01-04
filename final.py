import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import os
import numpy as np
import torch
import sys
import re

PREDICTION_SAMPLES = 16 * 4

sys.path.append(os.path.join(os.path.dirname(__file__), "network"))
from network.custom_recursive_CNN import CustomCNNv3
import label_extractors
from const_config import EQUATION_IMAGE_WIDTH
from const_config import YOLO_LABELS_PER_IMAGE

def extract_equations(model, image_filename):
    original_image = Image.open(image_filename).convert('L')
    image = np.asarray(original_image)
    if image.sum() * 2 > image.shape[0] * image.shape[1] * 255:
        image = image < 128
    else:
        image = image >= 128

    interresting_rows = image.sum(axis=1) > image.shape[1] * 0.009
    row_areas = []
    min_row_separator = 40
    area_start = 0
    area_end = 0
    ongoing_area = False
    separation_count = 0
    for i, row in enumerate(interresting_rows):
        if row:
            separation_count = 0
            area_end = i
            if not ongoing_area:
                area_start = i
                ongoing_area = True
        elif ongoing_area:
            separation_count += 1
            if separation_count == min_row_separator:
                ongoing_area = False
                row_areas.append((area_start, area_end + 1))
    
    if ongoing_area:
        row_areas.append((area_start, area_end + 1))
    
    areas = []
    min_col_separator = image.shape[0] * 0.15
    area_start = 0
    area_end = 0
    ongoing_area = False
    separation_count = 0
    for row1, row2 in row_areas:
        for i, col in enumerate(image[row1:row2].sum(axis=0) > (row2 - row1) * 0.025):
            if col:
                area_end = i
                separation_count = 0
                if not ongoing_area:
                    area_start = i
                    ongoing_area = True
            elif ongoing_area:
                separation_count += 1
                if separation_count == min_col_separator:
                    ongoing_area = False
                    areas.append((row1, row2, area_start, area_end + 1))
        
        if ongoing_area:
            areas.append((row1, row2, area_start, area_end + 1))

    equations = []
    for row1, row2, col1, col2 in areas:
        area = image[row1:row2, col1:col2]
        area_sum = area.sum()
        area_max = area.shape[0] * area.shape[1]
        if area_sum > area_max * 0.015 and area_sum < area_max * 0.25:
            final_images = np.zeros((PREDICTION_SAMPLES, 38, 288))
            for i, (y1, y2) in enumerate([(0, 38), (1, 37), (2, 36), (3, 35)]):
                resized_image = Image.fromarray((area * 255).astype(np.uint8), 'L')
                resized_image.thumbnail((resized_image.width, y2 - y1), Image.NEAREST)
                resized_image = np.asarray(resized_image)
                resized_image = resized_image[:, :288]
                width_shift = (EQUATION_IMAGE_WIDTH - resized_image.shape[1]) // 2
                for j, shift in enumerate(range(-8, 8)):
                    augmented_width_shift = width_shift + shift
                    if augmented_width_shift < 0:
                        augmented_width_shift = 0
                    elif augmented_width_shift + resized_image.shape[1] > EQUATION_IMAGE_WIDTH:
                        augmented_width_shift = EQUATION_IMAGE_WIDTH - resized_image.shape[1]
                    final_images[i*16 + j, y1:y2, augmented_width_shift:resized_image.shape[1] + augmented_width_shift] = resized_image

            samples = torch.tensor((final_images > 0).astype(np.float32))
            samples = samples.unsqueeze(1) # (64, 1, 288, 38) shape
            predictions = model(samples)   # 64 predictions

            # ---- take 'area' to do your prediction (you might need to resize it)

            classifications = [None] * PREDICTION_SAMPLES
            for i, _ in enumerate(samples):
                j = i * YOLO_LABELS_PER_IMAGE
                classifications[i] = label_extractors.yolo_prediction_only_class(predictions[j:j + YOLO_LABELS_PER_IMAGE], sep='')

            # ---- append results to 'classifications' 
            filtered_classifications = [ classified for classified in classifications if re.match(r"^(\d+[\+\-\*/])+\d+$", classified) ] # strings with syntactically valid equations

            try:
                # find the most same results
                equations.append(max(filtered_classifications, key=lambda x: sum([x == y for y in filtered_classifications])))
            except:
                continue
    
    return equations, original_image # prediction of equations detected on the image, the original not processed image

def select_file(model, objects):
    for tk_object in objects[0]:
        tk_object.destroy()
    objects[0] = []

    filetypes = [("images", "*.jpg"), ("images", "*.png")]
    filename = fd.askopenfilename(title="Choose an image", initialdir='~/', filetypes=filetypes)
    equations, image = extract_equations(model, filename)

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
    model = CustomCNNv3(batch_size=PREDICTION_SAMPLES, device="cpu")
    model.load()
    model = model.eval()

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
    open_button = ttk.Button(root, text="select", command=lambda:select_file(model, tk_objects))

    select_file_label.grid(row=0, columnspan=4, pady=5)
    open_button.grid(row=1, columnspan=4, pady=5)

    root.mainloop()
