import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import os
import numpy as np
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "network"))
from network.custom_CNN_v1 import CustomCNNv1
import label_extractors

def extract_equations(model, image_filename):
    original_image = Image.open(image_filename).convert('L')
    image = np.asarray(original_image)
    if image.sum() * 2 > image.shape[0] * image.shape[1] * 255:
        image = np.vectorize(lambda x: 0 if x >= 128 else 1)(image)
    
    else:
        image = np.vectorize(lambda x: 1 if x >= 128 else 0)(image)

    interresting_rows = image.sum(axis=1) > image.shape[0] * 0.005
    row_areas = []
    min_row_separator = image.shape[0] * 0.025
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
    min_col_separator = image.shape[1] * 0.15
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
        if area_sum > area_max * 0.025 and area_sum < area_max * 0.2:
            resized_image = Image.fromarray((area * 255).astype(np.uint8), 'L')
            resized_image.thumbnail((resized_image.width, 30), Image.NEAREST)
            final_image = np.zeros((38, 288))
            width_shift = (288 - resized_image.width) // 2
            try:
                final_image[4:34, width_shift:resized_image.width + width_shift] = np.asarray(resized_image)
            except:
                continue

            sample = torch.tensor((final_image > 0).astype(np.float32))
            sample = sample.unsqueeze(0).unsqueeze(0)
            prediction = model(sample)
    
            equations.append(label_extractors.yolo_prediction_only_class(prediction))
    
    return equations, original_image

def select_file(model, objects):
    for tk_object in objects["objects"]:
        tk_object.destroy()
    objects["objects"] = []

    filetypes = [("images", "*.jpg"), ("images", "*.png")]
    filename = fd.askopenfilename(title="Choose an image", initialdir='~/', filetypes=filetypes)
    equations, image = extract_equations(model, filename)

    image.thumbnail((800, 500))
    image = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=image)
    image_label.image = image
    image_label.grid(row=2, columnspan=2)
    objects["objects"].append(image_label)

    equation_count = len(equations)
    counter = 1
    for equation in equations:
        if not len(equation):
            continue

        try:
            result = eval(equation)
        except:
            result = "error"
        
        equation_input = tk.Entry(root)
        equation_input.insert(0, equation)
        equation_label = tk.Label(root, text="Equation:" if equation_count == 1 else f"Equation {counter}:")
        result_label = tk.Label(root, text="Result:" if equation_count == 1 else f"Result {counter}:")
        result_value = tk.Label(root, text=result)

        row = counter * 2
        equation_label.grid(row=row + 1, column=0, pady=5)
        equation_input.grid(row=row + 1, column=1, pady=5)
        result_label.grid(row=row + 2, column=0, pady=5)
        result_value.grid(row=row + 2, column=1, pady=5)
        counter += 1

        objects["objects"].append(equation_input)
        objects["objects"].append(equation_label)
        objects["objects"].append(result_label)
        objects["objects"].append(result_value)

if __name__ == "__main__":
    model = CustomCNNv1()
    model.load()
    model = model.eval()

    tk_objects = {"objects" : []}
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

    select_file_label.grid(row=0, columnspan=2, pady=5)
    open_button.grid(row=1, columnspan=2, pady=5)

    root.mainloop()

# /home/david/projs/visual_calc/data/equation_images/IMG_20221222_221848_3.jpg
