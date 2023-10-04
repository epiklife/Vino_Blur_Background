import os
import time
import sys
from collections import namedtuple
from pathlib import Path
import ipywidgets as widgets

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import torch
from IPython.display import HTML, FileLink, display


utils_file_path = Path("notebook_utils.py")


sys.path.append(str(utils_file_path.parent))

from notebook_utils import load_image


model_ir = ov.convert_model("u2netp.onnx")
input_folder = "input"

# Get a list of all .jpg files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

# Loop through each image file in the folder
for image_file in image_files:
    # Construct the full path to the image
    IMAGE = os.path.join(input_folder, image_file)

    input_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    input_scale = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

    image = cv2.cvtColor(
        src=load_image(IMAGE),
        code=cv2.COLOR_BGR2RGB,
    )

    resized_image = cv2.resize(src=image, dsize=(512, 512))

    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

    input_image = (input_image - input_mean) / input_scale

    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )

    compiled_model_ir = core.compile_model(model=model_ir, device_name=device.value)
    input_layer_ir = compiled_model_ir.input(0)
    output_layer_ir = compiled_model_ir.output(0)

    result = compiled_model_ir([input_image])[output_layer_ir]

    mask = np.rint(
        cv2.resize(src=np.squeeze(result), dsize=(image.shape[1], image.shape[0]))
    ).astype(np.uint8)

    mask_blur_radius = 101
    blur_intensity = 105
    blur_padding = 5

    mask = cv2.GaussianBlur(
        mask * 255, (mask_blur_radius, mask_blur_radius), blur_padding
    )

    blurred_image = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)

    image_c = image.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_c[mask == 0] = blurred_image[mask == 0]
    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)

    output_filename = os.path.join(output_directory, os.path.basename(IMAGE))
    image_c_rgb = cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB)

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cv2.rectangle(image_c_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(output_filename, image_c_rgb)

    print(f"Processed: {IMAGE}")
