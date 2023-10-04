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


model_ir = ov.convert_model("u2net.onnx")
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
    # Convert the image shape to a shape and a data type expected by the network
    # for OpenVINO IR model: (1, 3, 512, 512).
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

    blur_radius = 15
    mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

    blurred_image = cv2.GaussianBlur(image, (51, 51), 0)

    image_c = image.copy()

    image_c[mask == 0] = blurred_image[mask == 0]
    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)

    output_filename = os.path.join(output_directory, os.path.basename(IMAGE))
    image_c_rgb = cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_filename, image_c_rgb)

    print(f"Processed: {IMAGE}")
