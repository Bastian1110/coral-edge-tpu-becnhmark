import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file



interpreter = tf.lite.Interpreter(model_path="./ssd_mobilenet_v2_cpu.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = read_label_file("coco_labels.txt")
inference_size = input_size(interpreter)


import os
from time import time

directory = 'images'
file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
total_files = len(file_list)

print(f"Starting benchmark for {total_files} files ...")

# Initialize total time
total_time = 0

for i, filename in enumerate(file_list):
    file_path = os.path.join(directory, filename)
    
    start_time = time()

    image = cv2.imread(file_path)
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
    input_image = cv2.resize(image, (width, height))
    input_image = np.expand_dims(input_image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_image.astype(np.uint8))
    interpreter.invoke()

    get_objects(interpreter, 0.1)[:3]

    end_time = time()
    
    # Calculate time for this document and add to total time
    document_time = end_time - start_time
    total_time += document_time
    
    percentage = ((i + 1) / total_files) * 100
    print(f"\rProcessing file: {file_path} ({percentage:.2f}%)", end='')

print("\nAll files processed.")

# Convert total time to milliseconds
total_time_ms = total_time * 1000
average_time_ms = (total_time / total_files) * 1000

print(f"Total execution time: {total_time_ms:.2f} milliseconds")
print(f"Average time per document: {average_time_ms:.2f} milliseconds")

