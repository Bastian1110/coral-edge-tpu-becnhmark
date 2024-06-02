import cv2

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


interpreter = make_interpreter(model_path_or_content="./ssd_mobilenet_v2_cpu.tflite")
interpreter.allocate_tensors()

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
    cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
    run_inference(interpreter, cv2_im_rgb.tobytes())
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

