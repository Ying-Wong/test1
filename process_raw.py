import cv2
import numpy as np
import os

def convert_raw_to_png(input_folder, output_folder):
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".raw"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".raw", ".png"))

            # Determine the image type (color or depth) based on filename or other criteria
            if "Color" in filename:
                # Assuming color images are 640x480 and 3 channels (RGB)
                image = np.fromfile(input_path, dtype=np.uint8).reshape((720, 1280, 3))
            elif "Depth" in filename:
                # Assuming depth images are 640x400 and 1 channel (grayscale)
                image = np.fromfile(input_path, dtype=np.uint16).reshape((400, 640))
            else:
                print(f"Unknown image type for file: {filename}")
                continue

            # Save the image as PNG
            cv2.imwrite(output_path, image)

input_folder = "D:/wsy/626/7_3"  # 请确保此路径正确
output_folder = "D:/wsy/626/7_3_png"

convert_raw_to_png(input_folder, output_folder)
