import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 文件路径
raw_files_dir = r'C:\Users\Wong_\Downloads\OrbbecViewer for OpenNI2_v1.1.13_20220722_windows_x64\OrbbecViewer_1.1.13_202207221538_Windows\CapturedFrames'

# 假设深度图像尺寸为640x400，每个像素的格式为16位无符号整数
depth_image_width = 640
depth_image_height = 400
depth_data_type = np.uint16

# 假设彩色图像尺寸为640x480，每个像素的格式为RGB（8位无符号整数）
color_image_width = 1280
color_image_height = 720
color_channels = 3
color_data_type = np.uint8

# 创建保存目录
output_dir = os.path.join(raw_files_dir, 'ProcessedFrames')
os.makedirs(output_dir, exist_ok=True)

# 处理深度和彩色图像文件
for filename in os.listdir(raw_files_dir):
    if filename.endswith('.raw'):
        file_path = os.path.join(raw_files_dir, filename)
        output_file_base = os.path.join(output_dir, os.path.splitext(filename)[0])

        # 判断是深度图像还是彩色图像
        if 'Depth' in filename:
            # 读取原始深度数据
            with open(file_path, 'rb') as f:
                raw_data = f.read()

            # 将原始数据转换为numpy数组
            depth_image = np.frombuffer(raw_data, dtype=depth_data_type)
            depth_image = depth_image.reshape((depth_image_height, depth_image_width))

            # 保存深度图像
            depth_image_path = output_file_base + '.png'
            cv2.imwrite(depth_image_path, depth_image)

            print(f'Saved depth image: {depth_image_path}')

        elif 'Color' in filename:
            # 读取原始彩色数据
            with open(file_path, 'rb') as f:
                raw_color_data = f.read()

            # 将原始数据转换为numpy数组
            color_image = np.frombuffer(raw_color_data, dtype=color_data_type)
            color_image = color_image.reshape((color_image_height, color_image_width, color_channels))

            # 保存彩色图像
            color_image_path = output_file_base + '.png'
            cv2.imwrite(color_image_path, color_image)

            print(f'Saved color image: {color_image_path}')
