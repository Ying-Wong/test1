import cv2
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
raw_file_path = 'Depth_1719406077095_3.raw'

# 假设图像尺寸为640x400，像素深度值为16位无符号整数
image_width = 640
image_height = 400
data_type = np.uint16

# 读取原始数据
with open(raw_file_path, 'rb') as f:
    raw_data = f.read()

# 将原始数据转换为numpy数组
depth_image = np.frombuffer(raw_data, dtype=data_type)
depth_image = depth_image.reshape((image_height, image_width))

# 保存彩色图像
cv2.imwrite('data/depth_image.png', depth_image)



