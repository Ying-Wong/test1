import numpy as np
import cv2
import matplotlib.pyplot as plt

# 文件路径
raw_color_file_path = 'Color_1719406077096_3.raw'

# 假设图像尺寸为640x480，每个像素的格式为RGB（8位无符号整数）
color_image_width = 1280
color_image_height = 720
color_channels = 3
color_data_type = np.uint8

# 读取原始数据
with open(raw_color_file_path, 'rb') as f:
    raw_color_data = f.read()

# 将原始数据转换为numpy数组
color_image = np.frombuffer(raw_color_data, dtype=color_data_type)
color_image = color_image.reshape((color_image_height, color_image_width, color_channels))
# 保存彩色图像
# cv2.imwrite('color_image.png', color_image)

# 显示并保存彩色图像
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.title('Color Image')
# plt.savefig('data/color_image.png')  # 保存彩色图像
plt.show()

# 打印彩色图像的基本信息
print(f"Color image shape: {color_image.shape}")
print(f"Color image data type: {color_image.dtype}")

# 保存彩色图像为.npy文件
# np.save('data/color_image.npy', color_image)
