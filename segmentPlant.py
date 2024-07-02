import cv2
import numpy as np
import matplotlib.pyplot as plt

# 利用边缘检测进行图像的分割2024.7.1

# 加载图像
image_path = "01_colors_image_7.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 应用形态学操作来填充轮廓内部
kernel = np.ones((5, 5), np.uint8)
mask_dilated = cv2.dilate(edges, kernel, iterations=2)
mask_filled = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

# 找到轮廓
contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个空白图像用于绘制轮廓
output = np.zeros_like(image_rgb)

# 绘制轮廓
cv2.drawContours(output, contours, -1, (0, 255, 0), 3)

# 在原图上应用掩码
result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_filled)

# 显示原图、边缘检测结果、掩码和最终结果
plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Dilated Mask')
plt.imshow(mask_dilated, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Filled Mask')
plt.imshow(mask_filled, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Contour Drawing')
plt.imshow(output)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Segmented Plant')
plt.imshow(result)
plt.axis('off')

plt.show()
