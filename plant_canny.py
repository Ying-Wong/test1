import cv2
import numpy as np
import matplotlib.pyplot as plt
# 利用颜色阈值进行分割 2024.7.1
# 加载图像
image_path = "data/01_colors_image_7.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 转换为HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义找到的绿色范围的阈值
# lower_green = np.array([0, 0, 0])
# upper_green = np.array([129, 62, 255])
lower_green = np.array([127, 63, 0])
upper_green = np.array([179, 255, 255])
# 创建绿色范围的掩码
mask = cv2.inRange(hsv, lower_green, upper_green)

# 结合边缘检测结果和颜色阈值结果
combined_mask = cv2.bitwise_and(mask, edges)

# 反转掩码（使植物为白色，背景为黑色）
inverted_mask = cv2.bitwise_not(mask)

# 应用形态学操作来去噪
kernel = np.ones((5, 5), np.uint8)
mask_opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=2)
mask_closing = cv2.morphologyEx(mask_opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# 找到轮廓
contours, _ = cv2.findContours(mask_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个空白图像用于绘制轮廓
output = np.zeros_like(image_rgb)

# 绘制轮廓
cv2.drawContours(output, contours, -1, (0, 255, 0), 3)

# 在原图上应用掩码
result = cv2.bitwise_and(image_rgb, image_rgb, mask=inverted_mask)


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
plt.title('HSV Mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Combined Mask')
plt.imshow(combined_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Inverted Mask')
plt.imshow(inverted_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Segmented Plant')
plt.imshow(result)
plt.axis('off')

plt.show()
