import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载图像
image_path = 'data/01_colors_image_7.png'
image = cv2.imread(image_path)

# 裁剪图像，只保留黑色背景板部分
# 假设黑色背景板的范围已知，可以手动调整这些值
x, y, w, h = 50, 50, 300, 400  # 示例值，请根据实际情况调整
cropped_image = image[y:y+h, x:x+w]

# 转换为灰度图像
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# 应用二值化阈值分割植物
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建与原始图像大小相同的掩码
mask = np.zeros_like(cropped_image)

# 绘制最大轮廓
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# 使用掩码从原始图像中提取植物
result = cv2.bitwise_and(cropped_image, mask)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('裁剪后的图像')
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('分割后的植物')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.show()
