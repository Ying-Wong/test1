import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from pyntcloud import PyntCloud
import open3d as o3d

# 文件路径
depth_image_path = "7_3_png/Depth_1720008704844_4.png"  # 替换为你的深度图像路径
color_image_path = "7_3_png/Color_1720008704399_5.png"  # 替换为你的彩色图像路径

# IR 相机内参
ir_intrinsics = {
    'fx': 476.3,
    'fy': 476.3,
    'cx': 320.309,
    'cy': 198.168
}

# RGB 相机内参
rgb_intrinsics = {
    'fx': 451.886,
    'fy': 451.886,
    'cx': 324.326,
    'cy': 240.192
}

# 相机外参
rotation_matrix = np.array([
    [0.999998, 0.00176565, 0.000669865],
    [-0.00176923, 0.999984, 0.0053853],
    [-0.000660345, -0.00538647, 0.999985]
])

translation_vector = np.array([-10.198, -0.146565, -0.58615])


def depth_to_point_cloud(depth_image, intrinsics):
    height, width = depth_image.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image / 1000.0  # 假设深度图像以毫米为单位
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)
    return points


def transform_point_cloud(point_cloud, rotation_matrix, translation_vector):
    points_homogeneous = np.hstack(
        (point_cloud.reshape(-1, 3), np.ones((point_cloud.shape[0] * point_cloud.shape[1], 1))))
    transformed_points = np.dot(points_homogeneous, np.hstack((rotation_matrix, translation_vector.reshape(3, 1))).T)
    return transformed_points.reshape(point_cloud.shape)


def register_point_clouds(source, target):
    threshold = 0.02
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation


def save_point_cloud_to_txt(point_cloud, colors, filename):
    point_cloud_flat = point_cloud.reshape(-1, 3)
    points_with_colors = np.hstack((point_cloud_flat, colors))
    np.savetxt(filename, points_with_colors, fmt='%f %f %f %d %d %d')


# 读取深度图像和彩色图像
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
color_image = cv2.imread(color_image_path)
image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# 调整彩色图像的尺寸以匹配深度图像
color_image_resized = cv2.resize(image_rgb, (depth_image.shape[1], depth_image.shape[0]))

# 转换为HSV颜色空间
hsv = cv2.cvtColor(color_image_resized, cv2.COLOR_RGB2HSV)

lower_green = np.array([52, 96, 0])
upper_green = np.array([179, 255, 255])
# 创建绿色范围的掩码
mask = cv2.inRange(hsv, lower_green, upper_green)

# 应用形态学操作来去噪
kernel = np.ones((5, 5), np.uint8)
mask_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask_closing = cv2.morphologyEx(mask_opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# 将掩码转换为8位
mask_closing_8u = mask_closing.astype(np.uint8)

# 仅对分割出的植物部分进行处理
segmented_color = cv2.bitwise_and(color_image_resized, color_image_resized, mask=mask)

# 生成点云并转换
point_cloud = depth_to_point_cloud(depth_image.astype(np.float32), ir_intrinsics)
transformed_points = transform_point_cloud(point_cloud, rotation_matrix, translation_vector)

# 展平并添加颜色数据
colors = segmented_color.reshape(-1, 3)

# 过滤掉颜色为黑色的点（即未分割部分）
mask_non_black = np.any(colors != [0, 0, 0], axis=1)
transformed_points_filtered = transformed_points.reshape(-1, 3)[mask_non_black]
colors_filtered = colors[mask_non_black]

# 确保点云和颜色数组形状一致
assert transformed_points_filtered.shape[0] == colors_filtered.shape[
    0], "Point cloud and color arrays must have the same number of points."

# 保存点云到TXT文件
save_point_cloud_to_txt(transformed_points_filtered, colors_filtered, 'segmented_point_cloud_870.txt')

# 可视化点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(transformed_points_filtered)
pcd.colors = o3d.utility.Vector3dVector(colors_filtered / 255.0)

o3d.visualization.draw_geometries([pcd])

# 显示原图、HSV 掩码、分割后的彩色图像和最终结果
plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('HSV Mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Segmented Color Image')
plt.imshow(segmented_color)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Inverted Mask')
plt.imshow(mask_closing, cmap='gray')
plt.axis('off')

plt.show()
