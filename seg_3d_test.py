import os
import numpy as np
import cv2
import json
import open3d as o3d
from matplotlib import pyplot as plt

# 文件路径
depth_image_path = "ProcessedFrames/514_1/01_depth_image_7.png"  # 替换为你的深度图像路径
color_image_path = "ProcessedFrames/514_1/01_colors_image_7.png"  # 替换为你的彩色图像路径
json_path = "data/01_colors_image_7.json"  # 替换为你的检测框JSON文件路径

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

# 从 JSON 文件读取检测结果
with open(json_path, 'r') as f:
    detection_results = json.load(f)

# 找到得分最高的检测框
max_score_index = np.argmax(detection_results['scores'])
best_bbox = detection_results['bboxes'][max_score_index]


def depth_to_point_cloud(depth_image, intrinsics, scale=1.0):
    height, width = depth_image.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image * scale  # 深度图像可能需要一个比例因子，取决于单位
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)
    return points


def transform_point_cloud(point_cloud, rotation_matrix, translation_vector):
    points_homogeneous = np.hstack(
        (point_cloud.reshape(-1, 3), np.ones((point_cloud.shape[0] * point_cloud.shape[1], 1))))
    transformed_points = np.dot(points_homogeneous, np.hstack((rotation_matrix, translation_vector.reshape(3, 1))).T)
    return transformed_points.reshape(point_cloud.shape)


def save_point_cloud_to_txt(point_cloud, colors, filename):
    point_cloud_flat = point_cloud.reshape(-1, 3)
    points_with_colors = np.hstack((point_cloud_flat, colors))
    np.savetxt(filename, points_with_colors, fmt='%f %f %f %d %d %d')


# 读取深度图像和彩色图像
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
# Convert pseudo-color depth image to single channel grayscale
if len(depth_image.shape) == 3 and depth_image.shape[2] == 3:
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
color_image = cv2.imread(color_image_path)
image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# 获取原始彩色图像的尺寸
original_color_shape = image_rgb.shape

# 调整彩色图像的尺寸以匹配深度图像
color_image_resized = cv2.resize(image_rgb, (depth_image.shape[1], depth_image.shape[0]))

# 根据调整后的彩色图像尺寸调整检测框尺寸
scale_x = depth_image.shape[1] / original_color_shape[1]
scale_y = depth_image.shape[0] / original_color_shape[0]
x1, y1, x2, y2 = map(int, [best_bbox[0] * scale_x, best_bbox[1] * scale_y, best_bbox[2] * scale_x, best_bbox[3] * scale_y])

# 打印检测框的坐标，检查是否正确
print(f"Original bbox: {best_bbox}")
print(f"Scaled bbox: {x1, y1, x2, y2}")

# 转换为HSV颜色空间
hsv = cv2.cvtColor(color_image_resized, cv2.COLOR_BGR2HSV)

# 定义绿色范围的阈值
lower_green = np.array([127, 63, 0])
upper_green = np.array([179, 255, 255])

# 创建绿色范围的掩码
mask = cv2.inRange(hsv, lower_green, upper_green)

# 反转掩码（使植物为白色，背景为黑色）
inverted_mask = cv2.bitwise_not(mask)

# 仅对得分最高的检测框内进行处理
segmented_color = np.zeros_like(color_image_resized)
mask_roi = inverted_mask[y1:y2, x1:x2]
color_roi = color_image_resized[y1:y2, x1:x2]
segmented_color[y1:y2, x1:x2] = cv2.bitwise_and(color_roi, color_roi, mask=mask_roi)

# 可视化掩码和分割结果以调试
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
plt.title('Inverted Mask')
plt.imshow(inverted_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Segmented Color')
plt.imshow(segmented_color)
plt.axis('off')

plt.show()

# 生成点云并转换
point_cloud = depth_to_point_cloud(depth_image.astype(np.float32), ir_intrinsics, scale=0.1)  # 以100微米为单位，需要除以10000转换为米
transformed_points = transform_point_cloud(point_cloud, rotation_matrix, translation_vector)

# 展平并添加颜色数据
colors = segmented_color.reshape(-1, 3)

# 过滤掉颜色为黑色的点（即未分割部分）
mask_non_black = np.any(colors != [0, 0, 0], axis=1)
transformed_points_filtered = transformed_points.reshape(-1, 3)[mask_non_black]
colors_filtered = colors[mask_non_black]

# 确保点云和颜色数组形状一致
assert transformed_points_filtered.shape[0] == colors_filtered.shape[0], "Point cloud and color arrays must have the same number of points."

# 保存点云到TXT文件
save_point_cloud_to_txt(transformed_points_filtered, colors_filtered, 'segmented_point_cloud_607.txt')

# 创建坐标轴
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# 可视化点云和坐标轴
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(transformed_points_filtered)
pcd.colors = o3d.utility.Vector3dVector(colors_filtered / 255.0)

# 创建可视化窗口并设置视角
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(coordinate_frame)

# 获取相机参数
view_control = vis.get_view_control()
camera_params = view_control.convert_to_pinhole_camera_parameters()

# 设置相机内参
camera_params.intrinsic.set_intrinsics(width=depth_image.shape[1], height=depth_image.shape[0], fx=ir_intrinsics['fx'], fy=ir_intrinsics['fy'], cx=ir_intrinsics['cx'], cy=ir_intrinsics['cy'])

# 将相机参数应用于视图控制
view_control.convert_from_pinhole_camera_parameters(camera_params)

# 渲染并显示
vis.run()
vis.destroy_window()
