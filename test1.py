import os
import numpy as np
import cv2
import pandas as pd
from pyntcloud import PyntCloud

# 文件路径
raw_files_dir = 'ProcessedFrames/603'


# 读取文件名并提取时间戳
def get_files_with_timestamps(dir_path, prefix):
    files = []
    for filename in os.listdir(dir_path):
        if filename.startswith(prefix) and filename.endswith('.png'):
            timestamp = int(filename.split('_')[1])
            files.append((timestamp, os.path.join(dir_path, filename)))
    return sorted(files, key=lambda x: x[0])


# 获取深度图像和彩色图像文件
depth_files = get_files_with_timestamps(raw_files_dir, 'Depth')
color_files = get_files_with_timestamps(raw_files_dir, 'Color')


# 匹配最近的时间戳
matched_files = []
for depth_timestamp, depth_file in depth_files:
    closest_color_file = min(color_files, key=lambda x: abs(x[0] - depth_timestamp))
    matched_files.append((depth_file, closest_color_file[1]))
    color_files.remove(closest_color_file)

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


def save_point_cloud_to_ply(point_cloud, colors, filename):
    point_cloud_flat = point_cloud.reshape(-1, 3)
    points_with_colors = np.hstack((point_cloud_flat, colors))
    cloud = PyntCloud(pd.DataFrame(
        points_with_colors,
        columns=["x", "y", "z", "red", "green", "blue"]
    ))
    cloud.to_file(filename)


def save_point_cloud_to_txt(point_cloud, colors, filename):
    point_cloud_flat = point_cloud.reshape(-1, 3)
    points_with_colors = np.hstack((point_cloud_flat, colors))
    np.savetxt(filename, points_with_colors, fmt='%f %f %f %d %d %d')


# 处理匹配的图像
all_points = []
all_colors = []

for depth_file, color_file in matched_files:
    depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    color_image = cv2.imread(color_file)

    # 检查深度图像的形状，如果是伪彩色的，需要转换为灰度图像
    if len(depth_image.shape) == 3:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

    # 调整彩色图像的尺寸以匹配深度图像
    color_image_resized = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]))

    # 生成点云并转换
    point_cloud = depth_to_point_cloud(depth_image, ir_intrinsics)
    transformed_points = transform_point_cloud(point_cloud, rotation_matrix, translation_vector)

    # 展平并添加颜色数据
    all_points.append(transformed_points.reshape(-1, 3))
    all_colors.append(cv2.cvtColor(color_image_resized, cv2.COLOR_BGR2RGB).reshape(-1, 3))

# 合并所有点云和颜色
combined_points = np.vstack(all_points)
combined_colors = np.vstack(all_colors)

# 确保点云和颜色数组形状一致
assert combined_points.shape[0] == combined_colors.shape[
    0], "Point cloud and color arrays must have the same number of points."

# 保存原始合并的点云
# save_point_cloud_to_ply(combined_points, combined_colors, '/mnt/data/combined_point_cloud.ply')

# 保存原始合并的点云到TXT文件
save_point_cloud_to_txt(combined_points, combined_colors, 'combined_point_cloud_10.txt')

# 创建绿色版本的颜色数组
# green_colors = np.zeros_like(combined_colors)
# green_colors[:, 1] = 255  # 设置绿色通道为255，其他通道为0

# 保存绿色版本的点云
# save_point_cloud_to_ply(combined_points, green_colors, 'combined_point_cloud_green.ply')

# 保存绿色版本的点云到TXT文件
# save_point_cloud_to_txt(combined_points, green_colors, '/mnt/data/combined_point_cloud_green.txt')
