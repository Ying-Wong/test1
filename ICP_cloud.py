import os
import numpy as np
import cv2
import pandas as pd
from pyntcloud import PyntCloud
import open3d as o3d

# 文件路径
raw_files_dir = 'ProcessedFrames/2views'


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


def register_point_clouds(source, target):
    threshold = 0.02
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation


# 处理匹配的图像
all_point_clouds = []

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

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed_points.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(cv2.cvtColor(color_image_resized, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0)
    all_point_clouds.append(pcd)

# 配准点云
target = all_point_clouds[0]
combined_points = np.asarray(target.points)
combined_colors = np.asarray(target.colors) * 255

for i in range(1, len(all_point_clouds)):
    source = all_point_clouds[i]
    transformation = register_point_clouds(source, target)
    source.transform(transformation)
    target += source
    combined_points = np.vstack((combined_points, np.asarray(source.points)))
    combined_colors = np.vstack((combined_colors, np.asarray(source.colors) * 255))


# 保存配准后的点云到TXT文件
def save_point_cloud_to_txt(point_cloud, colors, filename):
    points_with_colors = np.hstack((point_cloud, colors))
    np.savetxt(filename, points_with_colors, fmt='%f %f %f %d %d %d')


save_point_cloud_to_txt(combined_points, combined_colors, 'registered_combined_point_cloud.txt')
