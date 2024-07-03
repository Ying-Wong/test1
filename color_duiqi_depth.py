import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_point_cloud_from_depth_image(depth_image, intrinsics, scale=1.0):
    height, width = depth_image.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image * scale  # 深度图像可能需要一个比例因子，取决于单位
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)
    return points


def transform_point_cloud_to_rgb_frame(point_cloud, rotation_matrix, translation_vector):
    points_homogeneous = np.hstack(
        (point_cloud.reshape(-1, 3), np.ones((point_cloud.shape[0] * point_cloud.shape[1], 1))))
    transformed_points = np.dot(points_homogeneous, np.hstack((rotation_matrix, translation_vector.reshape(3, 1))).T)
    return transformed_points.reshape(point_cloud.shape)


def project_point_cloud_to_image(point_cloud, intrinsics):
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x = point_cloud[:, :, 0]
    y = point_cloud[:, :, 1]
    z = point_cloud[:, :, 2]

    u = (x * fx / z + cx).astype(int)
    v = (y * fy / z + cy).astype(int)

    return u, v


def align_color_and_depth(color_image, depth_image, depth_intrinsics, rgb_intrinsics, rotation_matrix,
                          translation_vector, depth_scale=1.0):
    # Resize depth image to match color image resolution
    depth_image_resized = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]))

    point_cloud = load_point_cloud_from_depth_image(depth_image_resized, depth_intrinsics, scale=depth_scale)
    transformed_points = transform_point_cloud_to_rgb_frame(point_cloud, rotation_matrix, translation_vector)
    u, v = project_point_cloud_to_image(transformed_points, rgb_intrinsics)

    # 将深度图像中的点云投影到彩色图像上
    aligned_color_image = np.zeros_like(color_image)
    aligned_depth_image = np.zeros_like(depth_image_resized)

    valid_indices = (u >= 0) & (u < color_image.shape[1]) & (v >= 0) & (v < color_image.shape[0]) & (
                transformed_points[:, :, 2] > 0)

    # Ensure indices are within bounds
    u = np.clip(u, 0, color_image.shape[1] - 1)
    v = np.clip(v, 0, color_image.shape[0] - 1)

    aligned_color_image[v[valid_indices], u[valid_indices]] = color_image[v[valid_indices], u[valid_indices]]
    aligned_depth_image[v[valid_indices], u[valid_indices]] = transformed_points[valid_indices][:, 2] / depth_scale

    return aligned_color_image, aligned_depth_image


# 文件路径
depth_image_path = "ProcessedFrames/514_2/depth_image_530.png"  # 替换为你的深度图像路径
color_image_path = "ProcessedFrames/514_2/color_image_530.png"  # 替换为你的彩色图像路径

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

# 读取深度图像和彩色图像
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
# Convert pseudo-color depth image to single channel grayscale
if len(depth_image.shape) == 3 and depth_image.shape[2] == 3:
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
color_image = cv2.imread(color_image_path)
color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# 对齐彩色图像和深度图像
aligned_color_image, aligned_depth_image = align_color_and_depth(color_image_rgb, depth_image, ir_intrinsics,
                                                                 rgb_intrinsics, rotation_matrix, translation_vector,
                                                                 depth_scale=0.1)

# 保存对齐后的图像
cv2.imwrite('aligned_color_image.png', cv2.cvtColor(aligned_color_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('aligned_depth_image.png', (aligned_depth_image * 255 / np.max(aligned_depth_image)).astype(np.uint8))

# 可视化对齐结果
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Original Color Image')
plt.imshow(color_image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Aligned Color Image')
plt.imshow(aligned_color_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Aligned Depth Image')
plt.imshow(aligned_depth_image, cmap='gray')
plt.axis('off')

plt.show()
