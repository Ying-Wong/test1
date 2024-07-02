import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# 读取TXT格式的点云文件
def read_point_cloud_from_txt(filename):
    data = np.loadtxt(filename)
    points = data[:, :3]  # 仅使用前三列作为点云坐标
    return points


# 保存骨架化后的点云
def save_point_cloud_to_txt(points, filename):
    np.savetxt(filename, points, fmt='%f %f %f')


# 计算主轴
def calculate_principal_axis(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    principal_axis = pca.components_[0]
    return principal_axis


# 计算两条向量之间的夹角
def calculate_angle_between_vectors(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    return angle_degrees


# 可视化点云
def visualize_point_clouds(stem_points, leaf_points):
    stem_pcd = o3d.geometry.PointCloud()
    stem_pcd.points = o3d.utility.Vector3dVector(stem_points)
    stem_pcd.paint_uniform_color([1, 0, 0])  # 茎秆点云为红色

    leaf_pcd = o3d.geometry.PointCloud()
    leaf_pcd.points = o3d.utility.Vector3dVector(leaf_points)
    leaf_pcd.paint_uniform_color([0, 1, 0])  # 叶片点云为绿色

    o3d.visualization.draw_geometries([stem_pcd, leaf_pcd],
                                      window_name="Point Cloud Visualization",
                                      width=800, height=600)


# 文件路径
input_filename = 'combined_point_cloud_skeleton3.txt'

# 读取骨架化后的点云
points = read_point_cloud_from_txt(input_filename)

# 根据X轴变化范围分离叶片和茎秆的点云
x_range = np.ptp(points[:, 0])  # 计算点云在X轴上的变化范围
threshold = x_range * 0.1  # 设置一个阈值，比如总变化范围的10%

# 计算每个点云分片的X轴范围
leaf_points = []
stem_points = []

# 遍历点云并分离茎秆和叶片
for point in points:
    if np.ptp(points[:, 0]) < threshold:
        stem_points.append(point)
    else:
        leaf_points.append(point)

# 将分离后的点云转换为numpy数组
stem_points = np.array(stem_points)
leaf_points = np.array(leaf_points)

# 确保分离后的点云不为空
if stem_points.size == 0 or leaf_points.size == 0:
    raise ValueError("Failed to separate stem and leaf points. Please check the separation criteria.")

# 计算主轴
leaf_axis = calculate_principal_axis(leaf_points)
stem_axis = calculate_principal_axis(stem_points)

# 计算叶片与茎秆的角度
angle = calculate_angle_between_vectors(leaf_axis, stem_axis)
print(f"Angle between leaf and stem: {angle:.2f} degrees")

# 可视化点云
visualize_point_clouds(stem_points, leaf_points)
