import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# 读取TXT格式的点云文件
def read_point_cloud_from_txt(filename):
    data = np.loadtxt(filename)
    points = data[:, :3]  # 仅使用前三列作为点云坐标
    return points


# 保存骨架化后的点云
def save_point_cloud_to_txt(points, filename):
    np.savetxt(filename, points, fmt='%f %f %f')


# 骨架化点云
def skeletonize_point_cloud(points):
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 降采样点云（可选）
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # 估计法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 提取骨架（骨架化算法示例）
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    skeleton = o3d.geometry.PointCloud()
    skeleton.points = pcd.points
    skeleton = skeleton.uniform_down_sample(every_k_points=5)

    return np.asarray(skeleton.points)


# 文件路径
input_filename = 'combined_point_cloud - Cloud607SAM.txt'
output_filename = 'combined_point_cloud_skeleton3.txt'

# 读取点云
points = read_point_cloud_from_txt(input_filename)

# 骨架化点云
skeleton_points = skeletonize_point_cloud(points)

# 保存骨架化后的点云
save_point_cloud_to_txt(skeleton_points, output_filename)


# 可视化原始点云和骨架化点云
def visualize_point_clouds(original_points, skeleton_points):
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    original_pcd.paint_uniform_color([1, 0, 0])  # 原始点云为红色

    skeleton_pcd = o3d.geometry.PointCloud()
    skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_points)
    skeleton_pcd.paint_uniform_color([0, 1, 0])  # 骨架点云为绿色

    o3d.visualization.draw_geometries([ skeleton_pcd],
                                      window_name="Point Cloud Visualization",
                                      width=800, height=600)


# 可视化
visualize_point_clouds(points, skeleton_points)

print(f"Skeletonized point cloud saved to {output_filename}")
