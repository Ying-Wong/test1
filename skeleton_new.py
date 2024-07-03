import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors


def load_point_cloud(filename):
    data = np.loadtxt(filename)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0  # 将颜色值转换为0-1范围
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def skeletonize_point_cloud(pcd):
    # 下采样点云
    voxel_size = 0.05
    down_pcd = pcd.voxel_down_sample(voxel_size)

    # 法线估计
    down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 提取点云骨架
    distances = down_pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    # 使用邻域聚类法进行骨架化
    points = np.asarray(down_pcd.points)
    neigh = NearestNeighbors(radius=radius)
    neigh.fit(points)
    A = neigh.radius_neighbors_graph(points).toarray()
    A = (A + A.T) / 2  # 确保邻接矩阵是对称的

    # 基于邻接矩阵提取骨架
    skeleton_points = []
    for i in range(points.shape[0]):
        if np.sum(A[i]) < 6:  # 如果点的邻居少于6个，认为是骨架点
            skeleton_points.append(points[i])

    skeleton_points = np.array(skeleton_points)
    skeleton = o3d.geometry.PointCloud()
    skeleton.points = o3d.utility.Vector3dVector(skeleton_points)
    return skeleton


def save_skeleton_to_txt(skeleton, filename):
    points = np.asarray(skeleton.points)
    colors = np.zeros(points.shape)  # 颜色信息可以忽略或设为默认值
    data = np.hstack((points, colors))
    np.savetxt(filename, data, fmt='%f %f %f %d %d %d')


# 加载点云文件
pcd = load_point_cloud('segmented_point_cloud_607.txt')

# 骨架化点云
skeleton = skeletonize_point_cloud(pcd)

# 保存骨架化后的点云到TXT文件
save_skeleton_to_txt(skeleton, 'skeleton_point_cloud.txt')

# 可视化原始点云和骨架化后的点云
o3d.visualization.draw_geometries([pcd, skeleton])
