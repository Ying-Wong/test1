import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN


def load_point_cloud(filename):
    data = np.loadtxt(filename)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0  # 将颜色值转换为0-1范围
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def estimate_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


def save_point_cloud_with_normals(pcd, filename):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    colors = np.asarray(pcd.colors) * 255.0
    data = np.hstack((points, normals, colors))
    np.savetxt(filename, data, fmt='%f %f %f %f %f %f %d %d %d')


def cluster_point_cloud(pcd, eps=0.02, min_samples=10):
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)

    clusters = []
    for k in unique_labels:
        if k == -1:  # 忽略噪声点
            continue
        class_member_mask = (labels == k)
        cluster_points = points[class_member_mask]
        cluster_colors = np.asarray(pcd.colors)[class_member_mask]

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)
        clusters.append(cluster_pcd)
    return clusters


# 加载点云文件
pcd = load_point_cloud('segmented_point_cloud_607.txt')

# 可视化原始点云
o3d.visualization.draw_geometries([pcd], window_name='Original Point Cloud')

# 聚类点云以分离叶片和茎
clusters = cluster_point_cloud(pcd)

# 对每个簇进行法向量估计并保存
for i, cluster in enumerate(clusters):
    cluster = estimate_normals(cluster)
    save_point_cloud_with_normals(cluster, f'cluster_{i}_point_cloud_with_normals.txt')
    o3d.visualization.draw_geometries([cluster, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)],
                                      window_name=f'Cluster {i} Normals Visualization')
