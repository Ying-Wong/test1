import os
import numpy as np
import cv2
import open3d as o3d

# 文件路径
raw_files_dir = r'C:\Users\Wong_\Downloads\OrbbecViewer for OpenNI2_v1.1.13_20220722_windows_x64\OrbbecViewer_1.1.13_202207221538_Windows\CapturedFrames'

# 假设深度图像尺寸为640x400，每个像素的格式为16位无符号整数
depth_image_width = 640
depth_image_height = 400
depth_data_type = np.uint16

# 假设彩色图像尺寸为1280x720，每个像素的格式为RGB（8位无符号整数）
color_image_width = 1280
color_image_height = 720
color_channels = 3
color_data_type = np.uint8

# 创建保存目录
output_dir = os.path.join(raw_files_dir, 'ProcessedFrames')
os.makedirs(output_dir, exist_ok=True)

# 准备深度图像和彩色图像列表
depth_images = []
color_images = []
files = sorted(os.listdir(raw_files_dir))

# 处理深度和彩色图像文件
for filename in files:
    if filename.endswith('.raw'):
        file_path = os.path.join(raw_files_dir, filename)
        output_file_base = os.path.join(output_dir, os.path.splitext(filename)[0])

        # 判断是深度图像还是彩色图像
        if 'Depth' in filename:
            # 读取原始深度数据
            with open(file_path, 'rb') as f:
                raw_data = f.read()

            # 将原始数据转换为numpy数组
            depth_image = np.frombuffer(raw_data, dtype=depth_data_type)
            depth_image = depth_image.reshape((depth_image_height, depth_image_width))

            depth_images.append(depth_image)

            # 保存深度图像
            depth_image_path = output_file_base + '.png'
            cv2.imwrite(depth_image_path, depth_image)

        elif 'Color' in filename:
            # 读取原始彩色数据
            with open(file_path, 'rb') as f:
                raw_color_data = f.read()

            # 将原始数据转换为numpy数组
            color_image = np.frombuffer(raw_color_data, dtype=color_data_type)
            color_image = color_image.reshape((color_image_height, color_image_width, color_channels))

            color_images.append(color_image)

            # 保存彩色图像
            color_image_path = output_file_base + '.png'
            cv2.imwrite(color_image_path, color_image)

        # 每5帧进行一次三维重建
        if len(depth_images) == 5 and len(color_images) == 5:
            # 执行三维重建
            point_clouds = []

            for i in range(5):
                depth = depth_images[i]
                color = color_images[i]

                # 调整彩色图像大小以匹配深度图像大小
                color_resized = cv2.resize(color, (depth_image_width, depth_image_height))

                # 创建RGBD图像
                depth_o3d = o3d.geometry.Image(depth)
                color_o3d = o3d.geometry.Image(cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB))
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, convert_rgb_to_intensity=False
                )

                # 相机内参
                pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    depth_image_width, depth_image_height, 476.3, 476.3, 320.309, 198.168
                )

                # 生成点云
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, pinhole_camera_intrinsic
                )
                point_clouds.append(pcd)

            # 合并点云
            combined_pcd = point_clouds[0]
            for pcd in point_clouds[1:]:
                combined_pcd += pcd

            # 保存点云
            point_cloud_path = os.path.join(output_dir, f'combined_point_cloud_{filename[:-4]}.ply')
            o3d.io.write_point_cloud(point_cloud_path, combined_pcd)
            print(f'Saved combined point cloud: {point_cloud_path}')

            # 清空图像列表，准备下一次重建
            depth_images = []
            color_images = []

print('Processing completed.')
