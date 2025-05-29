import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import pandas as pd
from scipy.interpolate import interp1d
import shutil

def transform_points(points, translation, rotation_matrix):
    """
    将点从参考坐标系转换到目标坐标系
    
    参数:
    points: numpy数组，形状为(n, 3)，表示n个点在参考坐标系中的坐标
    translation: [x, y, z] 目标坐标系原点在参考坐标系中的位置
    rotation_matrix: 3x3旋转矩阵，表示从参考坐标系到目标坐标系的旋转
    
    返回:
    transformed_points: 点在目标坐标系中的坐标
    """
    # 将点从参考坐标系转换到目标坐标系
    # 公式: P_target = R^T @ (P_reference - T)
    translated_points = points - translation  # 平移
    transformed_points = (rotation_matrix.T @ translated_points.T).T  # 旋转
    
    return transformed_points

def plot_coordinate_system(ax, translation, rotation_matrix, label, color='k'):
    """
    绘制坐标系
    
    参数:
    ax: matplotlib的3D轴对象
    translation: [x, y, z] 坐标系原点在参考坐标系中的位置
    rotation_matrix: 3x3旋转矩阵，表示从参考坐标系到该坐标系的旋转
    label: 坐标系的标签
    color: 坐标轴的颜色
    """
    # 坐标系的基向量
    basis_vectors = np.eye(3)
    
    # 旋转基向量
    rotated_basis = rotation_matrix @ basis_vectors
    
    # 绘制坐标轴
    colors = ['r', 'g', 'b']
    for i in range(3):
        ax.quiver(
            translation[0], translation[1], translation[2],
            rotated_basis[0, i], rotated_basis[1, i], rotated_basis[2, i],
            color=colors[i], length=0.1, normalize=True
        )
    
    # 添加标签
    ax.text(translation[0], translation[1], translation[2], label, fontsize=12, color=color)

def load_marker_positions(file_path='data/microwave/microwave_66.txt', group_size=5):
    """
    读取marker位置数据，使用第一组完整的marker ID作为标准
    如果不匹配的组数超过总组数的20%，则放弃这组数据
    
    Args:
        file_path: 数据文件路径
        group_size: 每组期望的marker数量（默认为5）
    Returns:
        tuple: (reference_marker_ids, valid_groups, is_valid_data)
        reference_marker_ids: 第一组有效的marker ID列表
        valid_groups: 符合标准的数据组列表
        is_valid_data: 布尔值，表示这组数据是否有效
    """
    valid_groups = []
    current_group = {'group_id': 0, 'markers': {}}
    reference_marker_ids = None
    mismatch_count = 0
    total_groups = 0
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                if i + 2 < len(lines):
                    try:
                        # 读取时间和marker ID
                        time = float(lines[i].strip())
                        marker_id = int(lines[i+1].strip())
                        
                        # 处理位置数据行
                        pos_line = lines[i+2].strip()
                        # 提取方括号中的数字
                        if 'pos:' in pos_line:
                            # 找到方括号内的内容
                            start = pos_line.find('[') + 1
                            end = pos_line.find(']')
                            if start > 0 and end > start:
                                pos_str = pos_line[start:end]
                                # 分割为三个坐标值
                                pos = np.array([float(x.strip()) for x in pos_str.split(',')])
                            else:
                                i += 1
                                continue
                        else:
                            # 分割为三个坐标值
                            pos = np.array([float(x) for x in pos_line.split()])
                        
                        # 验证位置数据是否包含3个坐标
                        if len(pos) != 3:
                            i += 1
                            continue
                            
                        # 对xyz分别减去不同的值
                        pos[0] -= 0.01  # x坐标减去0.001
                        pos[1] -= 0.04  # y坐标减去0.002
                        pos[2] -= 0.01  # z坐标减去0.003
                        
                        # 然后再除以2
                        pos = pos / 2
                        
                        # 添加到当前组
                        current_group['markers'][marker_id] = {
                            'time': time,
                            'position': pos
                        }
                        
                        # 当前组收集满group_size个点时进行处理
                        if len(current_group['markers']) == group_size:
                            total_groups += 1
                            current_marker_ids = set(current_group['markers'].keys())
                            
                            # 如果还没有参考marker ID组，则将第一组完整的数据作为参考
                            if reference_marker_ids is None:
                                reference_marker_ids = sorted(list(current_marker_ids))
                                print(f"已确定参考marker IDs: {reference_marker_ids}")
                                valid_groups.append(current_group)
                            
                            # 否则检查当前组是否与参考组匹配
                            elif current_marker_ids == set(reference_marker_ids):
                                valid_groups.append(current_group)
                            else:
                                # print(f"警告：组 {current_group['group_id']} 的marker IDs与参考不匹配")
                                mismatch_count += 1
                            
                            # 准备下一组数据
                            current_group = {
                                'group_id': total_groups,
                                'markers': {}
                            }
                        
                        i += 3
                        
                    except ValueError as e:
                        print(f"警告：在处理第 {i+1} 行附近数据时出错: {str(e)}")
                        i += 1
                        continue
                else:
                    break
            
            if reference_marker_ids is None:
                print("错误：未能找到有效的参考marker组")
                return None, [], False
            
            # 计算不匹配比例并判断数据是否有效
            mismatch_ratio = mismatch_count / total_groups if total_groups > 0 else 1.0
            is_valid_data = mismatch_ratio <= 0.2
            
            print(f"总共读取到 {len(valid_groups)} 组有效数据")
            print(f"不匹配组数: {mismatch_count}, 总组数: {total_groups}, 不匹配比例: {mismatch_ratio:.2%}")
            
            if not is_valid_data:
                print(f"警告：不匹配比例 ({mismatch_ratio:0.15%}) 超过阈值 (20%)，该数据将被丢弃")
                
                # 创建 discarded_data 目录（如果不存在）
                discarded_dir = "discarded_data"
                if not os.path.exists(discarded_dir):
                    os.makedirs(discarded_dir)
                
                # 复制文件到 discarded_data 目录
                filename = os.path.basename(file_path)
                shutil.copy2(file_path, os.path.join(discarded_dir, filename))
                
                # 如果存在对应的视频文件，也复制到 discarded_data 目录
                video_path = file_path.rsplit('.', 1)[0] + '.mp4'
                if os.path.exists(video_path):
                    shutil.copy2(video_path, os.path.join(discarded_dir, os.path.basename(video_path)))
                camera_path = file_path.rsplit('.', 1)[0] + '_CameraTimestamps.txt'
                if os.path.exists(camera_path):
                    shutil.copy2(camera_path, os.path.join(discarded_dir, os.path.basename(camera_path)))
            return reference_marker_ids, valid_groups, is_valid_data
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None, [], False
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return None, [], False

def load_camera_timestamps(camera_timestamp_file):
    """
    读取相机时间戳文件
    
    参数:
    camera_timestamp_file: 相机时间戳文件路径
    
    返回:
    numpy数组，包含所有相机时间戳
    """
    try:
        with open(camera_timestamp_file, 'r') as f:
            timestamps = [float(line.strip()) for line in f.readlines() if line.strip()]
        return np.array(timestamps)
    except Exception as e:
        print(f"读取相机时间戳文件时出错：{str(e)}")
        return None

def interpolate_pose(pose_data, camera_timestamps):
    """
    将位姿数据插值到相机时间戳
    
    参数:
    pose_data: 包含原始位姿数据的DataFrame
    camera_timestamps: 相机时间戳数组
    
    返回:
    插值后的位姿DataFrame
    """
    # 创建插值器
    interpolators = {}
    columns_to_interpolate = ['TCP_pos_x', 'TCP_pos_y', 'TCP_pos_z', 
                            'TCP_euler_x', 'TCP_euler_y', 'TCP_euler_z',  # 改为XYZ顺序
                            'gripper_distance']
    
    for col in columns_to_interpolate:
        if col in pose_data.columns:  # 确保列存在
            interpolators[col] = interp1d(pose_data['timestamp'], 
                                        pose_data[col], 
                                        kind='linear',
                                        bounds_error=False,
                                        fill_value=np.nan)
    
    # 对每个相机时间戳进行插值
    interpolated_data = []
    for cam_time in camera_timestamps:
        row_data = {'timestamp': cam_time}
        for col in columns_to_interpolate:
            if col in interpolators:
                row_data[col] = interpolators[col](cam_time)
        interpolated_data.append(row_data)
    
    result_df = pd.DataFrame(interpolated_data)
    
    # 首先确保所有数值列都是float类型
    for col in columns_to_interpolate:
        if col in result_df.columns and col != 'timestamp':
            result_df[col] = result_df[col].astype(float)
    
    print("\n转换类型后的前5行数据:")
    print(result_df.head())
    
    # 现在处理nan值
    for col in columns_to_interpolate:
        if col in result_df.columns and col != 'timestamp':
            print(f"\n处理列: {col}")
            print(f"数据类型: {result_df[col].dtype}")
            print(f"该列前5个值:")
            print(result_df[col].head())
            
            first_valid = result_df[col].first_valid_index()
            print(f"第一个有效值索引: {first_valid}")
            
            if first_valid is not None and first_valid > 0:
                first_valid_value = result_df[col][first_valid]
                print(f"找到第一个有效值: {first_valid_value}，在索引 {first_valid}")
                result_df.loc[:first_valid-1, col] = first_valid_value
    
    return result_df

def make_continuous_angles(angles):
    """
    处理欧拉角的连续性
    """
    for i in range(1, len(angles)):
        for j in range(len(angles[i])):
            # 处理角度跳变
            diff = angles[i][j] - angles[i-1][j]
            if diff > np.pi:
                angles[i][j] -= 2 * np.pi
            elif diff < -np.pi:
                angles[i][j] += 2 * np.pi
    return angles

def convert_euler_to_quaternion(file_path):
    """
    读取CSV文件，将欧拉角转换为四元数并添加到新列
    
    参数:
    file_path: CSV文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取欧拉角 (xyz顺序)
    euler_angles = df[['TCP_euler_x', 'TCP_euler_y', 'TCP_euler_z']].values
    
    # 转换为四元数
    r = Rotation.from_euler('xyz', euler_angles)
    quaternions = r.as_quat()  # 返回顺序为 x,y,z,w
    
    # 添加四元数列 (调整为w,x,y,z顺序)
    df['quat_w'] = quaternions[:, 3]
    df['quat_x'] = quaternions[:, 0]
    df['quat_y'] = quaternions[:, 1]
    df['quat_z'] = quaternions[:, 2]
    
    # 保存到原文件
    df.to_csv(file_path, index=False)
    
    print(f"已将欧拉角转换为四元数并保存到文件: {file_path}")

def copy_mp4_files(data_dir, processed_dir, task_name, sequence_num):
    for camera_num in ['Camera1', 'Camera2']:
        # 使用os.path.join确保路径格式正确
        mp4_source = os.path.normpath(os.path.join(data_dir, f'{task_name}_{sequence_num}_{camera_num}.mp4'))
        mp4_dest = os.path.normpath(os.path.join(processed_dir, f'{task_name}_{sequence_num}_{camera_num}.mp4'))
        
        print(f"\n正在处理 {camera_num} 视频文件:")
        print(f"源文件路径: {mp4_source}")
        print(f"目标文件路径: {mp4_dest}")
        
        try:
            # 检查源文件是否存在
            if not os.path.exists(mp4_source):
                print(f"错误：找不到源视频文件: {mp4_source}")
                continue
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(mp4_dest), exist_ok=True)
            
            # 复制文件
            shutil.copy2(mp4_source, mp4_dest)
            print(f"成功：{camera_num} 视频文件已复制")
            
        except Exception as e:
            print(f"复制 {camera_num} 视频时出错: {str(e)}")

if __name__ == "__main__":
    # 定义任务名称
    task_name = 'unpackaddd'  # 只需要在这里修改任务名称
    
    # 获取data目录下所有的文件
    data_dir = f'data/{task_name}'
    processed_dir = f'processed_data/{task_name}'  # 新增处理后数据的目录
    
    # 确保输出目录存在
    os.makedirs(processed_dir, exist_ok=True)
    
    # 首先获取所有文件
    all_files = glob.glob(os.path.join(data_dir, f'{task_name}_*.txt'))
    
    # 修改文件分离逻辑
    marker_files = []
    timestamp_files = {}
    
    for f in all_files:
        # 统一使用 os.path.normpath 来处理路径
        f = os.path.normpath(f)
        base_name = os.path.basename(f)
        if 'Camera1Timestamps' in f:
            # 移除.txt后缀和Camera1Timestamps后缀
            base_name_no_ext = base_name.replace('_Camera1Timestamps.txt', '')
            # 移除任务名前缀
            seq_num = base_name_no_ext[len(task_name)+1:]  # +1 是为了去掉下划线
            timestamp_files[seq_num] = f
        elif 'Camera' not in f and f.endswith('.txt'):  # 添加.txt后缀检查
            marker_files.append(f)
    
    print(f"找到 {len(marker_files)} 个marker文件")
    print(f"找到 {len(timestamp_files)} 个Camera1时间戳文件")
    
    # 对每个marker文件进行处理
    for marker_file in marker_files:
        # 从文件名中提取序号
        base_name = os.path.basename(marker_file)
        # 移除.txt后缀
        base_name_no_ext = base_name.rsplit('.', 1)[0]
        # 移除任务名前缀
        sequence_num = base_name_no_ext[len(task_name)+1:]  # +1 是为了去掉下划线
        
        if sequence_num in timestamp_files:
            timestamp_file = timestamp_files[sequence_num]
            # 使用os.path.join来构建路径
            output_dir = os.path.join('processed_data', task_name)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\n处理文件: {marker_file}")
            
            # 读取相机时间戳
            camera_timestamps = load_camera_timestamps(timestamp_file)
            if camera_timestamps is None:
                print(f"跳过文件 {marker_file} - 无法读取相机时间戳")
                continue
            
            # 加载当前文件的marker位置数据
            reference_ids, valid_groups, is_valid_data = load_marker_positions(marker_file)
            
            if not is_valid_data:
                print(f"跳过文件 {marker_file} - 数据质量不符合要求")
                continue
            
            if reference_ids is None or not valid_groups:
                print(f"跳过文件 {marker_file} - 未能成功读取数据")
                continue
            
            # 复制对应的MP4文件到processed_data目录
            try:
                # 确保processed_dir存在
                os.makedirs(processed_dir, exist_ok=True)
                # 复制视频文件
                copy_mp4_files(data_dir, processed_dir, task_name, sequence_num)
            except Exception as e:
                print(f"视频复制过程发生错误: {str(e)}")
            
            # 获取输出文件名
            base_name = os.path.basename(marker_file)
            output_name = os.path.join(output_dir, f'local_coordinate_poses_{base_name.split(".")[0]}.csv')
            
            # 以下是原有的处理逻辑
            # 定义旋转角度（弧度制）
            rx = np.pi / 2  # 绕x轴旋转90度
            ry = np.pi      # 绕y轴旋转180度
            rz = 0          # 绕z轴旋转0度
            
            # 定义平移向量
            translation = [0.475, 0.01, 0]
            
            # 构建每一步的旋转矩阵 - 从参考坐标系到目标坐标系的旋转
            r_x = Rotation.from_euler('x', rx).as_matrix()  # 绕x轴旋转
            r_y = Rotation.from_euler('y', ry).as_matrix()  # 绕y轴旋转
            r_z = Rotation.from_euler('z', rz).as_matrix()  # 绕z轴旋转
            
            # 组合旋转矩阵 - 从参考坐标系到目标坐标系
            rotation_matrix_x = r_x  # 只绕x轴旋转
            rotation_matrix_xy = r_x @ r_y  # 先绕x轴再绕y轴旋转
            rotation_matrix_xyz = rotation_matrix_xy @ r_z  # 最后绕z轴旋转
            
            # 定义三个点在参考坐标系（原始坐标系）中的坐标
            points_reference = np.array([
                [0, 0, 0.001],
                [0.001, 0, 0],
                [0, 0.001, 0]
            ])
            
            # 在不同坐标系中表示这些点
            # 1. 点在原始坐标系中的坐标
            points_original = points_reference.copy()
            
            # 2. 点在绕x轴旋转后的坐标系中的坐标
            points_x_system = transform_points(points_reference, [0, 0, 0], rotation_matrix_x)
            
            # 3. 点在绕x轴和y轴旋转后的坐标系中的坐标
            points_xy_system = transform_points(points_reference, [0, 0, 0], rotation_matrix_xy)
            
            # 4. 点在绕x轴、y轴和z轴旋转后的坐标系中的坐标
            points_xyz_system = transform_points(points_reference, [0, 0, 0], rotation_matrix_xyz)
            
            # 5. 点在最终目标坐标系（平移后）中的坐标
            points_final_system = transform_points(points_reference, translation, rotation_matrix_xyz)
            
            # 打印不同坐标系下的点坐标
            print("Points in Reference Coordinate System:")
            print(points_reference)
            print("Points in X-Rotated Coordinate System:")
            print(points_x_system)
            print("Points in XY-Rotated Coordinate System:")
            print(points_xy_system)
            print("Points in XYZ-Rotated Coordinate System:")
            print(points_xyz_system)
            print("Points in Final Coordinate System (after Translation):")
            print(points_final_system)
            
            # 创建2行4列的图表布局
            fig = plt.figure(figsize=(20, 10))
            
            # 第一行：前4个步骤的可视化
            # 第一步：原始坐标系
            ax1 = fig.add_subplot(241, projection='3d')
            plot_coordinate_system(ax1, [0, 0, 0], np.eye(3), "Original", color='k')
            ax1.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax1.set_title("Step 1: Original Coordinates")
            ax1.set_xlim([-0.5, 1])
            ax1.set_ylim([-0.5, 1])
            ax1.set_zlim([-0.5, 1])
            
            # 第二步：绕x轴旋转的坐标系
            ax2 = fig.add_subplot(242, projection='3d')
            plot_coordinate_system(ax2, [0, 0, 0], rotation_matrix_x, "X-Rotated", color='r')
            ax2.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax2.set_title("Step 2: X-Rotated System")
            ax2.set_xlim([-0.5, 1])
            ax2.set_ylim([-0.5, 1])
            ax2.set_zlim([-0.5, 1])
            
            # 第三步：绕x轴和y轴旋转的坐标系
            ax3 = fig.add_subplot(243, projection='3d')
            plot_coordinate_system(ax3, [0, 0, 0], rotation_matrix_xy, "XY-Rotated", color='g')
            ax3.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax3.set_title("Step 3: XY-Rotated System")
            ax3.set_xlim([-0.5, 1])
            ax3.set_ylim([-0.5, 1])
            ax3.set_zlim([-0.5, 1])
            
            # 第四步：绕x轴、y轴和z轴旋转的坐标系
            ax4 = fig.add_subplot(244, projection='3d')
            plot_coordinate_system(ax4, [0, 0, 0], rotation_matrix_xyz, "XYZ-Rotated", color='b')
            ax4.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax4.set_title("Step 4: XYZ-Rotated System")
            ax4.set_xlim([-0.5, 1])
            ax4.set_ylim([-0.5, 1])
            ax4.set_zlim([-0.5, 1])
            
            # 第二行：第五步和marker点的可视化
            # 第五步：平移后的坐标系
            ax5 = fig.add_subplot(245, projection='3d')
            plot_coordinate_system(ax5, translation, rotation_matrix_xyz, "Translated", color='m')
            ax5.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax5.set_title("Step 5: Translated System")
            ax5.set_xlim([-0.5, 1])
            ax5.set_ylim([-0.5, 1])
            ax5.set_zlim([-0.5, 1])
            
            # Marker点的可视化
            if reference_ids is not None and valid_groups:
                ax6 = fig.add_subplot(246, projection='3d')
                plot_coordinate_system(ax6, [0, 0, 0], np.eye(3), "Markers in Original System", color='k')
                
                # 为每个marker使用不同的颜色
                colors = plt.cm.rainbow(np.linspace(0, 1, len(reference_ids)))
                
                # 为每个marker绘制所有时刻的点
                for idx, marker_id in enumerate(reference_ids):
                    # 收集该marker在所有时刻的位置
                    positions = []
                    for group in valid_groups:
                        if marker_id in group['markers']:
                            pos = group['markers'][marker_id]['position']
                            positions.append(pos)
                    
                    # 将位置转换为numpy数组
                    positions = np.array(positions)
                    
                    # 绘制该marker的所有位置
                    ax6.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                               color=colors[idx], alpha=0.5, s=2,#s是点的大小
                               label=f'Marker {marker_id}')
                    
                    # 绘制轨迹线
                    ax6.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                            color=colors[idx], alpha=0.3, linewidth=0.5)
                
                ax6.set_title("Markers Positions (All Timestamps)")
                ax6.legend()
                ax6.set_xlim([-0.5, 1])
                ax6.set_ylim([-0.5, 1])
                ax6.set_zlim([-0.5, 1])
            
            # 第七个图：转换后的marker点位置
            if reference_ids is not None and valid_groups:
                # 首先计算最稳定的三个点
                point_pairs_stats = {}
                for i in range(len(reference_ids)):
                    for j in range(i+1, len(reference_ids)):
                        id1 = reference_ids[i]
                        id2 = reference_ids[j]
                        distances = []
                        for group in valid_groups:
                            pos1 = group['markers'][id1]['position']
                            pos2 = group['markers'][id2]['position']
                            dist = np.linalg.norm(pos1 - pos2)
                            distances.append(dist)
                        
                        std = np.std(distances)
                        mean = np.mean(distances)
                        variation = std / mean * 100  # 变异系数（百分比）
                        point_pairs_stats[(id1, id2)] = {
                            'std': std,
                            'mean': mean,
                            'variation': variation,
                            'ids': (id1, id2)
                        }
                
                # 按变异系数排序，找出最稳定的点对
                sorted_pairs = sorted(point_pairs_stats.items(), key=lambda x: x[1]['variation'])
                
                # 选择变异系数最小的点对中的点
                selected_points = set()
                for pair, stats in sorted_pairs:
                    selected_points.add(pair[0])
                    selected_points.add(pair[1])
                    if len(selected_points) >= 3:
                        break
                
                selected_points = list(selected_points)[:3]
                print(f"\n选择的三个最稳定的点: {selected_points}")
                print("这些点之间的距离变异系数：")
                for i in range(3):
                    for j in range(i+1, 3):
                        id1, id2 = selected_points[i], selected_points[j]
                        if (id1, id2) in point_pairs_stats:
                            stats = point_pairs_stats[(id1, id2)]
                        else:
                            stats = point_pairs_stats[(id2, id1)]
                        print(f"点 {id1}-{id2}: 平均距离 = {stats['mean']:.3f}, 变异系数 = {stats['variation']:.3f}%")

                # 在create_coordinate_system之前添加初始帧点选择逻辑
                # 获取初始帧数据
                initial_frame = valid_groups[0]
                initial_positions = {}
                for marker_id in selected_points[:3]:
                    pos = initial_frame['markers'][marker_id]['position']
                    # 转换到translated坐标系
                    transformed_pos = transform_points(pos.reshape(1, 3), translation, rotation_matrix_xyz)[0]
                    initial_positions[marker_id] = transformed_pos

                # 根据y坐标排序
                sorted_markers = sorted(initial_positions.items(), key=lambda x: x[1][1])
                
                # 选择y坐标最小的点作为原点，y坐标最大的点作为y轴方向点
                origin_marker_id = sorted_markers[0][0]  # y坐标最小的点
                y_marker_id = sorted_markers[2][0]       # y坐标最大的点
                # 剩下的点作为z轴参考点
                z_ref_marker_id = sorted_markers[1][0]

                print(f"\n根据初始帧y坐标选择的点:")
                print(f"原点 (y最小): Marker {origin_marker_id}")
                print(f"y轴方向 (y最大): Marker {y_marker_id}")
                print(f"z轴参考: Marker {z_ref_marker_id}")

                # 修改坐标系创建函数
                def create_coordinate_system(positions_dict):
                    # 使用选定的marker ID
                    p_origin = positions_dict[origin_marker_id]
                    p_y = positions_dict[y_marker_id]
                    p_z_ref = positions_dict[z_ref_marker_id]
                    
                    # 其余计算保持不变
                    y_axis = p_y - p_origin
                    y_axis = y_axis / np.linalg.norm(y_axis)
                    
                    mid_point = (p_origin + p_y) / 2
                    
                    z_axis = mid_point - p_z_ref
                    z_axis = z_axis / np.linalg.norm(z_axis)
                    
                    x_axis = np.cross(y_axis, z_axis)
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    
                    z_axis = np.cross(x_axis, y_axis)
                    z_axis = z_axis / np.linalg.norm(z_axis)
                    
                    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
                    
                    return p_origin, rotation_matrix

                # 开始绘图
                ax7 = fig.add_subplot(247, projection='3d')
                plot_coordinate_system(ax7, [0, 0, 0], np.eye(3), "Transformed Markers", color='k')
                
                # 为每个marker使用不同的颜色
                colors = plt.cm.rainbow(np.linspace(0, 1, len(reference_ids)))
                
                # 为每个marker绘制所有时刻的转换后的点
                for idx, marker_id in enumerate(reference_ids):
                    positions = []
                    for group in valid_groups:
                        if marker_id in group['markers']:
                            pos = group['markers'][marker_id]['position']
                            # 将点转换到translated坐标系
                            transformed_pos = transform_points(pos.reshape(1, 3), translation, rotation_matrix_xyz)
                            positions.append(transformed_pos[0])
                    
                    positions = np.array(positions)
                    
                    # 绘制该marker的所有转换后的位置
                    ax7.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                               color=colors[idx], alpha=0.5, s=2,
                               label=f'Marker {marker_id}')
                    
                    # 绘制轨迹线
                    ax7.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                            color=colors[idx], alpha=0.3, linewidth=0.5)
                
                ax7.set_title("Transformed Markers Positions")
                ax7.legend()
                ax7.set_xlim([-0.5, 1])
                ax7.set_ylim([-0.5, 1])
                ax7.set_zlim([-0.5, 1])

                # 创建列表存储所有位姿数据
                pose_data = []
                
                # 在创建pose_data之前，找出非刚性体上的点
                rigid_body_points = selected_points[:3]  # 已经选择的三个刚性体点
                moving_points = [point for point in reference_ids if point not in rigid_body_points]
                print(f"\n相对运动的点: {moving_points}")

                # 遍历所有帧时使用新的create_coordinate_system
                for i in range(0, len(valid_groups)):
                    group = valid_groups[i]
                    positions_dict = {marker_id: group['markers'][marker_id]['position'] for marker_id in selected_points[:3]}
                    origin, rotation_matrix = create_coordinate_system(positions_dict)
                    
                    # 在局部坐标系中定义平移向量
                    local_translation = np.array([0.039, 0.06, 0.042])
                    
                    # 将局部平移向量转换到全局坐标系
                    global_translation = rotation_matrix @ local_translation
                    
                    # 更新原点位置
                    new_origin = origin + global_translation
                    
                    # 将坐标系原点和旋转矩阵转换到translated坐标系中
                    transformed_origin = transform_points(new_origin.reshape(1, 3), translation, rotation_matrix_xyz)[0]
                    transformed_rotation = rotation_matrix_xyz @ rotation_matrix
                    
                    # 计算相对于translated坐标系的欧拉角 (XYZ顺序)
                    euler_angles = Rotation.from_matrix(transformed_rotation).as_euler('xyz')
                    
                    # 获取时间戳时使用origin_marker_id
                    timestamp = group['markers'][origin_marker_id]['time']
                    
                    # 获取相对运动点的位置并转换到translated坐标系
                    moving_points_positions = {}
                    for point_id in moving_points:
                        if point_id in group['markers']:
                            pos = group['markers'][point_id]['position']
                            transformed_pos = transform_points(pos.reshape(1, 3), translation, rotation_matrix_xyz)[0]
                            moving_points_positions[point_id] = transformed_pos
                    
                    # 存储位姿数据
                    frame_data = {
                        'timestamp': timestamp,
                        'TCP_pos_x': transformed_origin[0],
                        'TCP_pos_y': transformed_origin[1],
                        'TCP_pos_z': transformed_origin[2],
                        'TCP_euler_x': euler_angles[0],  # 弧度制
                        'TCP_euler_y': euler_angles[1],  # 弧度制
                        'TCP_euler_z': euler_angles[2]   # 弧度制
                    }

                    # 计算两个gripper点之间的距离
                    if len(moving_points) >= 2 and all(point_id in moving_points_positions for point_id in moving_points[:2]):
                        pos1 = moving_points_positions[moving_points[0]]
                        pos2 = moving_points_positions[moving_points[1]]
                        distance = np.linalg.norm(pos1 - pos2)
                        frame_data['gripper_distance'] = distance
                    else:
                        frame_data['gripper_distance'] = np.nan

                    pose_data.append(frame_data)
                    
                    # 每隔100帧显示一次坐标系（可视化部分）
                    if i % 100 == 0:
                        plot_coordinate_system(ax7, transformed_origin, transformed_rotation, "", color='k')
                
                # 在保存CSV之前进行时间戳对齐
                original_df = pd.DataFrame(pose_data)
                
                # 检查并记录时间范围外的时间戳
                out_of_range_timestamps = camera_timestamps[
                    (camera_timestamps < original_df['timestamp'].min()) |
                    (camera_timestamps > original_df['timestamp'].max())
                ]

                if len(out_of_range_timestamps) > 0:
                    print(f"警告：有 {len(out_of_range_timestamps)} 个时间戳超出原始数据范围")
                    print(f"原始数据时间范围: [{original_df['timestamp'].min()}, {original_df['timestamp'].max()}]")
                    print(f"超出范围的时间戳: {out_of_range_timestamps}")
                    
                    # 根据具体需求选择处理方式，以下是几种可能的处理方案：
                    
                    # 方案1：使用最近的有效值填充
                    valid_camera_timestamps = camera_timestamps.copy()  # 保持原始长度
                    
                    # 方案2：抛出异常，要求输入数据满足时间范围
                    # raise ValueError("发现超出范围的时间戳，请检查数据")
                    
                    # 方案3：对超出范围的时间戳进行外推（可能不够准确）
                    # valid_camera_timestamps = camera_timestamps  # 直接使用原始时间戳，让插值函数处理外推
                else:
                    valid_camera_timestamps = camera_timestamps

                # 进行插值
                aligned_df = interpolate_pose(original_df, valid_camera_timestamps)
                
                # 移除包含NaN的行
                aligned_df = aligned_df.dropna()
                
                if len(aligned_df) == 0:
                    print(f"警告：{marker_file} 插值后没有有效数据")
                    continue
                
                # 在保存CSV之前添加处理
                euler_angles = np.array([[d['TCP_euler_x'], d['TCP_euler_y'], d['TCP_euler_z']] for d in pose_data])
                continuous_angles = make_continuous_angles(euler_angles)

                # 更新pose_data中的欧拉角
                for i in range(len(pose_data)):
                    pose_data[i]['TCP_euler_x'] = continuous_angles[i][0]
                    pose_data[i]['TCP_euler_y'] = continuous_angles[i][1]
                    pose_data[i]['TCP_euler_z'] = continuous_angles[i][2]
                
                # 保存对齐后的数据
                aligned_output_name = os.path.join(output_dir, 
                                                 f'aligned_local_coordinate_poses_{base_name.split(".")[0]}.csv')
                aligned_df.to_csv(aligned_output_name, index=False, float_format='%.15f')
                print(f"\n对齐后的位姿数据已保存到 {aligned_output_name}")
                print(f"原始帧数：{len(original_df)}，对齐后帧数：{len(aligned_df)}")
                
                # 在保存图片之前添加显示
                # plt.tight_layout()  # 调整子图之间的间距
                # plt.show()  # 添加这一行来显示交互式窗口
                
                # 保存图片
                # plt.savefig(os.path.join(processed_dir, f'visualization_{base_name.split(".")[0]}.png'))
                # plt.close()  # 关闭当前图形
                
                if reference_ids is not None and valid_groups:
                    print(f"\n成功处理文件 {marker_file}！")
                    print(f"参考Marker IDs: {reference_ids}")
                    
                
                # 在保存aligned_local_coordinate_poses文件后添加：
                csv_file_path = f"processed_data/{task_name}/aligned_local_coordinate_poses_{task_name}_{sequence_num}.csv"
                convert_euler_to_quaternion(csv_file_path)
