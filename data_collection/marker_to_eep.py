import shutil
import glob
import json
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

def transform_points(points, translation, rotation_matrix):
    translated_points = points - translation  
    transformed_points = (rotation_matrix.T @ translated_points.T).T  
    
    return transformed_points

def plot_coordinate_system(ax, translation, rotation_matrix, label, color='k'):
    basis_vectors = np.eye(3)
    rotated_basis = rotation_matrix @ basis_vectors
    colors = ['r', 'g', 'b']
    for i in range(3):
        ax.quiver(
            translation[0], translation[1], translation[2],
            rotated_basis[0, i], rotated_basis[1, i], rotated_basis[2, i],
            color=colors[i], length=0.1, normalize=True
        )
    
    ax.text(translation[0], translation[1], translation[2], label, fontsize=12, color=color)

def load_marker_positions(file_path='dataset/test/test_0.txt', group_size=5):
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
                        time = float(lines[i].strip())
                        marker_id = int(lines[i+1].strip())
                        pos_line = lines[i+2].strip()

                        if 'pos:' in pos_line:
                            start = pos_line.find('[') + 1
                            end = pos_line.find(']')
                            if start > 0 and end > start:
                                pos_str = pos_line[start:end]
                                pos = np.array([float(x.strip()) for x in pos_str.split(',')])
                            else:
                                i += 1
                                continue
                        else:
                            pos = np.array([float(x) for x in pos_line.split()])

                        if len(pos) != 3:
                            i += 1
                            continue
                            
                        # offset of optitrack
                        pos[0] -= 0.01  
                        pos[1] -= 0.04  
                        pos[2] -= 0.01  
                        pos = pos / 2

                        current_group['markers'][marker_id] = {
                            'time': time,
                            'position': pos
                        }
                        if len(current_group['markers']) == group_size:
                            total_groups += 1
                            current_marker_ids = set(current_group['markers'].keys())

                            if reference_marker_ids is None:
                                reference_marker_ids = sorted(list(current_marker_ids))
                                print(f"Reference marker IDs determined: {reference_marker_ids}")
                                valid_groups.append(current_group)

                            elif current_marker_ids == set(reference_marker_ids):
                                valid_groups.append(current_group)
                            else:
                                mismatch_count += 1
                            
                            current_group = {
                                'group_id': total_groups,
                                'markers': {}
                            }
                        i += 3
                        
                    except ValueError as e:
                        print(f"Warning: Error processing data near line {i+1}: {str(e)}")
                        i += 1
                        continue
                else:
                    break
            
            if reference_marker_ids is None:
                print("Error: Could not find valid reference marker group")
                return None, [], False
            

            mismatch_ratio = mismatch_count / total_groups if total_groups > 0 else 1.0
            is_valid_data = mismatch_ratio <= 0.2
            
            print(f"Total valid groups read: {len(valid_groups)}")
            print(f"Mismatched groups: {mismatch_count}, Total groups: {total_groups}, Mismatch ratio: {mismatch_ratio:.2%}")
            
            if not is_valid_data:
                print(f"Warning: Mismatch ratio ({mismatch_ratio:0.15%}) exceeds threshold (15%), this data will be discarded")
                

                discarded_dir = "discarded_data"
                if not os.path.exists(discarded_dir):
                    os.makedirs(discarded_dir)
                

                filename = os.path.basename(file_path)
                shutil.copy2(file_path, os.path.join(discarded_dir, filename))
                

                video_path = file_path.rsplit('.', 1)[0] + '.mp4'
                if os.path.exists(video_path):
                    shutil.copy2(video_path, os.path.join(discarded_dir, os.path.basename(video_path)))
                camera_path = file_path.rsplit('.', 1)[0] + '_CameraTimestamps.txt'
                if os.path.exists(camera_path):
                    shutil.copy2(camera_path, os.path.join(discarded_dir, os.path.basename(camera_path)))
            return reference_marker_ids, valid_groups, is_valid_data
        
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None, [], False
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, [], False

def load_camera_timestamps(camera_timestamp_file):

    try:
        with open(camera_timestamp_file, 'r') as f:
            timestamps = [float(line.strip()) for line in f.readlines() if line.strip()]
        return np.array(timestamps)
    except Exception as e:
        print(f"Error reading camera timestamp file: {str(e)}")
        return None

def interpolate_pose(pose_data, camera_timestamps):
    interpolators = {}
    columns_to_interpolate = ['TCP_pos_x', 'TCP_pos_y', 'TCP_pos_z', 
                            'TCP_euler_x', 'TCP_euler_y', 'TCP_euler_z',  
                            'gripper_distance']
    
    for col in columns_to_interpolate:
        if col in pose_data.columns:  
            interpolators[col] = interp1d(pose_data['timestamp'], 
                                        pose_data[col], 
                                        kind='linear',
                                        bounds_error=False,
                                        fill_value=np.nan)
    

    interpolated_data = []
    for cam_time in camera_timestamps:
        row_data = {'timestamp': cam_time}
        for col in columns_to_interpolate:
            if col in interpolators:
                row_data[col] = interpolators[col](cam_time)
        interpolated_data.append(row_data)
    
    result_df = pd.DataFrame(interpolated_data)
    
    for col in columns_to_interpolate:
        if col in result_df.columns and col != 'timestamp':
            result_df[col] = result_df[col].astype(float)
    
    print("\nFirst 5 rows of data after type conversion:")
    print(result_df.head())
    
    for col in columns_to_interpolate:
        if col in result_df.columns and col != 'timestamp':
            print(f"\nProcessing column: {col}")
            print(f"Data type: {result_df[col].dtype}")
            print(f"First 5 values of this column:")
            print(result_df[col].head())
            
            first_valid = result_df[col].first_valid_index()
            print(f"First valid value index: {first_valid}")
            
            if first_valid is not None and first_valid > 0:
                first_valid_value = result_df[col][first_valid]
                print(f"Found first valid value: {first_valid_value} at index {first_valid}")
                result_df.loc[:first_valid-1, col] = first_valid_value
    
    return result_df

def make_continuous_angles(angles):

    for i in range(1, len(angles)):
        for j in range(len(angles[i])):
            diff = angles[i][j] - angles[i-1][j]
            if diff > np.pi:
                angles[i][j] -= 2 * np.pi
            elif diff < -np.pi:
                angles[i][j] += 2 * np.pi
    return angles

def convert_euler_to_quaternion(file_path):

    df = pd.read_csv(file_path)
    

    euler_angles = df[['TCP_euler_x', 'TCP_euler_y', 'TCP_euler_z']].values
    

    r = Rotation.from_euler('xyz', euler_angles)
    quaternions = r.as_quat()  
    

    df['quat_w'] = quaternions[:, 3]
    df['quat_x'] = quaternions[:, 0]
    df['quat_y'] = quaternions[:, 1]
    df['quat_z'] = quaternions[:, 2]
    

    df.to_csv(file_path, index=False)
    
    print(f"Euler angles have been converted to quaternions and saved to file: {file_path}")

def copy_mp4_files(data_dir, processed_dir, task_name, sequence_num):
    for camera_num in ['Camera1', 'Camera2']:

        mp4_source = os.path.normpath(os.path.join(data_dir, f'{task_name}_{sequence_num}_{camera_num}.mp4'))
        mp4_dest = os.path.normpath(os.path.join(processed_dir, f'{task_name}_{sequence_num}_{camera_num}.mp4'))
        
        print(f"\nProcessing {camera_num} video file:")
        print(f"Source file path: {mp4_source}")
        print(f"Target file path: {mp4_dest}")
        
        try:

            if not os.path.exists(mp4_source):
                print(f"Error: Source video file not found: {mp4_source}")
                continue
            

            os.makedirs(os.path.dirname(mp4_dest), exist_ok=True)
            

            shutil.copy2(mp4_source, mp4_dest)
            print(f"Success: {camera_num} video file has been copied")
            
        except Exception as e:
            print(f"Error copying {camera_num} video: {str(e)}")

def process_marker_data(task_name):
    data_dir = f'dataset/raw/{task_name}'
    processed_dir = f'dataset/processed_data/{task_name}' 

    os.makedirs(processed_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(data_dir, f'{task_name}_*.txt'))

    marker_files = []
    timestamp_files = {}
    
    for f in all_files:

        f = os.path.normpath(f)
        base_name = os.path.basename(f)
        if 'Camera1Timestamps' in f:

            base_name_no_ext = base_name.replace('_Camera1Timestamps.txt', '')

            seq_num = base_name_no_ext[len(task_name)+1:]  
            timestamp_files[seq_num] = f
        elif 'Camera' not in f and f.endswith('.txt'):  
            marker_files.append(f)
    
    print(f"Found {len(marker_files)} marker files")
    print(f"Found {len(timestamp_files)} Camera1 timestamp files")
    

    for marker_file in marker_files:

        base_name = os.path.basename(marker_file)

        base_name_no_ext = base_name.rsplit('.', 1)[0]

        sequence_num = base_name_no_ext[len(task_name)+1:] 
        
        if sequence_num in timestamp_files:
            timestamp_file = timestamp_files[sequence_num]

            output_dir = os.path.join("dataset", 'processed_data', task_name)
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nProcessing file: {marker_file}")

            camera_timestamps = load_camera_timestamps(timestamp_file)
            if camera_timestamps is None:
                print(f"Skipping file {marker_file} - Unable to read camera timestamps")
                continue

            reference_ids, valid_groups, is_valid_data = load_marker_positions(marker_file)
            
            if not is_valid_data:
                print(f"Skipping file {marker_file} - Data quality does not meet requirements")
                continue
            
            if reference_ids is None or not valid_groups:
                print(f"Skipping file {marker_file} - Failed to read data successfully")
                continue

            try:

                os.makedirs(processed_dir, exist_ok=True)

                copy_mp4_files(data_dir, processed_dir, task_name, sequence_num)
            except Exception as e:
                print(f"Error occurred during video copying process: {str(e)}")
            

            base_name = os.path.basename(marker_file)
            output_name = os.path.join(output_dir, f'local_coordinate_poses_{base_name.split(".")[0]}.csv')

            rx = np.pi / 2  
            ry = np.pi      
            rz = 0          
            
            translation = [0.475, 0.01, 0]
            r_x = Rotation.from_euler('x', rx).as_matrix()  
            r_y = Rotation.from_euler('y', ry).as_matrix()  
            r_z = Rotation.from_euler('z', rz).as_matrix()  

            rotation_matrix_x = r_x  
            rotation_matrix_xy = r_x @ r_y  
            rotation_matrix_xyz = rotation_matrix_xy @ r_z  

            points_reference = np.array([
                [0, 0, 0.001],
                [0.001, 0, 0],
                [0, 0.001, 0]
            ])

            points_original = points_reference.copy()
            points_x_system = transform_points(points_reference, [0, 0, 0], rotation_matrix_x)
            points_xy_system = transform_points(points_reference, [0, 0, 0], rotation_matrix_xy)
            points_xyz_system = transform_points(points_reference, [0, 0, 0], rotation_matrix_xyz)
            points_final_system = transform_points(points_reference, translation, rotation_matrix_xyz)
            
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

            fig = plt.figure(figsize=(20, 10))

            ax1 = fig.add_subplot(241, projection='3d')
            plot_coordinate_system(ax1, [0, 0, 0], np.eye(3), "Original", color='k')
            ax1.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax1.set_title("Step 1: Original Coordinates")
            ax1.set_xlim([-0.5, 1])
            ax1.set_ylim([-0.5, 1])
            ax1.set_zlim([-0.5, 1])
            
            ax2 = fig.add_subplot(242, projection='3d')
            plot_coordinate_system(ax2, [0, 0, 0], rotation_matrix_x, "X-Rotated", color='r')
            ax2.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax2.set_title("Step 2: X-Rotated System")
            ax2.set_xlim([-0.5, 1])
            ax2.set_ylim([-0.5, 1])
            ax2.set_zlim([-0.5, 1])
            
            ax3 = fig.add_subplot(243, projection='3d')
            plot_coordinate_system(ax3, [0, 0, 0], rotation_matrix_xy, "XY-Rotated", color='g')
            ax3.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax3.set_title("Step 3: XY-Rotated System")
            ax3.set_xlim([-0.5, 1])
            ax3.set_ylim([-0.5, 1])
            ax3.set_zlim([-0.5, 1])
            
            ax4 = fig.add_subplot(244, projection='3d')
            plot_coordinate_system(ax4, [0, 0, 0], rotation_matrix_xyz, "XYZ-Rotated", color='b')
            ax4.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax4.set_title("Step 4: XYZ-Rotated System")
            ax4.set_xlim([-0.5, 1])
            ax4.set_ylim([-0.5, 1])
            ax4.set_zlim([-0.5, 1])

            ax5 = fig.add_subplot(245, projection='3d')
            plot_coordinate_system(ax5, translation, rotation_matrix_xyz, "Translated", color='m')
            ax5.scatter(points_reference[:, 0], points_reference[:, 1], points_reference[:, 2], c='r', marker='o')
            ax5.set_title("Step 5: Translated System")
            ax5.set_xlim([-0.5, 1])
            ax5.set_ylim([-0.5, 1])
            ax5.set_zlim([-0.5, 1])

            if reference_ids is not None and valid_groups:
                ax6 = fig.add_subplot(246, projection='3d')
                plot_coordinate_system(ax6, [0, 0, 0], np.eye(3), "Markers in Original System", color='k')

                colors = plt.cm.rainbow(np.linspace(0, 1, len(reference_ids)))

                for idx, marker_id in enumerate(reference_ids):

                    positions = []
                    for group in valid_groups:
                        if marker_id in group['markers']:
                            pos = group['markers'][marker_id]['position']
                            positions.append(pos)

                    positions = np.array(positions)

                    ax6.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                               color=colors[idx], alpha=0.5, s=2,
                               label=f'Marker {marker_id}')

                    ax6.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                            color=colors[idx], alpha=0.3, linewidth=0.5)
                
                ax6.set_title("Markers Positions (All Timestamps)")
                ax6.legend()
                ax6.set_xlim([-0.5, 1])
                ax6.set_ylim([-0.5, 1])
                ax6.set_zlim([-0.5, 1])

            if reference_ids is not None and valid_groups:

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
                        variation = std / mean * 100  
                        point_pairs_stats[(id1, id2)] = {
                            'std': std,
                            'mean': mean,
                            'variation': variation,
                            'ids': (id1, id2)
                        }
                

                sorted_pairs = sorted(point_pairs_stats.items(), key=lambda x: x[1]['variation'])
                

                selected_points = set()
                for pair, stats in sorted_pairs:
                    selected_points.add(pair[0])
                    selected_points.add(pair[1])
                    if len(selected_points) >= 3:
                        break
                
                selected_points = list(selected_points)[:3]
                print(f"\nSelected three most stable points: {selected_points}")
                print("Distance variation coefficients between these points:")
                for i in range(3):
                    for j in range(i+1, 3):
                        id1, id2 = selected_points[i], selected_points[j]
                        if (id1, id2) in point_pairs_stats:
                            stats = point_pairs_stats[(id1, id2)]
                        else:
                            stats = point_pairs_stats[(id2, id1)]
                        print(f"Points {id1}-{id2}: Average distance = {stats['mean']:.3f}, Variation coefficient = {stats['variation']:.3f}%")


                initial_frame = valid_groups[0]
                initial_positions = {}
                for marker_id in selected_points[:3]:
                    pos = initial_frame['markers'][marker_id]['position']

                    transformed_pos = transform_points(pos.reshape(1, 3), translation, rotation_matrix_xyz)[0]
                    initial_positions[marker_id] = transformed_pos


                sorted_markers = sorted(initial_positions.items(), key=lambda x: x[1][1])
                

                origin_marker_id = sorted_markers[0][0]  
                y_marker_id = sorted_markers[2][0]       

                z_ref_marker_id = sorted_markers[1][0]

                print(f"\nPoints selected based on initial frame y-coordinate:")
                print(f"Origin (minimum y): Marker {origin_marker_id}")
                print(f"Y-axis direction (maximum y): Marker {y_marker_id}")
                print(f"Z-axis reference: Marker {z_ref_marker_id}")


                def create_coordinate_system(positions_dict):

                    p_origin = positions_dict[origin_marker_id]
                    p_y = positions_dict[y_marker_id]
                    p_z_ref = positions_dict[z_ref_marker_id]
                    

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


                ax7 = fig.add_subplot(247, projection='3d')
                plot_coordinate_system(ax7, [0, 0, 0], np.eye(3), "Transformed Markers", color='k')
                

                colors = plt.cm.rainbow(np.linspace(0, 1, len(reference_ids)))
                

                for idx, marker_id in enumerate(reference_ids):
                    positions = []
                    for group in valid_groups:
                        if marker_id in group['markers']:
                            pos = group['markers'][marker_id]['position']

                            transformed_pos = transform_points(pos.reshape(1, 3), translation, rotation_matrix_xyz)
                            positions.append(transformed_pos[0])
                    
                    positions = np.array(positions)
                    

                    ax7.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                               color=colors[idx], alpha=0.5, s=2,
                               label=f'Marker {marker_id}')
                    

                    ax7.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                            color=colors[idx], alpha=0.3, linewidth=0.5)
                
                ax7.set_title("Transformed Markers Positions")
                ax7.legend()
                ax7.set_xlim([-0.5, 1])
                ax7.set_ylim([-0.5, 1])
                ax7.set_zlim([-0.5, 1])


                pose_data = []
                

                rigid_body_points = selected_points[:3] 
                moving_points = [point for point in reference_ids if point not in rigid_body_points]
                print(f"\nPoints in relative motion: {moving_points}")


                for i in range(0, len(valid_groups)):
                    group = valid_groups[i]
                    positions_dict = {marker_id: group['markers'][marker_id]['position'] for marker_id in selected_points[:3]}
                    origin, rotation_matrix = create_coordinate_system(positions_dict)
                    

                    local_translation = np.array([0.039, 0.06, 0.042])
                    

                    global_translation = rotation_matrix @ local_translation
                    

                    new_origin = origin + global_translation
                    

                    transformed_origin = transform_points(new_origin.reshape(1, 3), translation, rotation_matrix_xyz)[0]
                    transformed_rotation = rotation_matrix_xyz @ rotation_matrix
                    

                    euler_angles = Rotation.from_matrix(transformed_rotation).as_euler('xyz')
                    

                    timestamp = group['markers'][origin_marker_id]['time']
                    

                    moving_points_positions = {}
                    for point_id in moving_points:
                        if point_id in group['markers']:
                            pos = group['markers'][point_id]['position']
                            transformed_pos = transform_points(pos.reshape(1, 3), translation, rotation_matrix_xyz)[0]
                            moving_points_positions[point_id] = transformed_pos
                    

                    frame_data = {
                        'timestamp': timestamp,
                        'TCP_pos_x': transformed_origin[0],
                        'TCP_pos_y': transformed_origin[1],
                        'TCP_pos_z': transformed_origin[2],
                        'TCP_euler_x': euler_angles[0],  
                        'TCP_euler_y': euler_angles[1],  
                        'TCP_euler_z': euler_angles[2]  
                    }


                    if len(moving_points) >= 2 and all(point_id in moving_points_positions for point_id in moving_points[:2]):
                        pos1 = moving_points_positions[moving_points[0]]
                        pos2 = moving_points_positions[moving_points[1]]
                        distance = np.linalg.norm(pos1 - pos2)
                        frame_data['gripper_distance'] = distance
                    else:
                        frame_data['gripper_distance'] = np.nan

                    pose_data.append(frame_data)
                    

                    if i % 100 == 0:
                        plot_coordinate_system(ax7, transformed_origin, transformed_rotation, "", color='k')
                

                original_df = pd.DataFrame(pose_data)
                

                out_of_range_timestamps = camera_timestamps[
                    (camera_timestamps < original_df['timestamp'].min()) |
                    (camera_timestamps > original_df['timestamp'].max())
                ]

                if len(out_of_range_timestamps) > 0:
                    print(f"Warning: {len(out_of_range_timestamps)} timestamps are out of original data range")
                    print(f"Original data time range: [{original_df['timestamp'].min()}, {original_df['timestamp'].max()}]")
                    print(f"Out of range timestamps: {out_of_range_timestamps}")
                    
                    valid_camera_timestamps = camera_timestamps.copy() 
                    
                else:
                    valid_camera_timestamps = camera_timestamps


                aligned_df = interpolate_pose(original_df, valid_camera_timestamps)
                

                aligned_df = aligned_df.dropna()
                
                if len(aligned_df) == 0:
                    print(f"Warning: No valid data after interpolation for {marker_file}")
                    continue
                

                euler_angles = np.array([[d['TCP_euler_x'], d['TCP_euler_y'], d['TCP_euler_z']] for d in pose_data])
                continuous_angles = make_continuous_angles(euler_angles)


                for i in range(len(pose_data)):
                    pose_data[i]['TCP_euler_x'] = continuous_angles[i][0]
                    pose_data[i]['TCP_euler_y'] = continuous_angles[i][1]
                    pose_data[i]['TCP_euler_z'] = continuous_angles[i][2]
                

                aligned_output_name = os.path.join(output_dir, 
                                                 f'aligned_local_coordinate_poses_{base_name.split(".")[0]}.csv')
                aligned_df.to_csv(aligned_output_name, index=False, float_format='%.15f')
                print(f"\nAligned pose data has been saved to {aligned_output_name}")
                print(f"Original frame count: {len(original_df)}, Aligned frame count: {len(aligned_df)}")
                

                aligned_output_name_quat = os.path.join(output_dir, 
                                                     f'aligned_local_coordinate_poses_quat_{base_name.split(".")[0]}.csv')
                convert_euler_to_quaternion(aligned_output_name_quat)
                print(f"Euler angles converted to quaternions and saved to {aligned_output_name_quat}")

def extract_frames(task_name):
    input_folder = f'processed_data/{task_name}'
    output_folder = f'Data_frame_extraction/{task_name}'

    os.makedirs(output_folder, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.csv')])
    mp4_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')])

    min_frames = float('inf')
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(input_folder, csv_file))
        min_frames = min(min_frames, len(df))

    for i in range(0, len(mp4_files), 2):  
        if i + 1 >= len(mp4_files):
            continue  
        csv_file = csv_files[i//2]  
        mp4_file1 = mp4_files[i]    
        mp4_file2 = mp4_files[i+1]  
    
        df = pd.read_csv(os.path.join(input_folder, csv_file))
        df = df.iloc[:min_frames]
        df.to_csv(os.path.join(output_folder, csv_file), index=False)

        for mp4_file in [mp4_file1, mp4_file2]:
            cap = cv2.VideoCapture(os.path.join(input_folder, mp4_file))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(output_folder, mp4_file), 
                                fourcc, 
                                cap.get(cv2.CAP_PROP_FPS), 
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            frame_count = 0
            while cap.isOpened() and frame_count < min_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_count += 1

            cap.release()
            out.release()

    print("All frames have been extracted and saved to", output_folder)

    mp4_files = [f for f in os.listdir(output_folder) if f.endswith('.mp4')]

    for mp4_file in mp4_files:
        video_path = os.path.join(output_folder, mp4_file)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video file {mp4_file} has {frame_count} frames.")
        
        cap.release()

if __name__ == "__main__":
    try:
        with open('config/process.json', 'r') as cfg:
            cfg = json.load(cfg)
    except FileNotFoundError:
        cfg = {}
    task_name = cfg.get("task_name", "test")

    process_marker_data(task_name)
    extract_frames(task_name)