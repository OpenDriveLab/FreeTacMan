import h5py
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import os
import ikpy.chain
import cv2
import ikpy.utils.plot as plot_utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
import pandas as pd
from typing import Tuple, Any, Optional

# Load the configuration from the config.json file
with open('config/config.json', 'r') as config_file:
    config = json.load(config_file)
config = config["data_process_config"]

# Extract configuration values
START_QPOS = config["start_qpos"] # Initial joint positions for the robot (values specific to your robot's configuration)
PI = np.pi
ACTIVE_MASK = config["active_joint_mask"]
# upsample_factor, decide the number of frames
upsample_factor = config.get("upsample_factor", 2)

# load piper urdf
my_chain = ikpy.chain.Chain.from_urdf_file(
    urdf_file="./assets/piper_description.urdf", 
    active_links_mask=ACTIVE_MASK
)

print(f"Number of joints in the chain: {len(my_chain)}")
print(f"Joints in the chain {my_chain}")

def video_to_array(video_path: str) -> Tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames_array = np.empty((frame_count, height, width, 3), dtype=np.uint8)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                frames_array = frames_array[:i]
                frame_count = i
                break
            frames_array[i] = frame
        return frames_array, frame_count
    finally:
        cap.release()

def cartesian_to_joints(position, quaternion, initial_joint_angles=None, **kwargs):
    """
    Convert Cartesian coordinates to robot joint angles using inverse kinematics.
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    if initial_joint_angles is None:
        initial_joint_angles = [0] * len(my_chain)
    joint_angles = my_chain.inverse_kinematics(
        position,
        rotation_matrix,
        orientation_mode='all',
        initial_position=initial_joint_angles
    )
    return joint_angles

def upsample_frames(frames, target_frames):

    N, H, W, C = frames.shape
    x_old = np.arange(N)
    x_new = np.linspace(0, N - 1, target_frames)
    upsampled = np.empty((target_frames, H, W, C), dtype=np.uint8)
    for i, xi in enumerate(x_new):
        lower = int(np.floor(xi))
        upper = min(lower + 1, N - 1)
        weight = xi - lower
        interp_frame = (1 - weight) * frames[lower].astype(np.float32) + weight * frames[upper].astype(np.float32)
        upsampled[i] = np.clip(interp_frame, 0, 255).astype(np.uint8)
    return upsampled

def interpolate_trajectory(data, target_frames, kind="cubic"):

    N, D = data.shape
    x_old = np.arange(N)
    x_new = np.linspace(0, N - 1, target_frames)
    new_data = np.zeros((target_frames, D))
    for j in range(D):
        f = interp1d(x_old, data[:, j], kind=kind)
        new_data[:, j] = f(x_new)
    return new_data

def process_single_file(csv_file, input_dir, output_dir, episode_index):
    error_logs = []
    df = pd.read_csv(os.path.join(input_dir, csv_file))
    traj_name = csv_file.split("poses_")[1].split(".csv")[0]
    mp4_file = os.path.join(input_dir, f"{traj_name}.mp4")
    
    if not os.path.exists(mp4_file):
        error_logs.append(f"MP4 file not found: {mp4_file}")
        print(f"MP4 file not found: {mp4_file}")
        return
    
    frames, frame_count = video_to_array(mp4_file)
    
    xyz = df.iloc[:, 1:4].values
    euler_angles = df.iloc[:, 4:7].values
    gripper_widths = df.iloc[:, 7].values
    rotations = R.from_euler('zyx', euler_angles, degrees=False)
    quaternions = rotations.as_quat()
    
    initial_joint_angles = None
    end_poses = []
    new_joint_angles = []
    valid_indices = []
    data_size = len(xyz)
    if data_size != frame_count:
        msg = f"Warning: Number of frames ({frame_count}) does not match data points ({data_size}) in {csv_file}"
        error_logs.append(msg)
        print(msg)
    
    for i in range(data_size):
        x, y, z = xyz[i]
        qx, qy, qz, qw = quaternions[i]
        end_pose = [x, y, z, qx, qy, qz, qw]
        end_poses.append(end_pose)
        direction, quaternion = [x, y, z], [qx, qy, qz, qw]
        try:
            if i == 0:
                initial_joint_angles = np.array(START_QPOS)
                full_joint_angles = cartesian_to_joints(direction, quaternion, initial_joint_angles)
            else:
                full_joint_angles = cartesian_to_joints(direction, quaternion, initial_joint_angles)
            if initial_joint_angles is not None:
                error = np.linalg.norm(np.array(full_joint_angles) - np.array(initial_joint_angles))
                threshold = 0.5 
                
                # Warn if the error between current and previous joint angles is large
                if error > threshold:
                    log_msg = f"Warning: Large joint update error at frame {i}: error={error}"
                    print(log_msg)
                    error_logs.append(log_msg)
            initial_joint_angles = full_joint_angles
            six_dof_joint_angles = full_joint_angles[1:7]
            # minus the real flag gripper width, 0.02 here
            joint_angles = np.append(six_dof_joint_angles, gripper_widths[i] - 0.02)
            new_joint_angles.append(joint_angles)
            valid_indices.append(i)
        except Exception as e:
            error_msg = f"Exception at frame {i}: {str(e)}"
            print(error_msg)
            error_logs.append(error_msg)
            continue
    
    frames = frames[valid_indices]
    
    joint_angles_arr = np.array(new_joint_angles)    # shape (N, d)
    end_poses_arr = np.array(end_poses)                # shape (N, 7)
    
    original_N = joint_angles_arr.shape[0]
    target_frames = original_N * upsample_factor
    
    smoothed_joint_angles = interpolate_trajectory(joint_angles_arr, target_frames, kind="cubic")
    smoothed_end_poses = interpolate_trajectory(end_poses_arr, target_frames, kind="cubic")
    smoothed_frames = upsample_frames(frames, target_frames)
    
    output_file = os.path.join(output_dir, f"episode_{episode_index}.hdf5")
    with h5py.File(output_file, 'w') as f_out:
        f_out.attrs['compress'] = False
        f_out.attrs['sim'] = False
        f_out.create_dataset('action', data=smoothed_joint_angles)
        obs = f_out.create_group('observations')
        images = obs.create_group('images')
        images.create_dataset(name='front', dtype='uint8', data=smoothed_frames)
        obs.create_dataset('end_pose', data=smoothed_end_poses)
        obs.create_dataset('qpos', data=smoothed_joint_angles)
        print(f"Data saved to {output_file}")
    
    if error_logs:
        log_file = os.path.join(output_dir, "../logs", f"episode_{episode_index}_error.log")
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        with open(log_file, 'w') as lf:
            for log in error_logs:
                lf.write(log + "\n")
        print(f"Error logs saved to {log_file}")
    
    print(f"Processed file: {csv_file}")


def process_file_wrapper(args):
    return process_single_file(*args)

def read_csv_ik_save_hdf5_parallel(input_dir, output_dir, use_multiprocessing=True):
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    if use_multiprocessing:
        pool = Pool(cpu_count())
        tasks = [(csv_file, input_dir, output_dir, idx) for idx, csv_file in enumerate(csv_files)]
        for _ in tqdm(pool.imap_unordered(process_file_wrapper, tasks), total=len(tasks)):
            pass
        pool.close()
        pool.join()
    else:
        idx = 0
        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            process_single_file(csv_file, input_dir, output_dir, idx)
            idx += 1

if __name__ == "__main__":
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    use_multiprocessing = config.get("use_multiprocessing", True)
    read_csv_ik_save_hdf5_parallel(input_dir, output_dir, use_multiprocessing=use_multiprocessing)

    print("Processing completed.")
