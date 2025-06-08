import os
import cv2
import h5py
import json
import numpy as np
import pandas as pd
import ikpy.chain
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Optional

# Load configuration
try:
    with open('config/process.json', 'r') as cfg:
        cfg = json.load(cfg)
except FileNotFoundError:
    cfg = {}

# Settings
START_QPOS = cfg["start_qpos"]
ACTIVE_MASK = cfg["active_joint_mask"]
SMOOTHING = cfg.get("enable_smoothing", False)
UPSAMPLE_FACTOR = cfg.get("upsample_factor", 1)
URDF_PATH = cfg.get("urdf_path", "./assets/piper_description.urdf")

def video_to_array(path: str) -> Tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    arr = np.empty((count, h, w, 3), dtype=np.uint8)
    for i in range(count):
        ret, frame = cap.read()
        if not ret:
            arr = arr[:i]
            count = i
            break
        arr[i] = frame
    cap.release()
    return arr, count

# IK chain
chain = ikpy.chain.Chain.from_urdf_file(
    urdf_file=URDF_PATH,
    active_links_mask=ACTIVE_MASK
)

def cartesian_to_joints(pos: List[float], quat: List[float], init: Optional[List[float]] = None) -> np.ndarray:
    rot = R.from_quat(quat).as_matrix()
    init = init or [0] * len(chain)
    return chain.inverse_kinematics(pos, rot, orientation_mode='all', initial_position=init)

# Trajectory interpolation
def interpolate(data: np.ndarray, frames: int, kind="cubic") -> np.ndarray:
    orig, dim = data.shape
    x_old = np.arange(orig)
    x_new = np.linspace(0, orig - 1, frames)
    out = np.zeros((frames, dim))
    for d in range(dim):
        f = interp1d(x_old, data[:, d], kind=kind)
        out[:, d] = f(x_new)
    return out

# Upsample frames
def upsample_frames(frames: np.ndarray, frames_out: int) -> np.ndarray:
    return interpolate(frames.reshape(frames.shape[0], -1), frames_out).reshape((frames_out, *frames.shape[1:]))

# Core processing
def process_file(csv_file: str, in_dir: str, out_dir: str, idx: int, qpos_mode: str = "joint", camera_names: List[str] = ["front", "tactile"]):
    error_logs = []
    df = pd.read_csv(os.path.join(in_dir, csv_file))
    name = csv_file.split("poses_")[1].split(".csv")[0]
    vid_names = [f"{name}_{cam_name}.mp4" for cam_name in camera_names]
    # find videos
    vids = [f for f in vid_names if os.path.exists(os.path.join(in_dir, f))]
    if not vids:
        raise FileNotFoundError(f"No video for {name}")

    camera_frames = []
    for vid in vids:
        frames, frame_count = video_to_array(os.path.join(in_dir, vid))
        camera_frames.append(frames)

    # kinematics data
    xyz = df.iloc[:, 1:4].values
    euler_angles = df.iloc[:, 4:7].values
    gripper_widths = df.iloc[:, 7].values
    rotations = R.from_euler('xyz', euler_angles, degrees=False)
    quaternions = rotations.as_quat()

    initial_joint_angles = None
    end_poses = []
    new_joint_angles = []
    new_poses = []
    valid_indices = []
    data_size = len(xyz)
    if data_size != frame_count:
        msg = f"Warning: Number of frames ({frame_count}) does not match data points ({data_size}) in {csv_file}"
        error_logs.append(msg)
        print(msg)
        if data_size > frame_count:
            frames = upsample_frames(frames, data_size)
            tac_frames = upsample_frames(tac_frames, data_size)
    
    for i in range(data_size):
        x, y, z = xyz[i]
        qx, qy, qz, qw = quaternions[i]
        roll, pitch, yaw = euler_angles[i]
        end_pose = [x, y, z, qx, qy, qz, qw]
        end_poses.append(end_pose)
        direction, quaternion = [x, y, z], [qx, qy, qz, qw]
        seven_dof_pose = [x, y, z, roll, pitch, yaw, gripper_widths[i]]
        new_poses.append(seven_dof_pose)
        
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
            # minus the real gap betweein gripper markers, 0.02m here
            joint_angles = np.append(six_dof_joint_angles, gripper_widths[i])
            new_joint_angles.append(joint_angles)
            valid_indices.append(i)
        except Exception as e:
            error_msg = f"Exception at frame {i}: {str(e)}"
            print(error_msg)
            error_logs.append(error_msg)
            continue

    # smoothing
    if SMOOTHING and UPSAMPLE_FACTOR > 1:
        target = len(joints) * UPSAMPLE_FACTOR
        joints = interpolate(joints, target)
        endposes = interpolate(endposes, target)
        for i, frames in enumerate(camera_frames):
            camera_frames[i] = upsample_frames(frames, target)

    # save
    output_file = os.path.join(out_dir, f"episode_{idx}.hdf5")
    with h5py.File(output_file, 'w') as f_out:
        f_out.attrs['compress'] = False
        f_out.attrs['sim'] = False
        obs = f_out.create_group('observations')
        images = obs.create_group('images')
        for i in range(len(camera_names)):
            images.create_dataset(name=camera_names[i], dtype='uint8', data=camera_frames[i])
        if qpos_mode == 'joint':
            obs.create_dataset('qpos', data=new_joint_angles)
            f_out.create_dataset('action', data=new_joint_angles)
        else:
            obs.create_dataset('qpos', data=new_poses)
            f_out.create_dataset('action', data=new_poses)
        obs.create_dataset('end_pose', data=np.array(end_poses))
        print(f"Data saved to {output_file}")

    if error_logs:
        log_file = os.path.join(out_dir, "../logs", f"episode_{idx}_error.log")
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        with open(log_file, 'w') as lf:
            for log in error_logs:
                lf.write(log + "\n")
        print(f"Error logs saved to {log_file}")
    
    print(f"Processed file: {csv_file}")

def process_file_wrapper(args):
    return process_file(*args)

def read_csv_ik_save_hdf5_parallel(input_dir, output_dir, use_multiprocessing=True, qpos_mode="joint", camera_names=["front", "tactile"]):
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    if use_multiprocessing:
        pool = Pool(cpu_count())
        tasks = [(csv_file, input_dir, output_dir, idx, qpos_mode, camera_names) for idx, csv_file in enumerate(csv_files)]
        for _ in tqdm(pool.imap_unordered(process_file_wrapper, tasks), total=len(tasks)):
            pass
        pool.close()
        pool.join()
    else:
        idx = 0
        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            process_file(csv_file, input_dir, output_dir, idx, qpos_mode, camera_names)
            idx += 1

# Parallel entry
if __name__ == '__main__':
    task_name = cfg.get("task_name", "test")
    camera_dict = cfg.get("camera_dict", { "0": "front", "3": "tactile" })
    qpos_mode = cfg.get("qpos_mode", "joint")
    camera_names = list(camera_dict.values())
    assert camera_names, "No cameras found in the camera_dict"
    inp, out = os.path.join('dataset', 'extracted_eep_data', task_name), os.path.join('dataset', 'processed', task_name)
    os.makedirs(out, exist_ok=True)
    use_multiprocessing = cfg.get("use_multiprocessing", True)
    read_csv_ik_save_hdf5_parallel(inp, out, use_multiprocessing=use_multiprocessing, qpos_mode=qpos_mode, camera_names=camera_names)

    print("Processing completed.")