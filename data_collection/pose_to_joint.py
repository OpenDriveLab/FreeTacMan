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
def process_file(csv_file: str, in_dir: str, out_dir: str, idx: int, qpos_mode: str = "joint"):
    df = pd.read_csv(os.path.join(in_dir, csv_file))
    name = csv_file.replace("poses_", "").replace(".csv", "")
    # find videos
    vids = [f for f in [f"{name}.mp4", f"{name}_Camera1.mp4", f"{name}_Camera2.mp4"]
            if os.path.exists(os.path.join(in_dir, f))]
    if not vids:
        raise FileNotFoundError(f"No video for {name}")
    frames, fc = video_to_array(os.path.join(in_dir, vids[0]))
    tac = None
    if len(vids) > 1:
        tac, _ = video_to_array(os.path.join(in_dir, vids[1]))

    # kinematics data
    xyz = df.iloc[:, 1:4].values
    euler = df.iloc[:, 4:7].values
    widths = df.iloc[:, 7].values
    quats = R.from_euler('xyz', euler, degrees=False).as_quat()

    joints, endposes, valid = [], [], []
    prev = None
    for i, (pos, quat, gr) in enumerate(zip(xyz, quats, widths)):
        if i == 0:
            prev = np.array(START_QPOS)
        full = cartesian_to_joints(pos.tolist(), quat.tolist(), prev)
        # warn large jump
        if prev is not None and np.linalg.norm(full - prev) > 0.5:
            print(f"Large joint jump at {i}")
        prev = full
        j6 = np.append(full[1:7], gr)
        joints.append(j6)
        endposes.append(np.append(pos, quat))
        valid.append(i)
    joints = np.array(joints)
    endposes = np.array(endposes)
    frames = frames[valid]
    if tac is not None:
        tac = tac[valid]

    # smoothing
    if SMOOTHING and UPSAMPLE_FACTOR > 1:
        target = len(joints) * UPSAMPLE_FACTOR
        joints = interpolate(joints, target)
        endposes = interpolate(endposes, target)
        frames = upsample_frames(frames, target)
        if tac is not None:
            tac = upsample_frames(tac, target)

    # save
    out = os.path.join(out_dir, f"episode_{idx}.hdf5")
    with h5py.File(out, 'w') as f:
        f.attrs.update({
            'compress': False,
            'sim': False
        })
        obs = f.create_group('observations')
        imgs = obs.create_group('images')
        imgs.create_dataset('front', data=frames, dtype='uint8')
        if tac is not None:
            imgs.create_dataset('tactile', data=tac, dtype='uint8')
        data = joints if qpos_mode == 'joint' else endposes
        obs.create_dataset('qpos', data=data)
        f.create_dataset('action', data=data)

    print(f"Saved: {out}")

# Parallel entry
if __name__ == '__main__':
    task_name = cfg.get("task_name", "test")
    inp, out = os.path.join('dataset', 'Data_frame_extraction', task_name), os.path.join('dataset', 'processed', task_name)
    os.makedirs(out, exist_ok=True)
    pool = Pool(cpu_count()) if cfg.get('use_multiprocessing', True) else None
    files = sorted([f for f in os.listdir(inp) if f.endswith('.csv')])
    tasks = [(f, inp, out, i, cfg.get('qpos_mode','joint')) for i, f in enumerate(files)]
    if pool:
        list(tqdm(pool.imap_unordered(lambda args: process_file(*args), tasks), total=len(tasks)))
        pool.close(); pool.join()
    else:
        for args in tqdm(tasks): process_file(*args)