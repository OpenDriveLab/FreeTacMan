import os
import cv2
import time
import winsound
import json
import numpy as np
from NatNetClient import NatNetClient

# Load configuration from JSON file
with open("config/collect.json", "r") as cfg_file:
    config = json.load(cfg_file)

time_duration = config.get("time_duration", 8) # in seconds
camera_frame_rate = config.get("camera_frame_rate", 30) # in frames per second
camera_dict = config.get("camera_dict", { "0": "front", "3": "tactile" }) # e.g., { "0": "front", "2": "tactile_left", "4": "tactile_right" }, check the real indice by running test_collection.py
camera_names = list(camera_dict.values())
camera_width = config.get("camera_width", 640)
camera_height = config.get("camera_height", 480)
exposures = config.get("exposures", {})  # e.g., { "0": -5, "3": -6 }
task_name = config.get("task_name", "default_task")
suffix = config.get("initial_suffix", 1)

cameras = []
assert camera_names, "No camera names provided in the config file."
for cam_idx, cam_name in camera_dict.items():
    cam_idx = int(cam_idx)
    camera = cv2.VideoCapture(cam_idx)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        camera.set(cv2.CAP_PROP_FPS, camera_frame_rate)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        exp_value = exposures.get(str(cam_idx), exposures.get(cam_idx, None))
        if exp_value is not None:
            camera.set(cv2.CAP_PROP_EXPOSURE, exp_value)
        cameras.append(camera)
        print(f"Camera {cam_idx} opened successfully.")
    else:
        print(f"Failed to open camera {cam_idx}")

if len(cameras) == 0:
    print("No cameras opened successfully. Exiting.")
    exit()

# Create directory for saving data
data_dir = os.path.join(".", "dataset", "raw", task_name)
os.makedirs(data_dir, exist_ok=True)

# Initialize NatNet client
natnet_client = NatNetClient(task_name, suffix)
natnet_client.set_client_address('127.0.0.1')
natnet_client.set_server_address('127.0.0.1')
natnet_client.set_use_multicast(True)

def collect_data(suffix):
    # Prepare video writers and timestamp storage
    frame_timestamps = [[] for _ in range(len(cameras))]
    video_writers = []
    timestamps_files = []
    video_filenames = []

    frame_interval = 1.0 / camera_frame_rate
    targetime_duration_frames = int(time_duration * camera_frame_rate)

    for cam_id, cam_name in camera_dict.items():
        video_filename = os.path.join(data_dir, f"{task_name}_{suffix}_{cam_name}.mp4")
        video_filenames.append(video_filename)

        timestamps_txt = os.path.join(data_dir, f"{task_name}_{suffix}_{cam_name}_timestamps.txt")
        timestamps_files.append(timestamps_txt)

        writer = cv2.VideoWriter(
            video_filename,
            cv2.VideoWriter_fourcc(*'mp4v'),
            camera_frame_rate,
            (camera_width, camera_height)
        )
        video_writers.append(writer)

    if not natnet_client.run():
        print("Failed to start NatNet client.")
        return False

    start_time = time.time()
    frame_count = 0
    print(f"Starting data collection: aiming for {targetime_duration_frames} frames (~{time_duration} seconds).")

    next_frame_time = time.time()
    try:
        while True:
            current_time = time.time()

            if frame_count >= targetime_duration_frames:
                print(f"Reached target frame count {targetime_duration_frames}. Stopping collection.")
                break

            if current_time < next_frame_time:
                time.sleep(0.001)
                continue

            print(f"Collected {frame_count}/{targetime_duration_frames} frames...")

            for cam_idx, camera in enumerate(cameras):
                ret, frame = camera.read()
                if not ret:
                    print(f"Failed to capture frame from camera {cam_idx + 1}")
                    continue

                timestamp = time.time()
                frame_timestamps[cam_idx].append(timestamp)
                video_writers[cam_idx].write(frame)

            frame_count += 1
            next_frame_time += frame_interval

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Collection stopped by user.")
                break

    except KeyboardInterrupt:
        print("Collection interrupted via Ctrl+C.")
    finally:
        # Release video writers
        for writer in video_writers:
            writer.release()

        # Save timestamps to files
        for cam_idx, timestamps in enumerate(frame_timestamps):
            with open(timestamps_files[cam_idx], 'w') as f:
                for ts in timestamps:
                    f.write(f"{ts:.3f}\n")

        actual_duration = time.time() - start_time
        print(f"Data collection complete: {frame_count} frames in {actual_duration:.2f} seconds.")
        print(f"Actual frame rate: {frame_count / actual_duration:.2f} fps")

        for cam_idx in range(len(cameras)):
            print(f"Camera {cam_idx + 1} video saved as {os.path.basename(video_filenames[cam_idx])}")
            print(f"Camera {cam_idx + 1} timestamps saved as {os.path.basename(timestamps_files[cam_idx])}")
        print(f"OptiTrack data saved as {task_name}_{suffix}_optitrack.npy")

        winsound.Beep(1000, 500)
    return True

# Start data collection
if __name__ == "__main__":
    cmd = input("Press 'Enter' to start data collection or 'q' to quit: ")
    if cmd.lower() == 'q':
        exit()
    
    while True:
        if not collect_data(suffix):
            break

        natnet_client.shutdown()
        user_input = input("Press 'Enter' to continue or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        elif user_input != '':
            print("Invalid input. Please press 'Enter' to continue or 'q' to quit.")
            continue

        suffix += 1
        natnet_client = NatNetClient(task_name, suffix)
        natnet_client.set_client_address('127.0.0.1')
        natnet_client.set_server_address('127.0.0.1')
        natnet_client.set_use_multicast(True)

    # Release camera resources and shut down NatNet
    for camera in cameras:
        camera.release()
    natnet_client.shutdown()
