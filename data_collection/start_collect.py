import time
import numpy as np
import cv2
from NatNetClient import NatNetClient
import os  # 添加在文件开头的import部分
import winsound

# 参数设置
t_total =   8# 总时间（秒）

# 相机帧率
camera_frame_rate = 30

# 相机配置
cameras = []

# 打印出电脑可用的相机索引
# camera_indices = []
# for i in range(10):  # 假设最多有10个相机
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         camera_indices.append(i)
#         cap.release()
# print(f"Available camera indices: {camera_indices}")
camera_indices = [0 ,3]  # 三个相机的索引

# 设置相机分辨率
camera_width = 640  # 设置相机宽度
camera_height = 480  # 设置相机高度

for cam_idx in camera_indices:
    camera = cv2.VideoCapture(cam_idx)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 禁用自动对焦
        camera.set(cv2.CAP_PROP_FPS, camera_frame_rate)  # 设置相机帧率
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        camera.set(cv2.CAP_PROP_EXPOSURE, -6)  # 设置相机曝光为固定值
        cameras.append(camera)
        print(f"Camera {cam_idx} opened successfully.")
    else:
        print(f"Failed to open camera {cam_idx}")
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_EXPOSURE, -5)  # 设置相机曝光为固定值
# camera = cv2.VideoCapture(0)
# if camera.isOpened():
#     camera.set(cv2.CAP_PROP_EXPOSURE, -6)  # 设置相机曝光为固定值
#     print(f"Camera {cam_idx} opened successfully.")
# else:
#     print(f"Failed to open camera {cam_idx}")
        

if len(cameras) == 0:
    print("No cameras opened successfully.")
    exit()

# 用户输入任务名称
task_name = input("Please input task name: ")
suffix = 1

# 创建保存数据的目录
data_dir = os.path.join(".", "data", task_name)
os.makedirs(data_dir, exist_ok=True)

# OptiTrack 配置
natnet_client = NatNetClient(task_name, suffix)  # 传递 task_name 和 suffix
natnet_client.set_client_address('127.0.0.1')
natnet_client.set_server_address('127.0.0.1')
natnet_client.set_use_multicast(True)

def collect_data(suffix):
    # 初始化存储
    frame_timestamps = [[] for _ in range(len(cameras))]
    video_writers = []
    timestamps_files = []
    video_filenames = []
    
    # 添加帧率控制
    frame_interval = 1.0 / camera_frame_rate  # 计算每帧的理想时间间隔
    next_frame_time = 0
    
    # 计算目标总帧数
    target_total_frames = int(t_total * camera_frame_rate)
    
    for i in range(len(cameras)):
        video_filename = os.path.join(data_dir, f"{task_name}_{suffix}_camera{i+1}.mp4")
        video_filenames.append(video_filename)  # 保存文件名
        timestamps_txt_filename = os.path.join(data_dir, f"{task_name}_{suffix}_Camera{i+1}Timestamps.txt")
        
        video_writer = cv2.VideoWriter(
            video_filename,
            cv2.VideoWriter_fourcc(*'mp4v'),
            camera_frame_rate,
            (camera_width, camera_height)
        )
        video_writers.append(video_writer)
        timestamps_files.append(timestamps_txt_filename)

    if not natnet_client.run():
        print("Failed to start NatNet client")
        return False

    # 开始采集
    start_time = time.time()
    frame_count = 0  # 记录已采集的帧数
    print(f"开始采集数据，计划采集{target_total_frames}帧(约{t_total}秒)...")
    
    try:
        next_frame_time = time.time()  # 初始化下一帧的时间
        while True:
            current_time = time.time()
            
            # 检查是否达到目标帧数
            if frame_count >= target_total_frames:
                print(f"已达到目标帧数 {target_total_frames} 帧，停止采集")
                break

            # 等待直到达到下一帧的时间
            if current_time < next_frame_time:
                time.sleep(0.001)  # 小睡避免CPU过度使用
                continue

            print(f"已采集：{frame_count}/{target_total_frames}帧")

            # 采集所有相机的帧
            for cam_idx, camera in enumerate(cameras):
                # tic=time.time()
                ret, frame = camera.read()
                # print("check",time.time()-tic)
                if not ret:
                    print(f"Failed to capture frame from camera {cam_idx+1}")
                    continue

                current_frame_time = time.time()
                frame_timestamps[cam_idx].append(current_frame_time)

                video_writers[cam_idx].write(frame)

            # 更新下一帧的时间和帧计数器
            next_frame_time += frame_interval
            frame_count += 1

            # 检查是否按下 'q' 键提前退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户手动停止采集")
                break

    except KeyboardInterrupt:
        print("用户通过Ctrl+C中断采集")
    finally:
        # 确保资源正确释放
        for writer in video_writers:
            writer.release()

        # 保存所有相机的时间戳
        for cam_idx, timestamps in enumerate(frame_timestamps):
            with open(timestamps_files[cam_idx], 'w') as f:
                for ts in timestamps:
                    f.write(f"{ts:.3f}\n")

        actual_duration = time.time() - start_time
        print(f"数据采集完成，共{frame_count}帧，用时{actual_duration:.2f}秒")
        print(f"实际帧率: {frame_count/actual_duration:.2f} fps")
        for cam_idx in range(len(cameras)):
            print(f"相机{cam_idx+1}视频已保存为 {os.path.basename(video_filenames[cam_idx])}")
            print(f"相机{cam_idx+1}时间戳已保存为 {os.path.basename(timestamps_files[cam_idx])}")
        print(f"OptiTrack数据已保存为 {task_name}_{suffix}_optitrack.npy")
        winsound.Beep(1000, 500)
    return True

while True:
    # 进行数据采集
    if not collect_data(suffix):
        break

    # 暂停采集，等待用户输入
    natnet_client.shutdown()
    user_input = input("请输入回车以继续下一组采集，输入 'q' 退出: ")
    if user_input.lower() == 'q':
        break
    elif user_input.lower() != '':
        print("输入无效，请输入回车继续或 'q' 退出。")
        continue

    # 重新启动 NatNet 客户端
    natnet_client = NatNetClient(task_name, suffix + 1)  # 更新 suffix
    natnet_client.set_client_address('127.0.0.1')
    natnet_client.set_server_address('127.0.0.1')
    natnet_client.set_use_multicast(True)

    suffix += 1

# 释放所有相机资源和关闭 NatNet 客户端
for camera in cameras:
    camera.release()
natnet_client.shutdown()