import os
import pandas as pd
import cv2

# 定义任务名
task_name = 'unpack'

# 定义文件夹路径
input_folder = f'processed_data/{task_name}'
output_folder = f'Data_frame_extraction/{task_name}'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取所有CSV和MP4文件
csv_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.csv')])
mp4_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')])

# 找到最小的帧数
min_frames = float('inf')
for csv_file in csv_files:
    df = pd.read_csv(os.path.join(input_folder, csv_file))
    min_frames = min(min_frames, len(df))

# 处理每个CSV文件和对应的两个MP4文件
for i in range(0, len(mp4_files), 2):  # 每次处理两个MP4文件
    csv_file = csv_files[i//2]  # 对应的CSV文件
    mp4_file1 = mp4_files[i]    # 第一个视频
    mp4_file2 = mp4_files[i+1]  # 第二个视频
    
    # 处理CSV文件
    df = pd.read_csv(os.path.join(input_folder, csv_file))
    df = df.iloc[:min_frames]  # 保留最小帧数的行
    df.to_csv(os.path.join(output_folder, csv_file), index=False)

    # 处理两个MP4文件
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

print("处理完成，文件已保存到", output_folder)

# 获取所有处理后的MP4文件
mp4_files = [f for f in os.listdir(output_folder) if f.endswith('.mp4')]

# 验证每个视频的帧数
for mp4_file in mp4_files:
    video_path = os.path.join(output_folder, mp4_file)
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频文件 {mp4_file} 的帧数为: {frame_count}")
    
    cap.release()
