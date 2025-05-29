import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from datetime import datetime
from collections import defaultdict

# 读取数据
with open('data/classify/classify_3.txt', 'r') as f:
    lines = f.readlines()

# 解析数据
raw_data = defaultdict(lambda: defaultdict(list))  # 按时间和marker_id分组的数据

# 时间格式转换函数
def parse_time(time_str):
    return float(time_str.strip())  # 直接将字符串转换为浮点数

# 首先按时间和marker_id分组收集数据
for i in range(0, len(lines), 3):
    if i+2 < len(lines):
        time = parse_time(lines[i])
        marker_id = int(lines[i+1].strip())
        pos_line = lines[i+2].strip()
        if pos_line.startswith('pos:'):
            pos_line = pos_line[4:].strip()
        pos_line = pos_line.strip('[]')
        pos = [float(x.strip()) for x in pos_line.split(',')]
        raw_data[time][marker_id].append(pos)

# 处理后的数据
times = []
marker_ids = []
positions = []

# 对每个时间戳的数据进行处理
for time in sorted(raw_data.keys()):
    for marker_id in raw_data[time]:
        pos_list = raw_data[time][marker_id]
        n_points = len(pos_list)
        if n_points > 1:
            # 在1秒内均匀分布时间点
            sub_times = np.linspace(time, time + 0.999, n_points)
            for sub_t, pos in zip(sub_times, pos_list):
                times.append(sub_t)
                marker_ids.append(marker_id)
                positions.append(pos)
        else:
            # 单个点直接添加
            times.append(time)
            marker_ids.append(marker_id)
            positions.append(pos_list[0])

# 转换为numpy数组
times = np.array(times)
marker_ids = np.array(marker_ids)
positions = np.array(positions)

# 按时间排序所有数据
sort_idx = np.argsort(times)
times = times[sort_idx]
marker_ids = marker_ids[sort_idx]
positions = positions[sort_idx]

# 标准化时间（使其从0开始）
times = times - times[0]

# 获取坐标轴的范围
x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

# 添加边距（每个方向增加20%的范围）
x_padding = (x_max - x_min) * 0.2
y_padding = (y_max - y_min) * 0.2
z_padding = (z_max - z_min) * 0.2

x_min -= x_padding
x_max += x_padding
y_min -= y_padding
y_max += y_padding
z_min -= z_padding
z_max += z_padding

# 创建图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 为每个marker创建散点对象
unique_ids = np.unique(marker_ids)
# 使用 matplotlib 的颜色循环
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_ids)))
scatters = []
trails = []

for i, marker_id in enumerate(unique_ids):
    scatter = ax.scatter([], [], [], c=colors[i], label=f'Marker {marker_id}')
    scatters.append(scatter)
    trail, = ax.plot([], [], [], c=colors[i], alpha=0.5)
    trails.append(trail)

# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Marker运动轨迹动画')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
ax.legend()

# 添加时间显示文本
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

# 添加网格
ax.grid(True)

# 定义动画更新函数
def update(frame):
    trail_length = 20  # 轨迹长度
    
    # 计算当前时间
    current_time = times[0] + frame * (times[-1] - times[0]) / 200
    
    # 更新时间显示
    time_text.set_text(f'运行时间: {current_time:.2f} 秒')
    
    for i, marker_id in enumerate(unique_ids):
        mask = marker_ids == marker_id
        marker_positions = positions[mask]
        marker_times = times[mask]
        
        # 找到当前时间点之前的所有位置
        time_mask = marker_times <= current_time
        
        if np.any(time_mask):
            current_pos = marker_positions[time_mask]
            # 更新散点位置
            scatters[i]._offsets3d = (current_pos[-1:, 0], 
                                     current_pos[-1:, 1], 
                                     current_pos[-1:, 2])
            
            # 更新轨迹
            trail_start = max(0, len(current_pos) - trail_length)
            trails[i].set_data(current_pos[trail_start:, 0], 
                             current_pos[trail_start:, 1])
            trails[i].set_3d_properties(current_pos[trail_start:, 2])
    
    return scatters + trails + [time_text]

# 创建动画
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()
