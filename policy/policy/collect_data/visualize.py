import os

# 定义变量
dataset_dir = "~/data"
task_name = "wipe"

# 循环episode_idx从0到18
for episode_idx in range(19):  # range(19)生成从0到18的数字序列
    # 构建命令字符串
    command = f"python visualize_episodes.py --dataset_dir {dataset_dir} --task_name {task_name} --episode_idx {episode_idx}"
    
    # 打印当前执行的命令（可选）
    print(f"Executing: {command}")
    
    # 执行命令
    os.system(command)

print("所有命令执行完毕")