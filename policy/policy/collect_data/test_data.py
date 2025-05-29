import h5py

def print_hdf5_structure(file_path):
    try:
        # 打开 HDF5 文件
        with h5py.File(file_path, 'r') as file:
            # 定义一个递归函数来打印文件结构
            def print_group(group, indent=0, level=0):
                # 若层级大于 3 则停止递归
                if level > 3:
                    return
                # 打印当前组的名称
                print('  ' * indent + f'Group: {group.name}')
                # 遍历当前组中的所有项
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        # 如果是组，递归调用 print_group 函数，层级加 1
                        print_group(item, indent + 1, level + 1)
                    elif isinstance(item, h5py.Dataset):
                        # 如果是数据集，打印数据集的名称、形状和数据类型
                        print('  ' * (indent + 1) + f'Dataset: {item.name}, Shape: {item.shape}, Dtype: {item.dtype}')

            # 从根组开始打印文件结构，初始层级为 0
            print_group(file)
    except Exception as e:
        print(f"读取文件时出现错误: {e}")

# 替换为你的 HDF5 文件路径
file_path = '/home/agilex/cobot_magic/collect_data/test_data/aloha_mobile_dummy/episode_0.hdf5'
print_hdf5_structure(file_path)