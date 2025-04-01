import os
import pandas as pd
import random
import shutil
def count_csv_dimensions(folder_path):
    # 用于存储不同维度组合的计数
    dimension_counts = {}

    # 遍历指定目录下的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # 构造完整的文件路径
            file_path = os.path.join(folder_path, file_name)
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                # 获取行数和列数
                rows, cols = df.shape
                # 创建维度字符串
                dimension_str = f"{cols}x{rows}"
                # 按照维度字符串计数
                if dimension_str in dimension_counts:
                    dimension_counts[dimension_str] += 1
                else:
                    print(file_path)
                    dimension_counts[dimension_str] = 1
            except Exception as e:
                print(f"无法处理文件 {file_name}: {e}")

    # 输出结果
    for dim, count in dimension_counts.items():
        print(f"{dim}（长宽）的有 {count} 个")



def randomly_select_and_copy_csvs(source_folder, selected_folder, remaining_folder, num_samples=2000):
    # 收集所有csv文件路径
    csv_files = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    # 如果找到的csv文件少于num_samples，则调整num_samples为实际数量
    if len(csv_files) < num_samples:
        print(f"注意：只找到了{len(csv_files)}个CSV文件，小于指定的抽样数量")
        num_samples = len(csv_files)

    # 随机选取指定数量的csv文件
    selected_files = set(random.sample(csv_files, num_samples))
    remaining_files = set(csv_files) - selected_files

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(selected_folder):
        os.makedirs(selected_folder)
    if not os.path.exists(remaining_folder):
        os.makedirs(remaining_folder)

    def copy_files(file_list, destination_folder):
        for file_path in file_list:
            try:
                # 构造新的文件路径
                relative_path = os.path.relpath(file_path, source_folder)
                new_file_path = os.path.join(destination_folder, os.path.basename(relative_path))
                # 复制文件
                shutil.copy2(file_path, new_file_path)
                print(f"复制文件: {file_path} -> {new_file_path}")
            except Exception as e:
                print(f"无法复制文件 {file_path}: {e}")

    # 复制选定的csv文件到selected_folder
    copy_files(selected_files, selected_folder)

    # 复制剩余的csv文件到remaining_folder
    copy_files(remaining_files, remaining_folder)


# 设置源文件夹和目标文件夹路径
source_folder = 'json/pos'
selected_folder = 'data/train/pos'
remaining_folder = 'data/test/pos'

# 随机抽取个CSV文件，并将剩余的移动到另一个文件夹
randomly_select_and_copy_csvs(source_folder, selected_folder, remaining_folder)

# 设置你想检查的文件夹路径
# folder_path = 'json/neg'
# count_csv_dimensions(folder_path)
