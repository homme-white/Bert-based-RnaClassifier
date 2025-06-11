import os
import shutil

def count_seq_lengths(folder_path):
    length_counts = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.seq'):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    bases = content.split()
                    length = len(bases)
                    if length == 0:
                        continue  # 跳过空文件

                    # 计算区间
                    interval_start = ((length - 1) // 500) * 500 + 1
                    interval_end = interval_start + 499
                    interval_str = f"{interval_start}-{interval_end}"

                    length_counts[interval_str] = length_counts.get(interval_str, 0) + 1
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")

    # 按区间排序输出
    sorted_intervals = sorted(length_counts.keys(), key=lambda x: int(x.split('-')[0]))
    for interval in sorted_intervals:
        print(f"{interval}: {length_counts[interval]} 个")


def extract_files_in_range(source_folder, target_folder, min_length=1, max_length=3000):
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    copied_files = 0  # 统计成功复制的文件数

    for file_name in os.listdir(source_folder):
        if not file_name.endswith('.seq'):
            continue  # 仅处理.seq文件

        file_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(target_folder, file_name)

        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                bases = content.split()
                length = len(bases)

                # 检查长度是否在范围内（包含边界）
                if min_length <= length <= max_length:
                    shutil.copy2(file_path, target_path)
                    copied_files += 1
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")

    print(f"共复制符合条件的文件：{copied_files} 个到 {target_folder}")


# 使用示例（替换为实际文件夹路径）
folder_path = 'seq/Final/Bert/train/neg'  # 根据实际路径修改
count_seq_lengths(folder_path)

# # 使用示例
# source_folder = 'seq/ori_seq/lncRNA'  # 原始数据路径
# target_folder = 'seq/0t3k/lnc_3000'  # 目标存放路径
# extract_files_in_range(source_folder, target_folder, 1, 3000)
