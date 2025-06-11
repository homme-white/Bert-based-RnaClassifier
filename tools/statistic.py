import os
import random
import shutil
from collections import defaultdict


def stratified_sample_and_split(
        source_folder,
        target_train,
        target_val,
        total_samples=30000,
        min_per_interval=0,
        split_ratio=(1, 0)
):
    # 按区间分组统计文件
    intervals = defaultdict(list)
    for file_name in os.listdir(source_folder):
        if not file_name.endswith('.seq'):
            continue
        file_path = os.path.join(source_folder, file_name)
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                bases = content.split()
                length = len(bases)
                if not (1 <= length <= 3000):
                    continue  # 跳过超出范围的文件

                # 计算区间
                interval_start = ((length - 1) // 500) * 500 + 1
                interval_end = interval_start + 499
                interval_str = f"{interval_start}-{interval_end}"
                intervals[interval_str].append(file_path)
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")

    # 检查每个区间是否满足最小文件数要求
    for interval, files in intervals.items():
        if len(files) < min_per_interval:
            raise ValueError(f"区间 {interval} 的文件不足 {min_per_interval} 个")

    # 计算基础采样量和剩余需求
    num_intervals = len(intervals)
    base_total = num_intervals * min_per_interval
    remaining = total_samples - base_total
    if remaining < 0:
        raise ValueError("总样本数不足，无法满足分层要求")

    # 计算各区间可分配的额外文件数
    remaining_available = {
        interval: len(files) - min_per_interval
        for interval, files in intervals.items()
    }
    total_available = sum(remaining_available.values())

    # 分配额外文件数
    extra_alloc = {}
    for interval in intervals:
        avail = remaining_available[interval]
        extra = int(avail / total_available * remaining)
        extra_alloc[interval] = extra

    # 调整余数（四舍五入误差）
    total_extra = sum(extra_alloc.values())
    diff = remaining - total_extra
    if diff != 0:
        # 随机分配余数
        for _ in range(abs(diff)):
            interval = random.choice(list(intervals.keys()))
            extra_alloc[interval] += 1 if diff > 0 else -1
            if sum(extra_alloc.values()) == remaining:
                break

    # 收集选中的文件
    selected_files = []
    for interval, files in intervals.items():
        total_take = min_per_interval + extra_alloc[interval]
        selected = random.sample(files, total_take)
        selected_files.extend(selected)

    # 打乱顺序并分割数据集
    random.shuffle(selected_files)
    split_idx = int(len(selected_files) * split_ratio[0])
    train_set = selected_files[:split_idx]
    val_set = selected_files[split_idx:]

    # 创建目标文件夹
    os.makedirs(target_train, exist_ok=True)
    os.makedirs(target_val, exist_ok=True)

    # 定义复制函数
    def copy_files(file_list, target_dir):
        for src in file_list:
            fname = os.path.basename(src)
            dst = os.path.join(target_dir, fname)
            shutil.copy2(src, dst)
        print(f"成功复制 {len(file_list)} 个文件到 {target_dir}")

    # 执行复制操作
    copy_files(train_set, target_train)
    copy_files(val_set, target_val)

    print(f"总样本：{len(selected_files)}（训练集：{len(train_set)}，验证集：{len(val_set)}）")


# 使用示例
source_folder = 'seq/0t3k_all/lnc_3000'  # 原始数据路径
stratified_sample_and_split(
    source_folder,
    target_train='seq/Final/Bert/train/pos',
    target_val='seq/Final/Bert/train/',
    total_samples=30000,
    min_per_interval=0
)
