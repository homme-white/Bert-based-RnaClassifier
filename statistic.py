import os

def get_text_length_without_spaces(file_path):
    """读取文件并返回不包括空格的字符长度"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # 去除所有空格（包括普通空格、换行符和制表符）
            content_no_spaces = ''.join(content.split())
            return len(content_no_spaces)
    except Exception as e:
        print(f"无法读取文件 {file_path}: {e}")
        return 0

def main(directory):
    text_lengths = []
    file_names = []

    # 遍历指定目录下的所有文件
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.seq'):
                file_path = os.path.join(root, filename)
                length = get_text_length_without_spaces(file_path)
                if length > 0:  # 只有当文件成功读取且长度大于0时才添加到列表
                    text_lengths.append(length)
                    file_names.append(file_path)

    if not text_lengths:
        print("没有找到任何文本文件或所有文本文件为空。")
        return

    # 计算最长、最短和平均字符长度
    max_length = max(text_lengths)
    min_length = min(text_lengths)
    avg_length = sum(text_lengths) / len(text_lengths)

    # 找到最短字符长度对应的文件名
    min_length_index = text_lengths.index(min_length)
    min_length_file = file_names[min_length_index]

    # 输出结果
    print(f"最长字符长度 (不包括空格): {max_length}")
    print(f"最短字符长度 (不包括空格): {min_length}")
    print(f"最短字符长度的文件: {min_length_file}")
    print(f"平均字符长度 (不包括空格): {avg_length:.2f}")

    # 统计每个字符长度区间内的文件数量
    interval_size = 1000
    max_interval = (max_length // interval_size + 1) * interval_size
    intervals = [(i, i + interval_size) for i in range(0, max_interval, interval_size)]

    interval_counts = {interval: 0 for interval in intervals}

    for length in text_lengths:
        for start, end in intervals:
            if start <= length < end:
                interval_counts[(start, end)] += 1
                break

    # 输出每个字符长度区间的文件数量
    print("\n字符长度区间统计:")
    for (start, end), count in interval_counts.items():
        print(f"{start}-{end}: {count} 个")

if __name__ == "__main__":
    # 硬编码的目标目录路径
    target_directory = 'seq/pc_mRNA'
    main(target_directory)