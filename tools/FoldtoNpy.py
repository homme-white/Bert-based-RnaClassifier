import os
import numpy as np


def extract_structure(file_path):
    """
    提取 RNAfold 输出文件中的二级结构。
    :param file_path: RNAfold 输出文件路径。
    :return: 二级结构字符串。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 提取最后一行的结构信息（忽略自由能）
    structure_line = lines[-1].strip()
    structure = structure_line.split()[0]  # 去除结尾的自由能值
    return structure


def map_to_one_hot(structure):
    """
    将二级结构映射为三维 one-hot 向量。
    :param structure: 二级结构字符串。
    :return: 三维 one-hot 数组（shape: (len(structure), 3)）。
    """
    mapping = {
        '(': [0, 1, 0],
        '.': [1, 0, 0],
        ')': [0, 0, 1]
    }
    one_hot = []
    for char in structure:
        one_hot.append(mapping[char])
    return np.array(one_hot, dtype=np.float32)


def pad_to_fixed_length(one_hot_array, target_length=3000):
    """
    将 one-hot 数组补全到指定长度，不足则用零填充。
    :param one_hot_array: 原始 one-hot 数组（shape: (n, 3)）。
    :param target_length: 目标长度（默认 3000）。
    :return: 补全后的数组（shape: (target_length, 3)）。
    """
    current_length = one_hot_array.shape[0]
    if current_length > target_length:
        raise ValueError("结构长度超过目标长度！")

    # 创建零填充数组
    padded = np.zeros((target_length, 3), dtype=np.float32)
    padded[:current_length] = one_hot_array
    return padded


def process_files(input_dir, output_npy):
    """
    批量处理文件并保存为 .npy 文件。
    :param input_dir: 包含 RNAfold 输出文件的文件夹路径。
    :param output_npy: 输出的 .npy 文件路径。
    """
    # 获取文件列表并按字典顺序排序
    folded_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.folded')])

    if not folded_files:
        print(f"输入文件夹 '{input_dir}' 中没有找到 .folded 文件！")
        return

    # 收集所有处理后的数据
    all_data = []
    for folded_file in folded_files:
        file_path = os.path.join(input_dir, folded_file)

        try:
            # 提取结构
            structure = extract_structure(file_path)
            # 映射为 one-hot
            one_hot = map_to_one_hot(structure)
            # 补全到固定长度
            padded = pad_to_fixed_length(one_hot)
            # 保存到列表
            all_data.append(padded)

            print(f"处理完成：{folded_file} → shape: {padded.shape}")

        except Exception as e:
            print(f"处理文件 '{folded_file}' 失败：{e}")

    # 将列表转换为 numpy 数组（shape: (文件数, 3000, 3)）
    final_array = np.stack(all_data, axis=0)

    # 保存为 .npy 文件
    np.save(output_npy, final_array)
    print(f"数据已保存到：{output_npy}，形状：{final_array.shape}")


if __name__ == "__main__":
    # 定义输入文件夹和输出路径
    input_directory = "Final/test/pc_neg"  # 替换为你的输入文件夹路径
    output_npy_file = "output/Final/pc_neg_test"  # 替换为你的输出 .npy 文件路径

    # 执行处理
    process_files(input_directory, output_npy_file)
