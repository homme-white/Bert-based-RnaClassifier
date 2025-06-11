import os
import subprocess
import numpy as np
import tempfile


def save_rna_sequence_to_seq(sequence, file_id, output_dir):
    """
    将RNA序列保存为 .seq 文件。
    """
    seq_file_path = os.path.join(output_dir, f"{file_id}.seq")
    with open(seq_file_path, 'w') as f:
        f.write(sequence.strip().replace(' ', '').replace('\n', ''))
    return seq_file_path


def run_rnafold(seq_file_path, output_dir):
    """
    使用 RNAfold 生成 .folded 文件。
    """
    base_name = os.path.splitext(os.path.basename(seq_file_path))[0]
    folded_file = os.path.join(output_dir, f"{base_name}.folded")

    try:
        with open(folded_file, 'w') as out_f:
            subprocess.run(
                ["RNAfold", "-i", seq_file_path],
                stdout=out_f,
                stderr=subprocess.PIPE,
                check=True
            )
        return folded_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"RNAfold 执行失败: {e.stderr.decode('utf-8')}")


def extract_structure(file_path):
    """
    提取 RNAfold 输出文件中的二级结构。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    structure_line = lines[-1].strip()
    structure = structure_line.split()[0]
    return structure


def map_to_one_hot(structure):
    """
    将二级结构映射为三维 one-hot 向量。
    """
    mapping = {
        '(': [0, 1, 0],
        '.': [1, 0, 0],
        ')': [0, 0, 1]
    }
    return np.array([mapping[char] for char in structure], dtype=np.float32)


def pad_to_fixed_length(one_hot_array, target_length=3000):
    """
    补零至固定长度。
    """
    current_length = one_hot_array.shape[0]
    if current_length > target_length:
        raise ValueError("结构长度超过目标长度！")

    padded = np.zeros((target_length, 3), dtype=np.float32)
    padded[:current_length] = one_hot_array
    return padded


def rna_sequence_to_npy(rna_sequence, file_id, output_npy_file="output.npy"):
    """
    端到端处理：从RNA序列到.npy文件。
    """
    # 创建临时目录存储中间文件
    with tempfile.TemporaryDirectory() as tmpdir:
        print("保存 .seq 文件...")
        seq_file = save_rna_sequence_to_seq(rna_sequence, file_id, tmpdir)

        print("运行 RNAfold...")
        folded_file = run_rnafold(seq_file, tmpdir)

        print("提取结构并编码...")
        structure = extract_structure(folded_file)
        one_hot = map_to_one_hot(structure)
        padded = pad_to_fixed_length(one_hot)

        print("保存为 .npy 文件...")
        np.save(output_npy_file, padded[np.newaxis, ...])  # 添加 batch 维度

        print(f"结果已保存至: {output_npy_file}")
        print(f"输出形状: {padded.shape}")


if __name__ == "__main__":
    # 用户输入RNA序列和标识符
    rna_seq = input("请输入RNA序列：")
    identifier = input("请输入序列标识符（如 ENST00000702787.2）：")
    output_file = input("请输入输出 .npy 文件路径（默认 output.npy）：") or "output.npy"

    # 执行端到端流程
    rna_sequence_to_npy(rna_seq, identifier, output_file)
