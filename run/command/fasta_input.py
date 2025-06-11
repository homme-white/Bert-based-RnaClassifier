import time
import subprocess
import pandas as pd

def rna_sequence_to_fasta(sequence, file_id, output_file="output.fasta"):
    """
    将RNA序列转换为FASTA格式并保存到文件中

    参数:
    sequence (str): RNA序列
    file_id (str): 序列标识符（文件名）
    output_file (str): 输出文件名，默认为 "output.fasta"
    """
    # 去除所有空格和换行符
    sequence = sequence.replace(' ', '').replace('\n', '')

    # 构建FASTA条目
    fasta_entry = f">{file_id}|1|rna_sequence\n{sequence}\n"

    # 写入文件
    with open(output_file, 'w') as fout:
        fout.write(fasta_entry)

    print(f"FASTA文件已生成：{output_file}")


def extract_features_with_ilearn(input_fasta="output.fasta", output_csv="out.csv"):
    """
    使用 iLearn-nucleotide-basic.py 提取特征
    """
    feature_extraction_cmd = [
        'python', 'ilearn/iLearn-nucleotide-basic.py',
        '--file', input_fasta,
        '--method', 'PseEIIP',
        '--format', 'csv',
        '--out', output_csv
    ]
    print("开始提取特征...")
    subprocess.run(feature_extraction_cmd, check=True)
    print(f"特征提取完成，结果已保存至 {output_csv}")


def remove_first_column(input_csv="out.csv", output_csv="processed_out.csv"):
    """
    按行读取CSV文件并删除第一列，保留原始数值格式
    """
    with open(input_csv, 'r') as fin:
        lines = fin.readlines()

    # 处理每一行，去掉第一个字段
    processed_lines = [','.join(line.strip().split(',')[1:]) + '\n' for line in lines]

    with open(output_csv, 'w') as fout:
        fout.writelines(processed_lines)

    print(f"第一列已删除，处理后的文件保存为 {output_csv}")


if __name__ == "__main__":
    # 步骤1: 输入RNA序列并生成FASTA文件
    rna_sequence = input("请输入RNA序列：")
    file_id = input("请输入序列标识符（例如：ENST00000702787.2）：")

    rna_sequence_to_fasta(rna_sequence, file_id)

    # 步骤2: 提取特征
    extract_features_with_ilearn()

    # 步骤3: 处理CSV文件，删除第一列
    remove_first_column()
