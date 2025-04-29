import os


def main():
    # 输入FASTA文件路径（替换为你的FASTA文件路径）
    input_fasta = "ori_data/Human B/pct.train.human.B.fa"

    # 输出文件夹路径（存放生成的.seq文件）
    output_dir = "seq/human/human_pct"

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化计数器
    total_sequences = 0

    # 打开FASTA文件并逐行读取
    with open(input_fasta, 'r') as fin:
        current_id = None
        current_sequence = []

        for line in fin:
            line = line.strip()

            if line.startswith('>'):  # 描述符行
                # 如果当前有未保存的序列，则先保存
                if current_id is not None:
                    save_sequence(output_dir, current_id, current_sequence)
                    total_sequences += 1

                # 提取新的ID（去掉 '>' 符号）
                current_id = line[1:].split()[0]  # 取第一个空格前的部分作为ID
                current_sequence = []  # 重置序列内容
            else:
                # 添加序列行内容
                current_sequence.append(line)

        # 处理最后一条序列（循环结束时可能还有未保存的序列）
        if current_id is not None:
            save_sequence(output_dir, current_id, current_sequence)
            total_sequences += 1

    print(f"拆分完成，共处理 {total_sequences} 条序列")
    print(f"所有 .seq 文件已保存至文件夹：{output_dir}")


def save_sequence(output_dir, seq_id, sequence_lines):
    """
    将单条序列保存为一个 .seq 文件
    :param output_dir: 输出文件夹路径
    :param seq_id: 序列ID
    :param sequence_lines: 序列内容（列表形式，每行为一个元素）
    """
    # 合并序列内容为单行字符串
    full_sequence = ''.join(sequence_lines)

    # 将连续碱基插入空格，每3个字符加一个空格
    formatted_sequence = ' '.join([full_sequence[i:i + 3] for i in range(0, len(full_sequence), 3)])

    # 写入文件
    output_file = os.path.join(output_dir, f"{seq_id}.seq")
    with open(output_file, 'w') as fout:
        fout.write(formatted_sequence)


if __name__ == "__main__":
    main()