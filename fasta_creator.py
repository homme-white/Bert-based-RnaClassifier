import os
import time

def main():
    # 输入文件夹路径（替换为你的文件夹路径）
    input_dir = "seq/complete/pc_train"
    # 输出文件路径
    output_file = "fasta/pc_com_train.fasta"

    # 获取所有以 .seq 结尾的文件名
    files = [f for f in os.listdir(input_dir) if f.endswith('.seq')]

    # 按文件名排序
    files.sort()

    # # 新增：输入起始文件名
    # start_file = "ENST00000702787.2.seq"
    #
    # # 新增：处理起始文件逻辑
    # if start_file in files:
    #     start_index = files.index(start_file)
    #     files = files[start_index+1:]  # 从该文件开始处理后续所有文件
    # else:
    #     print(f"错误：未找到文件 {start_file}！")
    #     return

    # 初始化计数器
    total_sequences = 0

    # 记录开始时间
    start_time = time.time()

    with open(output_file, 'w') as fout:
        for filename in files:
            # 提取文件名（去除扩展名）
            file_id = filename.replace('.seq', '')

            # 读取序列内容
            with open(os.path.join(input_dir, filename), 'r') as fin:
                sequence = fin.read().strip()

                # 去除所有空格和换行符（合并为连续的碱基字符串）
                sequence = sequence.replace(' ', '').replace('\n', '')

            # 构建FASTA格式的描述符行和序列行
            fasta_entry = f">{file_id}|1|testing\n{sequence}\n"
            fout.write(fasta_entry)

            # 更新计数器
            total_sequences += 1

    # 计算总耗时
    elapsed_time = time.time() - start_time

    # 输出结果
    print(f"合并完成，输出文件：{output_file}")
    print(f"共处理 {total_sequences} 个序列")
    print(f"总耗时：{elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()
