import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file(seq_file_path, output_file):
    """
    使用 RNAfold 处理单个文件。
    """
    try:
        print(f"正在处理文件: {seq_file_path}")
        with open(output_file, 'w') as out_f:
            subprocess.run(
                ["RNAfold", "-i", seq_file_path],
                stdout=out_f,
                stderr=subprocess.PIPE,
                check=True
            )
        print(f"处理完成，输出已保存到: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"处理文件 '{seq_file_path}' 时出错！")
        print(f"错误信息: {e.stderr.decode('utf-8')}")

def batch_rnafold(input_dir, output_dir, max_workers=12):
    """
    批量处理指定文件夹中的 .seq 文件，并将结果保存到指定的输出文件夹。
    :param input_dir: 包含 .seq 文件的文件夹路径。
    :param output_dir: 保存处理结果的文件夹路径。
    :param max_workers: 最大线程数。
    """
    # 检查输入文件夹是否存在
    if not os.path.isdir(input_dir):
        print(f"错误：输入文件夹 '{input_dir}' 不存在！")
        return

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    # 获取文件夹中所有的 .seq 文件
    seq_files = [f for f in os.listdir(input_dir) if f.endswith('.seq')]

    if not seq_files:
        print(f"输入文件夹 '{input_dir}' 中没有找到 .seq 文件！")
        return

    # 使用多线程处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for seq_file in seq_files:
            # 构造完整路径
            seq_file_path = os.path.join(input_dir, seq_file)
            base_name = os.path.splitext(seq_file)[0]  # 去掉扩展名
            output_file = os.path.join(output_dir, f"{base_name}.folded")

            # 提交任务到线程池
            future = executor.submit(process_file, seq_file_path, output_file)
            futures.append(future)

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()  # 捕获异常
            except Exception as e:
                print(f"线程中发生错误: {e}")

    print("所有文件处理完成！")

if __name__ == "__main__":
    # 直接在代码中定义输入和输出路径
    input_directory = "../seq/Final/test/nc_pos_ns"  # 替换为你的输入文件夹路径
    output_directory = "Final/test/nc_pos"  # 替换为你的输出文件夹路径

    # 设置最大线程数（根据系统资源调整）
    max_threads = 16

    # 执行批量处理
    batch_rnafold(input_directory, output_directory, max_threads)
