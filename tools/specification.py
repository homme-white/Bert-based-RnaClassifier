import os
def process_seq_files(source_dir, target_dir):
    """
    处理指定目录下的所有 .seq 文件，去除空格后输出到目标目录
    :param source_dir: 源文件目录路径（需手动设置）
    :param target_dir: 目标输出目录路径（需手动设置）
    """
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith('.seq'):
            # 构建文件路径
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, filename)

            # 读取并处理内容
            with open(src_path, 'r') as f_in, open(dst_path, 'w') as f_out:
                content = f_in.read()
                processed = content.replace(' ', '')  # 去除所有空格
                f_out.write(processed)

    print(f"处理完成！共处理 {len([name for name in os.listdir(source_dir) if name.endswith('.seq')])} 个文件")


if __name__ == "__main__":
    # === 直接修改以下两个目录路径 === #
    SOURCE_DIRECTORY = "Final/test/pc_neg"  # 源文件目录
    TARGET_DIRECTORY = "Final/test/pc_neg_ns"  # 输出目录
    # ================================ #

    process_seq_files(SOURCE_DIRECTORY, TARGET_DIRECTORY)