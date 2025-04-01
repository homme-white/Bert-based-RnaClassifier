import os
import shutil


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


def copy_files_in_range(source_dir, target_dir, min_length=1000, max_length=3000):
    """从源目录中复制符合条件的文件到目标目录"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith('.seq'):
                file_path = os.path.join(root, filename)
                length = get_text_length_without_spaces(file_path)

                if min_length <= length < max_length:
                    # 构建目标文件路径
                    target_file_path = os.path.join(target_dir, filename)

                    # 复制文件
                    try:
                        shutil.copy(file_path, target_file_path)
                        print(f"已复制文件: {filename} (字符长度: {length})")
                    except Exception as e:
                        print(f"无法复制文件 {file_path} 到 {target_file_path}: {e}")


if __name__ == "__main__":
    # 硬编码的目标目录路径
    source_directory = 'seq/lncRNA'
    target_directory = 'seq/lnc_1kt3k'

    # 调用函数进行处理
    copy_files_in_range(source_directory, target_directory)