# 定义输入和输出文件路径
input_file_path = 'lnc_test.fasta'  # 替换为你的输入文件路径
output_file_path = 'lncc_testu.fasta'  # 替换为你的输出文件路径

# 读取文件内容
with open(input_file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 替换所有 T 为 U（区分大小写）
content = content.replace('T', 'U')

# 写入新文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print(f"替换完成！输出文件已保存到 {output_file_path}")