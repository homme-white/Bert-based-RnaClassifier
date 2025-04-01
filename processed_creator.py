import os
import json

# 配置参数
output_folder = 'json/neg'
processed_file = os.path.join(output_folder, 'processed_files.json')

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 读取已处理文件列表
if os.path.exists(processed_file):
    with open(processed_file, 'r') as f:
        processed_files = set(json.load(f))
else:
    processed_files = set()

# 获取输出文件夹中的所有 .csv 文件
csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]

# 提取对应的 .seq 文件名
new_processed_files = {os.path.splitext(f)[0] + '.seq' for f in csv_files}

# 更新已处理文件列表
processed_files.update(new_processed_files)

# 写入更新后的已处理文件列表
with open(processed_file, 'w') as f:
    json.dump(list(processed_files), f)

print('All CSV files have been added to processed_files.')
