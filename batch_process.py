import os
import subprocess
import json
from multiprocessing import Pool, cpu_count

# 配置参数（与原代码相同）
input_folder = 'seq/lnc_1kt3k'
output_folder = 'json/pos'
vocab_file = 'vocab.txt'
bert_config_file = 'bert/bert_config.json'
init_checkpoint = 'bert/bert_model.ckpt.index'
do_lower_case = 'False'
layers = '-1,-2,-3,-4'
max_seq_length = '512'
batch_size = '64'
processed_file = os.path.join(output_folder, 'processed_files.json')

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 读取已处理文件列表
if os.path.exists(processed_file):
    with open(processed_file, 'r') as f:
        processed_files = set(json.load(f))
else:
    processed_files = set()

# 获取输入文件夹中的所有 .seq 文件
seq_files = [f for f in os.listdir(input_folder) if f.endswith('.seq')]


def process_file(seq_file):
    if seq_file in processed_files:
        print(f'Skipping already processed file: {seq_file}')
        return None

    input_file = os.path.join(input_folder, seq_file)
    output_jsonl_file = os.path.join(output_folder, f'{os.path.splitext(seq_file)[0]}.jsonl')
    output_csv_file = os.path.join(output_folder, f'{os.path.splitext(seq_file)[0]}.csv')

    # 运行 extract_features.py
    extract_features_cmd = [
        'python', 'extract_features.py',
        '--input_file', input_file,
        '--output_file', output_jsonl_file,
        '--vocab_file', vocab_file,
        '--bert_config_file', bert_config_file,
        '--init_checkpoint', init_checkpoint,
        '--do_lower_case', do_lower_case,
        '--layers', layers,
        '--max_seq_length', max_seq_length,
        '--batch_size', batch_size
    ]
    print(f'Running: {" ".join(extract_features_cmd)}')
    subprocess.run(extract_features_cmd, check=True)

    # 运行 jsonl2csv.py
    jsonl2csv_cmd = ['python', 'jsonl2csv.py', output_jsonl_file, output_csv_file]
    print(f'Running: {" ".join(jsonl2csv_cmd)}')
    subprocess.run(jsonl2csv_cmd, check=True)

    print(f'Processed {seq_file} successfully.')
    return seq_file


if __name__ == '__main__':
    # 使用所有可用CPU核心进行并行处理
    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, seq_files)

    # 更新已处理文件列表
    new_processed_files = set(filter(None, results))  # 过滤掉None值
    processed_files.update(new_processed_files)

    with open(processed_file, 'w') as f:
        json.dump(list(processed_files), f)

    print('All files processed.')
