import os
import glob
import torch
from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==== 1. 加载自定义词汇表 ====
def load_vocab(vocab_file):
    """
    加载自定义的词汇表文件。

    :param vocab_file: 自定义词汇表文件路径
    :return: 词汇表字典
    """
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            kmer = line.strip()
            if not kmer:  # 跳过空行
                continue
            vocab[kmer] = idx
    # 确保特殊标记存在（如[PAD]、[UNK]）
    assert '[PAD]' in vocab and '[UNK]' in vocab, "词汇表需包含'[PAD]'和'[UNK]'"
    return vocab


# ==== 2. 定义BERT模型 ====
def init_bert_model(vocab_size, pretrained_path="bert-base-uncased"):
    """
    初始化BERT模型。

    :param vocab_size: 词汇表大小
    :param pretrained_path: 预训练模型路径，默认为"bert-base-uncased"
    :return: BERT模型对象
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = BertModel.from_pretrained(pretrained_path)
        model.resize_token_embeddings(vocab_size)  # 调整模型嵌入层大小以匹配新的词汇表
        model.to(device)
        logging.info(f"成功加载并调整了预训练BERT模型，设备：{device}")
    except Exception as e:
        logging.error(f"加载或调整预训练BERT模型时出错：{e}")
        raise
    return model, device


# ==== 3. 特征提取函数 ====
def extract_bert_features(rna_sequence, k=6, max_length=512, vocab_dict=None):
    """
    提取RNA序列的BERT特征。

    :param rna_sequence: RNA序列字符串
    :param k: k-mer长度
    :param max_length: 最大序列长度
    :param vocab_dict: 词汇表字典
    :return: 特征矩阵
    """
    if len(rna_sequence) < k:
        raise ValueError(f"RNA序列长度 ({len(rna_sequence)}) 小于k-mer长度 ({k})")

    kmer_ids = []
    for i in range(len(rna_sequence) - k + 1):
        kmer = rna_sequence[i:i + k]
        kmer_ids.append(vocab_dict.get(kmer, vocab_dict['[UNK]']))

    # 处理长度（padding或截断）
    if len(kmer_ids) > max_length:
        kmer_ids = kmer_ids[:max_length]
    else:
        pad_id = vocab_dict['[PAD]']
        kmer_ids += [pad_id] * (max_length - len(kmer_ids))

    input_tensor = torch.tensor([kmer_ids])
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        features = outputs.last_hidden_state[0].cpu().numpy()  # [max_length, 768]

    return features


# ==== 4. 批量特征提取函数 ====
def batch_extract_bert_features(rna_sequences, k=6, max_length=512, batch_size=32, vocab_dict=None):
    """
    批量提取RNA序列的BERT特征。

    :param rna_sequences: RNA序列列表
    :param k: k-mer长度
    :param max_length: 最大序列长度
    :param batch_size: 批处理大小
    :param vocab_dict: 词汇表字典
    :return: 特征矩阵
    """
    all_features = []
    total_sequences = len(rna_sequences)
    for i in range(0, total_sequences, batch_size):
        batch_sequences = rna_sequences[i:i + batch_size]
        batch_kmer_ids = []
        for seq in batch_sequences:
            kmer_ids = []
            for j in range(len(seq) - k + 1):
                kmer = seq[j:j + k]
                kmer_ids.append(vocab_dict.get(kmer, vocab_dict['[UNK]']))
            if len(kmer_ids) > max_length:
                kmer_ids = kmer_ids[:max_length]
            else:
                pad_id = vocab_dict['[PAD]']
                kmer_ids += [pad_id] * (max_length - len(kmer_ids))
            batch_kmer_ids.append(kmer_ids)

        input_tensor = torch.tensor(batch_kmer_ids).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            features = outputs.last_hidden_state.cpu().numpy()  # [batch_size, max_length, 768]
        all_features.extend(features)
        logging.info(f"已处理 {min(i + batch_size, total_sequences)} / {total_sequences} 条RNA序列")

    return all_features


# ==== 5. 批量读取文件夹中的seq文件 ====
def read_sequences_from_folder(folder_path):
    """
    从文件夹中读取所有RNA序列。

    :param folder_path: 包含RNA序列文件的文件夹路径
    :return: RNA序列列表
    """
    rna_sequences = []
    file_paths = glob.glob(os.path.join(folder_path, "*.seq"))
    total_files = len(file_paths)
    for i, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'r') as f:
                seq = f.read().replace('\n', '').replace(' ', '')
                rna_sequences.append(seq)
            logging.info(f"已读取文件 {i + 1}/{total_files}: {file_path}")
        except Exception as e:
            logging.error(f"无法读取文件 {file_path}: {e}")
    return rna_sequences


# ==== 6. 主流程 ====
if __name__ == "__main__":
    # 设置路径
    custom_vocab_path = "../vocab.txt"  # 替换为你的自定义词汇表路径
    seq_folder = "../seq/test/pc_13"  # 替换为你的RNA序列文件夹路径

    # 加载自定义词汇表
    vocab_dict = load_vocab(custom_vocab_path)
    vocab_size = len(vocab_dict)

    # 初始化BERT模型
    model, device = init_bert_model(vocab_size)

    # 读取所有RNA序列
    rna_sequences = read_sequences_from_folder(seq_folder)

    # 批量提取特征
    features = batch_extract_bert_features(rna_sequences, vocab_dict=vocab_dict)

    # 转换为numpy数组（形状：[样本数, max_length, 768]）
    feature_matrix = np.array(features)

    # 保存结果
    output_path = "torch_data/neg_features.npy"
    np.save(output_path, feature_matrix)
    logging.info(f"特征矩阵已保存到 {output_path}，形状：{feature_matrix.shape}")
