import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
import tempfile

from PySide6.QtCore import QThread, Signal  # PredictionWorker 是一个 QThread

# 确保可以导入本地模块
# 根据项目结构，这个路径调整可能是必要的
# 如果 fasta_input、rnafold_processor 等文件在当前目录或可发现的路径中，
# 则这里可能不需要此设置，但为了与原脚本保持一致，仍保留。
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 后端功能模块导入（特征提取、模型定义）
from command.fasta_input import rna_sequence_to_fasta, extract_features_with_ilearn, remove_first_column
from command.rnafold_processor import rna_sequence_to_npy as generate_rnafold_feature
from command.bert_extract import extract_features as extract_bert_features_external  # 重命名为避免冲突

# 模型定义（假设路径正确或已加入 PYTHONPATH）
from command.evaluate import MultiModalEfficientNet
from model_definition.eiip_model import EfficientNetB1_1D
from model_definition.fold_model import EfficientNet1D
from model_definition.bert_model import KmerBERT


class PredictionWorker(QThread):
    """
    在后台线程中处理 RNA 序列预测任务。
    此类封装了所有的特征提取和模型推理逻辑。
    """
    progress = Signal(str)  # 信号：用于更新 GUI 中的进度信息
    result_ready = Signal(dict)  # 信号：用于向 GUI 发送预测结果
    error_occurred = Signal(str)  # 信号：用于向 GUI 报告错误

    def __init__(self, model, rna_sequence, file_id, temp_dir_path, device, model_type, tokenizer=None):
        """
        初始化 PredictionWorker。

        参数:
            model: 预加载的 PyTorch 模型。
            rna_sequence (str): 要预测的 RNA 序列。
            file_id (str): RNA 序列的标识符。
            temp_dir_path (str): 临时目录路径，用于中间文件。
            device (torch.device): 用于张量运算的设备（CPU/GPU）。
            model_type (str): 模型类型（"HYBRID", "BERT", "EIIP", "FOLD"）。
            tokenizer: BERT 模型使用的分词器（如适用）。
        """
        super().__init__()
        self.model = model
        self.rna_sequence = rna_sequence
        self.file_id = file_id
        self.temp_dir = temp_dir_path  # 直接使用提供的路径
        self.device = device
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.bert_kmer_size = 6  # BERT 的 k-mer 大小
        self.bert_max_len = 512  # BERT 的最大序列长度

    def run(self):
        """
        执行预测任务。该方法在线程启动时调用。
        根据模型类型调度到相应的处理方法。
        """
        try:
            # 根据模型类型调度到正确的处理方法
            if self.model_type == "HYBRID":
                result = self._process_hybrid()
            elif self.model_type == "BERT":
                result = self._process_bert()
            elif self.model_type == "EIIP":
                result = self._process_eiip()
            elif self.model_type == "FOLD":
                result = self._process_fold()
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")

            self.result_ready.emit(result)  # 准备好后发出结果

        except Exception as e:
            self.error_occurred.emit(
                f"PredictionWorker 中发生错误 ({self.model_type}): {str(e)}")  # 如果出错则发出错误

    def _process_hybrid(self):
        """使用 HYBRID 模型处理 RNA 序列。"""
        self.progress.emit("HYBRID - 第一步：生成 FASTA 并提取 PSE 特征...")
        fasta_path = os.path.join(self.temp_dir, f"{self.file_id}_output.fasta")
        pse_csv_path = os.path.join(self.temp_dir, f"{self.file_id}_out.csv")
        processed_pse_path = os.path.join(self.temp_dir, f"{self.file_id}_processed_out.csv")

        rna_sequence_to_fasta(self.rna_sequence, self.file_id, fasta_path)
        extract_features_with_ilearn(fasta_path, pse_csv_path)
        remove_first_column(pse_csv_path, processed_pse_path)

        self.progress.emit("HYBRID - 第二步：提取 RNAfold 特征...")
        rnafold_npy_path = os.path.join(self.temp_dir, f"{self.file_id}_rnafold.npy")
        generate_rnafold_feature(self.rna_sequence, self.file_id, rnafold_npy_path)

        self.progress.emit("HYBRID - 第三步：提取 BERT 特征...")
        # 对于 HYBRID，BERT 特征是通过外部脚本提取的
        # 外部脚本期望包含 .seq 文件的目录
        seq_file_path = os.path.join(self.temp_dir, f"{self.file_id}.seq")
        with open(seq_file_path, 'w') as f:
            f.write(self.rna_sequence.strip().replace(' ', '').replace('\n', ''))

        # 定义外部 BERT 特征提取所需的路径
        # 这些路径可能需要相对于 bert_extract.py 脚本而言，或者使用绝对路径
        # 假设 bert_extract.py 能找到其模型和分词器
        # 为了与原始 GUI 一致，我们使用以下路径
        tokenizer_path_for_external = "../model/tokenizer.pkl"  # 外部 BERT 使用的分词器路径
        bert_model_path_for_external = "../model/kmer_bert_model.pth"  # 外部 BERT 使用的模型路径

        extract_bert_features_external(
            input_dir=self.temp_dir,  # 包含 .seq 文件的目录
            output_dir=self.temp_dir,  # 保存 .npy 输出的目录
            tokenizer_path=tokenizer_path_for_external,
            model_path=bert_model_path_for_external,
            k=self.bert_kmer_size,
            max_len=self.bert_max_len
        )
        bert_npy_path = os.path.join(self.temp_dir, f"{self.file_id}.npy")  # 外部脚本预期输出路径

        self.progress.emit("HYBRID - 第四步：加载所有特征...")
        bert_tensor, pse_tensor, rnafold_tensor = load_features_for_hybrid_prediction(
            processed_pse_path, rnafold_npy_path, bert_npy_path, self.device
        )

        # 调整张量维度以匹配 HYBRID 模型的输入要求
        pse_tensor = pse_tensor.unsqueeze(0)
        rnafold_tensor = rnafold_tensor.permute(0, 1, 2)  # 原始: rnafold_tensor.permute(0, 2, 1)，检查模型输入格式

        self.progress.emit("HYBRID - 第五步：执行模型预测...")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(bert_tensor, pse_tensor, rnafold_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        return self._create_result_dict(outputs, probs, pred_class)

    def _process_bert(self):
        """使用 BERT 模型处理 RNA 序列。"""
        self.progress.emit("BERT - 第一步：准备 BERT 所需的序列...")
        # BERT 模型在此处使用内部特征提取和提供的分词器

        self.progress.emit("BERT - 第二步：内部提取 BERT 特征...")
        input_ids, attention_mask = extract_bert_features_internal(
            self.rna_sequence,
            self.tokenizer,  # 使用 GUI 加载的分词器
            k=self.bert_kmer_size,
            max_len=self.bert_max_len
        )

        self.progress.emit("BERT - 第三步：执行 BERT 模型预测...")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids.unsqueeze(0).to(self.device),
                attention_mask.unsqueeze(0).to(self.device)
            )
            # KmerBERT 的输出可能是包含 "logits" 的字典
            logits = outputs.get("logits", outputs) if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        return self._create_result_dict(logits, probs, pred_class)

    def _process_eiip(self):
        """使用 EIIP 模型处理 RNA 序列。"""
        self.progress.emit("EIIP - 第一步：生成 FASTA 并提取 PSE 特征...")
        fasta_path = os.path.join(self.temp_dir, f"{self.file_id}_output.fasta")
        pse_csv_path = os.path.join(self.temp_dir, f"{self.file_id}_out.csv")
        processed_pse_path = os.path.join(self.temp_dir, f"{self.file_id}_processed_out.csv")

        rna_sequence_to_fasta(self.rna_sequence, self.file_id, fasta_path)
        extract_features_with_ilearn(fasta_path, pse_csv_path)
        remove_first_column(pse_csv_path, processed_pse_path)

        self.progress.emit("EIIP - 第二步：加载 PSE 特征...")
        pse_tensor = load_pse_feature_for_eiip(processed_pse_path, self.device)
        pse_tensor = pse_tensor.unsqueeze(0)  # 添加 batch 维度

        self.progress.emit("EIIP - 第三步：执行 EIIP 模型预测...")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pse_tensor)  # 输出形状: [1, 1] 或 [1]
            # 直接使用模型输出（已包含 Sigmoid）
            if outputs.ndim == 1 or outputs.shape[1] == 1:
                prob_class_1 = outputs.item()  # 直接获取概率
                probabilities_np = np.array([1 - prob_class_1, prob_class_1], dtype=np.float32)
                pred_class = 1 if prob_class_1 >= 0.5 else 0

        return self._create_result_dict(outputs, probabilities_np, pred_class)

    def _process_fold(self):
        """使用 FOLD 模型处理 RNA 序列。"""
        self.progress.emit("FOLD - 第一步：提取 RNAfold 特征...")
        rnafold_npy_path = os.path.join(self.temp_dir, f"{self.file_id}_rnafold.npy")
        generate_rnafold_feature(self.rna_sequence, self.file_id, rnafold_npy_path)

        self.progress.emit("FOLD - 第二步：加载 RNAfold 特征...")
        rnafold_tensor = load_rnafold_feature_for_fold(rnafold_npy_path, self.device)
        # 调整维度：[batch, channels, length]
        # 原始: rnafold_tensor.unsqueeze(0).permute(0, 2, 1)
        # 假设 EfficientNet1D 期望 [batch_size, num_features, sequence_length]
        rnafold_tensor = rnafold_tensor.unsqueeze(0).permute(0, 2, 1)

        self.progress.emit("FOLD - 第三步：执行 FOLD 模型预测...")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(rnafold_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        return self._create_result_dict(outputs, probs, pred_class)

    def _create_result_dict(self, raw_output, probabilities, pred_class):
        """
        创建标准化的预测结果字典。

        参数:
            raw_output (torch.Tensor or np.ndarray): 模型的原始输出。
            probabilities (torch.Tensor or np.ndarray): 类别概率。
            pred_class (int): 预测类别索引。

        返回:
            dict: 包含预测类别、概率和原始输出的字典。
        """
        if isinstance(raw_output, torch.Tensor):
            raw_output_np = raw_output.cpu().numpy()
            if raw_output_np.ndim > 1 and raw_output_np.shape[0] == 1:
                raw_output_np = raw_output_np[0]
        elif isinstance(raw_output, dict):  # 如 BERT 返回的字典
            raw_output_np = raw_output.get("logits", raw_output)[0].cpu().numpy()
        else:  # 如 EIIP 模型返回的浮点数
            raw_output_np = np.array(raw_output).flatten()

        if isinstance(probabilities, torch.Tensor):
            probabilities_np = probabilities.cpu().numpy()
            if probabilities_np.ndim > 1 and probabilities_np.shape[0] == 1:
                probabilities_np = probabilities_np[0]
        else:  # 已经是 numpy 数组（例如来自 EIIP 的特殊处理）
            probabilities_np = np.array(probabilities).flatten()

        # 确保 probabilities_np 是两个元素的一维数组 [prob_class_0, prob_class_1]
        if probabilities_np.shape[0] == 1:  # 处理单一概率输出（如 EIIP 模型后 sigmoid）
            prob_class_1 = probabilities_np[0]
            probabilities_np = np.array([1 - prob_class_1, prob_class_1], dtype=np.float32)

        return {
            "class": pred_class,
            "probabilities": probabilities_np,  # 应为一维数组 [prob_class_0, prob_class_1]
            "raw_output": raw_output_np  # 应为一维数组，原始得分/logits
        }


# 特征加载和处理函数，适配后端使用

def extract_bert_features_internal(sequence, tokenizer, k=6, max_len=512):
    """
    使用提供的分词器内部提取 BERT 特征。
    用于独立的 BERT 模型选项。
    """
    raw_seq = sequence.strip().upper().replace('U', 'T')  # 规范 DNA/RNA 碱基
    seq_len = len(raw_seq)

    # 填充序列以确保长度能被 k 整除
    if seq_len % k != 0:
        padded_len = ((seq_len // k) + 1) * k
        padded_seq = raw_seq + 'N' * (padded_len - seq_len)  # 使用 'N' 填充未知碱基
    else:
        padded_seq = raw_seq

    kmers = [padded_seq[i:i + k] for i in range(0, len(padded_seq) - k + 1, 1)]  # 滑动窗口提取 k-mer

    if not kmers:  # 处理极短序列
        kmers = [padded_seq] if padded_seq else []

    # 分词 k-mer
    input_ids = [tokenizer.get("[CLS]", tokenizer.get("<CLS>", 0))]  # CLS token
    for kmer in kmers:
        input_ids.append(tokenizer.get(kmer, tokenizer.get("[UNK]", tokenizer.get("<UNK>", 1))))  # UNK token
    input_ids.append(tokenizer.get("[SEP]", tokenizer.get("<SEP>", 2)))  # SEP token

    # 填充至 max_len
    padding_length = max_len - len(input_ids)
    attention_mask = [1] * len(input_ids)  # 注意力掩码

    if padding_length > 0:
        input_ids += [tokenizer.get("[PAD]", tokenizer.get("<PAD>", 3))] * padding_length  # PAD token
        attention_mask += [0] * padding_length
    elif padding_length < 0:  # 截断超过 max_len 的部分
        input_ids = input_ids[:max_len - 1] + [tokenizer.get("[SEP]", tokenizer.get("<SEP>", 2))]
        attention_mask = attention_mask[:max_len]

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

    return input_ids_tensor, attention_mask_tensor


def load_pse_feature_for_eiip(pse_path, device):
    """加载并预处理 EIIP 模型所需的 PSE 特征。"""
    max_val = 0.114714286  # 来自原始脚本的归一化值
    pse_feat = pd.read_csv(pse_path, header=None).values
    pse_feat_normalized = pse_feat.astype(np.float32) / max_val  # 归一化
    pse_tensor = torch.from_numpy(pse_feat_normalized).to(device)
    return pse_tensor  # 形状: [num_features] 或 [1, num_features]


def load_rnafold_feature_for_fold(rnafold_path, device):
    """加载 FOLD 模型所需的 RNAfold 特征。"""
    rnafold_feat = np.load(rnafold_path)  # 预期形状如 (length, num_fold_features)
    if rnafold_feat.ndim == 3 and rnafold_feat.shape[0] == 1:  # 删除 batch 维度（如果存在）
        rnafold_feat = rnafold_feat.squeeze(0)
    rnafold_tensor = torch.from_numpy(rnafold_feat.astype(np.float32)).to(device)
    return rnafold_tensor  # 形状: (length, num_fold_features)


def load_features_for_hybrid_prediction(pse_path, rnafold_path, bert_path, device):
    """加载 HYBRID 模型预测所需的所有特征。"""
    # 加载 PSE 特征
    pse_feat = pd.read_csv(pse_path, header=None).values
    pse_tensor = torch.from_numpy(pse_feat.astype(np.float32)).squeeze().to(device)  # 形状: [pse_features]

    # 加载 RNAfold 特征
    rnafold_feat = np.load(rnafold_path)  # 预期如 (1, length, num_fold_features) 或 (length, num_fold_features)
    if rnafold_feat.ndim == 3 and rnafold_feat.shape[0] == 1:
        rnafold_feat = rnafold_feat.squeeze(0)  # 变成 (length, num_fold_features)
    rnafold_tensor = torch.from_numpy(rnafold_feat.astype(np.float32)).unsqueeze(0).to(device)

    # 加载 BERT 特征
    bert_feat = np.load(bert_path)  # 预期如 (1, bert_embedding_dim) 或 (bert_embedding_dim)
    bert_tensor = torch.from_numpy(bert_feat.astype(np.float32)).unsqueeze(0).to(device)

    return bert_tensor, pse_tensor, rnafold_tensor
