import itertools


def generate_k_mers(k):
    """生成所有可能的k-mer组合"""
    bases = ['A', 'C', 'G', 'T']
    k_mers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    return k_mers


def create_vocab_file(k_mers, vocab_file_path):
    """创建词汇表文件"""
    # 特殊标记
    special_tokens = [
        "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"
    ]

    # 将特殊标记和k-mer写入文件
    with open(vocab_file_path, 'w') as f:
        for token in special_tokens:
            f.write(f"{token}\n")
        for k_mer in k_mers:
            f.write(f"{k_mer}\n")


if __name__ == "__main__":
    k = 6  # 设置k-mer大小
    vocab_file_path = 'vocab_rna_6mer.txt'  # 输出文件路径

    # 生成所有6-mer组合
    k_mers = generate_k_mers(k)

    # 创建词汇表文件
    create_vocab_file(k_mers, vocab_file_path)
    print(f"词汇表文件已生成: {vocab_file_path}")