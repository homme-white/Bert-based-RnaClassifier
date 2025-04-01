from sympy import false
from transformers import BertTokenizer, BertModel


# ==== 1. 加载自定义词汇表 ====
def load_custom_vocab(vocab_file):
    """
    加载自定义的词汇表文件。

    :param vocab_file: 自定义词汇表文件路径
    :return: 分词器对象
    """
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=false)
    return tokenizer


# ==== 2. 检查分词器是否正确识别k-mer ====
def check_tokenizer_recognition(tokenizer, kmers_to_check):
    """
    检查分词器是否能够正确识别给定的k-mer。

    :param tokenizer: 已加载的BERT分词器
    :param kmers_to_check: 需要检查的k-mer列表
    """
    for kmer in kmers_to_check:
        tokens = tokenizer.tokenize(kmer)
        print(f"K-mer: {kmer} -> Tokens: {tokens}")
        if len(tokens) != 1 or tokens[0] != kmer:
            print(f"警告：'{kmer}' 未被正确识别！")


# ==== 3. 主流程 ====
if __name__ == "__main__":
    # 设置路径
    custom_vocab_path = "../vocab.txt"  # 替换为你的自定义词汇表路径
    kmers_to_check = ["AAAAAA", "ACGTCA", "TACGAA","TACGAAACCGGT","A"]  # 示例k-mer列表，可根据需要调整

    # 加载自定义词汇表并初始化分词器
    tokenizer = load_custom_vocab(custom_vocab_path)

    # 加载预训练BERT模型（使用自定义词汇表）
    model = BertModel.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))  # 调整模型嵌入层大小以匹配新的词汇表

    # 检查分词器是否正确识别k-mer
    check_tokenizer_recognition(tokenizer, kmers_to_check)

    # 可选：测试一些标准英文单词以确保分词器对标准词汇的处理正常
    standard_words = ["hospitalization", "hello"]
    print("\n检查标准英文单词的分词结果：")
    check_tokenizer_recognition(tokenizer, standard_words)