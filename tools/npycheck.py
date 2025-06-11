import numpy as np
import os

# 定义文件路径
pos_npy_path = '/home/bronya/Desktop/毕设用/Bert-based-RnaClassifier/run/a.npy'


def check_npy_content(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)

        # 设置 numpy 打印选项（避免截断显示）
        np.set_printoptions(threshold=np.inf, suppress=True)

        print(f"\n文件 {file_path} 的内容：")
        print(data)
        print(f"文件形状: {data.shape}")

        print("\n--- 分割线 ---")

    except Exception as e:
        print(f"无法加载文件 {file_path}: {str(e)}")


# 查看 pos_npy 文件内容和形状
check_npy_content(pos_npy_path)



