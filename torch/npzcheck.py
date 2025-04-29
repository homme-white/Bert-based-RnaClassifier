import numpy as np
import os
# 定义文件路径
pos_npy_path = '/home/bronya/Desktop/ENST00000026218.npy'
neg_npy_path = 'torch_data/phase1（1kt3k）/neg_features.npy'

# 加载 .npy 文件并查看形状
def check_npy_content(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)

        # 设置numpy打印选项（避免截断显示）
        np.set_printoptions(threshold=np.inf, suppress=True)

        print(f"\n文件 {file_path} 的内容：")
        print(data)
        print("\n--- 分割线 ---")

    except Exception as e:
        print(f"无法加载文件 {file_path}: {str(e)}")

# 查看 pos_npy 文件内容
check_npy_content(pos_npy_path)

# # 查看 neg_npy 文件形状（如果存在）
# if os.path.exists(neg_npy_path):
#     check_npy_shape(neg_npy_path)
# else:
#     print(f"文件 {neg_npy_path} 不存在")

