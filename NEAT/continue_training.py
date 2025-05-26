"""
从现有checkpoint继续训练NEAT模型的示例脚本
"""

import os
from train_neat import run


# 查找最新的checkpoint文件
def find_latest_checkpoint(checkpoint_dir="saved_models"):
    """查找最新的checkpoint文件"""
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("neat-checkpoint-")]
    if not checkpoint_files:
        return None

    # 提取checkpoint编号并找到最大值
    checkpoint_numbers = []
    for f in checkpoint_files:
        try:
            num = int(f.split("-")[-1].split(".")[0])
            checkpoint_numbers.append((num, f))
        except (ValueError, IndexError):
            continue

    if not checkpoint_numbers:
        return None

    _, latest_file = max(checkpoint_numbers)
    return os.path.join(checkpoint_dir, latest_file)


if __name__ == "__main__":
    # 查找最新的checkpoint
    latest_checkpoint = find_latest_checkpoint()

    if latest_checkpoint:
        print(f"找到最新的checkpoint文件: {latest_checkpoint}")
        # 继续训练50代
        run("neat-config.txt", n_generations=10000, checkpoint_path=latest_checkpoint)
    else:
        print("未找到checkpoint文件，将从头开始训练")
        run("neat-config.txt", n_generations=50)
