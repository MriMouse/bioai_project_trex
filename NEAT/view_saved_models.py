#!/usr/bin/env python3
"""
示例脚本：如何加载和使用保存的NEAT训练数据
"""

import os
import pickle
import glob
from datetime import datetime


def list_saved_models(models_dir="saved_models"):
    """列出所有保存的模型文件"""
    print("=== 保存的模型文件 ===")

    # 列出所有checkpoint文件
    checkpoints = glob.glob(os.path.join(models_dir, "neat-best-checkpoint-*"))
    print(f"\n完整Checkpoint文件 ({len(checkpoints)}个):")
    for cp in sorted(checkpoints):
        basename = os.path.basename(cp)
        print(f"  {basename}")

    # 列出所有基因组文件
    genomes = glob.glob(os.path.join(models_dir, "neat-best-genome-gen*.pkl"))
    print(f"\n最佳基因组文件 ({len(genomes)}个):")
    for genome in sorted(genomes):
        basename = os.path.basename(genome)
        print(f"  {basename}")

    # 列出所有统计数据文件
    stats = glob.glob(os.path.join(models_dir, "neat-stats-gen*.pkl"))
    print(f"\n统计数据文件 ({len(stats)}个):")
    for stat in sorted(stats):
        basename = os.path.basename(stat)
        print(f"  {basename}")


def load_best_genome(models_dir="saved_models"):
    """加载最佳基因组"""
    try:
        # 尝试加载最新的最佳基因组
        overall_path = os.path.join(models_dir, "neat-best-genome-overall.pkl")
        if os.path.exists(overall_path):
            with open(overall_path, "rb") as f:
                genome = pickle.load(f)
            print(f"成功加载最佳基因组，适应度: {genome.fitness}")
            return genome
        else:
            print("未找到最佳基因组文件")
            return None
    except Exception as e:
        print(f"加载最佳基因组失败: {e}")
        return None


def load_training_stats(stats_file=None, models_dir="saved_models"):
    """加载训练统计数据"""
    try:
        if stats_file is None:
            # 加载默认的统计数据文件
            stats_path = os.path.join(models_dir, "neat_stats_data.pkl")
        else:
            stats_path = stats_file

        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                stats_data = pickle.load(f)

            print(f"成功加载统计数据: {os.path.basename(stats_path)}")
            print(f"  训练代数: {len(stats_data.get('best_scores', []))}")
            if stats_data.get("best_scores"):
                print(f"  最佳适应度: {max(stats_data['best_scores']):.2f}")
                print(f"  最终适应度: {stats_data['best_scores'][-1]:.2f}")

            return stats_data
        else:
            print(f"统计数据文件不存在: {stats_path}")
            return None
    except Exception as e:
        print(f"加载统计数据失败: {e}")
        return None


def load_specific_checkpoint(checkpoint_file, models_dir="saved_models"):
    """加载特定的checkpoint文件信息"""
    import neat

    try:
        checkpoint_path = (
            os.path.join(models_dir, checkpoint_file) if not os.path.isabs(checkpoint_file) else checkpoint_file
        )

        # 这里只是演示如何获取checkpoint信息，实际加载需要config
        print(f"Checkpoint文件: {checkpoint_file}")
        print(f"文件大小: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")

        # 从文件名解析信息
        if "gen" in checkpoint_file and "fitness" in checkpoint_file:
            parts = checkpoint_file.split("-")
            for part in parts:
                if part.startswith("gen"):
                    print(f"  代数: {part[3:]}")
                elif part.startswith("fitness"):
                    print(f"  适应度: {part[7:]}")
                elif len(part) == 15 and "_" in part:  # 时间戳格式
                    try:
                        timestamp = datetime.strptime(part, "%Y%m%d_%H%M%S")
                        print(f"  保存时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        pass

        return True
    except Exception as e:
        print(f"处理checkpoint文件失败: {e}")
        return False


def main():
    """主函数"""
    models_dir = "saved_models"

    print("NEAT T-Rex 训练数据查看器")
    print("=" * 40)

    # 列出所有保存的模型
    list_saved_models(models_dir)

    print("\n" + "=" * 40)

    # 加载最佳基因组
    best_genome = load_best_genome(models_dir)

    print("\n" + "=" * 40)

    # 加载训练统计数据
    stats_data = load_training_stats(models_dir=models_dir)

    # 查看特定的统计数据文件
    specific_stats = glob.glob(os.path.join(models_dir, "neat-stats-gen*.pkl"))
    if specific_stats:
        print(f"\n加载最新的特定统计数据文件:")
        latest_stats = sorted(specific_stats)[-1]
        load_training_stats(latest_stats)

    print("\n" + "=" * 40)

    # 查看checkpoint信息
    checkpoints = glob.glob(os.path.join(models_dir, "neat-best-checkpoint-*"))
    if checkpoints:
        print(f"\n最新checkpoint信息:")
        latest_checkpoint = sorted(checkpoints)[-1]
        load_specific_checkpoint(latest_checkpoint, models_dir)


if __name__ == "__main__":
    main()
