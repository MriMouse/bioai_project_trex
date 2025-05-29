import neat
import pickle
import os
import graphviz  # 确保 graphviz 已安装且在 PATH 中
from visualize import draw_net  # 从 visualize.py 导入绘图函数

# 定义节点名称，与 train_neat.py 中的一致
node_names = {
    -1: "player_y",
    -2: "player_v_vel",
    -3: "is_crouch",
    -4: "obs1_dx",
    -5: "obs1_y",
    -6: "obs1_w",
    -7: "obs1_h",
    -8: "obs1_bird",
    -9: "obs2_dx",
    -10: "obs2_y",
    -11: "obs2_w",
    -12: "obs2_h",
    -13: "obs2_bird",
    -14: "game_speed",
    0: "NoOp",
    1: "Jump",
    2: "Duck",
}

# 配置文件路径
config_path = "neat-config.txt"
# 最佳基因组文件路径
genome_path = "neat-best-genome-overall.pkl"
# 输出图片的文件名（不含扩展名）
output_image_filename = os.path.join("saved_models", "best_genome_network")


def analyze_best_genome(config_file, best_genome_file, output_filename):
    """
    加载最佳基因组并绘制其网络图。
    """
    # 加载配置
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # 加载最佳基因组
    try:
        with open(best_genome_file, "rb") as f:
            genome = pickle.load(f)
        print(f"成功加载最佳基因组: {best_genome_file}")
    except Exception as e:
        print(f"加载最佳基因组失败 ({best_genome_file}): {e}")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 绘制网络图
    try:
        print(f"正在绘制网络图，将保存为 {output_filename}.png ...")
        draw_net(
            config,
            genome,
            filename=output_filename,
            view=False,  # 设置为 True 可以直接打开图片
            node_names=node_names,
            show_disabled=False,  # 可以设置为 True 来显示禁用的连接
            prune_unused=True,  # 移除未使用的节点
            fmt="png",
        )
        print(f"网络图已成功保存到 {output_filename}.png")
    except Exception as e:
        print(f"绘制网络图失败: {e}")
        print("请确保已经安装了 Graphviz 并且将其添加到了系统的 PATH 环境变量中。")
        print("对于 Windows 用户，通常需要从 Graphviz 官网下载并安装，然后将安装目录下的 bin 文件夹添加到 PATH。")
        print(
            "对于 Linux/macOS 用户，可以使用包管理器安装，例如：sudo apt-get install graphviz 或 brew install graphviz"
        )


if __name__ == "__main__":
    analyze_best_genome(config_path, genome_path, output_image_filename)
