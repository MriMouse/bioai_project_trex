import neat
import pickle
import matplotlib.pyplot as plt
from aiGame import TRexGame, DEFAULT_SEED
import os
import argparse
import re


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = TRexGame(human_mode=False, random_seed=DEFAULT_SEED)
        state = game.reset()
        fitness = 0
        done = False
        while not done:
            inputs = [
                state["player_y"],
                state["player_vertical_velocity"],
                int(state["is_crouching"]),
                state["obs1_dist_x"],
                state["obs1_y"],
                state["obs1_width"],
                state["obs1_height"],
                int(state["obs1_is_bird"]),
                state["obs2_dist_x"],
                state["obs2_y"],
                state["obs2_width"],
                state["obs2_height"],
                int(state["obs2_is_bird"]),
                state["game_speed"],
            ]
            output = net.activate(inputs)
            action = output.index(max(output))  # 0:无操作 1:跳 2:蹲

            # 只有鸟和恐龙重合且无操作时才惩罚
            bird_penalty = 0
            overlap_threshold = 5  # 判定重合的距离阈值

            # 检查第一个障碍物
            if state["obs1_is_bird"] and abs(state["obs1_dist_x"]) < overlap_threshold and action == 0:
                bird_y = state["obs1_y"]
                if bird_y < 85:  # 高飞的鸟应该蹲下通过
                    bird_penalty -= 10
                elif bird_y > 95:  # 低飞的鸟应该跳跃通过
                    bird_penalty -= 10
                else:  # 中等高度的鸟，蹲或跳都可以
                    bird_penalty -= 5

            # 检查第二个障碍物
            if state["obs2_is_bird"] and abs(state["obs2_dist_x"]) < overlap_threshold and action == 0:
                bird_y = state["obs2_y"]
                if bird_y < 85:
                    bird_penalty -= 10
                elif bird_y > 95:
                    bird_penalty -= 10
                else:
                    bird_penalty -= 5

            state, reward, done = game.step(action)
            fitness += reward + bird_penalty  # 将鸟类惩罚加入适应度计算

            if fitness > 6000:
                done = True
        genome.fitness = fitness


def run(config_file, n_generations=10000, checkpoint_path=None):
    # Create directory for saving models and results
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file
    )

    # 加载checkpoint或创建新的种群
    if checkpoint_path:
        print(f"加载checkpoint: {checkpoint_path}")
        try:
            p = neat.Checkpointer.restore_checkpoint(checkpoint_path)

            # 提取checkpoint的生成代数
            checkpoint_num_match = re.search(r"checkpoint-(\d+)", checkpoint_path)
            start_gen = int(checkpoint_num_match.group(1)) if checkpoint_num_match else 0

            # 尝试加载已有的分数记录
            scores_path = os.path.join(models_dir, "neat_scores.pkl")
            best_scores = []
            if os.path.exists(scores_path):
                try:
                    with open(scores_path, "rb") as f:
                        best_scores = pickle.load(f)
                    print(f"已加载{len(best_scores)}代的历史分数记录")
                except Exception as e:
                    print(f"加载分数记录失败: {e}")
                    best_scores = []

            print(f"成功加载checkpoint，从第{start_gen}代继续训练...")
        except Exception as e:
            print(f"加载checkpoint失败: {e}")
            print("创建新的种群...")
            p = neat.Population(config)
            start_gen = 0
            best_scores = []
    else:
        p = neat.Population(config)
        start_gen = 0
        best_scores = []

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Update checkpoint path
    p.add_reporter(neat.Checkpointer(100, filename_prefix=os.path.join(models_dir, "neat-checkpoint-")))

    overall_best_genome = None
    overall_best_fitness = -float("inf")

    # 查找已有的最佳基因组
    best_genome_path = os.path.join(models_dir, "neat-best-genome-overall.pkl")
    if os.path.exists(best_genome_path):
        try:
            with open(best_genome_path, "rb") as f:
                overall_best_genome = pickle.load(f)
                if hasattr(overall_best_genome, "fitness"):
                    overall_best_fitness = overall_best_genome.fitness
                    print(f"已加载历史最佳基因组，适应度：{overall_best_fitness:.2f}")
        except Exception as e:
            print(f"加载最佳基因组失败: {e}")

    print(f"开始NEAT训练，共{n_generations}代（第{start_gen}代到第{start_gen + n_generations - 1}代）...")

    for gen in range(start_gen, start_gen + n_generations):
        winner = p.run(eval_genomes, 1)  # winner is the best genome of this generation

        if winner is not None and winner.fitness is not None:
            current_fitness = winner.fitness
            best_scores.append(current_fitness)

            if current_fitness > overall_best_fitness:
                overall_best_fitness = current_fitness
                overall_best_genome = winner
                print(
                    f"第{gen}代（已完成{gen - start_gen + 1}/{n_generations}）: 适应度 = {current_fitness:.2f} *** 新的历史最佳，已保存 ***"
                )
                # 立即保存最佳基因组
                best_genome_path = os.path.join(models_dir, "neat-best-genome-overall.pkl")
                with open(best_genome_path, "wb") as f:
                    pickle.dump(overall_best_genome, f)
            else:
                print(
                    f"第{gen}代（已完成{gen - start_gen + 1}/{n_generations}）: 适应度 = {current_fitness:.2f}（历史最佳: {overall_best_fitness:.2f}）"
                )
        else:
            best_scores.append(0)
            print(f"第{gen}代（已完成{gen - start_gen + 1}/{n_generations}）: 未找到有效的基因组。")

        # 保存分数
        scores_path = os.path.join(models_dir, "neat_scores.pkl")
        with open(scores_path, "wb") as f:
            pickle.dump(best_scores, f)

    # 训练结束提示
    if overall_best_genome:
        print(
            f"\n训练完成。历史最佳基因组已保存到{os.path.join(models_dir, 'neat-best-genome-overall.pkl')}，适应度：{overall_best_fitness:.2f}"
        )
    else:
        print("\n训练完成。未找到可保存的历史最佳基因组。")

    # 绘制分数曲线
    plot_path = os.path.join(models_dir, "neat_training_curve.png")
    plt.figure(figsize=(10, 6))
    plt.plot(best_scores)
    plt.xlabel("代数")
    plt.ylabel("最佳适应度")
    plt.title("NEAT训练进度")
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"训练进度图已保存到{plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练NEAT模型")
    parser.add_argument("--config", type=str, default="neat-config.txt", help="配置文件路径")
    parser.add_argument("--generations", type=int, default=50000, help="要训练的代数")
    parser.add_argument("--checkpoint", type=str, help="要加载的checkpoint文件路径")

    args = parser.parse_args()

    run(args.config, args.generations, args.checkpoint)
