from matplotlib.colors import Normalize
import neat
import pickle
import matplotlib.pyplot as plt
from aiGame import TRexGame
import os
import argparse
import re
import visualize as visualize  # <--- 添加可视化模块
import numpy as np  # <--- 添加numpy用于统计


def eval_genomes(genomes, config, fixed_seed_enabled=True):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = TRexGame(human_mode=False, use_fixed_seed=fixed_seed_enabled)
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

            # 添加鸟类惩罚逻辑，优化适应度计算
            bird_penalty = 0
            overlap_threshold = 5 + (state["obs1_width"] if state["obs1_is_bird"] else 0)  # 判定重合的距离阈值

            # 检查第一个障碍物
            if state["obs1_is_bird"] and (abs(state["obs1_dist_x"]) < overlap_threshold):
                bird_y = state["obs1_y"]
                if bird_y <= 85:  # 高飞的鸟应该蹲下通过
                    if action == 1:
                        bird_penalty -= 20  # 让你蹲你不准跳
                    elif action == 0:
                        bird_penalty -= 10
                    elif action == 2:
                        bird_penalty += 0  # 跳起来大大有赏
                elif bird_y >= 125:  # 低飞的鸟应该跳跃通过
                    if action == 2:
                        bird_penalty -= 20  # 让你跳你不准蹲
                    elif action == 0:
                        bird_penalty -= 10
                    elif action == 1:
                        bird_penalty += 0  # 蹲起来大大有赏
                else:  # 中等高度的鸟，蹲或跳都可以
                    if action == 0:
                        bird_penalty -= 10  # 干点啥都行你别不干

            # # 检查第二个障碍物
            # if state["obs2_is_bird"] and abs(state["obs2_dist_x"]) < overlap_threshold and action == 0:
            #     bird_y = state["obs2_y"]
            #     if bird_y <= 85:
            #         bird_penalty -= 10
            #     elif bird_y >= 95:
            #         bird_penalty -= 10
            #     else:
            #         bird_penalty -= 5

            state, reward, done = game.step(action)
            fitness += reward + bird_penalty  # 将鸟类惩罚加入适应度计算

            if fitness > 6000:
                done = True
        genome.fitness = fitness


def run(
    config_file, n_generations=10000, checkpoint_path=None, fixed_seed_generations=None, min_fitness_for_random=3000
):
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    log_file_path = os.path.join(models_dir, "neat_training_log.txt")

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file
    )

    if checkpoint_path:
        print(f"加载checkpoint: {checkpoint_path}")
        try:
            p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
            checkpoint_num_match = re.search(r"saved_models\\neat-checkpoint-(\d+)", checkpoint_path)
            start_gen = int(checkpoint_num_match.group(1)) if checkpoint_num_match else 0
            # import code

            # code.interact(local=locals())

            # 尝试加载已有的统计数据
            stats_path = os.path.join(models_dir, "neat_stats_data.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    saved_stats = pickle.load(f)
                    best_scores = saved_stats.get("best_scores", [])
                    avg_scores = saved_stats.get("avg_scores", [])
                    stdev_scores = saved_stats.get("stdev_scores", [])
                    species_counts = saved_stats.get("species_counts", [])
                    best_genome_nodes = saved_stats.get("best_genome_nodes", [])
                    best_genome_conns = saved_stats.get("best_genome_conns", [])
                # import code

                # code.interact(local=locals())
                print(f"已加载{len(best_scores)}代的历史统计数据")
            else:
                best_scores, avg_scores, stdev_scores, species_counts, best_genome_nodes, best_genome_conns = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
            print(f"成功加载checkpoint，从第{start_gen}代继续训练...")
        except Exception as e:
            print(f"加载checkpoint失败: {e}")
            p = neat.Population(config)
            start_gen = 0
            best_scores, avg_scores, stdev_scores, species_counts, best_genome_nodes, best_genome_conns = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
    else:
        p = neat.Population(config)
        start_gen = 0
        best_scores, avg_scores, stdev_scores, species_counts, best_genome_nodes, best_genome_conns = (
            [],
            [],
            [],
            [],
            [],
            [],
        )  # 如果是新的训练，清空或创建新的日志文件
        with open(log_file_path, "w") as log_f:
            log_f.write("NEAT Training Log\n")
            log_f.write("=" * 30 + "\n")

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))  # <--- 添加标准输出报告器，每代打印简报

    # 创建checkpointer对象，便于手动保存
    checkpointer = neat.Checkpointer(100, filename_prefix=os.path.join(models_dir, "neat-checkpoint-"))
    p.add_reporter(checkpointer)

    overall_best_genome = None
    overall_best_fitness = -float("inf")

    best_genome_path = os.path.join(models_dir, "neat-best-genome-overall.pkl")

    try:
        with open(best_genome_path, "rb") as f:
            overall_best_genome = pickle.load(f)
            if hasattr(overall_best_genome, "fitness") and overall_best_genome.fitness is not None:
                overall_best_fitness = overall_best_genome.fitness
                print(f"已加载历史最佳基因组，适应度：{overall_best_fitness:.2f}")
            else:  # 如果加载的genome没有fitness属性或为None，重置
                print("加载的历史最佳基因组无有效适应度，将重新评估。")
                overall_best_genome = None
                overall_best_fitness = -float("inf")
    except Exception as e:
        print(f"加载最佳基因组失败: {e}")

    print(f"开始NEAT训练，共{n_generations}代（第{start_gen}代到第{start_gen + n_generations -1}代）...")

    # 添加种子策略控制逻辑
    if fixed_seed_generations is None:
        fixed_seed_generations = float("inf")  # 如果不设置，则一直使用固定种子

    print(f"种子策略: 前{fixed_seed_generations}代使用固定种子, 适应度达到{min_fitness_for_random}后切换到随机种子")

    if start_gen == 0:  # 确保从头开始训练时日志文件头部被写入
        with open(log_file_path, "a") as log_f:
            if log_f.tell() == 0:  # 检查文件是否为空，避免重复写入头部
                log_f.write("NEAT Training Log\n")
                log_f.write("=" * 30 + "\n")
                log_f.write(
                    f"Seed Strategy: Fixed seed for first {fixed_seed_generations} generations, switch to random when fitness > {min_fitness_for_random}\n"
                )
                log_f.write("=" * 30 + "\n")

    prev_use_fixed_seed = True  # 初始化变量
    for gen in range(start_gen, start_gen + n_generations):
        # 决定是否使用固定种子
        use_fixed_seed = True
        if gen >= fixed_seed_generations:
            use_fixed_seed = False
        elif overall_best_fitness > min_fitness_for_random:
            use_fixed_seed = False

        # 如果种子策略改变，打印信息
        if gen == 0 or (gen > 0 and use_fixed_seed != prev_use_fixed_seed):
            seed_status = "固定种子" if use_fixed_seed else "随机种子"
            print(f"第{gen}代开始使用{seed_status}")
            with open(log_file_path, "a") as log_f:
                log_f.write(f"Generation {gen}: Switching to {'fixed seed' if use_fixed_seed else 'random seed'}\n")

        prev_use_fixed_seed = use_fixed_seed

        # 使用lambda来传递额外参数给eval_genomes
        def eval_wrapper(genomes, config):
            return eval_genomes(genomes, config, use_fixed_seed)

        winner = p.run(eval_wrapper, 1)  # winner is the best genome of this generation

        # --- 收集统计数据 ---
        if winner is not None and winner.fitness is not None:
            current_fitness = winner.fitness
            best_scores.append(current_fitness)

            # 从 StatisticsReporter 获取平均和标准差 (确保stats在p.run后更新)
            # 注意：stats.get_fitness_mean() 返回的是历史列表，取最后一个
            if stats.get_fitness_mean():
                avg_scores.append(stats.get_fitness_mean()[-1])
            if stats.get_fitness_stdev():
                stdev_scores.append(stats.get_fitness_stdev()[-1])  # 基因组复杂度
            nodes, conns = winner.size()
            best_genome_nodes.append(nodes)
            best_genome_conns.append(conns)

            if current_fitness > overall_best_fitness:
                overall_best_fitness = current_fitness
                overall_best_genome = winner

                # 创建带时间戳的文件名
                import time

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                fitness_str = f"{current_fitness:.0f}"  # 保存完整checkpoint（包含整个种群）
                # 注意：NEAT checkpointer自动使用filename_prefix + generation作为文件名
                # 我们先调用标准保存，然后复制到我们的自定义文件名
                checkpointer.save_checkpoint(config, p.population, p.species, gen)

                # 复制到带时间戳的自定义文件名
                import shutil

                standard_checkpoint = os.path.join(models_dir, f"neat-checkpoint-{gen}")
                checkpoint_filename = f"neat-best-checkpoint-gen{gen}-fitness{fitness_str}-{timestamp}"
                checkpoint_path = os.path.join(models_dir, checkpoint_filename)
                if os.path.exists(standard_checkpoint):
                    shutil.copy2(standard_checkpoint, checkpoint_path)

                # 保存最佳基因组（单独保存便于快速加载）
                best_genome_filename = f"neat-best-genome-gen{gen}-fitness{fitness_str}-{timestamp}.pkl"
                best_genome_path = os.path.join(models_dir, best_genome_filename)
                with open(best_genome_path, "wb") as f:
                    pickle.dump(overall_best_genome, f)

                # 保存当前的统计数据
                stats_filename = f"neat-stats-gen{gen}-fitness{fitness_str}-{timestamp}.pkl"
                stats_path = os.path.join(models_dir, stats_filename)
                current_stats_data = {
                    "best_scores": best_scores,
                    "avg_scores": avg_scores,
                    "stdev_scores": stdev_scores,
                    "species_counts": species_counts,
                    "best_genome_nodes": best_genome_nodes,
                    "best_genome_conns": best_genome_conns,
                    "generation": gen,
                    "best_fitness": current_fitness,
                    "timestamp": timestamp,
                }
                with open(stats_path, "wb") as f:
                    pickle.dump(current_stats_data, f)

                # 同时更新overall最佳文件（保持兼容性）
                overall_best_genome_path = os.path.join(models_dir, "neat-best-genome-overall.pkl")
                with open(overall_best_genome_path, "wb") as f:
                    pickle.dump(overall_best_genome, f)

                print(
                    f"第{gen}代（已完成{gen - start_gen + 1}/{n_generations}）: 适应度 = {current_fitness:.2f} *** 新的历史最佳 ***"
                )
                print(f"  完整checkpoint已保存: {checkpoint_filename}")
                print(f"  最佳基因组已保存: {best_genome_filename}")
                print(f"  统计数据已保存: {stats_filename}")
            else:
                print(
                    f"第{gen}代（已完成{gen - start_gen + 1}/{n_generations}）: 适应度 = {current_fitness:.2f}（历史最佳: {overall_best_fitness:.2f}）"
                )  # Log current generation's detailed stats
            with open(log_file_path, "a") as log_f:
                log_f.write(f"--- Generation {gen} ---\n")
                log_f.write(f"  Best Fitness: {current_fitness:.2f}\n")
                if stats.get_fitness_mean():
                    log_f.write(f"  Average Fitness: {stats.get_fitness_mean()[-1]:.2f}\n")
                if stats.get_fitness_stdev():
                    log_f.write(f"  Std Dev Fitness: {stats.get_fitness_stdev()[-1]:.2f}\n")
                log_f.write(f"  Species Count: {len(p.species.species)}\n")
                log_f.write(f"  Best Genome Nodes: {nodes}\n")
                log_f.write(f"  Best Genome Connections: {conns}\n")
                if current_fitness > (overall_best_fitness - current_fitness):  # 检查是否是新的最佳
                    log_f.write(f"  *** New Overall Best Genome Saved ***\n")
                log_f.write("-" * 20 + "\n")

        else:
            best_scores.append(0)  # 或 None, 或上一个值
            avg_scores.append(avg_scores[-1] if avg_scores else 0)
            stdev_scores.append(stdev_scores[-1] if stdev_scores else 0)
            best_genome_nodes.append(best_genome_nodes[-1] if best_genome_nodes else 0)
            best_genome_conns.append(best_genome_conns[-1] if best_genome_conns else 0)
            print(f"第{gen}代（已完成{gen - start_gen + 1}/{n_generations}）: 未找到有效的基因组。")
            # Log if no valid genome found
            with open(log_file_path, "a") as log_f:
                log_f.write(f"--- Generation {gen} ---\n")
                log_f.write("  No valid genome found in this generation.\n")
                log_f.write("-" * 20 + "\n")

        # 物种数量
        species_counts.append(len(p.species.species))
        # ---------------------

        # 保存所有统计数据
        all_stats_data = {
            "best_scores": best_scores,
            "avg_scores": avg_scores,
            "stdev_scores": stdev_scores,
            "species_counts": species_counts,
            "best_genome_nodes": best_genome_nodes,
            "best_genome_conns": best_genome_conns,
        }
        stats_path = os.path.join(models_dir, "neat_stats_data.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(all_stats_data, f)

    # 训练结束提示
    if overall_best_genome:
        print(
            f"\\n训练完成。历史最佳基因组已保存到{os.path.join(models_dir, 'neat-best-genome-overall.pkl')}，适应度：{overall_best_fitness:.2f}"
        )
        # 可视化最佳基因组
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
        try:
            visualize.draw_net(
                config,
                overall_best_genome,
                True,
                node_names=node_names,
                filename=os.path.join(models_dir, "neat-best-genome-net"),  # 修改这里
            )
            visualize.plot_stats(
                stats, ylog=False, view=False, filename=os.path.join(models_dir, "neat_fitness_stats.png")
            )
            visualize.plot_species(stats, view=False, filename=os.path.join(models_dir, "neat_speciation.png"))
            print(f"最佳基因组网络图、适应度统计图和物种图已保存到 {models_dir}")
            # Log final best genome details
            with open(log_file_path, "a") as log_f:
                log_f.write("\n=== Overall Best Genome ===\n")
                log_f.write(f"  Fitness: {overall_best_fitness:.2f}\n")
                if overall_best_genome:
                    nodes, conns = overall_best_genome.size()
                    log_f.write(f"  Nodes: {nodes}\n")
                    log_f.write(f"  Connections: {conns}\n")
                log_f.write("===========================\n")
        except Exception as e:
            print(f"保存网络图或统计图失败 (可能需要安装 graphviz 和 python-graphviz): {e}")

    else:
        print("\\n训练完成。未找到可保存的历史最佳基因组。")
        with open(log_file_path, "a") as log_f:
            log_f.write("\n=== Training Complete ===\n")
            log_f.write("  No overall best genome found to save.\n")
            log_f.write("===========================\n")

    last_genome_path = os.path.join(models_dir, "neat-last-genome.pkl")
    try:
        if "winner" in locals() and winner is not None:  # 确保winner存在
            with open(last_genome_path, "wb") as f:
                pickle.dump(winner, f)
            print(f"最后一代模型已保存到 {last_genome_path}")
    except Exception as e:
        print(f"保存最后一代模型失败: {e}")

    # --- 绘制自定义图表 ---
    generations_axis = range(len(best_scores))

    fig1 = plt.figure(figsize=(12, 8))  # Capture figure object
    plt.plot(generations_axis, best_scores, "b-", label="Best Fitness")
    if avg_scores:
        plt.plot(generations_axis, avg_scores, "g--", label="Average Fitness")
    if stdev_scores and avg_scores:  # 确保avg_scores存在以便填充
        avg_plus_std = [m + s for m, s in zip(avg_scores, stdev_scores)]
        avg_minus_std = [m - s for m, s in zip(avg_scores, stdev_scores)]
        plt.fill_between(generations_axis, avg_minus_std, avg_plus_std, color="gray", alpha=0.3, label="Std Dev")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("NEAT Training Fitness Over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(models_dir, "neat_custom_fitness_curve.png"))
    print(f"自定义适应度曲线图已保存到 {os.path.join(models_dir, 'neat_custom_fitness_curve.png')}")
    plt.close(fig1)  # Close the figure

    if species_counts:
        fig2 = plt.figure(figsize=(10, 6))  # Capture figure object
        plt.plot(generations_axis, species_counts)
        plt.xlabel("Generation")
        plt.ylabel("Number of Species")
        plt.title("Number of Species Over Generations")
        plt.grid(True)
        plt.savefig(os.path.join(models_dir, "neat_species_count.png"))
        print(f"物种数量图已保存到 {os.path.join(models_dir, 'neat_species_count.png')}")
        plt.close(fig2)  # Close the figure

    if best_genome_nodes and best_genome_conns:
        fig3, ax1 = plt.subplots(figsize=(10, 6))  # Capture figure object (fig3)
        color = "tab:red"
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Nodes (Best Genome)", color=color)
        ax1.plot(generations_axis, best_genome_nodes, color=color, linestyle="--")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # 共享x轴
        color = "tab:blue"
        ax2.set_ylabel("Connections (Best Genome)", color=color)
        ax2.plot(generations_axis, best_genome_conns, color=color, linestyle=":")
        ax2.tick_params(axis="y", labelcolor=color)

        fig3.tight_layout()  # 否则右边的y轴标签可能会被剪掉
        plt.title("Complexity of Best Genome Over Generations")
        plt.grid(True)
        plt.savefig(os.path.join(models_dir, "neat_genome_complexity.png"))
        print(f"基因组复杂度图已保存到 {os.path.join(models_dir, 'neat_genome_complexity.png')}")
        plt.close(fig3)  # Close the figure

    # plt.show()  # 一次性显示所有图 (注释掉，因为我们已经保存并关闭了它们)


def draw_checkpoint(config_file, checkpoint_path, top_n=5):
    print(f"--- Analyzing Checkpoint: {checkpoint_path} ---")

    # 加载配置
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file
    )
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

    # 加载 Checkpoint
    try:
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
        print(f"Successfully loaded checkpoint. Population size: {len(p.population)}")
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return

    # 创建输出目录
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    analysis_dir = os.path.join("saved_models", f"{checkpoint_name}_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Analysis results will be saved in: {analysis_dir}")

    # 评估所有基因组的适应度
    print("\nEvaluating fitness for all genomes in the checkpoint...")
    genomes_to_eval = list(p.population.items())
    eval_genomes(genomes_to_eval, config)
    print("Fitness evaluation complete.")

    # 提取所有基因组和它们的适应度
    evaluated_genomes = [genome for _, genome in p.population.items() if genome.fitness is not None]
    if not evaluated_genomes:
        print("No genomes with valid fitness scores after evaluation. Cannot proceed.")
        return

    # 选出 Top N
    evaluated_genomes.sort(key=lambda g: g.fitness, reverse=True)
    top_n_genomes = evaluated_genomes[:top_n]

    print(f"\n--- Top {len(top_n_genomes)} Genomes from Checkpoint ---")
    for i, genome in enumerate(top_n_genomes):
        nodes_count, conns_count = genome.size()
        print(
            f"\nRank {i+1}: Genome ID {genome.key}, Fitness: {genome.fitness:.2f}, Nodes: {nodes_count}, Connections: {conns_count}"
        )

        # 可视化网络
        net_filename = os.path.join(analysis_dir, f"genome_{genome.key}_rank{i+1}_net")
        try:
            visualize.draw_net(config, genome, view=False, node_names=node_names, filename=net_filename, fmt="png")
            print(f"  Network diagram saved to {net_filename}.png")
        except Exception as e:
            print(f"  Could not generate network diagram for genome {genome.key}: {e}")

    # 绘制种群整体统计图
    print("\n--- Checkpoint Population Statistics ---")
    all_fitnesses = [g.fitness for g in evaluated_genomes]
    all_nodes = [g.size()[0] for g in evaluated_genomes]
    all_conns = [g.size()[1] for g in evaluated_genomes]

    # 适应度分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_fitnesses, bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Fitness Distribution in Checkpoint {checkpoint_name} (N={len(all_fitnesses)})")
    plt.xlabel("Fitness")
    plt.ylabel("Number of Genomes")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(analysis_dir, "fitness_distribution.png"))
    plt.close()
    print(
        f"Fitness distribution histogram saved to {os.path.join(analysis_dir, 'fitness_distribution.png')}"
    )  # 基因组复杂度散点图
    if all_nodes and all_conns:
        plt.figure(figsize=(12, 8))
        norm = Normalize(vmin=min(all_fitnesses), vmax=max(all_fitnesses))
        scatter = plt.scatter(
            all_nodes, all_conns, c=all_fitnesses, cmap="viridis", norm=norm, alpha=0.7, edgecolors="w", s=50
        )
        plt.colorbar(scatter, label="Fitness")
        plt.title(f"Genome Complexity in Checkpoint {checkpoint_name}")
        plt.xlabel("Number of Nodes")
        plt.ylabel("Number of Connections")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(analysis_dir, "complexity_scatterplot.png"))
        plt.close()
        print(f"Complexity scatter plot saved to {os.path.join(analysis_dir, 'complexity_scatterplot.png')}")
    else:
        print("Could not generate complexity scatter plot due to missing data.")

    print(f"--- Analysis for {checkpoint_path} complete. ---")


# 修改主函数以支持新的命令
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or analyze NEAT models for TRex Game")
    subparsers = parser.add_subparsers(dest="command", help="Action to perform")
    subparsers.required = True

    train_parser = subparsers.add_parser("train", help="Train a new model or continue from checkpoint")
    train_parser.add_argument("--config", type=str, default="neat-config.txt", help="NEAT configuration file path")
    train_parser.add_argument("--generations", type=int, default=100, help="Number of generations to train")
    train_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to continue training from")
    train_parser.add_argument(
        "--fixed-seed-generations",
        type=int,
        default=None,
        help="Number of generations to use fixed seed before switching to random seed",
    )
    train_parser.add_argument(
        "--min-fitness-for-random",
        type=int,
        default=3000,
        help="Minimum fitness threshold to switch from fixed seed to random seed",
    )

    draw_parser = subparsers.add_parser("draw", help="Analyze and draw top genomes from a checkpoint")
    draw_parser.add_argument("--config", type=str, default="neat-config.txt", help="NEAT configuration file path")
    draw_parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file to analyze")
    draw_parser.add_argument("--top_n", type=int, default=5, help="Number of top genomes to visualize and analyze")

    args = parser.parse_args()
    if args.command == "train":
        run(args.config, args.generations, args.checkpoint, args.fixed_seed_generations, args.min_fitness_for_random)
    elif args.command == "draw":
        draw_checkpoint(args.config, args.checkpoint_path, args.top_n)
