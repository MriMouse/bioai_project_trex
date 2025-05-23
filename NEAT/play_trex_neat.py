import neat
import pickle
import os
import pygame  # Pygame is used by aiGame.py for rendering
from aiGame import TRexGame, DEFAULT_SEED  # 导入DEFAULT_SEED

# --- Configuration ---
CONFIG_PATH = "neat-config.txt"
# Path to the saved genome you want to use - 可选多种格式
SAVED_GENOME_PATH = os.path.join("saved_models", "neat-best-genome-overall.pkl")  # 最佳模型
# 或者使用特定的checkpoint
# SAVED_GENOME_PATH = os.path.join("saved_models", "neat-checkpoint-199")  # 不带.pkl的checkpoint
FPS = 90  # Frames per second for game display


def play_game(genome, config):
    """
    Plays the TRex game using the provided genome and NEAT configuration.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    game = TRexGame(human_mode=True, random_seed=DEFAULT_SEED)  # 使用固定随机种子确保游戏环境一致
    state = game.reset()
    # MODIFICATION: Ensure the game starts its internal logic for AI play in human_mode
    game.start_game_action = True

    # Send an initial jump action to start the game, as per user finding
    print("Sending initial jump to start the game...")

    clock = pygame.time.Clock()

    running = True
    frame_num = 0  # For tracking frames

    while running:
        frame_num += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

        if not running:
            break

        # Prepare inputs for the neural network
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
        # Removed: print(f"[DEBUG Frame {frame_num}] Inputs to NN: {inputs}")

        # Get action from the neural network
        output = net.activate(inputs)
        action = output.index(max(output))  # 0: No-op, 1: Jump, 2: Duck

        if action == 1:
            print(f"[Frame {frame_num}] AI chose JUMP. Raw output: {output}")
        elif action == 2:
            print(f"[Frame {frame_num}] AI chose DUCK. Raw output: {output}")

        # Perform the action in the game
        state, reward, done = game.step(action)

        # Render the game
        game._render()

        clock.tick(FPS)

        if done:
            print(f"Game Over! Final Score: {int(game.get_score())}")
            running = False
            pygame.time.wait(2000)

    pygame.quit()
    print("Game window closed.")


if __name__ == "__main__":
    # Ensure the script is running from the NEAT directory or adjust paths
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths
    abs_config_path = os.path.join(script_dir, CONFIG_PATH)
    abs_saved_genome_path = os.path.join(script_dir, SAVED_GENOME_PATH)
    if not os.path.exists(abs_config_path):
        print(f"Error: NEAT config file not found at {abs_config_path}")
        exit()
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, abs_config_path
    )

    if not os.path.exists(abs_saved_genome_path):
        print(f"Error: Saved genome file not found at {abs_saved_genome_path}")
        exit()

    print(f"Loading genome from {abs_saved_genome_path}...")

    # 根据文件扩展名和命名方式判断处理方法
    if abs_saved_genome_path.endswith(".pkl"):
        # 直接加载.pkl文件（可能是最佳基因组或checkpoint）
        with open(abs_saved_genome_path, "rb") as f:
            saved_data = pickle.load(f)

            # 判断是否是最佳基因组文件
            if hasattr(saved_data, "fitness"):
                best_genome = saved_data
                print(f"已加载单个基因组，适应度: {saved_data.fitness if hasattr(saved_data, 'fitness') else '未知'}")
            else:
                # 可能是checkpoint字典，尝试提取种群并获取最佳基因组
                try:
                    population = saved_data.population
                    best_genome = max(
                        population.values(),
                        key=lambda x: x.fitness if hasattr(x, "fitness") and x.fitness is not None else -float("inf"),
                    )
                    print(
                        f"从checkpoint中提取最佳基因组，适应度: {best_genome.fitness if hasattr(best_genome, 'fitness') else '未知'}"
                    )
                except:
                    print("无法从文件中提取有效的基因组，请检查文件格式")
                    exit()
    else:
        # 尝试作为NEAT checkpoint加载
        try:
            # 使用neat的Checkpointer加载
            checkpoint = neat.Checkpointer.restore_checkpoint(abs_saved_genome_path)
            # 从checkpoint的种群中找出最佳基因组
            population = checkpoint.population
            best_genome = max(
                population.values(),
                key=lambda x: x.fitness if hasattr(x, "fitness") and x.fitness is not None else -float("inf"),
            )
            print(
                f"从checkpoint中加载最佳基因组，适应度: {best_genome.fitness if hasattr(best_genome, 'fitness') else '未知'}"
            )
        except Exception as e:
            print(f"加载checkpoint失败: {e}")
            exit()

    print("Genome loaded. Starting game...")
    play_game(best_genome, config)
