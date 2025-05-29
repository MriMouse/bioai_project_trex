import torch
import numpy as np
from collections import deque
import os
import time

from aiGame import TRexGame  # Your game environment
from ppo_agent import PPOAgent  # Your PPO agent

# --- Configuration ---
STATE_DIM = 14  # player_y, player_v_vel, is_crouch, 2x(obs_dist, obs_y, obs_w, obs_h, obs_is_bird), game_speed
ACTION_DIM = 3  # 0: no-op, 1: jump, 2: crouch/fast-fall
HIDDEN_DIM = 256
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_EPOCHS = 10
MINIBATCH_SIZE = 64
UPDATE_TIMESTEP = 2048  # Collect this many steps before PPO update

# Decay parameters
TOTAL_TRAINING_STEPS = 2_000_000  # Total environment steps over which LR and entropy decay
LR_DECAY = True
FINAL_LR_FACTOR = 0.01  # Actor/Critic LR will decay to initial_lr * final_lr_factor
INITIAL_ENTROPY_COEF = 0.01
FINAL_ENTROPY_COEF = 0.001
ENTROPY_DECAY = True
MAX_GRAD_NORM = 0.5

MAX_EPISODES = 1000
SAVE_INTERVAL = 500  # Save model every 100 episodes
MODEL_PATH = "./trex_ppo_models"
LOG_INTERVAL = 10  # Print stats every 10 episodes
LOAD_MODEL = True  # Flag to load a pre-trained model
BEST_MODEL_ACTOR_PATH = os.path.join(MODEL_PATH, "actor_best.pkl")
BEST_MODEL_CRITIC_PATH = os.path.join(MODEL_PATH, "critic_best.pkl")

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def normalize_state(state_dict):
    """Converts game state dictionary to a normalized numpy array."""
    s = state_dict
    # Normalization constants (approximate, might need tuning)
    norm_player_y = (s["player_y"] - 65) / 45.0  # Peak jump ~20, ground 110. Mid 65, Range 90.
    norm_player_v_vel = (s["player_vertical_velocity"]) / 12.0  # Approx max absolute velocity
    norm_is_crouching = 1.0 if s["is_crouching"] else 0.0

    # Use 1000 as a general large distance for normalization, actual max can be smaller
    norm_obs1_dist_x = s["obs1_dist_x"] / 600.0  # Screen width is 600
    norm_obs1_y = (s["obs1_y"] - 107.5) / 22.5  # Range ~85-130. Mid 107.5, Range 45.
    norm_obs1_width = s["obs1_width"] / 50.0
    norm_obs1_height = s["obs1_height"] / 50.0
    norm_obs1_is_bird = 1.0 if s["obs1_is_bird"] else 0.0

    norm_obs2_dist_x = s["obs2_dist_x"] / 600.0
    norm_obs2_y = (s["obs2_y"] - 107.5) / 22.5
    norm_obs2_width = s["obs2_width"] / 50.0
    norm_obs2_height = s["obs2_height"] / 50.0
    norm_obs2_is_bird = 1.0 if s["obs2_is_bird"] else 0.0

    norm_game_speed = (s["game_speed"] - 4.0) / 6.0  # Speed 4 to ~10 (0 to 1)

    state_vector = np.array(
        [
            norm_player_y,
            norm_player_v_vel,
            norm_is_crouching,
            norm_obs1_dist_x,
            norm_obs1_y,
            norm_obs1_width,
            norm_obs1_height,
            norm_obs1_is_bird,
            norm_obs2_dist_x,
            norm_obs2_y,
            norm_obs2_width,
            norm_obs2_height,
            norm_obs2_is_bird,
            norm_game_speed,
        ],
        dtype=np.float32,
    )
    return state_vector


def main():
    env = TRexGame(human_mode=False, use_fixed_seed=True)
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ppo_epsilon=PPO_EPSILON,
        ppo_epochs=PPO_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        device=DEVICE,
        total_training_steps=TOTAL_TRAINING_STEPS,
        lr_decay=LR_DECAY,
        final_lr_factor=FINAL_LR_FACTOR,
        initial_entropy_coef=INITIAL_ENTROPY_COEF,
        final_entropy_coef=FINAL_ENTROPY_COEF,
        entropy_decay=ENTROPY_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
    )

    start_episode = 1
    if LOAD_MODEL and os.path.exists(BEST_MODEL_ACTOR_PATH) and os.path.exists(BEST_MODEL_CRITIC_PATH):
        try:
            agent.load_model(BEST_MODEL_ACTOR_PATH, BEST_MODEL_CRITIC_PATH)
            print(f"成功加载模型: {BEST_MODEL_ACTOR_PATH} 和 {BEST_MODEL_CRITIC_PATH}")
            # Potentially load other training states like episode number, optimizer states, etc.
            # For simplicity, we'll just load model weights here.
            # If you saved episode number, you could set:
            # start_episode = loaded_episode_number + 1
        except Exception as e:
            print(f"加载模型失败: {e}. 将从头开始训练。")
    else:
        print("未找到预训练模型或LOAD_MODEL为False。将从头开始训练。")

    global_timestep_counter = 0  # Should be loaded if continuing training properly
    best_avg_score = -float("inf")  # Should be loaded if continuing training
    episode_rewards_deque = deque(maxlen=100)  # For calculating average score over last 100 episodes

    start_time = time.time()

    for episode in range(start_episode, MAX_EPISODES + 1):
        state_dict = env.reset()
        current_state = normalize_state(state_dict)
        episode_reward = 0
        done = False

        while not done:
            action, value, log_prob = agent.get_action(current_state)

            # The game's step function expects actions 0, 1, or 2
            # action from agent.get_action is already an integer in this range
            next_state_dict, reward, done = env.step(action)

            next_state = normalize_state(next_state_dict)

            agent.replay_buffer.add_memo(current_state, action, reward, value, done)

            current_state = next_state
            episode_reward += reward
            global_timestep_counter += 1

            if (
                global_timestep_counter % UPDATE_TIMESTEP == 0 and agent.replay_buffer.size() >= MINIBATCH_SIZE
            ):  # Ensure enough samples
                # Calculate value of the last state for GAE
                if not done:
                    _, last_val, _ = agent.get_action(next_state, evaluate=True)  # Get value from critic
                else:
                    last_val = 0.0

                lr_a, lr_c, ent_c = agent.update(last_val, global_timestep_counter)
                # print(f"Update at timestep {global_timestep_counter}. LR_A: {lr_a:.2e}, LR_C: {lr_c:.2e}, EntC: {ent_c:.3f}")

            if done:
                break

        # If episode ended and buffer has data not yet used for update (e.g. episode < UPDATE_TIMESTEP)
        if agent.replay_buffer.size() > MINIBATCH_SIZE and agent.replay_buffer.size() % UPDATE_TIMESTEP != 0:
            last_val = 0.0  # done is true
            lr_a, lr_c, ent_c = agent.update(last_val, global_timestep_counter)
            # print(f"Update at end of episode {episode}. LR_A: {lr_a:.2e}, LR_C: {lr_c:.2e}, EntC: {ent_c:.3f}")

        episode_rewards_deque.append(episode_reward)
        avg_score_100_eps = np.mean(episode_rewards_deque)

        if episode % LOG_INTERVAL == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Ep: {episode}, Score: {episode_reward:.2f}, Avg Score (100ep): {avg_score_100_eps:.2f}, Timesteps: {global_timestep_counter}, Time: {elapsed_time:.2f}s"
            )
            # For more detailed logging if needed:
            # print(f"  LR_Actor: {agent.actor_optimizer.param_groups[0]['lr']:.2e}, LR_Critic: {agent.critic_optimizer.param_groups[0]['lr']:.2e}, EntropyCoef: {ent_c if 'ent_c' in locals() else agent.initial_entropy_coef:.3f}")

        if episode % SAVE_INTERVAL == 0:
            actor_path = os.path.join(MODEL_PATH, f"actor_ep{episode}.pkl")
            critic_path = os.path.join(MODEL_PATH, f"critic_ep{episode}.pkl")
            agent.save_model(actor_path, critic_path)
            print(f"Saved model at episode {episode}: {actor_path}, {critic_path}")

        if (
            avg_score_100_eps > best_avg_score and len(episode_rewards_deque) >= 100
        ):  # Ensure enough episodes for stable average
            best_avg_score = avg_score_100_eps
            actor_path_best = os.path.join(MODEL_PATH, "actor_best.pkl")
            critic_path_best = os.path.join(MODEL_PATH, "critic_best.pkl")
            agent.save_model(actor_path_best, critic_path_best)
            print(f"New best average score: {best_avg_score:.2f}. Saved best model.")

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
