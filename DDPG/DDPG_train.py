import torch
import numpy as np
from collections import deque
import os
import time
import math
import json  # Added import

from aiGame import TRexGame  # Your game environment
from DDPG_agent import DDPGAgent  # Your DDPG agent

# --- Configuration ---
STATE_DIM = 14
ACTION_DIM = 3
HIDDEN_DIM = 256
LR_ACTOR = 1e-4  # DDPG often uses smaller LRs
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005  # Soft update factor for target networks
REPLAY_BUFFER_SIZE = 100_000  # DDPG needs a large replay buffer
BATCH_SIZE = 64
GUMBEL_SOFTMAX_TAU = 1.0  # Temperature for Gumbel-Softmax in actor loss

# Exploration (Epsilon-greedy for discrete actions)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 500_000  # Steps over which epsilon decays

# Training settings
TOTAL_TRAINING_STEPS = 2_000_000  # Total environment steps for LR decay
LR_DECAY = True
FINAL_LR_FACTOR = 0.01
MAX_GRAD_NORM = 1.0  # Max gradient norm for clipping

MAX_EPISODES = 10000
SAVE_INTERVAL = 500
MODEL_PATH = "./trex_ddpg_models"
LOG_INTERVAL = 10
LOAD_MODEL = False  # Flag to load a pre-trained model
BEST_MODEL_ACTOR_PATH = os.path.join(MODEL_PATH, "actor_best.pkl")
BEST_MODEL_CRITIC_PATH = os.path.join(MODEL_PATH, "critic_best.pkl")
TRAINING_PROGRESS_PATH = os.path.join(MODEL_PATH, "training_progress.json")  # Path to save/load training progress
LEARNING_STARTS = 10000  # Number of steps to fill buffer before starting updates
UPDATE_EVERY_N_STEPS = 4  # Update networks every N environment steps

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def normalize_state(state_dict):
    """Converts game state dictionary to a normalized numpy array."""
    s = state_dict
    norm_player_y = (s["player_y"] - 65) / 45.0
    norm_player_v_vel = (s["player_vertical_velocity"]) / 12.0
    norm_is_crouching = 1.0 if s["is_crouching"] else 0.0
    norm_obs1_dist_x = s["obs1_dist_x"] / 600.0
    norm_obs1_y = (s["obs1_y"] - 107.5) / 22.5
    norm_obs1_width = s["obs1_width"] / 50.0
    norm_obs1_height = s["obs1_height"] / 50.0
    norm_obs1_is_bird = 1.0 if s["obs1_is_bird"] else 0.0
    norm_obs2_dist_x = s["obs2_dist_x"] / 600.0
    norm_obs2_y = (s["obs2_y"] - 107.5) / 22.5
    norm_obs2_width = s["obs2_width"] / 50.0
    norm_obs2_height = s["obs2_height"] / 50.0
    norm_obs2_is_bird = 1.0 if s["obs2_is_bird"] else 0.0
    norm_game_speed = (s["game_speed"] - 4.0) / 6.0

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


def get_epsilon(current_step, start=EPSILON_START, end=EPSILON_END, decay_steps=EPSILON_DECAY_STEPS):
    if current_step >= decay_steps:
        return end
    return start - (start - end) * (current_step / decay_steps)


def main():
    env = TRexGame(human_mode=False, use_fixed_seed=True)  # Fixed seed for reproducibility
    agent = DDPGAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=REPLAY_BUFFER_SIZE,
        device=DEVICE,
        total_training_steps=TOTAL_TRAINING_STEPS,
        lr_decay=LR_DECAY,
        final_lr_factor=FINAL_LR_FACTOR,
        max_grad_norm=MAX_GRAD_NORM,
        gumbel_softmax_tau=GUMBEL_SOFTMAX_TAU,
    )

    start_episode = 1
    global_timestep_counter = 0
    best_avg_score = -float("inf")

    if LOAD_MODEL:
        if os.path.exists(BEST_MODEL_ACTOR_PATH) and os.path.exists(BEST_MODEL_CRITIC_PATH):
            try:
                agent.load_model(BEST_MODEL_ACTOR_PATH, BEST_MODEL_CRITIC_PATH)
                agent.actor.train()
                agent.critic.train()
                agent.actor_target.eval()
                agent.critic_target.eval()
                print(f"Successfully loaded models: {BEST_MODEL_ACTOR_PATH} and {BEST_MODEL_CRITIC_PATH}")

                if os.path.exists(TRAINING_PROGRESS_PATH):
                    with open(TRAINING_PROGRESS_PATH, "r") as f:
                        progress = json.load(f)
                        start_episode = progress.get("episode", 1) + 1  # Start from the next episode
                        global_timestep_counter = progress.get("global_timestep_counter", 0)
                        best_avg_score = progress.get("best_avg_score", -float("inf"))
                        # Load deque requires a bit more care if you want to preserve its exact state
                        # For simplicity, we'll just use the best_avg_score and continue building the deque
                        print(
                            f"Loaded training progress: Episode {start_episode-1}, Timesteps {global_timestep_counter}, Best Avg Score {best_avg_score:.2f}"
                        )
                else:
                    print("No training progress file found. Starting from default values.")

            except Exception as e:
                print(f"Failed to load models or progress: {e}. Training from scratch.")
                start_episode = 1
                global_timestep_counter = 0
                best_avg_score = -float("inf")
        else:
            print("Best model files not found. Training from scratch.")
    else:
        print("LOAD_MODEL is False. Training from scratch.")

    episode_rewards_deque = deque(maxlen=100)
    start_time = time.time()
    current_lr_a, current_lr_c = LR_ACTOR, LR_CRITIC  # For logging

    for episode in range(start_episode, MAX_EPISODES + 1):
        state_dict = env.reset()
        current_state = normalize_state(state_dict)
        episode_reward = 0
        done = False

        while not done:
            epsilon = get_epsilon(global_timestep_counter)
            action = agent.get_action(current_state, epsilon=epsilon, evaluate=False)

            next_state_dict, reward, done = env.step(action)
            next_state = normalize_state(next_state_dict)

            agent.replay_buffer.add_memo(current_state, action, reward, next_state, done)

            current_state = next_state
            episode_reward += reward
            global_timestep_counter += 1

            if global_timestep_counter >= LEARNING_STARTS and global_timestep_counter % UPDATE_EVERY_N_STEPS == 0:
                update_results = agent.update(BATCH_SIZE, global_timestep_counter)
                if update_results:  # If update happened (buffer had enough samples)
                    current_lr_a, current_lr_c = update_results

            if done:
                break

        episode_rewards_deque.append(episode_reward)
        avg_score_100_eps = np.mean(episode_rewards_deque)

        if episode % LOG_INTERVAL == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Ep: {episode}, Score: {episode_reward:.2f}, Avg Score (100ep): {avg_score_100_eps:.2f}, "
                f"Timesteps: {global_timestep_counter}, Epsilon: {epsilon:.3f}, Time: {elapsed_time:.2f}s"
            )
            print(f"  LR_Actor: {current_lr_a:.2e}, LR_Critic: {current_lr_c:.2e}")

        if episode % SAVE_INTERVAL == 0:
            actor_path = os.path.join(MODEL_PATH, f"actor_ep{episode}.pkl")
            critic_path = os.path.join(MODEL_PATH, f"critic_ep{episode}.pkl")
            agent.save_model(actor_path, critic_path)
            print(f"Saved model at episode {episode}: {actor_path}, {critic_path}")

        if avg_score_100_eps > best_avg_score and len(episode_rewards_deque) >= 100:
            best_avg_score = avg_score_100_eps
            agent.save_model(BEST_MODEL_ACTOR_PATH, BEST_MODEL_CRITIC_PATH)
            print(f"New best average score: {best_avg_score:.2f}. Saved best model.")
            # Save training progress
            progress_data = {
                "episode": episode,
                "global_timestep_counter": global_timestep_counter,
                "best_avg_score": best_avg_score,
            }
            with open(TRAINING_PROGRESS_PATH, "w") as f:
                json.dump(progress_data, f)
            print(f"Saved training progress to {TRAINING_PROGRESS_PATH}")

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
