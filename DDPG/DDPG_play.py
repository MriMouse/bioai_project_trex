import torch
import numpy as np
import os
import time
import pygame  # For QUIT event and clock

from aiGame import TRexGame  # Your game environment
from DDPG_agent import DDPGAgent  # Your DDPG agent

# --- Configuration ---
STATE_DIM = 14  # Must match the training configuration
ACTION_DIM = 3  # Must match the training configuration
HIDDEN_DIM = 256  # Must match the training configuration

MODEL_DIR = "./trex_ddpg_models"  # Directory where models are saved
CKTAG = "best"  # Example checkpoint tag
ACTOR_MODEL_FILE = f"actor_{CKTAG}.pkl"
CRITIC_MODEL_FILE = f"critic_{CKTAG}.pkl"

NUM_EPISODES_TO_PLAY = 5
FPS = 60  # Frames per second for rendering

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def normalize_state(state_dict):
    """Converts game state dictionary to a normalized numpy array.
    This MUST be identical to the normalization used during training."""
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


def play_game(actor_model_path, critic_model_path, num_episodes):
    env = TRexGame(human_mode=False, use_fixed_seed=True)  # Fixed seed for reproducibility

    agent = DDPGAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr_actor=0,
        lr_critic=0,  # Not used for inference
        gamma=0,
        tau=0,  # Not used for inference
        buffer_size=1,  # Not used for inference
        device=DEVICE,
        total_training_steps=1,  # Not used
        lr_decay=False,
        max_grad_norm=0,  # Not used
    )

    try:
        agent.load_model(actor_model_path, critic_model_path)
        print(f"Successfully loaded models: \n  Actor: {actor_model_path}\n  Critic: {critic_model_path}")
    except FileNotFoundError:
        print(f"Error: Model files not found at {actor_model_path} or {critic_model_path}")
        print("Please ensure the model files exist or provide correct paths.")
        env.close()
        return
    except Exception as e:
        print(f"Error loading models: {e}")
        env.close()
        return

    # agent.actor.eval() is called in load_model

    for episode in range(1, num_episodes + 1):
        print(f"\n--- Starting Episode {episode}/{num_episodes} ---")
        state_dict = env.reset()
        current_state_normalized = normalize_state(state_dict)
        episode_reward = 0
        done = False
        running = True

        while running and not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Quit event detected. Closing game.")
                    running = False
                    done = True
                    break
            if not running:
                break

            action = agent.get_action(current_state_normalized, evaluate=True)  # Deterministic action

            next_state_dict, reward, game_done = env.step(action)
            done = game_done

            current_state_normalized = normalize_state(next_state_dict)
            episode_reward += reward

            env._render()
            env.clock.tick(FPS)

        if not running:
            break

        print(f"Episode {episode} finished. Score: {int(episode_reward)}")
        if env.get_score() != int(episode_reward):
            print(f"  (Game internal score: {env.get_score()})")

    print("\n--- Finished Playing ---")
    env.close()


if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory {MODEL_DIR} not found. Please train models first or specify correct path.")
    else:
        actor_path = os.path.join(MODEL_DIR, ACTOR_MODEL_FILE)
        critic_path = os.path.join(MODEL_DIR, CRITIC_MODEL_FILE)
        play_game(actor_path, critic_path, NUM_EPISODES_TO_PLAY)
