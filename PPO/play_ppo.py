import torch
import numpy as np
import os
import time
import pygame  # For QUIT event and clock

from aiGame import TRexGame  # Your game environment
from ppo_agent import PPOAgent  # Your PPO agent

# --- Configuration ---
STATE_DIM = 14  # Must match the training configuration
ACTION_DIM = 3  # Must match the training configuration
HIDDEN_DIM = 256  # Must match the training configuration

MODEL_DIR = "./trex_ppo_models"  # Directory where models are saved
CKTAG = "best"
ACTOR_MODEL_FILE = f"actor_{CKTAG}.pkl"  # "actor_epISODE_NUM.pkl" or "actor_best.pkl"
CRITIC_MODEL_FILE = f"critic_{CKTAG}.pkl"  # "critic_epISODE_NUM.pkl" or "critic_best.pkl"

NUM_EPISODES_TO_PLAY = 5
FPS = 60  # Frames per second for rendering

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Initialize environment
    # human_mode=False means the game loop won't try to get human input for actions
    env = TRexGame(human_mode=False)

    # Initialize agent
    # Dummy values for LR, gamma etc., as they are not used for inference
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr_actor=0,
        lr_critic=0,  # Not used for inference
        gamma=0,
        gae_lambda=0,
        ppo_epsilon=0,  # Not used
        ppo_epochs=0,
        minibatch_size=0,  # Not used
        device=DEVICE,
        total_training_steps=1,  # Not used
        lr_decay=False,
        entropy_decay=False,  # Not used
    )

    # Load trained models
    try:
        agent.load_model(actor_model_path, critic_model_path)
        print(f"Successfully loaded models: \n  Actor: {actor_model_path}\n  Critic: {critic_model_path}")
    except FileNotFoundError:
        print(f"Error: Model files not found at {actor_model_path} or {critic_model_path}")
        print("Please ensure the model files exist in the specified directory or provide correct paths.")
        env.close()
        return
    except Exception as e:
        print(f"Error loading models: {e}")
        env.close()
        return

    agent.actor.eval()  # Set actor to evaluation mode
    agent.critic.eval()  # Set critic to evaluation mode (though not strictly needed for action selection)

    for episode in range(1, num_episodes + 1):
        print(f"\n--- Starting Episode {episode}/{num_episodes} ---")
        state_dict = env.reset()
        current_state_normalized = normalize_state(state_dict)
        episode_reward = 0
        done = False
        # env.start_game_action is automatically True in reset() when human_mode=False

        running = True
        while running and not done:
            # Handle Pygame events (e.g., closing the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Quit event detected. Closing game.")
                    running = False  # Exit inner loop
                    done = True  # Signal to stop episode
                    # To stop all episodes, you might need another flag or exit strategy
                    break
            if not running:
                break

            # Get action from the agent (deterministic)
            action, _, _ = agent.get_action(current_state_normalized, evaluate=True)

            # Perform action in the environment
            next_state_dict, reward, game_done = env.step(action)
            done = game_done  # Update done status for the episode loop

            current_state_normalized = normalize_state(next_state_dict)
            episode_reward += reward

            # Render the game
            env._render()  # Manually call render as human_mode is False

            # Control game speed / FPS
            env.clock.tick(FPS)
            # time.sleep(0.01) # Optional small delay to make it more watchable

        if not running:  # If quit event happened
            break  # Exit episodes loop

        print(f"Episode {episode} finished. Score: {int(episode_reward)}")
        if env.get_score() != int(episode_reward):  # Just a sanity check if reward structure matches score
            print(f"  (Game internal score: {env.get_score()})")

    print("\n--- Finished Playing ---")
    env.close()  # Properly close Pygame window and resources


if __name__ == "__main__":
    actor_path = os.path.join(MODEL_DIR, ACTOR_MODEL_FILE)
    critic_path = os.path.join(MODEL_DIR, CRITIC_MODEL_FILE)

    play_game(actor_path, critic_path, NUM_EPISODES_TO_PLAY)
