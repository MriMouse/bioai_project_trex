import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from aiGame import TRexGame


class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)


def preprocess_state(state):
    return np.array(
        [
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
        ],
        dtype=np.float32,
    )


def train_ppo(episodes=300):
    state_dim = 13
    action_dim = 3
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    gamma = 0.99
    eps_clip = 0.2
    all_rewards, all_losses = [], []

    for ep in range(episodes):
        game = TRexGame(human_mode=False)
        state = preprocess_state(game.reset())
        done = False
        ep_reward = 0
        log_probs, values, rewards, states, actions = [], [], [], [], []

        while not done:
            s = torch.tensor(state).unsqueeze(0)
            logits, value = agent(s)
            prob = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()
            next_state, reward, done = game.step(action.item())
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            states.append(s)
            actions.append(action)
            state = preprocess_state(next_state)
            ep_reward += reward

        # 计算 advantage 和 returns
        returns, advs = [], []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        values = torch.cat(values).squeeze()
        advs = returns - values.detach()

        # PPO 损失
        log_probs = torch.stack(log_probs)
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advs
        loss = -torch.min(surr1, surr2).mean() + (returns - values).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(ep_reward)
        all_losses.append(loss.item())
        print(f"Episode {ep}, Reward: {ep_reward}, Loss: {loss.item()}")

        # 保存模型
        if (ep + 1) % 50 == 0:
            torch.save(agent.state_dict(), f"ppo_agent_ep{ep+1}.pth")
            np.save("ppo_rewards.npy", np.array(all_rewards))
            np.save("ppo_losses.npy", np.array(all_losses))

    # 绘图
    plt.plot(all_rewards, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Reward")
    plt.legend()
    plt.savefig("ppo_training_reward.png")
    plt.show()

    plt.plot(all_losses, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("PPO Training Loss")
    plt.legend()
    plt.savefig("ppo_training_loss.png")
    plt.show()


if __name__ == "__main__":
    train_ppo(episodes=300)
