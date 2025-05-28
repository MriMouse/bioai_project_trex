import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from collections import deque


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # Outputs logits/scores for each discrete action
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):  # action_dim is for one-hot encoded action
        super(Critic, self).__init__()
        # Q(s,a)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),  # Concatenate state and action
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action_one_hot):
        x = torch.cat([state, action_one_hot], dim=1)
        return self.net(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add_memo(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)  # Actions are integers
        rewards_np = np.array(rewards, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)

        # Expand dims for rewards and dones to be [B, 1]
        rewards_np = np.expand_dims(rewards_np, axis=1)
        dones_np = np.expand_dims(dones_np, axis=1)
        # Actions will be [B], needs to be [B,1] for unsqueezing later or direct use if expected.
        # Let's make actions_np [B,1]
        actions_np = np.expand_dims(actions_np, axis=1)

        states_tensor = torch.from_numpy(states_np).to(device)
        actions_tensor = torch.from_numpy(actions_np).long().to(device)  # [B, 1]
        rewards_tensor = torch.from_numpy(rewards_np).to(device)
        next_states_tensor = torch.from_numpy(next_states_np).to(device)
        dones_tensor = torch.from_numpy(dones_np).to(device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        return len(self.memory)


class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        lr_actor,
        lr_critic,
        gamma,
        tau,  # For soft target updates
        buffer_size,
        device,
        total_training_steps,  # For LR decay
        lr_decay=True,
        final_lr_factor=0.01,
        max_grad_norm=0.5,
        gumbel_softmax_tau=1.0,  # Temperature for Gumbel-Softmax
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.max_grad_norm = max_grad_norm
        self.gumbel_softmax_tau = gumbel_softmax_tau

        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic
        self.total_training_steps = total_training_steps
        self.lr_decay = lr_decay
        self.final_lr_factor = final_lr_factor
        self.min_lr = 1e-6  # Minimum learning rate

        # Actor Network
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        # Critic Network
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)  # action_dim for one_hot
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.initial_lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.initial_lr_critic)

        self.replay_buffer = ReplayMemory(buffer_size)
        self.mse_loss = nn.MSELoss()

    def _update_learning_rates(self, current_timestep):
        frac = min(1.0, current_timestep / self.total_training_steps)

        current_lr_actor = self.initial_lr_actor
        current_lr_critic = self.initial_lr_critic
        if self.lr_decay:
            final_lr_actor = self.initial_lr_actor * self.final_lr_factor
            final_lr_critic = self.initial_lr_critic * self.final_lr_factor
            current_lr_actor = self.initial_lr_actor + (final_lr_actor - self.initial_lr_actor) * frac
            current_lr_critic = self.initial_lr_critic + (final_lr_critic - self.initial_lr_critic) * frac
            current_lr_actor = max(current_lr_actor, self.min_lr)
            current_lr_critic = max(current_lr_critic, self.min_lr)

            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = current_lr_actor
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = current_lr_critic
        return current_lr_actor, current_lr_critic

    def get_action(self, state, epsilon=0.0, evaluate=False):
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_scores = self.actor(state_tensor)  # [1, action_dim]

        if evaluate:  # Deterministic action for evaluation
            action = torch.argmax(action_scores, dim=1).item()
        else:  # Epsilon-greedy for exploration during training
            if random.random() < epsilon:
                action = random.randrange(self.action_dim)
            else:
                action = torch.argmax(action_scores, dim=1).item()
        return action  # Returns an integer action

    def _update_targets(self, soft=True):
        if soft:
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        else:  # Hard update (not typical for DDPG)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

    def update(self, batch_size, current_total_timestep_counter):
        if len(self.replay_buffer) < batch_size:
            return None, None  # Not enough samples to train

        lr_a, lr_c = self._update_learning_rates(current_total_timestep_counter)

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, self.device)
        # actions is [B, 1], containing integer actions. Convert to one-hot for critic.
        actions_one_hot = F.one_hot(actions.squeeze(1), num_classes=self.action_dim).float()

        # --- Critic Update ---
        with torch.no_grad():
            next_action_scores = self.actor_target(next_states)
            # For discrete actions, target actor selects action deterministically (argmax)
            next_actions_indices = torch.argmax(next_action_scores, dim=1, keepdim=True)
            next_actions_one_hot = F.one_hot(next_actions_indices.squeeze(1), num_classes=self.action_dim).float()

            target_q_values = self.critic_target(next_states, next_actions_one_hot)
            y_i = rewards + self.gamma * target_q_values * (1.0 - dones)

        current_q_values = self.critic(states, actions_one_hot)
        critic_loss = self.mse_loss(current_q_values, y_i)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Actor aims to output actions that maximize Q value from critic
        # Use Gumbel-Softmax for differentiable sampling of discrete actions
        actor_action_scores = self.actor(states)  # Logits
        # `hard=True` makes gumbel_softmax output one-hot vectors
        # `tau` here is the Gumbel-Softmax temperature, not polyak averaging tau
        actor_actions_gs_one_hot = F.gumbel_softmax(actor_action_scores, tau=self.gumbel_softmax_tau, hard=True, dim=1)

        actor_loss = -self.critic(states, actor_actions_gs_one_hot).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # --- Soft update target networks ---
        self._update_targets()

        return lr_a, lr_c  # Return current learning rates for logging

    def save_model(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        # To save optimizers for resuming training:
        # torch.save(self.actor_optimizer.state_dict(), actor_path + "_optimizer")
        # torch.save(self.critic_optimizer.state_dict(), critic_path + "_optimizer")

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

        # Sync target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.eval()  # Set to eval mode if only for inference
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
        # To load optimizers:
        # self.actor_optimizer.load_state_dict(torch.load(actor_path + "_optimizer", map_location=self.device))
        # self.critic_optimizer.load_state_dict(torch.load(critic_path + "_optimizer", map_location=self.device))
