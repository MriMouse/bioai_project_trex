import torch
from torch import nn
from torch.distributions import Categorical  # Changed from Normal
import numpy as np
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):  # Removed action_scale
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_logits = nn.Linear(hidden_dim, action_dim)  # Outputs logits for discrete actions

    def forward(self, x):
        hidden_out = self.net(x)
        logits = self.fc_logits(hidden_out)
        return logits

    def get_distribution(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits)

    def get_log_prob(self, state, action):
        dist = self.get_distribution(state)
        # Ensure action is correctly shaped (e.g., squeeze if it's [B, 1] for Categorical)
        if action.ndim > 1 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        return dist.log_prob(action).unsqueeze(-1)  # Return with shape [B, 1]


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    def __init__(self):
        self.clear()

    def add_memo(self, state, action, reward, value, done):  # action is an integer
        self.states.append(state)
        self.actions.append(action)  # Store integer action
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get_tensors(self, device):
        states_np = np.array(self.states, dtype=np.float32)
        # Actions are discrete integers, ensure they are long type for Categorical.log_prob
        actions_np = np.array(self.actions, dtype=np.int64)
        rewards_np = np.array(self.rewards, dtype=np.float32)
        values_np = np.array(self.values, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)

        rewards_np = np.expand_dims(rewards_np, axis=1)
        values_np = np.expand_dims(values_np, axis=1)
        dones_np = np.expand_dims(dones_np, axis=1)

        states_tensor = torch.from_numpy(states_np).to(device)
        # Actions need to be unsqueezed to [B, 1] if PPO structure expects it,
        # or kept as [B] if log_prob handles it.
        # Actor.get_log_prob expects action as [B] or [B,1] and returns [B,1].
        # Let's make action_tensor [B, 1] of type long.
        actions_tensor = torch.from_numpy(actions_np).long().unsqueeze(-1).to(device)
        rewards_tensor = torch.from_numpy(rewards_np).to(device)
        values_tensor = torch.from_numpy(values_np).to(device)
        dones_tensor = torch.from_numpy(dones_np).to(device)
        return states_tensor, actions_tensor, rewards_tensor, values_tensor, dones_tensor

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []

    def size(self):
        return len(self.states)


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        lr_actor,
        lr_critic,
        gamma,
        gae_lambda,
        ppo_epsilon,
        ppo_epochs,
        minibatch_size,
        device,
        total_training_steps,
        lr_decay=True,
        final_lr_factor=0.0,
        initial_entropy_coef=0.01,
        final_entropy_coef=0.0,
        entropy_decay=True,
        max_grad_norm=0.5,
    ):  # Removed action_scale from agent's own params if not used elsewhere

        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = ppo_epsilon
        self.epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        # self.action_scale = action_scale # Not used for discrete action actor

        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic
        self.total_training_steps = total_training_steps
        self.lr_decay = lr_decay
        self.final_lr_factor = final_lr_factor
        self.min_lr = 1e-6

        self.initial_entropy_coef = initial_entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.entropy_decay = entropy_decay
        self.final_entropy_coef = max(0.0, self.final_entropy_coef)

        # Actor Network (discrete actions, no action_scale)
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.old_actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_actor.eval()

        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.initial_lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.initial_lr_critic)

        self.replay_buffer = ReplayMemory()
        self.mse_loss = nn.MSELoss()

    def save_model(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        # Consider saving optimizer states as well for robust checkpointing
        # torch.save(self.actor_optimizer.state_dict(), actor_path + "_optimizer")
        # torch.save(self.critic_optimizer.state_dict(), critic_path + "_optimizer")

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.old_actor.load_state_dict(self.actor.state_dict())  # Sync old_actor
        self.old_actor.eval()
        # Consider loading optimizer states as well
        # self.actor_optimizer.load_state_dict(torch.load(actor_path + "_optimizer", map_location=self.device))
        # self.critic_optimizer.load_state_dict(torch.load(critic_path + "_optimizer", map_location=self.device))

    def get_action(self, state, evaluate=False):
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.actor.get_distribution(state_tensor)  # Categorical distribution
            if evaluate:
                action = torch.argmax(dist.logits, dim=-1, keepdim=True)  # Choose action with max prob
            else:
                action = dist.sample().unsqueeze(-1)  # Sample action, ensure shape [1,1]

            # log_prob of the CHOSEN action.
            # dist.log_prob expects action to be of shape [B] if dist is batch [B,num_actions]
            # action here is [1,1], squeeze to [1] for log_prob, then unsqueeze to [1,1] for consistency
            log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
            value = self.critic(state_tensor)
        # action is a tensor like tensor([[int_val]]), .item() extracts the int_val
        return action.cpu().item(), value.cpu().item(), log_prob.cpu().item()

    def _calculate_gae(self, rewards, values, dones, last_value):
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards).to(self.device)
        # returns = torch.zeros_like(rewards).to(self.device) # GAE returns are advantages + values
        last_gae_lam = 0
        # Ensure last_value is correctly shaped for concatenation or direct use
        if not isinstance(last_value, torch.Tensor):
            last_value_tensor = torch.tensor([[last_value]], dtype=torch.float32).to(self.device)
        else:
            last_value_tensor = last_value.to(self.device)
        if last_value_tensor.ndim == 0:  # if scalar
            last_value_tensor = last_value_tensor.unsqueeze(0).unsqueeze(0)
        elif last_value_tensor.ndim == 1:  # if [1]
            last_value_tensor = last_value_tensor.unsqueeze(0)

        full_values = torch.cat((values, last_value_tensor), dim=0)

        for t in reversed(range(num_steps)):
            delta = rewards[t] + self.gamma * full_values[t + 1] * (1.0 - dones[t]) - full_values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae_lam
        returns = advantages + values  # Calculate returns for critic loss
        return advantages, returns

    def update(self, last_value, current_timestep):
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

        current_entropy_coef = self.initial_entropy_coef
        if self.entropy_decay:
            current_entropy_coef = (
                self.initial_entropy_coef + (self.final_entropy_coef - self.initial_entropy_coef) * frac
            )
            current_entropy_coef = max(self.final_entropy_coef, current_entropy_coef)

        states, actions, rewards, values, dones = self.replay_buffer.get_tensors(self.device)
        advantages, returns = self._calculate_gae(rewards, values, dones, last_value)
        # raw_advantages = advantages.clone() # For debugging
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.old_actor.load_state_dict(self.actor.state_dict())
        num_samples = len(states)
        indices = np.arange(num_samples)

        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]
                if len(batch_indices) == 0:
                    continue

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]  # Shape [B, 1], dtype long
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                new_dist = self.actor.get_distribution(batch_states)  # Categorical
                # get_log_prob expects actions [B] or [B,1] and returns [B,1]
                batch_log_probs = self.actor.get_log_prob(batch_states, batch_actions)
                entropy = new_dist.entropy().mean()  # new_dist.entropy() is [B], .mean() is scalar

                with torch.no_grad():
                    # old_dist = self.old_actor.get_distribution(batch_states)
                    batch_old_log_probs = self.old_actor.get_log_prob(batch_states, batch_actions)

                ratio = torch.exp(batch_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - current_entropy_coef * entropy

                current_values = self.critic(batch_states)
                critic_loss = self.mse_loss(current_values, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        self.replay_buffer.clear()
        return current_lr_actor, current_lr_critic, current_entropy_coef

    def save_model(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        # Consider saving optimizer states as well for robust checkpointing
        # torch.save(self.actor_optimizer.state_dict(), actor_path + "_optimizer")
        # torch.save(self.critic_optimizer.state_dict(), critic_path + "_optimizer")

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.old_actor.load_state_dict(self.actor.state_dict())  # Sync old_actor
        self.old_actor.eval()
        # Consider loading optimizer states as well
        # self.actor_optimizer.load_state_dict(torch.load(actor_path + "_optimizer", map_location=self.device))
        # self.critic_optimizer.load_state_dict(torch.load(critic_path + "_optimizer", map_location=self.device))
