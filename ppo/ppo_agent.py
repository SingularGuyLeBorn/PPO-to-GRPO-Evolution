"""
PPO (Proximal Policy Optimization) Implementation
Reference: Schulman et al. 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    """Policy network for PPO"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        return mean, self.log_std

    def get_log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)

    def get_entropy(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.entropy().mean()

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class CriticNetwork(nn.Module):
    """Value network for PPO"""

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)


class PPOAgent:
    """PPO Agent with GAE and clipped objective"""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=10,
        batch_size=64,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        gae = 0
        advantages = []
        returns = []

        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, dones, next_state):
        """Update policy and value networks"""
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        dones = torch.FloatTensor(np.array(dones))

        # Compute advantages and returns
        with torch.no_grad():
            values = [self.critic(s.unsqueeze(0)).item() for s in states]
            next_value = self.critic(
                torch.FloatTensor(next_state).unsqueeze(0)
            ).item()

        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = torch.FloatTensor(np.array(advantages))
        returns = torch.FloatTensor(np.array(returns))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of update
        dataset_size = len(states)
        for _ in range(self.epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Current log probs
                log_probs = self.actor.get_log_prob(batch_states, batch_actions)
                entropy = self.actor.get_entropy(batch_states)

                # Ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = self.critic(batch_states).squeeze()
                value_loss = F.mse_loss(values_pred, batch_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=0.5
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=0.5
                )
                self.optimizer.step()

        return loss.item()
