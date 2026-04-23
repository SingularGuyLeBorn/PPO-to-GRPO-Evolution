"""
PPO Example: Training on a simple continuous control task
"""
import torch
import numpy as np
from ppo_agent import PPOAgent


def create_env():
    """Simple 2D environment for demonstration"""
    class SimpleEnv:
        def __init__(self):
            self.state = np.zeros(4)
            self.target = np.ones(4) * 2.0

        def reset(self):
            self.state = np.random.randn(4) * 0.1
            return self.state.copy()

        def step(self, action):
            self.state = self.state + action * 0.1
            dist = np.linalg.norm(self.state - self.target)
            reward = -dist  # Negative distance as reward
            done = dist < 0.1
            return self.state.copy(), reward, done, {}

    return SimpleEnv()


def main():
    env = create_env()
    state_dim = 4
    action_dim = 4

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
    )

    num_episodes = 100
    max_steps = 200

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        states, actions, log_probs, rewards, dones = [], [], [], [], []

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = agent.actor.sample(state_tensor)

            next_state, reward, done, _ = env.step(
                action.detach().numpy().squeeze()
            )

            states.append(state)
            actions.append(action.detach().numpy().squeeze())
            log_probs.append(log_prob.detach().numpy())
            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Update after each episode
        loss = agent.update(
            states, actions, log_probs, rewards, dones, next_state
        )

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Loss: {loss:.4f}")

    print("PPO training completed!")


if __name__ == "__main__":
    main()
