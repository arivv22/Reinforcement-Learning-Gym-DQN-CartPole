import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import trange


# ================================
# 🔹 DQN Network Definition
# ================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# 🔹 Replay Buffer
# ================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)


# ================================
# 🔹 Training Function
# ================================
def train_dqn(env, episodes=300, gamma=0.99, batch_size=64, lr=1e-3):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(10000)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    all_rewards = []

    print("🚀 Starting DQN training on", env.spec.id)
    for ep in trange(episodes, desc="Training"):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32))
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = model(next_states).max(1)[0]
                    target = rewards + gamma * max_next_q * (1 - dones)

                loss = loss_fn(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            print(f"Episode {ep + 1}: avg reward (last 50 eps) = {avg_reward:.2f}")

    print("✅ Training completed.")
    return model, all_rewards


# ================================
# 🔹 Evaluation Function
# ================================
def evaluate_agent(env, model, episodes=5, render=False):
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            if render:
                env.render()
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            done = done or truncated
        total_rewards.append(total_reward)
        print(f"🎮 Episode {ep + 1} reward: {total_reward}")
    env.close()
    print(f"Average test reward: {np.mean(total_rewards):.2f}")


# ================================
# 🔹 Main Execution
# ================================
if __name__ == "__main__":
    # Setup
    os.makedirs("assets", exist_ok=True)
    env = gym.make("CartPole-v1")

    # Train model
    model, rewards = train_dqn(env, episodes=300)

    # Save reward plot
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Progress on CartPole-v1")
    plt.grid()
    plt.savefig("assets/training_plot.png")
    plt.show()

    # Evaluate trained agent
    evaluate_agent(env, model, episodes=5, render=False)

    # Save model
    torch.save(model.state_dict(), "assets/dqn_cartpole.pth")
    print("💾 Model saved to assets/dqn_cartpole.pth")
