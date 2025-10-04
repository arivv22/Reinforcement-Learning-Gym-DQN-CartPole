# 🧠 Reinforcement Learning – CartPole DQN Agent

This mini project implements a **Deep Q-Network (DQN)** agent to solve the classic control problem **CartPole-v1** from OpenAI Gym.

The goal is to balance a pole on a moving cart using reinforcement learning techniques.  
The agent learns optimal control behavior through trial and error by interacting with the environment.

---

## 🎯 Objectives
- Understand basic Reinforcement Learning concepts.
- Implement a DQN agent using PyTorch.
- Train and evaluate the agent in a Gym environment.
- Visualize the learning curve and performance.

---

## 🧩 Project Structure

```bash
Reinforcement-Learning-Gym-DQN-CartPole/
│
├── main.py # Main training script
├── requirements.txt # Dependencies
├── README.md # Documentation
└── assets/
└── training_plot.png # (optional) Saved reward plot
```
---

## 🧠 Environment Details
- **Environment:** `CartPole-v1`
- **Action Space:** Discrete (2 actions)
- **State Space:** Continuous (4-dimensional)
- **Reward:** +1 per timestep if the pole remains balanced

---

## 🧰 Installation

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/Reinforcement-Learning-Gym-DQN-CartPole.git
cd Reinforcement-Learning-Gym-DQN-CartPole
```

--- 
### 2. Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)
```

--- 
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
---

## 🚀 Run Training

```bash
python main.py
```

During training, the script will display progress for each episode.
After training completes, a reward plot will be generated.

---

## 📊 Example Result
After around 300 episodes, the DQN agent typically achieves:

- Average reward > 150

- Stable balancing behavior

---

## 🎬 Visualize Agent

To watch the trained agent play:

```bash
# inside main.py
env = gym.make("CartPole-v1", render_mode="human")
```

--- 

## 🔬 Research & Learning Outcomes

- Reinforcement Learning fundamentals (Markov Decision Process)

- Q-learning & Deep Q-Network implementation

- Epsilon-Greedy exploration

- Experience Replay Buffer

- Gradient descent optimization in PyTorch

---

## 🧩 Next Steps

You can extend this mini project to:

- Double DQN or Dueling DQN
- Other Gym environments (e.g., LunarLander, MountainCar)
- Integrate visualization with TensorBoard or Weights & Biases (wandb)

---

## 🧑‍💻 Author    

M. Afdhal Arief Malik
Aspiring AI Researcher | Backend Developer 