# 🐍 Snake AI Learning Simulator

A modular, educational reinforcement learning project that demonstrates how different AI algorithms learn to play the classic Snake game.

## 🎯 Overview

This project implements multiple reinforcement learning algorithms that learn to play Snake:
- **Q-Learning** (Tabular, Off-Policy)
- **SARSA** (Tabular, On-Policy)  
- **DQN** (Deep Q-Network with Experience Replay)

Each algorithm can be easily swapped, and performance can be compared across different difficulty levels.

## 📁 Project Structure

```
snake_ai/
├── main.py                # Entry point
├── environment.py         # Snake game environment
├── visualize.py          # Visualization and plotting
├── config.py             # All configuration parameters
├── compare.py            # Compare all algorithms
│
├── agents/
│   ├── agent_qlearning.py  # Q-Learning implementation
│   ├── agent_sarsa.py      # SARSA implementation
│   └── agent_dqn.py        # Deep Q-Network implementation
│
└── utils/
    ├── replay_memory.py   # Experience replay buffer for DQN
    └── model.py          # Neural network architecture for DQN
```

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install numpy pygame matplotlib

# For DQN (optional)
pip install torch torchvision
```

### Running the Project

1. **Configure** - Edit `config.py`:
   ```python
   ALGORITHM = "Q_LEARNING"  # or "SARSA", "DQN"
   DIFFICULTY = "MEDIUM"      # or "EASY", "HARD"
   MODE = "TRAIN"            # or "PLAY"
   EPISODES = 500
   ```

2. **Train** an agent:
   ```bash
   python main.py
   ```

3. **Watch** the trained agent play:
   ```python
   # In config.py, change:
   MODE = "PLAY"
   ```
   ```bash
   python main.py
   ```

4. **Compare** all algorithms:
   ```bash
   python compare.py
   ```

## ⚙️ Configuration Options

### Algorithms
- `Q_LEARNING` - Classic tabular Q-learning (off-policy)
- `SARSA` - On-policy temporal difference learning
- `DQN` - Deep Q-Network with neural network approximation

### Difficulty Levels
- `EASY` - 6×6 grid, slower speed
- `MEDIUM` - 10×10 grid, moderate speed
- `HARD` - 14×14 grid, faster speed

### Key Parameters (in `config.py`)
```python
LEARNING_RATE = 0.1          # Alpha
DISCOUNT_FACTOR = 0.9        # Gamma
EPSILON_START = 1.0          # Initial exploration rate
EPSILON_DECAY = 0.995        # Epsilon decay rate
EPISODES = 500               # Number of training episodes
SHOW_TRAINING = True         # Visualize during training
```

## 🎮 How It Works

### State Representation
The agent observes:
- Danger in 3 directions (straight, left, right)
- Current movement direction (4 one-hot values)
- Food location relative to head (4 binary flags)

Total: **11-dimensional state vector**

### Actions
- `0` - Continue straight
- `1` - Turn left
- `2` - Turn right

### Rewards
- `+10` - Eating food
- `-10` - Collision (wall or self)
- `-0.1` - Each step (encourages efficiency)

## 📊 Training Progress

The system automatically generates:
- Real-time game visualization (if enabled)
- Training curves (scores and steps over episodes)
- Moving average plots
- Algorithm comparison charts

## 🧠 Algorithm Differences

### Q-Learning (Off-Policy)
- Updates Q-values using the maximum Q-value of the next state
- Can learn optimal policy even when following exploratory policy
- Generally converges faster in tabular settings

### SARSA (On-Policy)
- Updates Q-values using the actual action taken in the next state
- More conservative, considers exploration in learning
- Better for environments with dangerous exploration

### DQN (Deep Learning)
- Uses neural network to approximate Q-values
- Handles larger state spaces
- Uses experience replay for stability
- Requires more episodes to converge

## 🎯 Performance Tips

1. **Start with Q-Learning on EASY** difficulty to see quick results
2. **Increase EPISODES** for better performance (1000+ recommended)
3. **Adjust EPSILON_DECAY** if agent explores too much/little
4. **For DQN**, increase `MEMORY_SIZE` and `BATCH_SIZE` for better stability
5. **Disable SHOW_TRAINING** for faster training

## 📈 Expected Results

After ~500 episodes on MEDIUM difficulty:
- **Q-Learning**: Average score 3-7
- **SARSA**: Average score 2-6  
- **DQN**: Average score 4-8 (needs more episodes)

Results improve significantly with more training!

## 🔧 Extending the Project

### Add a New Algorithm
1. Create `agents/agent_yourname.py`
2. Implement the same interface as `BaseAgent`
3. Add to `config.py` and `main.py`

### Modify Rewards
Edit `REWARD_*` values in `config.py` to experiment with different reward structures.

### Change State Representation
Modify `get_state()` in `environment.py` to include additional information.

## 🐛 Troubleshooting

**Agent performs poorly:**
- Increase training episodes
- Adjust learning rate
- Check epsilon decay rate
- Try different algorithm

**Training too slow:**
- Set `SHOW_TRAINING = False`
- Reduce `EPISODES` for initial testing
- Use smaller grid size

**DQN not working:**
- Ensure PyTorch is installed
- Increase `MEMORY_SIZE` and `BATCH_SIZE`
- Train for more episodes (1000+)

## 📚 Learning Resources

- [Sutton & Barto - Reinforcement Learning Book](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep Q-Learning Paper](https://www.nature.com/articles/nature14236)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

## 🤝 Contributing

Feel free to:
- Add new algorithms
- Improve visualization
- Optimize performance
- Add new features

## 📝 License

This is an educational project - free to use and modify!

---

**Happy Learning! 🐍🤖**
