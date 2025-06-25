# Genesis Humanoid RL with Stable-Baselines3

This project implements humanoid robot reinforcement learning using the Genesis physics engine and Stable-Baselines3 for robust, easy-to-use PPO training.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Test the Setup
```bash
uv run python quick_sb3_test.py
```

### 3. Start Training
```bash
# Quick test training (5k steps)
uv run python scripts/train_sb3.py --config config_sb3_test.json

# Full training (1M steps)
uv run python scripts/train_sb3.py
```

### 4. Monitor Training
```bash
tensorboard --logdir ./logs/test_sb3
```

### 5. Evaluate Trained Model
```bash
uv run python scripts/evaluate_sb3.py ./models/test_sb3/final_model --episodes 10 --render
```

## 📁 Project Structure

```
genesis_humanoid_rl/
├── scripts/
│   ├── train_sb3.py          # Main SB3 training script
│   └── evaluate_sb3.py       # Model evaluation script
├── src/genesis_humanoid_rl/
│   └── environments/
│       ├── humanoid_env.py   # Core environment
│       └── sb3_wrapper.py    # SB3-compatible wrapper
├── config_sb3_test.json     # Test configuration
└── quick_sb3_test.py         # Setup verification
```

## 🎯 Features

### ✅ Complete Implementation
- **Genesis Physics**: High-fidelity 100+ FPS simulation
- **Unitree G1 Robot**: 35 DOF humanoid with automatic grounding
- **Stable-Baselines3**: Production-ready PPO implementation
- **Parallel Training**: Multi-environment support for faster learning
- **TensorBoard Logging**: Real-time training monitoring
- **Model Management**: Automatic saving, loading, and evaluation

### 🤖 Environment Details
- **Observation Space**: 113-dimensional robot state vector
- **Action Space**: 35-dimensional continuous joint control
- **Reward Function**: Walking performance with stability and energy efficiency
- **Episode Length**: Configurable (default: 1000 steps)
- **Physics Rate**: 100 FPS simulation, 20 Hz control

### 🧠 Algorithm Configuration
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network**: Multi-layer perceptron with customizable architecture
- **Default Settings**: Optimized for continuous control humanoid tasks
- **Parallel Environments**: 4 environments by default for efficient training

## 📊 Training Configuration

### Default Configuration
```json
{
    "env": {
        "episode_length": 1000,
        "n_envs": 4,
        "target_velocity": 1.0
    },
    "algorithm": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "policy_kwargs": {
            "net_arch": [256, 256, 128]
        }
    },
    "training": {
        "total_timesteps": 1000000,
        "save_freq": 50000,
        "eval_freq": 10000
    }
}
```

### Custom Configuration
Create a JSON file with your desired settings and pass it with `--config`:

```bash
uv run python scripts/train_sb3.py --config my_config.json
```

## 🎮 Usage Examples

### Basic Training
```bash
# Start training with default settings
uv run python scripts/train_sb3.py
```

### Custom Training
```bash
# Train with custom configuration
uv run python scripts/train_sb3.py --config my_config.json

# Train with rendering (slower)
uv run python scripts/train_sb3.py --render
```

### Model Evaluation
```bash
# Evaluate best model
uv run python scripts/evaluate_sb3.py ./models/sb3/best_model

# Evaluate with rendering
uv run python scripts/evaluate_sb3.py ./models/sb3/final_model --render --episodes 5

# Compare multiple models
uv run python scripts/evaluate_sb3.py --compare ./models/sb3/best_model ./models/sb3/final_model
```

## 📈 Expected Performance

### Training Progress
- **Initial Performance**: Random actions (~-10 to 0 reward)
- **Learning Phase**: Gradual improvement over 100k-500k steps
- **Convergence**: Stable walking behavior around 1M steps
- **Target Performance**: >100 reward per episode for good walking

### Training Time
- **Test Config**: ~5 minutes (5k steps)
- **Full Training**: ~2-8 hours (1M steps, depending on hardware)
- **Hardware**: Optimized for GPU acceleration (CUDA)

## 🔧 Troubleshooting

### Common Issues
1. **Genesis Compilation**: First run takes 5-10 minutes to compile kernels
2. **GPU Memory**: Reduce `n_envs` if running out of GPU memory
3. **Slow Training**: Ensure GPU acceleration is working (`device="auto"`)

### Performance Tips
1. **Use GPU**: Ensure PyTorch CUDA is installed for GPU acceleration
2. **Parallel Environments**: Increase `n_envs` for faster data collection
3. **Hyperparameters**: Adjust `n_steps` and `batch_size` based on hardware

## 🏆 Advantages over Acme

- ✅ **No Dependency Issues**: No dm-launchpad problems
- ✅ **Simple Setup**: Single pip install, works immediately
- ✅ **Better Documentation**: Extensive SB3 community and docs
- ✅ **Proven Algorithms**: Battle-tested PPO implementation
- ✅ **Easy Customization**: Simple configuration and callback system
- ✅ **Model Management**: Built-in save/load/evaluation tools

## 🎯 Next Steps

1. **Run Training**: Start with test config to verify everything works
2. **Monitor Progress**: Use TensorBoard to track learning curves
3. **Tune Hyperparameters**: Adjust learning rate, network size, etc.
4. **Scale Up**: Increase `n_envs` and `total_timesteps` for better performance
5. **Experiment**: Try different reward functions or network architectures

Start training your humanoid walking policy now! 🚶‍♂️🤖