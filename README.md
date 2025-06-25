# Genesis Humanoid RL 🤖🚶‍♂️

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Genesis](https://img.shields.io/badge/Genesis-v0.2.1-green.svg)](https://genesis-world.readthedocs.io)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-v2.0+-orange.svg)](https://stable-baselines3.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Train humanoid robots to walk using reinforcement learning with Genesis physics engine and Stable-Baselines3.**

This project implements a complete pipeline for training the Unitree G1 humanoid robot to walk using PPO (Proximal Policy Optimization) in high-fidelity physics simulation.

## 🚀 Features

- **🤖 Unitree G1 Integration**: Full 35-DOF humanoid robot simulation
- **🏃‍♂️ Walking Behaviors**: Train natural, stable walking gaits
- **⚡ High Performance**: 100+ FPS physics simulation with GPU acceleration
- **🔄 Parallel Training**: Multiple environments for faster learning
- **📊 Real-time Monitoring**: TensorBoard integration with detailed metrics
- **🎯 Production Ready**: Stable-Baselines3 for robust, proven algorithms
- **🛡️ Automatic Safety**: Robot grounding system prevents penetration issues

## 📦 Installation

### Prerequisites
- Python 3.10
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for parallel training

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/genesis_humanoid_rl.git
cd genesis_humanoid_rl

# Automated setup (recommended)
uv run python tools/setup_environment.py

# Or manual setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Verify Installation
```bash
# Comprehensive verification
uv run python tools/verify_installation.py

# Quick verification
uv run python -c "
from stable_baselines3 import PPO
import genesis as gs
print('✓ Installation successful!')
"
```

## 🏃‍♂️ Quick Start

### 1. Test Training (5 minutes)
```bash
# Quick test with 5k steps
uv run python scripts/train_sb3.py --config configs/test.json
```

### 2. Full Training (2-8 hours)
```bash
# Train for 1M steps to achieve walking
uv run python scripts/train_sb3.py --config configs/default.json
```

### 3. Monitor Progress
```bash
# Open TensorBoard
tensorboard --logdir ./logs/sb3
# Navigate to http://localhost:6006
```

### 4. Evaluate Trained Model
```bash
# Watch your robot walk!
uv run python scripts/evaluate_sb3.py ./models/sb3/best_model --render --episodes 5
```

## 📁 Project Structure

```
genesis_humanoid_rl/
├── 📁 scripts/                   # Training and evaluation scripts
│   ├── train_sb3.py             # Main training script
│   ├── evaluate_sb3.py          # Model evaluation
│   └── create_demo_video.py     # Demo video creation
├── 📁 src/genesis_humanoid_rl/   # Core source code
│   ├── 📁 environments/         # RL environments
│   ├── 📁 rewards/              # Reward functions
│   ├── 📁 config/               # Configuration management
│   └── 📁 utils/                # Utility functions
├── 📁 configs/                   # Training configurations
│   ├── default.json             # Production config
│   ├── test.json                # Quick test config
│   └── high_performance.json    # High-performance config
├── 📁 assets/                    # Robot assets and models
├── 📁 tools/                     # Setup and verification tools
├── 📁 docs/                      # Documentation
└── 📁 robot_grounding/           # Automatic positioning system
```

## 🎯 Training Pipeline

### Learning Progression
Our training follows a natural learning progression:

1. **🍼 Baby Steps (0-50k steps)**
   - Learning basic balance and coordination
   - Reducing falls and unstable behavior
   - Average reward: -10 to 0

2. **🚶‍♂️ First Steps (50k-200k steps)**
   - Discovering forward motion
   - Developing primitive walking patterns
   - Average reward: 0 to 50

3. **🏃‍♂️ Walking (200k-500k steps)**
   - Consistent forward locomotion
   - Improved stability and efficiency
   - Average reward: 50 to 100

4. **🏆 Mastery (500k+ steps)**
   - Smooth, natural walking gaits
   - Robust to disturbances
   - Average reward: 100+

### Environment Details
- **Robot**: Unitree G1 (35 DOF)
- **Observation Space**: 113-dimensional state vector
- **Action Space**: 35-dimensional continuous joint control
- **Physics**: Genesis engine at 100 FPS
- **Control Frequency**: 20 Hz
- **Episode Length**: 1000 steps (50 seconds)

## ⚙️ Configuration

Training configurations are in the `configs/` directory:

- **`configs/test.json`**: Quick test (5k steps, small networks)
- **`configs/default.json`**: Production training (1M steps)
- **`configs/high_performance.json`**: Advanced training (2M steps, large networks)

### Example Configuration
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
        "policy_kwargs": {
            "net_arch": [256, 256, 128]
        }
    },
    "training": {
        "total_timesteps": 1000000,
        "save_freq": 50000
    }
}
```

## 📊 Performance Optimization

### Hardware Recommendations
- **RTX 3060 Ti**: `n_envs: 4` (recommended)
- **RTX 4080**: `n_envs: 8-12`
- **RTX 4090**: `n_envs: 16+`

### Training Time
- **Test Config**: ~5 minutes (5k steps)
- **Default Config**: ~4-6 hours (1M steps)
- **High Performance**: ~8-12 hours (2M steps)

## 🛠️ Advanced Usage

### Custom Training
```bash
# Create custom configuration
cp configs/default.json configs/my_experiment.json
# Edit my_experiment.json

# Train with custom config
uv run python scripts/train_sb3.py --config configs/my_experiment.json
```

### Hyperparameter Tuning
```bash
# Different network architectures in config
"policy_kwargs": {
    "net_arch": [512, 512, 256],  # Larger network
    "activation_fn": "relu"       # Different activation
}
```

## 📚 Documentation

- **[docs/README.md](docs/README.md)**: Complete documentation index
- **[docs/stable_baselines3_guide.md](docs/stable_baselines3_guide.md)**: Detailed SB3 guide
- **[configs/README.md](configs/README.md)**: Configuration documentation

## 🔧 Tools

- **`tools/setup_environment.py`**: Automated environment setup
- **`tools/verify_installation.py`**: Installation verification
- **`scripts/create_demo_video.py`**: Generate demonstration videos

## 🤝 Contributing

Contributions welcome! Areas of interest:

- **New locomotion behaviors** (running, stairs, rough terrain)
- **Different humanoid robots** (beyond Unitree G1)
- **Advanced training algorithms** (SAC, TD3, etc.)
- **Sim-to-real improvements** (domain adaptation)

### Development Setup
```bash
# Install development dependencies
uv sync --dev

# Run verification
uv run python tools/verify_installation.py

# Format code
uv run black src/ scripts/ tools/
uv run isort src/ scripts/ tools/
```

## 🙏 Acknowledgments

- **[Genesis](https://genesis-world.readthedocs.io)**: High-performance physics simulation
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io)**: Reliable RL algorithms
- **[Unitree Robotics](https://www.unitree.com)**: G1 humanoid robot platform

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/genesis_humanoid_rl/issues)
- **Documentation**: [docs/README.md](docs/README.md)
- **Verification**: `uv run python tools/verify_installation.py`

---

**Start training your walking humanoid today!** 🚀🤖

```bash
git clone https://github.com/yourusername/genesis_humanoid_rl.git
cd genesis_humanoid_rl
uv run python tools/setup_environment.py
uv run python scripts/train_sb3.py --config configs/test.json
```