# Training Configurations

This directory contains training configuration files for different scenarios.

## Available Configurations

### `default.json`
Production training configuration for 1M timesteps with optimal hyperparameters.

```bash
uv run python scripts/train_sb3.py --config configs/default.json
```

### `test.json`
Quick test configuration for validation (5k timesteps, small networks).

```bash
uv run python scripts/train_sb3.py --config configs/test.json
```

### `high_performance.json`
High-performance training with larger networks and more parallel environments.

```bash
uv run python scripts/train_sb3.py --config configs/high_performance.json
```

## Configuration Structure

```json
{
    "env": {
        "episode_length": 1000,      // Steps per episode
        "simulation_fps": 100,       // Physics simulation rate
        "control_freq": 20,          // Robot control frequency
        "target_velocity": 1.0,      // Target walking speed (m/s)
        "n_envs": 4                  // Parallel environments
    },
    "algorithm": {
        "learning_rate": 3e-4,       // PPO learning rate
        "n_steps": 2048,             // Steps per environment per update
        "batch_size": 64,            // Batch size for training
        "n_epochs": 10,              // Training epochs per update
        "policy_kwargs": {
            "net_arch": [256, 256, 128]  // Neural network architecture
        }
    },
    "training": {
        "total_timesteps": 1000000,  // Total training steps
        "save_freq": 50000,          // Model save frequency
        "eval_freq": 10000,          // Evaluation frequency
        "experiment_name": "exp_name" // Experiment identifier
    }
}
```

## Creating Custom Configurations

1. Copy an existing configuration:
   ```bash
   cp configs/default.json configs/my_experiment.json
   ```

2. Modify parameters as needed

3. Run with custom config:
   ```bash
   uv run python scripts/train_sb3.py --config configs/my_experiment.json
   ```

## Hyperparameter Guidelines

### Environment Parameters
- **episode_length**: 500-2000 (longer = more learning per episode)
- **n_envs**: 1-16 (more = faster training, but needs more GPU memory)
- **target_velocity**: 0.5-2.0 m/s (walking speed target)

### Algorithm Parameters
- **learning_rate**: 1e-5 to 1e-3 (3e-4 is usually good)
- **n_steps**: 1024-4096 (more = more stable but slower updates)
- **net_arch**: [128,128] to [512,512,256] (bigger = more capacity)

### Hardware Optimization
- **RTX 3060 Ti**: n_envs: 4, net_arch: [256,256,128]
- **RTX 4080**: n_envs: 8, net_arch: [512,256,128] 
- **RTX 4090**: n_envs: 16, net_arch: [512,512,256]