"""
Training configuration for humanoid RL experiments.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class EnvironmentConfig:
    """Configuration for the humanoid environment."""

    render_mode: Optional[str] = None
    simulation_fps: int = 100
    control_freq: int = 20
    episode_length: int = 1000
    target_velocity: float = 1.0

    # Reward weights
    velocity_reward_weight: float = 1.0
    stability_reward_weight: float = 0.5
    energy_penalty_weight: float = 0.01
    action_smoothness_weight: float = 0.1

    # Termination conditions
    max_height_deviation: float = 0.3
    max_orientation_deviation: float = 0.5


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""

    learning_rate: float = 3e-4
    entropy_cost: float = 0.01
    value_cost: float = 0.5
    max_gradient_norm: float = 0.5
    num_epochs: int = 10
    num_minibatches: int = 32
    unroll_length: int = 16
    batch_size: int = 256

    # Network architecture
    policy_layers: list = None
    value_layers: list = None
    activation: str = "tanh"

    def __post_init__(self):
        if self.policy_layers is None:
            self.policy_layers = [256, 128, 64]
        if self.value_layers is None:
            self.value_layers = [256, 128, 64]


@dataclass
class TrainingConfig:
    """Main training configuration."""

    # Training parameters
    total_steps: int = 1_000_000
    eval_frequency: int = 10_000
    save_frequency: int = 50_000
    log_frequency: int = 1000

    # Environment settings
    num_parallel_envs: int = 16
    max_episode_steps: int = 1000

    # Experiment settings
    experiment_name: str = "humanoid_walking"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    # Random seed
    seed: int = 42

    # GPU settings
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for training."""
    return {
        "environment": EnvironmentConfig(),
        "agent": PPOConfig(),
        "training": TrainingConfig(),
    }


def load_config_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration from dictionary."""
    config = get_default_config()

    # Update environment config
    if "environment" in config_dict:
        env_config = config_dict["environment"]
        for key, value in env_config.items():
            if hasattr(config["environment"], key):
                setattr(config["environment"], key, value)

    # Update agent config
    if "agent" in config_dict:
        agent_config = config_dict["agent"]
        for key, value in agent_config.items():
            if hasattr(config["agent"], key):
                setattr(config["agent"], key, value)

    # Update training config
    if "training" in config_dict:
        training_config = config_dict["training"]
        for key, value in training_config.items():
            if hasattr(config["training"], key):
                setattr(config["training"], key, value)

    return config
