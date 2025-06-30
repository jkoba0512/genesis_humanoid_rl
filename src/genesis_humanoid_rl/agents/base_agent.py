"""
Base agent interface for humanoid RL agents.
Provides common functionality for Acme-based agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import acme
from acme import specs


class BaseHumanoidAgent(ABC):
    """
    Base class for humanoid RL agents using Acme framework.
    """

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        agent_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.environment_spec = environment_spec
        self.agent_config = agent_config or {}
        self._agent = None

    @abstractmethod
    def build_agent(self) -> acme.Actor:
        """Build and return the Acme agent."""
        pass

    @abstractmethod
    def get_actor(self) -> acme.Actor:
        """Get the actor for inference."""
        pass

    @abstractmethod
    def get_learner(self) -> acme.Learner:
        """Get the learner for training."""
        pass

    def train(self, num_steps: int) -> Dict[str, Any]:
        """Train the agent for specified number of steps."""
        if self._agent is None:
            self._agent = self.build_agent()

        # TODO: Implement training loop
        # This will be specific to each agent type
        return {}

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent performance."""
        if self._agent is None:
            raise ValueError("Agent must be built before evaluation")

        # TODO: Implement evaluation logic
        return {"average_return": 0.0, "success_rate": 0.0}

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save agent checkpoint."""
        # TODO: Implement checkpoint saving
        pass

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load agent checkpoint."""
        # TODO: Implement checkpoint loading
        pass
