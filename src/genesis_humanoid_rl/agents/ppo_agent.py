"""
PPO agent implementation for humanoid walking using Acme.
"""

from typing import Any, Dict, Optional, Sequence
import numpy as np
import jax
import jax.numpy as jnp
import acme
from acme import specs
from acme.agents.jax import ppo
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import optax

from .base_agent import BaseHumanoidAgent


class PPOHumanoidAgent(BaseHumanoidAgent):
    """
    PPO agent optimized for humanoid walking tasks.
    Uses continuous action space for joint control.
    """

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        network_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 3e-4,
        entropy_cost: float = 0.01,
        value_cost: float = 0.5,
        max_gradient_norm: float = 0.5,
        num_epochs: int = 10,
        num_minibatches: int = 32,
        unroll_length: int = 16,
        batch_size: int = 256,
        **kwargs,
    ):
        super().__init__(environment_spec, **kwargs)

        self.network_kwargs = network_kwargs or {}
        self.learning_rate = learning_rate
        self.entropy_cost = entropy_cost
        self.value_cost = value_cost
        self.max_gradient_norm = max_gradient_norm
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.unroll_length = unroll_length
        self.batch_size = batch_size

        self._networks = None
        self._agent = None

    def _build_networks(self) -> ppo.PPONetworks:
        """Build neural networks for PPO agent."""

        def policy_network(x):
            """Policy network for continuous control."""
            # Extract default layer sizes
            layer_sizes = self.network_kwargs.get("policy_layers", [256, 128, 64])
            activation = self.network_kwargs.get("activation", jax.nn.tanh)

            # Policy network
            policy_trunk = hk.nets.MLP(
                layer_sizes, activation=activation, name="policy_trunk"
            )

            # Get trunk output
            trunk_output = policy_trunk(x)

            # Action distribution parameters
            action_dim = self.environment_spec.actions.shape[0]

            # Mean of action distribution
            mean = hk.Linear(action_dim, name="policy_mean")(trunk_output)
            mean = jnp.tanh(mean)  # Bound actions to [-1, 1]

            # Log standard deviation (learnable)
            log_std = hk.get_parameter(
                "policy_log_std",
                shape=[action_dim],
                init=hk.initializers.Constant(-0.5),
            )
            log_std = jnp.broadcast_to(log_std, mean.shape)

            return mean, log_std

        def value_network(x):
            """Value network for critic."""
            layer_sizes = self.network_kwargs.get("value_layers", [256, 128, 64])
            activation = self.network_kwargs.get("activation", jax.nn.tanh)

            value_net = hk.nets.MLP(
                layer_sizes + [1], activation=activation, name="value_network"
            )

            return jnp.squeeze(value_net(x), axis=-1)

        # Transform networks
        policy_network = hk.without_apply_rng(hk.transform(policy_network))
        value_network = hk.without_apply_rng(hk.transform(value_network))

        # Create dummy input for initialization
        dummy_obs = utils.add_batch_dim(
            utils.zeros_like(self.environment_spec.observations)
        )

        return ppo.PPONetworks(
            policy_network=policy_network,
            value_network=value_network,
            log_prob=lambda params, actions, observations: self._log_prob(
                params, actions, observations, policy_network
            ),
            entropy=lambda params, observations: self._entropy(
                params, observations, policy_network
            ),
            sample=lambda params, observations, key: self._sample(
                params, observations, key, policy_network
            ),
            sample_eval=lambda params, observations, key: self._sample_eval(
                params, observations, key, policy_network
            ),
        )

    def _log_prob(self, params, actions, observations, policy_network):
        """Calculate log probability of actions."""
        mean, log_std = policy_network.apply(params, observations)
        std = jnp.exp(log_std)

        # Gaussian log probability
        log_prob = -0.5 * jnp.sum(
            ((actions - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1
        )
        return log_prob

    def _entropy(self, params, observations, policy_network):
        """Calculate policy entropy."""
        _, log_std = policy_network.apply(params, observations)
        # Entropy of multivariate Gaussian
        entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
        return entropy

    def _sample(self, params, observations, key, policy_network):
        """Sample actions from policy."""
        mean, log_std = policy_network.apply(params, observations)
        std = jnp.exp(log_std)

        # Sample from Gaussian distribution
        noise = jax.random.normal(key, mean.shape)
        actions = mean + std * noise

        # Clip to action bounds
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions

    def _sample_eval(self, params, observations, key, policy_network):
        """Sample actions for evaluation (deterministic)."""
        mean, _ = policy_network.apply(params, observations)
        return jnp.clip(mean, -1.0, 1.0)

    def build_agent(self) -> ppo.PPO:
        """Build PPO agent."""
        if self._networks is None:
            self._networks = self._build_networks()

        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_gradient_norm),
            optax.adam(self.learning_rate),
        )

        # PPO agent configuration
        config = ppo.PPOConfig(
            unroll_length=self.unroll_length,
            num_minibatches=self.num_minibatches,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            entropy_cost=self.entropy_cost,
            baseline_cost=self.value_cost,
            max_gradient_norm=self.max_gradient_norm,
        )

        # Create PPO agent
        self._agent = ppo.PPO(
            spec=self.environment_spec,
            networks=self._networks,
            config=config,
            seed=42,
            optimizer=optimizer,
        )

        return self._agent

    def get_actor(self) -> acme.Actor:
        """Get actor for inference."""
        if self._agent is None:
            self.build_agent()
        return self._agent

    def get_learner(self) -> acme.Learner:
        """Get learner for training."""
        if self._agent is None:
            self.build_agent()
        return self._agent._learner
