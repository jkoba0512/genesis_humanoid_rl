"""
Simple dependency injection container for genesis_humanoid_rl.
Enables clean separation of concerns and better testability.
"""

from typing import TypeVar, Generic, Dict, Any, Callable, Type, Optional
import logging
from dataclasses import dataclass

from ..protocols import (
    PhysicsManagerProtocol,
    ObservationManagerProtocol,
    RewardCalculatorProtocol,
    TerminationCheckerProtocol,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DIContainer:
    """
    Simple dependency injection container.

    Supports:
    - Singleton registration
    - Factory registration
    - Lazy initialization
    - Type-safe retrieval
    """

    def __init__(self):
        self._singletons: Dict[type, Any] = {}
        self._factories: Dict[type, Callable[[], Any]] = {}
        self._instances: Dict[type, Any] = {}
        self._initialized = False

    def register_singleton(self, interface: Type[T], implementation: T) -> None:
        """
        Register a singleton service.

        Args:
            interface: Interface type (usually a Protocol)
            implementation: Concrete implementation instance
        """
        self._singletons[interface] = implementation
        logger.debug(
            f"Registered singleton {interface.__name__} -> {type(implementation).__name__}"
        )

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """
        Register a factory function for creating services.

        Args:
            interface: Interface type (usually a Protocol)
            factory: Factory function that creates instances
        """
        self._factories[interface] = factory
        logger.debug(f"Registered factory for {interface.__name__}")

    def register_type(self, interface: Type[T], implementation_type: Type[T]) -> None:
        """
        Register a type that will be instantiated when requested.

        Args:
            interface: Interface type
            implementation_type: Concrete implementation type
        """
        self.register_factory(interface, lambda: implementation_type())

    def get(self, interface: Type[T]) -> T:
        """
        Get service instance.

        Args:
            interface: Interface type to retrieve

        Returns:
            Service instance

        Raises:
            ValueError: If no registration found for interface
        """
        # Check singletons first
        if interface in self._singletons:
            return self._singletons[interface]

        # Check if we already have an instance (lazy singleton)
        if interface in self._instances:
            return self._instances[interface]

        # Check factories
        if interface in self._factories:
            instance = self._factories[interface]()
            self._instances[interface] = instance  # Cache as lazy singleton
            logger.debug(f"Created instance of {interface.__name__}")
            return instance

        raise ValueError(f"No registration found for {interface.__name__}")

    def has(self, interface: Type[T]) -> bool:
        """
        Check if interface is registered.

        Args:
            interface: Interface type to check

        Returns:
            True if interface is registered
        """
        return (
            interface in self._singletons
            or interface in self._factories
            or interface in self._instances
        )

    def clear(self) -> None:
        """Clear all registrations and instances."""
        self._singletons.clear()
        self._factories.clear()
        self._instances.clear()
        logger.debug("DI container cleared")

    def get_registrations(self) -> Dict[str, str]:
        """
        Get information about registered services.

        Returns:
            Dictionary mapping interface names to registration types
        """
        registrations = {}

        for interface in self._singletons:
            registrations[interface.__name__] = "singleton"

        for interface in self._factories:
            registrations[interface.__name__] = "factory"

        for interface in self._instances:
            registrations[interface.__name__] = "instance"

        return registrations


def create_container() -> DIContainer:
    """
    Create a production DI container with default implementations.

    Returns:
        Configured DI container
    """
    container = DIContainer()

    # Import here to avoid circular dependencies
    from ..physics import GenesisPhysicsManager
    from ..observations import HumanoidObservationManager
    from ..rewards.walking_rewards import WalkingRewardFunction

    # Register default implementations
    container.register_factory(PhysicsManagerProtocol, lambda: GenesisPhysicsManager())

    container.register_factory(
        ObservationManagerProtocol, lambda: HumanoidObservationManager()
    )

    container.register_factory(
        RewardCalculatorProtocol, lambda: WalkingRewardFunction()
    )

    # TODO: Add TerminationCheckerProtocol implementation

    logger.info("Created production DI container")
    return container


def create_test_container() -> DIContainer:
    """
    Create a test DI container with mock implementations.

    Returns:
        DI container configured for testing
    """
    container = DIContainer()

    # Import here to avoid circular dependencies
    from ..physics import MockPhysicsManager
    from ..observations import MockObservationManager

    # Register mock implementations
    container.register_factory(PhysicsManagerProtocol, lambda: MockPhysicsManager())

    container.register_factory(
        ObservationManagerProtocol, lambda: MockObservationManager()
    )

    # TODO: Add mock reward calculator and termination checker

    logger.info("Created test DI container")
    return container


def create_custom_container(config: Dict[str, Any]) -> DIContainer:
    """
    Create a custom DI container from configuration.

    Args:
        config: Configuration dictionary with service mappings

    Returns:
        Configured DI container
    """
    container = DIContainer()

    # This could be expanded to support configuration-driven DI
    # For now, return default container
    logger.warning("Custom container configuration not implemented, using default")
    return create_container()


@dataclass
class ServiceConfig:
    """Configuration for a service registration."""

    interface: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    singleton: Optional[Any] = None

    def validate(self) -> None:
        """Validate service configuration."""
        count = sum(
            [
                self.implementation is not None,
                self.factory is not None,
                self.singleton is not None,
            ]
        )

        if count != 1:
            raise ValueError(
                "Exactly one of implementation, factory, or singleton must be provided"
            )


class ContainerBuilder:
    """
    Builder for creating DI containers with fluent API.

    Example:
        container = (ContainerBuilder()
                    .with_physics_manager(CustomPhysicsManager)
                    .with_observation_manager(CustomObservationManager)
                    .build())
    """

    def __init__(self):
        self._configs: Dict[Type, ServiceConfig] = {}

    def with_physics_manager(self, implementation: Type) -> "ContainerBuilder":
        """Configure physics manager implementation."""
        self._configs[PhysicsManagerProtocol] = ServiceConfig(
            interface=PhysicsManagerProtocol, implementation=implementation
        )
        return self

    def with_observation_manager(self, implementation: Type) -> "ContainerBuilder":
        """Configure observation manager implementation."""
        self._configs[ObservationManagerProtocol] = ServiceConfig(
            interface=ObservationManagerProtocol, implementation=implementation
        )
        return self

    def with_reward_calculator(self, implementation: Type) -> "ContainerBuilder":
        """Configure reward calculator implementation."""
        self._configs[RewardCalculatorProtocol] = ServiceConfig(
            interface=RewardCalculatorProtocol, implementation=implementation
        )
        return self

    def with_singleton(self, interface: Type, instance: Any) -> "ContainerBuilder":
        """Register a singleton instance."""
        self._configs[interface] = ServiceConfig(
            interface=interface, singleton=instance
        )
        return self

    def with_factory(self, interface: Type, factory: Callable) -> "ContainerBuilder":
        """Register a factory function."""
        self._configs[interface] = ServiceConfig(interface=interface, factory=factory)
        return self

    def build(self) -> DIContainer:
        """
        Build the DI container.

        Returns:
            Configured DI container
        """
        container = DIContainer()

        for config in self._configs.values():
            config.validate()

            if config.singleton is not None:
                container.register_singleton(config.interface, config.singleton)
            elif config.factory is not None:
                container.register_factory(config.interface, config.factory)
            elif config.implementation is not None:
                container.register_type(config.interface, config.implementation)

        logger.info(f"Built DI container with {len(self._configs)} services")
        return container
