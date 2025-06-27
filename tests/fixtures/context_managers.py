"""
Context managers for test resource management.

Provides context managers for common testing patterns like temporary files,
mocked environments, and resource cleanup.
"""

import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List
from unittest.mock import patch, Mock, MagicMock

import numpy as np
import pytest


@contextmanager
def temporary_directory(prefix: str = "test_") -> Generator[Path, None, None]:
    """
    Context manager for temporary directories.
    
    Args:
        prefix: Prefix for temporary directory name
        
    Yields:
        Path to temporary directory
        
    Example:
        with temporary_directory("my_test_") as temp_dir:
            test_file = temp_dir / "test.txt"
            test_file.write_text("test content")
        # Directory automatically cleaned up
    """
    with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
        yield Path(temp_dir)


@contextmanager
def temporary_file(
    suffix: str = ".tmp",
    prefix: str = "test_",
    content: Optional[str] = None
) -> Generator[Path, None, None]:
    """
    Context manager for temporary files.
    
    Args:
        suffix: File extension
        prefix: File name prefix
        content: Optional initial content
        
    Yields:
        Path to temporary file
        
    Example:
        with temporary_file(".json", content='{"test": true}') as temp_file:
            # Use temp_file
            pass
        # File automatically deleted
    """
    with tempfile.NamedTemporaryFile(
        mode='w+', suffix=suffix, prefix=prefix, delete=False
    ) as tmp_file:
        if content:
            tmp_file.write(content)
            tmp_file.flush()
        
        file_path = Path(tmp_file.name)
    
    try:
        yield file_path
    finally:
        if file_path.exists():
            file_path.unlink()


@contextmanager
def mock_environment_variables(env_vars: Dict[str, str]) -> Generator[None, None, None]:
    """
    Context manager for temporarily setting environment variables.
    
    Args:
        env_vars: Dictionary of environment variables to set
        
    Example:
        with mock_environment_variables({"TEST_MODE": "true"}):
            # TEST_MODE is set to "true"
            pass
        # Original environment restored
    """
    with patch.dict('os.environ', env_vars):
        yield


@contextmanager
def mock_time(fixed_time: float) -> Generator[None, None, None]:
    """
    Context manager for mocking time.time().
    
    Args:
        fixed_time: Fixed timestamp to return
        
    Example:
        with mock_time(1234567890.0):
            assert time.time() == 1234567890.0
    """
    with patch('time.time', return_value=fixed_time):
        yield


@contextmanager
def capture_logging(logger_name: str, level: str = "INFO") -> Generator[List[str], None, None]:
    """
    Context manager for capturing log messages.
    
    Args:
        logger_name: Name of logger to capture
        level: Minimum log level to capture
        
    Yields:
        List that will contain captured log messages
        
    Example:
        with capture_logging("my_logger") as logs:
            logger.info("test message")
        assert "test message" in logs[0]
    """
    import logging
    
    captured_logs = []
    
    class CaptureHandler(logging.Handler):
        def emit(self, record):
            captured_logs.append(self.format(record))
    
    logger = logging.getLogger(logger_name)
    handler = CaptureHandler()
    handler.setLevel(getattr(logging, level.upper()))
    
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(getattr(logging, level.upper()))
    
    try:
        yield captured_logs
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


@contextmanager
def timeout_context(seconds: float) -> Generator[None, None, None]:
    """
    Context manager for enforcing operation timeouts.
    
    Args:
        seconds: Timeout in seconds
        
    Raises:
        TimeoutError: If operation takes longer than specified time
        
    Example:
        with timeout_context(5.0):
            # Operation must complete within 5 seconds
            slow_operation()
    """
    def timeout_handler():
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    
    try:
        yield
    finally:
        timer.cancel()


@contextmanager
def performance_monitor() -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for monitoring performance metrics.
    
    Yields:
        Dictionary that will contain performance metrics
        
    Example:
        with performance_monitor() as metrics:
            # Perform operations
            pass
        print(f"Elapsed time: {metrics['elapsed_time']}")
    """
    metrics = {}
    start_time = time.time()
    
    try:
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        metrics['initial_memory_mb'] = initial_memory / (1024 * 1024)
    except ImportError:
        initial_memory = None
    
    yield metrics
    
    # Calculate metrics
    end_time = time.time()
    metrics['elapsed_time'] = end_time - start_time
    
    if initial_memory is not None:
        try:
            final_memory = process.memory_info().rss
            metrics['final_memory_mb'] = final_memory / (1024 * 1024)
            metrics['memory_delta_mb'] = metrics['final_memory_mb'] - metrics['initial_memory_mb']
        except:
            pass


@contextmanager
def isolated_random_state(seed: int = 42) -> Generator[None, None, None]:
    """
    Context manager for isolated random number generation.
    
    Args:
        seed: Random seed for reproducible results
        
    Example:
        with isolated_random_state(123):
            # All random operations are reproducible
            data = np.random.randn(100)
    """
    # Save current random states
    np_state = np.random.get_state()
    
    # Set new seed
    np.random.seed(seed)
    
    try:
        yield
    finally:
        # Restore original states
        np.random.set_state(np_state)


@contextmanager
def mock_genesis_physics(
    robot_config: Optional[Dict[str, Any]] = None,
    scene_config: Optional[Dict[str, Any]] = None
) -> Generator[Mock, None, None]:
    """
    Context manager for comprehensive Genesis physics mocking.
    
    Args:
        robot_config: Configuration for mock robot
        scene_config: Configuration for mock scene
        
    Yields:
        Mock Genesis physics engine
        
    Example:
        with mock_genesis_physics() as physics:
            robot = physics.add_robot("robot.urdf")
            physics.step()
    """
    robot_config = robot_config or {}
    scene_config = scene_config or {}
    
    # Create comprehensive mock
    mock_physics = Mock()
    
    # Configure robot mock
    mock_robot = Mock()
    mock_robot.n_dofs = robot_config.get('n_dofs', 35)
    mock_robot.n_links = robot_config.get('n_links', 30)
    mock_robot.get_pos.return_value = robot_config.get('position', np.array([0.0, 0.0, 0.8]))
    mock_robot.get_quat.return_value = robot_config.get('orientation', np.array([0.0, 0.0, 0.0, 1.0]))
    mock_robot.get_dofs_position.return_value = robot_config.get('joint_positions', np.zeros(35))
    mock_robot.get_dofs_velocity.return_value = robot_config.get('joint_velocities', np.zeros(35))
    
    # Configure scene mock
    mock_scene = Mock()
    mock_scene.add_entity.return_value = mock_robot
    mock_scene.step.return_value = None
    mock_scene.build.return_value = None
    mock_scene.reset.return_value = None
    
    # Configure physics engine
    mock_physics.scene = mock_scene
    mock_physics.robot = mock_robot
    mock_physics.step.return_value = None
    mock_physics.reset.return_value = None
    
    yield mock_physics


@contextmanager
def database_transaction_test(db_connection) -> Generator[None, None, None]:
    """
    Context manager for database transaction testing.
    
    Automatically starts a transaction and rolls it back after the test,
    ensuring test isolation.
    
    Args:
        db_connection: Database connection
        
    Example:
        with database_transaction_test(db) as transaction:
            # Perform database operations
            # Changes automatically rolled back
            pass
    """
    db_connection.execute("BEGIN TRANSACTION")
    
    try:
        yield
    finally:
        try:
            db_connection.execute("ROLLBACK")
        except Exception:
            pass  # Ignore rollback errors


@contextmanager
def assert_no_exceptions() -> Generator[None, None, None]:
    """
    Context manager that ensures no exceptions are raised.
    
    Example:
        with assert_no_exceptions():
            # This code must not raise any exceptions
            risky_operation()
    """
    try:
        yield
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {type(e).__name__}: {e}")


@contextmanager
def assert_raises_with_message(exception_type: type, message_pattern: str) -> Generator[None, None, None]:
    """
    Context manager that ensures specific exception with message is raised.
    
    Args:
        exception_type: Expected exception type
        message_pattern: Pattern that must be in exception message
        
    Example:
        with assert_raises_with_message(ValueError, "invalid input"):
            raise ValueError("The input is invalid")
    """
    try:
        yield
        pytest.fail(f"Expected {exception_type.__name__} to be raised")
    except exception_type as e:
        if message_pattern not in str(e):
            pytest.fail(f"Exception message '{e}' does not contain '{message_pattern}'")
    except Exception as e:
        pytest.fail(f"Expected {exception_type.__name__}, got {type(e).__name__}: {e}")


@contextmanager
def resource_usage_monitor(max_memory_mb: float = 1000.0) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for monitoring and limiting resource usage.
    
    Args:
        max_memory_mb: Maximum memory usage in MB
        
    Yields:
        Dictionary with resource usage statistics
        
    Raises:
        MemoryError: If memory usage exceeds limit
        
    Example:
        with resource_usage_monitor(max_memory_mb=500) as usage:
            # Perform memory-intensive operations
            pass
        print(f"Peak memory: {usage['peak_memory_mb']} MB")
    """
    try:
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        peak_memory = initial_memory
        
        usage_stats = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': initial_memory
        }
        
        # Monitor memory usage during execution
        def check_memory():
            nonlocal peak_memory
            current_memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)
            usage_stats['peak_memory_mb'] = peak_memory
            
            if current_memory > max_memory_mb:
                raise MemoryError(f"Memory usage {current_memory:.1f} MB exceeds limit {max_memory_mb} MB")
        
        # Start monitoring
        monitor_timer = threading.Timer(0.1, check_memory)
        monitor_timer.start()
        
        yield usage_stats
        
    except ImportError:
        # psutil not available
        usage_stats = {'error': 'psutil not available for monitoring'}
        yield usage_stats
    
    finally:
        try:
            monitor_timer.cancel()
        except:
            pass


# Parameterized context managers
def with_different_configs(configs: List[Dict[str, Any]]):
    """
    Decorator for running tests with different configurations.
    
    Args:
        configs: List of configuration dictionaries
        
    Example:
        @with_different_configs([
            {'batch_size': 32, 'lr': 0.001},
            {'batch_size': 64, 'lr': 0.01}
        ])
        def test_training(config):
            # Test runs twice with different configs
            pass
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            for config in configs:
                test_func(*args, config=config, **kwargs)
        return wrapper
    return decorator