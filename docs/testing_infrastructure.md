# Testing Infrastructure Guide

## Overview

This document describes the improved testing infrastructure for the genesis_humanoid_rl project. The new infrastructure provides comprehensive fixtures, context managers, and utilities to make testing more reliable, maintainable, and efficient.

## Key Improvements

### 1. **Centralized Fixture Management**
- **Location**: `tests/fixtures/`
- **Purpose**: Modular, reusable test fixtures organized by domain
- **Benefits**: Reduced duplication, consistent setup patterns, easier maintenance

### 2. **Context Managers for Resource Management**
- **Automatic cleanup** of temporary files, databases, and external resources
- **Isolation** between tests to prevent side effects
- **Performance monitoring** and resource usage tracking

### 3. **Domain-Driven Test Organization**
- **Domain fixtures** for business logic testing
- **Database fixtures** for persistence layer testing
- **Simulation fixtures** for physics and environment testing

### 4. **Comprehensive Test Utilities**
- **Validation functions** for common assertions
- **Mock builders** for complex object creation
- **Performance benchmarking** tools

## Architecture

```
tests/
├── fixtures/
│   ├── __init__.py                  # Fixture module exports
│   ├── database_fixtures.py         # Database testing infrastructure
│   ├── domain_fixtures.py           # Domain object builders and scenarios
│   ├── simulation_fixtures.py       # Physics and environment mocks
│   └── context_managers.py          # Resource management utilities
├── examples/
│   └── test_improved_patterns.py    # Example usage patterns
├── conftest.py                      # Pytest configuration and imports
└── ...existing test files...
```

## Usage Patterns

### Domain Object Testing

```python
def test_robot_creation(domain_builder, domain_validator):
    """Test robot creation using domain builder."""
    robot = domain_builder.robot(
        robot_id="test-robot",
        robot_type=RobotType.UNITREE_G1,
        joint_count=35
    )
    
    domain_validator.assert_valid_robot(robot)
    assert robot.joint_count == 35
```

### Database Testing with Context Managers

```python
def test_database_operations():
    """Test with automatic database cleanup."""
    with database_connection() as db:
        # Database automatically created and cleaned up
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        db.execute("INSERT INTO test (id) VALUES (1)")
        
        result = db.fetch_one("SELECT COUNT(*) FROM test")
        assert result[0] == 1
    # Database automatically closed and cleaned up
```

### Isolated Transaction Testing

```python
@pytest.mark.database
def test_with_transaction_isolation(isolated_transaction):
    """Test with automatic rollback."""
    uow = isolated_transaction
    
    # Create test data
    robot = create_test_robot()
    uow.robots.add(robot)
    
    # Data exists within transaction
    saved_robot = uow.robots.get(robot.robot_id)
    assert saved_robot is not None
    
    # Transaction automatically rolled back after test
```

### Physics Simulation Testing

```python
def test_physics_simulation(mock_physics_engine, environment_test_case):
    """Test with realistic physics simulation."""
    # Simulate robot actions
    for i in range(100):
        action = create_walking_action(i)
        state = mock_physics_engine.step(action)
        
        # Validate physics stability
        environment_test_case.assert_physics_stability(state)
```

### Performance Testing

```python
def test_training_performance():
    """Test with performance monitoring."""
    with performance_monitor() as metrics:
        # Perform training operations
        run_training_episode()
    
    # Automatically collected metrics
    assert metrics['elapsed_time'] < 5.0
    assert metrics['memory_delta_mb'] < 100.0
```

## Fixture Categories

### Domain Fixtures (`domain_fixtures.py`)

#### DomainObjectBuilder
Fluent interface for creating domain objects with sensible defaults:

```python
# Create objects with minimal configuration
robot = builder.robot(robot_id="test-robot")
session = builder.learning_session(session_id="test-session")
episode = builder.learning_episode(step_count=1000)

# Or with full customization
motion_command = builder.motion_command(
    motion_type=MotionType.WALK_FORWARD,
    velocity=1.5,
    duration=10.0
)
```

#### Scenario Fixtures
Pre-configured scenarios for integration testing:

- `training_scenario`: Complete training setup with robot, session, episodes, curriculum
- `skill_progression_scenario`: Skills progression from beginner to expert
- `curriculum_stage_scenario`: Multi-stage curriculum progression

#### Domain Validators
Comprehensive validation utilities:

```python
domain_validator.assert_valid_robot(robot)
domain_validator.assert_valid_session(session)
domain_validator.assert_valid_performance_metrics(metrics)
```

### Database Fixtures (`database_fixtures.py`)

#### Context Managers
- `temporary_database()`: Creates isolated temporary database
- `database_connection()`: Managed database connection with cleanup
- `database_transaction_test()`: Automatic transaction rollback

#### Pytest Fixtures
- `temp_database_path`: Temporary database file path
- `database_connection_fixture`: Connected database instance
- `unit_of_work_fixture`: Configured Unit of Work
- `training_service_fixture`: Training transaction service
- `isolated_transaction`: Transaction with automatic rollback

#### Database Test Utilities
```python
class DatabaseTestCase:
    def create_test_robot(self, robot_id="test-robot")
    def create_test_session(self, session_id="test-session")
    def setup_database_test(self, db_connection)
```

### Simulation Fixtures (`simulation_fixtures.py`)

#### MockPhysicsEngine
Realistic physics simulation without Genesis dependency:

```python
engine = MockPhysicsEngine()
for i in range(100):
    action = create_action()
    state = engine.step(action)
    observation = engine.get_observation()
```

#### Features:
- **Deterministic behavior** for reproducible tests
- **Realistic physics** including forward motion, joint dynamics
- **State history tracking** for analysis
- **Configurable parameters** (timestep, robot configuration)

#### Context Managers
- `mock_genesis_environment()`: Complete Genesis environment mock
- `mock_genesis_physics()`: Physics engine with custom configuration

#### Environment Testing Utilities
```python
class EnvironmentTestCase:
    @staticmethod
    def assert_valid_observation(obs)
    @staticmethod
    def assert_valid_action(action)
    @staticmethod
    def assert_physics_stability(state)
```

### Context Manager Utilities (`context_managers.py`)

#### Resource Management
- `temporary_file()`: Managed temporary files
- `temporary_directory()`: Managed temporary directories
- `mock_environment_variables()`: Environment variable mocking

#### Testing Utilities
- `isolated_random_state()`: Reproducible random number generation
- `capture_logging()`: Log message capture and validation
- `timeout_context()`: Operation timeout enforcement
- `performance_monitor()`: Resource usage tracking

#### Error Testing
- `assert_no_exceptions()`: Ensure code doesn't raise exceptions
- `assert_raises_with_message()`: Validate specific exceptions

## Test Organization

### Test Markers
Configure test categories using pytest markers:

```python
@pytest.mark.unit          # Fast, isolated unit tests
@pytest.mark.integration   # Integration tests with dependencies
@pytest.mark.database      # Tests requiring database
@pytest.mark.simulation    # Tests requiring physics simulation
@pytest.mark.performance   # Performance benchmark tests
@pytest.mark.slow          # Long-running tests
```

Run specific test categories:
```bash
# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Parameterized Testing
Use pytest parameterization for comprehensive coverage:

```python
@pytest.mark.parametrize("robot_type,expected_joints", [
    (RobotType.UNITREE_G1, 35),
    (RobotType.GENERIC_HUMANOID, 30),
])
def test_robot_configuration(robot_type, expected_joints, domain_builder):
    robot = domain_builder.robot(
        robot_type=robot_type,
        joint_count=expected_joints
    )
    assert robot.joint_count == expected_joints
```

## Performance and Isolation

### Automatic Cleanup
All fixtures and context managers provide automatic resource cleanup:

- **Temporary files** automatically deleted
- **Database connections** properly closed
- **Transactions** rolled back for isolation
- **Mock states** reset between tests

### Session-Level Fixtures
Expensive setup operations use session-scoped fixtures:

```python
@pytest.fixture(scope="session")
def shared_test_database():
    """Database shared across all tests in session."""
    with temporary_database("shared.db") as db_path:
        with database_connection(db_path) as db:
            yield db
```

### Performance Monitoring
Built-in performance monitoring for identifying bottlenecks:

```python
with performance_monitor() as metrics:
    run_expensive_operation()

print(f"Operation took {metrics['elapsed_time']:.2f}s")
print(f"Memory usage: {metrics['memory_delta_mb']:.1f}MB")
```

## Migration Guide

### From unittest.TestCase to pytest

**Before:**
```python
class TestRobot(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.db = DatabaseConnection(self.temp_file.name)
        
    def tearDown(self):
        self.db.close()
        os.unlink(self.temp_file.name)
        
    def test_robot_creation(self):
        robot = HumanoidRobot(...)
        self.assertEqual(robot.joint_count, 35)
```

**After:**
```python
class TestRobot:
    def test_robot_creation(self, domain_builder, domain_validator):
        robot = domain_builder.robot(joint_count=35)
        domain_validator.assert_valid_robot(robot)
        assert robot.joint_count == 35
```

### Using New Fixtures

**Replace manual setup:**
```python
# Old pattern
def test_database():
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    db = DatabaseConnection(temp_file.name)
    try:
        # test code
        pass
    finally:
        db.close()
        os.unlink(temp_file.name)
```

**With context managers:**
```python
# New pattern
def test_database():
    with database_connection() as db:
        # test code - automatic cleanup
        pass
```

## Best Practices

### 1. **Use Appropriate Fixtures**
- **Domain fixtures** for business logic tests
- **Database fixtures** for persistence tests
- **Simulation fixtures** for environment tests

### 2. **Leverage Context Managers**
- Always use context managers for resource management
- Prefer `with` statements over manual setup/teardown

### 3. **Maintain Test Isolation**
- Use `isolated_transaction` for database tests
- Use `isolated_random_state` for reproducible randomness
- Reset state between tests

### 4. **Validate Comprehensively**
- Use domain validators for business object validation
- Use environment validators for simulation state
- Include performance assertions where relevant

### 5. **Organize Tests Logically**
- Group related tests in classes
- Use descriptive test names
- Mark tests with appropriate categories

### 6. **Monitor Performance**
- Use `performance_monitor` for expensive operations
- Set reasonable performance expectations
- Profile tests that are slower than expected

## Example Test Structure

```python
class TestRobotLearning:
    """Test robot learning functionality."""
    
    def test_single_episode_learning(self, domain_builder, mock_physics_engine):
        """Test learning in a single episode."""
        # Arrange
        robot = domain_builder.robot()
        
        # Act
        with performance_monitor() as metrics:
            for step in range(100):
                action = create_action(step)
                state = mock_physics_engine.step(action)
        
        # Assert
        assert metrics['elapsed_time'] < 1.0
        assert state.step_count == 100
    
    @pytest.mark.integration
    def test_full_training_pipeline(self, training_scenario, database_connection_fixture):
        """Test complete training pipeline."""
        scenario = training_scenario
        
        with isolated_random_state(42):
            # Run deterministic training
            results = run_training(scenario)
        
        assert results['success_rate'] > 0.8
    
    @pytest.mark.performance
    def test_training_performance(self, benchmark_config):
        """Test training performance benchmarks."""
        with resource_usage_monitor(max_memory_mb=500) as usage:
            run_performance_test(benchmark_config)
        
        assert usage['peak_memory_mb'] < 500
```

This improved testing infrastructure provides a solid foundation for reliable, maintainable, and comprehensive testing across all layers of the application.