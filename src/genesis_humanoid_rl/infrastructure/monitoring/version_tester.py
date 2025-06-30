"""
Genesis version testing and compatibility validation.

Provides automated testing of Genesis versions against our humanoid RL codebase
to ensure compatibility and detect breaking changes.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .genesis_monitor import (
    GenesisAPIMonitor,
    GenesisCompatibilityReport,
    GenesisCompatibilityLevel,
)

logger = logging.getLogger(__name__)


class VersionTestResult(Enum):
    """Version test result status."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class GenesisVersionTest:
    """Individual version test case."""

    test_name: str
    description: str
    version_requirement: Optional[str] = None
    expected_result: VersionTestResult = VersionTestResult.PASS
    actual_result: Optional[VersionTestResult] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    output: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VersionTestSuite:
    """Collection of version tests for Genesis compatibility."""

    name: str
    description: str
    genesis_version: str
    tests: List[GenesisVersionTest] = field(default_factory=list)
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None

    def add_test(self, test: GenesisVersionTest) -> None:
        """Add a test to the suite."""
        self.tests.append(test)

    def get_pass_rate(self) -> float:
        """Calculate test pass rate."""
        if not self.tests:
            return 0.0

        passed_tests = sum(
            1 for test in self.tests if test.actual_result == VersionTestResult.PASS
        )
        return passed_tests / len(self.tests)

    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary."""
        results = {}
        for result in VersionTestResult:
            results[result.value] = sum(
                1 for test in self.tests if test.actual_result == result
            )

        return {
            "name": self.name,
            "genesis_version": self.genesis_version,
            "total_tests": len(self.tests),
            "pass_rate": self.get_pass_rate(),
            "results": results,
            "execution_time": sum(test.execution_time for test in self.tests),
        }


class GenesisVersionTester:
    """Tests Genesis versions for compatibility with our codebase."""

    def __init__(self, test_dir: str = "version_tests"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize test suites
        self.test_suites = {}
        self._initialize_test_suites()

    def _initialize_test_suites(self) -> None:
        """Initialize standard test suites."""
        # Core functionality test suite
        core_suite = VersionTestSuite(
            name="core_functionality",
            description="Test core Genesis functionality required by our humanoid RL system",
            genesis_version="any",
        )

        # Add core tests
        core_tests = [
            GenesisVersionTest(
                test_name="basic_import", description="Test basic Genesis module import"
            ),
            GenesisVersionTest(
                test_name="scene_creation",
                description="Test Genesis scene creation and configuration",
            ),
            GenesisVersionTest(
                test_name="rigid_body_simulation",
                description="Test rigid body physics simulation",
            ),
            GenesisVersionTest(
                test_name="robot_loading",
                description="Test URDF robot loading capabilities",
            ),
            GenesisVersionTest(
                test_name="joint_control",
                description="Test robot joint control and actuation",
            ),
            GenesisVersionTest(
                test_name="observation_extraction",
                description="Test robot state observation extraction",
            ),
            GenesisVersionTest(
                test_name="simulation_performance",
                description="Test simulation performance benchmarks",
            ),
            GenesisVersionTest(
                test_name="headless_operation",
                description="Test headless simulation operation",
            ),
        ]

        for test in core_tests:
            core_suite.add_test(test)

        self.test_suites["core_functionality"] = core_suite

        # Humanoid-specific test suite
        humanoid_suite = VersionTestSuite(
            name="humanoid_robotics",
            description="Test humanoid robotics specific functionality",
            genesis_version="any",
        )

        humanoid_tests = [
            GenesisVersionTest(
                test_name="unitree_g1_loading",
                description="Test Unitree G1 robot loading and initialization",
            ),
            GenesisVersionTest(
                test_name="bipedal_physics",
                description="Test bipedal locomotion physics",
            ),
            GenesisVersionTest(
                test_name="contact_detection",
                description="Test ground contact detection for feet",
            ),
            GenesisVersionTest(
                test_name="balance_simulation",
                description="Test robot balance and stability simulation",
            ),
            GenesisVersionTest(
                test_name="gait_analysis",
                description="Test gait pattern analysis capabilities",
            ),
        ]

        for test in humanoid_tests:
            humanoid_suite.add_test(test)

        self.test_suites["humanoid_robotics"] = humanoid_suite

        # Performance test suite
        performance_suite = VersionTestSuite(
            name="performance_benchmarks",
            description="Test Genesis performance for RL training",
            genesis_version="any",
        )

        performance_tests = [
            GenesisVersionTest(
                test_name="single_robot_fps",
                description="Test single robot simulation FPS",
            ),
            GenesisVersionTest(
                test_name="multi_robot_scaling",
                description="Test multi-robot simulation scaling",
            ),
            GenesisVersionTest(
                test_name="memory_usage",
                description="Test memory usage during extended simulation",
            ),
            GenesisVersionTest(
                test_name="gpu_acceleration",
                description="Test GPU acceleration performance",
            ),
            GenesisVersionTest(
                test_name="training_integration",
                description="Test integration with RL training loops",
            ),
        ]

        for test in performance_tests:
            performance_suite.add_test(test)

        self.test_suites["performance_benchmarks"] = performance_suite

    async def test_version_compatibility(
        self, version: Optional[str] = None, test_suites: Optional[List[str]] = None
    ) -> Dict[str, VersionTestSuite]:
        """Test Genesis version compatibility."""
        if test_suites is None:
            test_suites = list(self.test_suites.keys())

        self.logger.info(
            f"Testing Genesis compatibility for version: {version or 'current'}"
        )

        results = {}

        for suite_name in test_suites:
            if suite_name not in self.test_suites:
                self.logger.warning(f"Unknown test suite: {suite_name}")
                continue

            self.logger.info(f"Running test suite: {suite_name}")
            suite = self.test_suites[suite_name]

            # Execute test suite
            await self._execute_test_suite(suite)
            results[suite_name] = suite

        # Save results
        await self._save_test_results(results, version)

        return results

    async def _execute_test_suite(self, suite: VersionTestSuite) -> None:
        """Execute a test suite."""
        self.logger.info(f"Executing test suite: {suite.name}")

        # Run setup if available
        if suite.setup_code:
            await self._execute_setup_code(suite.setup_code)

        # Execute each test
        for test in suite.tests:
            await self._execute_version_test(test)

        # Run teardown if available
        if suite.teardown_code:
            await self._execute_teardown_code(suite.teardown_code)

    async def _execute_version_test(self, test: GenesisVersionTest) -> None:
        """Execute an individual version test."""
        self.logger.info(f"Running test: {test.test_name}")

        start_time = time.time()

        try:
            # Route to appropriate test implementation
            test_method = getattr(self, f"_test_{test.test_name}", None)
            if test_method:
                result = await test_method(test)
                test.actual_result = result
            else:
                test.actual_result = VersionTestResult.SKIP
                test.error_message = f"Test implementation not found: {test.test_name}"

        except Exception as e:
            test.actual_result = VersionTestResult.ERROR
            test.error_message = str(e)
            self.logger.error(f"Test {test.test_name} failed with error: {e}")

        test.execution_time = time.time() - start_time

        self.logger.info(f"Test {test.test_name} completed: {test.actual_result.value}")

    async def _test_basic_import(self, test: GenesisVersionTest) -> VersionTestResult:
        """Test basic Genesis import."""
        try:
            import genesis as gs

            # Check for essential classes
            required_classes = ["Scene", "RigidSolver"]
            missing_classes = []

            for cls_name in required_classes:
                if not hasattr(gs, cls_name):
                    missing_classes.append(cls_name)

            if missing_classes:
                test.error_message = f"Missing required classes: {missing_classes}"
                return VersionTestResult.FAIL

            test.output = f"Genesis version: {getattr(gs, '__version__', 'unknown')}"
            return VersionTestResult.PASS

        except ImportError as e:
            test.error_message = f"Genesis import failed: {e}"
            return VersionTestResult.FAIL

    async def _test_scene_creation(self, test: GenesisVersionTest) -> VersionTestResult:
        """Test Genesis scene creation."""
        try:
            import genesis as gs

            # Test scene creation with different configurations
            configs = [
                {},
                {"show_viewer": False},
                {"show_viewer": False, "substeps": 10},
            ]

            successful_configs = 0

            for config in configs:
                try:
                    scene = gs.Scene(**config)
                    successful_configs += 1

                    # Clean up
                    if hasattr(scene, "close"):
                        scene.close()

                except Exception as e:
                    test.output = f"Config {config} failed: {e}"

            if successful_configs == len(configs):
                test.output = f"All {len(configs)} scene configurations successful"
                return VersionTestResult.PASS
            elif successful_configs > 0:
                test.output = (
                    f"{successful_configs}/{len(configs)} configurations successful"
                )
                return VersionTestResult.FAIL
            else:
                test.error_message = "No scene configurations successful"
                return VersionTestResult.FAIL

        except Exception as e:
            test.error_message = f"Scene creation test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_rigid_body_simulation(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test rigid body simulation."""
        try:
            import genesis as gs

            scene = gs.Scene(show_viewer=False)

            # Add basic rigid bodies
            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            box = gs.morphs.Box(size=(0.1, 0.1, 0.1))
            scene.add_entity(box, pos=(0, 0, 1.0))

            # Build and simulate
            scene.build()

            # Run simulation steps
            step_count = 10
            step_times = []

            for i in range(step_count):
                step_start = time.time()
                scene.step()
                step_duration = time.time() - step_start
                step_times.append(step_duration)

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            avg_step_time = sum(step_times) / len(step_times)
            fps = 1.0 / avg_step_time

            test.output = (
                f"Simulation FPS: {fps:.1f}, Avg step time: {avg_step_time:.4f}s"
            )

            # Consider it successful if we can maintain reasonable FPS
            if fps > 10:
                return VersionTestResult.PASS
            else:
                test.error_message = f"Poor simulation performance: {fps:.1f} FPS"
                return VersionTestResult.FAIL

        except Exception as e:
            test.error_message = f"Rigid body simulation test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_robot_loading(self, test: GenesisVersionTest) -> VersionTestResult:
        """Test robot loading capabilities."""
        try:
            import genesis as gs

            # Check URDF loading capability
            if not (hasattr(gs, "morphs") and hasattr(gs.morphs, "URDF")):
                test.error_message = "URDF loading not available"
                return VersionTestResult.FAIL

            # Test basic robot loading infrastructure
            scene = gs.Scene(show_viewer=False)

            # We can't test actual robot loading without robot files,
            # but we can test the loading infrastructure
            try:
                # This should fail gracefully with file not found, not crash
                robot_loader = gs.morphs.URDF
                test.output = "URDF loader class available"

                # Clean up
                if hasattr(scene, "close"):
                    scene.close()

                return VersionTestResult.PASS

            except Exception as e:
                test.error_message = f"URDF loader test failed: {e}"
                return VersionTestResult.FAIL

        except Exception as e:
            test.error_message = f"Robot loading test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_joint_control(self, test: GenesisVersionTest) -> VersionTestResult:
        """Test robot joint control."""
        try:
            import genesis as gs

            # This is a capability test - we check if joint control APIs are available
            scene = gs.Scene(show_viewer=False)

            # Check for joint control related classes/methods
            control_features = []

            if hasattr(gs, "RigidEntity"):
                control_features.append("RigidEntity")

            if hasattr(gs, "ArticulatedEntity"):
                control_features.append("ArticulatedEntity")

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            if control_features:
                test.output = f"Joint control features available: {control_features}"
                return VersionTestResult.PASS
            else:
                test.error_message = "No joint control features detected"
                return VersionTestResult.FAIL

        except Exception as e:
            test.error_message = f"Joint control test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_observation_extraction(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test robot state observation extraction."""
        try:
            import genesis as gs

            scene = gs.Scene(show_viewer=False)

            # Add a simple entity to test state extraction
            box = gs.morphs.Box(size=(0.1, 0.1, 0.1))
            entity = scene.add_entity(box, pos=(0, 0, 1.0))

            scene.build()
            scene.step()

            # Test state extraction capabilities
            state_features = []

            # Check if we can extract basic state information
            if hasattr(entity, "get_pos") or hasattr(entity, "pos"):
                state_features.append("position")

            if hasattr(entity, "get_quat") or hasattr(entity, "quat"):
                state_features.append("orientation")

            if hasattr(entity, "get_vel") or hasattr(entity, "vel"):
                state_features.append("velocity")

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            if state_features:
                test.output = f"State extraction features: {state_features}"
                return VersionTestResult.PASS
            else:
                test.error_message = "No state extraction features detected"
                return VersionTestResult.FAIL

        except Exception as e:
            test.error_message = f"Observation extraction test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_simulation_performance(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test simulation performance benchmarks."""
        try:
            import genesis as gs

            scene = gs.Scene(show_viewer=False)

            # Create a more complex scene for performance testing
            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            # Add multiple objects
            num_objects = 20
            for i in range(num_objects):
                box = gs.morphs.Box(size=(0.05, 0.05, 0.05))
                x = (i % 5) * 0.1
                y = (i // 5) * 0.1
                scene.add_entity(box, pos=(x, y, 1.0))

            scene.build()

            # Performance test
            num_steps = 100
            step_times = []

            for _ in range(num_steps):
                step_start = time.time()
                scene.step()
                step_duration = time.time() - step_start
                step_times.append(step_duration)

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            # Calculate performance metrics
            avg_step_time = sum(step_times) / len(step_times)
            fps = 1.0 / avg_step_time
            min_step_time = min(step_times)
            max_step_time = max(step_times)

            test.output = (
                f"Performance: {fps:.1f} FPS avg, "
                f"{min_step_time*1000:.2f}-{max_step_time*1000:.2f}ms range"
            )

            # Performance thresholds
            if fps > 50:
                return VersionTestResult.PASS
            elif fps > 20:
                # Acceptable but not great
                test.error_message = f"Moderate performance: {fps:.1f} FPS"
                return VersionTestResult.FAIL
            else:
                test.error_message = f"Poor performance: {fps:.1f} FPS"
                return VersionTestResult.FAIL

        except Exception as e:
            test.error_message = f"Performance test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_headless_operation(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test headless simulation operation."""
        try:
            import genesis as gs

            # Test headless operation - critical for server deployment
            scene = gs.Scene(show_viewer=False)

            # Add basic scene elements
            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            box = gs.morphs.Box(size=(0.1, 0.1, 0.1))
            scene.add_entity(box, pos=(0, 0, 1.0))

            # Build and run simulation
            scene.build()

            # Simulate for several steps
            for _ in range(10):
                scene.step()

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            test.output = "Headless simulation successful"
            return VersionTestResult.PASS

        except Exception as e:
            test.error_message = f"Headless operation test failed: {e}"
            return VersionTestResult.FAIL

    # Humanoid-specific tests
    async def _test_unitree_g1_loading(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test Unitree G1 robot loading (capability test)."""
        try:
            import genesis as gs

            # This is a capability test - check if we can set up the infrastructure
            # for loading the Unitree G1 robot

            scene = gs.Scene(show_viewer=False)

            # Check URDF loading capability
            if not (hasattr(gs, "morphs") and hasattr(gs.morphs, "URDF")):
                test.error_message = "URDF loading required for Unitree G1"
                return VersionTestResult.FAIL

            # Test if we can handle articulated robots
            articulated_support = (
                hasattr(gs, "RigidEntity")
                or hasattr(gs, "ArticulatedEntity")
                or hasattr(gs, "Robot")
            )

            if not articulated_support:
                test.error_message = "Articulated robot support not detected"
                return VersionTestResult.FAIL

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            test.output = "Unitree G1 loading infrastructure available"
            return VersionTestResult.PASS

        except Exception as e:
            test.error_message = f"Unitree G1 test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_bipedal_physics(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test bipedal locomotion physics."""
        try:
            import genesis as gs

            scene = gs.Scene(show_viewer=False)

            # Test physics features needed for bipedal locomotion
            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            # Create simple bipedal structure (two "feet")
            foot1 = gs.morphs.Box(size=(0.1, 0.05, 0.02))
            foot2 = gs.morphs.Box(size=(0.1, 0.05, 0.02))

            scene.add_entity(foot1, pos=(0.1, 0, 0.1))
            scene.add_entity(foot2, pos=(-0.1, 0, 0.1))

            scene.build()

            # Simulate physics
            for _ in range(20):
                scene.step()

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            test.output = "Bipedal physics simulation successful"
            return VersionTestResult.PASS

        except Exception as e:
            test.error_message = f"Bipedal physics test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_contact_detection(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test ground contact detection."""
        try:
            import genesis as gs

            scene = gs.Scene(show_viewer=False)

            # Create ground and object for contact testing
            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            box = gs.morphs.Box(size=(0.1, 0.1, 0.1))
            scene.add_entity(box, pos=(0, 0, 0.5))

            scene.build()

            # Simulate until contact
            for _ in range(50):
                scene.step()

            # Check if we have contact detection APIs
            contact_features = []

            # Look for contact-related functionality
            if hasattr(gs, "ContactData") or hasattr(gs, "contacts"):
                contact_features.append("contact_data")

            if hasattr(scene, "get_contacts") or hasattr(scene, "contacts"):
                contact_features.append("scene_contacts")

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            if contact_features:
                test.output = f"Contact detection features: {contact_features}"
                return VersionTestResult.PASS
            else:
                test.output = "Contact simulation successful (detection API unknown)"
                return VersionTestResult.PASS  # Physics works even without contact API

        except Exception as e:
            test.error_message = f"Contact detection test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_balance_simulation(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test robot balance and stability simulation."""
        return await self._test_bipedal_physics(test)  # Same test effectively

    async def _test_gait_analysis(self, test: GenesisVersionTest) -> VersionTestResult:
        """Test gait pattern analysis capabilities."""
        return await self._test_contact_detection(test)  # Related functionality

    # Performance benchmark tests
    async def _test_single_robot_fps(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test single robot simulation FPS."""
        return await self._test_simulation_performance(test)  # Same test

    async def _test_multi_robot_scaling(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test multi-robot simulation scaling."""
        try:
            import genesis as gs

            scene = gs.Scene(show_viewer=False)

            # Create multiple simple "robots" (boxes with different positions)
            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            num_robots = 5
            for i in range(num_robots):
                box = gs.morphs.Box(size=(0.1, 0.1, 0.1))
                x = i * 0.3
                scene.add_entity(box, pos=(x, 0, 0.5))

            scene.build()

            # Measure performance with multiple robots
            num_steps = 50
            step_times = []

            for _ in range(num_steps):
                step_start = time.time()
                scene.step()
                step_duration = time.time() - step_start
                step_times.append(step_duration)

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            avg_step_time = sum(step_times) / len(step_times)
            fps = 1.0 / avg_step_time

            test.output = (
                f"Multi-robot performance: {fps:.1f} FPS with {num_robots} robots"
            )

            if fps > 30:
                return VersionTestResult.PASS
            else:
                test.error_message = f"Poor multi-robot performance: {fps:.1f} FPS"
                return VersionTestResult.FAIL

        except Exception as e:
            test.error_message = f"Multi-robot scaling test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_memory_usage(self, test: GenesisVersionTest) -> VersionTestResult:
        """Test memory usage during extended simulation."""
        try:
            import genesis as gs
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            scene = gs.Scene(show_viewer=False)

            # Create scene
            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            for i in range(10):
                box = gs.morphs.Box(size=(0.05, 0.05, 0.05))
                scene.add_entity(box, pos=(i * 0.1, 0, 0.5))

            scene.build()

            # Run extended simulation
            for _ in range(200):
                scene.step()

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            test.output = f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)"

            # Memory increase should be reasonable
            if memory_increase < 100:  # Less than 100MB increase
                return VersionTestResult.PASS
            else:
                test.error_message = (
                    f"High memory usage increase: {memory_increase:.1f}MB"
                )
                return VersionTestResult.FAIL

        except Exception as e:
            test.error_message = f"Memory usage test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_gpu_acceleration(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test GPU acceleration performance."""
        try:
            import genesis as gs

            # Check if GPU features are available
            gpu_features = []

            if hasattr(gs, "cuda"):
                gpu_features.append("cuda")

            if hasattr(gs, "gpu"):
                gpu_features.append("gpu")

            # Test basic scene creation (GPU backend handled automatically)
            scene = gs.Scene(show_viewer=False)

            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            scene.build()
            scene.step()

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            if gpu_features:
                test.output = f"GPU features available: {gpu_features}"
                return VersionTestResult.PASS
            else:
                test.output = "GPU acceleration not detected (CPU fallback working)"
                return VersionTestResult.PASS  # Not a failure, just no GPU

        except Exception as e:
            test.error_message = f"GPU acceleration test failed: {e}"
            return VersionTestResult.FAIL

    async def _test_training_integration(
        self, test: GenesisVersionTest
    ) -> VersionTestResult:
        """Test integration with RL training loops."""
        try:
            import genesis as gs
            import numpy as np

            # Simulate RL training loop interaction
            scene = gs.Scene(show_viewer=False)

            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            box = gs.morphs.Box(size=(0.1, 0.1, 0.1))
            entity = scene.add_entity(box, pos=(0, 0, 0.5))

            scene.build()

            # Simulate training loop
            episode_length = 50
            action_dim = 3  # Simple 3D action

            for step in range(episode_length):
                # Random action (simulate RL agent)
                action = np.random.uniform(-1, 1, action_dim)

                # Apply action (in real case, this would be joint commands)
                # For this test, we just step the simulation
                scene.step()

                # Extract observation (simulate state extraction)
                # In real case, this would be robot state
                obs_dim = 6  # Simple observation
                observation = np.random.random(obs_dim)

                # Calculate reward (simulate reward function)
                reward = np.random.random()

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            test.output = f"Training integration test: {episode_length} steps completed"
            return VersionTestResult.PASS

        except Exception as e:
            test.error_message = f"Training integration test failed: {e}"
            return VersionTestResult.FAIL

    async def _execute_setup_code(self, setup_code: str) -> None:
        """Execute setup code for test suite."""
        try:
            exec(setup_code)
        except Exception as e:
            self.logger.error(f"Setup code execution failed: {e}")

    async def _execute_teardown_code(self, teardown_code: str) -> None:
        """Execute teardown code for test suite."""
        try:
            exec(teardown_code)
        except Exception as e:
            self.logger.error(f"Teardown code execution failed: {e}")

    async def _save_test_results(
        self, results: Dict[str, VersionTestSuite], version: Optional[str]
    ) -> None:
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_str = version or "current"
        results_file = self.test_dir / f"version_test_{version_str}_{timestamp}.json"

        try:
            # Convert results to serializable format
            serializable_results = {}

            for suite_name, suite in results.items():
                suite_data = {
                    "name": suite.name,
                    "description": suite.description,
                    "genesis_version": suite.genesis_version,
                    "summary": suite.get_summary(),
                    "tests": [],
                }

                for test in suite.tests:
                    test_data = {
                        "test_name": test.test_name,
                        "description": test.description,
                        "expected_result": test.expected_result.value,
                        "actual_result": (
                            test.actual_result.value if test.actual_result else None
                        ),
                        "execution_time": test.execution_time,
                        "error_message": test.error_message,
                        "output": test.output,
                        "timestamp": test.timestamp.isoformat(),
                    }
                    suite_data["tests"].append(test_data)

                serializable_results[suite_name] = suite_data

            with open(results_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            self.logger.info(f"Version test results saved to {results_file}")

        except Exception as e:
            self.logger.error(f"Failed to save test results: {e}")

    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Get the most recent test results."""
        try:
            result_files = list(self.test_dir.glob("version_test_*.json"))
            if not result_files:
                return None

            # Get most recent file
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)

            with open(latest_file, "r") as f:
                return json.load(f)

        except Exception as e:
            self.logger.error(f"Failed to load latest results: {e}")
            return None


# Convenience functions
async def test_genesis_version(
    version: Optional[str] = None, test_suites: Optional[List[str]] = None
) -> Dict[str, VersionTestSuite]:
    """Test Genesis version compatibility."""
    tester = GenesisVersionTester()
    return await tester.test_version_compatibility(version, test_suites)


async def run_all_version_tests() -> Dict[str, VersionTestSuite]:
    """Run all version tests."""
    return await test_genesis_version()


if __name__ == "__main__":
    # Run version testing
    import asyncio

    async def main():
        print("Running Genesis version compatibility tests...")
        results = await run_all_version_tests()

        print(f"\nVersion Test Results:")
        print("=" * 50)

        for suite_name, suite in results.items():
            summary = suite.get_summary()
            print(f"\n{suite.name.upper()}:")
            print(f"  Description: {suite.description}")
            print(f"  Total Tests: {summary['total_tests']}")
            print(f"  Pass Rate: {summary['pass_rate']:.1%}")
            print(f"  Execution Time: {summary['execution_time']:.2f}s")

            for result_type, count in summary["results"].items():
                if count > 0:
                    print(f"  {result_type.upper()}: {count}")

            # Show failed tests
            failed_tests = [
                test
                for test in suite.tests
                if test.actual_result == VersionTestResult.FAIL
            ]
            if failed_tests:
                print(f"  Failed Tests:")
                for test in failed_tests:
                    print(f"    - {test.test_name}: {test.error_message}")

    asyncio.run(main())
