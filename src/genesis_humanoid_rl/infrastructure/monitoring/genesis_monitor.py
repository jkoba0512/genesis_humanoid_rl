"""
Genesis API monitoring and version compatibility testing.

Provides comprehensive monitoring of Genesis physics engine API compatibility,
version tracking, and performance metrics for production deployments.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

try:
    import genesis as gs

    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    gs = None

logger = logging.getLogger(__name__)


class GenesisCompatibilityLevel(Enum):
    """Genesis API compatibility levels."""

    FULLY_COMPATIBLE = "fully_compatible"
    MOSTLY_COMPATIBLE = "mostly_compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class GenesisFeatureStatus(Enum):
    """Status of individual Genesis features."""

    WORKING = "working"
    DEPRECATED = "deprecated"
    BROKEN = "broken"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass
class GenesisVersionInfo:
    """Genesis version and build information."""

    version: str
    build_date: Optional[str] = None
    commit_hash: Optional[str] = None
    api_level: Optional[int] = None
    features: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_genesis(cls) -> "GenesisVersionInfo":
        """Extract version info from Genesis module."""
        if not GENESIS_AVAILABLE:
            return cls(version="unavailable")

        version = getattr(gs, "__version__", "unknown")
        build_date = getattr(gs, "__build_date__", None)
        commit_hash = getattr(gs, "__commit__", None)

        # Try to get API level from Genesis
        api_level = None
        if hasattr(gs, "__api_level__"):
            api_level = gs.__api_level__
        elif hasattr(gs, "API_VERSION"):
            api_level = gs.API_VERSION

        # Extract feature information
        features = {}
        if hasattr(gs, "__features__"):
            features = gs.__features__
        else:
            # Manually detect key features
            features = {
                "rigid_solver": hasattr(gs, "RigidSolver"),
                "soft_solver": hasattr(gs, "SoftSolver"),
                "particle_solver": hasattr(gs, "ParticleSolver"),
                "scene_management": hasattr(gs, "Scene"),
                "material_system": hasattr(gs, "materials"),
                "visualization": hasattr(gs, "visualizer"),
                "gpu_acceleration": hasattr(gs, "cuda") or hasattr(gs, "gpu"),
            }

        return cls(
            version=version,
            build_date=build_date,
            commit_hash=commit_hash,
            api_level=api_level,
            features=features,
        )


@dataclass
class GenesisFeatureTest:
    """Individual Genesis feature test result."""

    feature_name: str
    status: GenesisFeatureStatus
    execution_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GenesisCompatibilityReport:
    """Comprehensive Genesis compatibility assessment."""

    version_info: GenesisVersionInfo
    compatibility_level: GenesisCompatibilityLevel
    feature_tests: List[GenesisFeatureTest]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    tested_at: datetime = field(default_factory=datetime.now)
    test_duration: float = 0.0

    def get_working_features(self) -> List[str]:
        """Get list of working features."""
        return [
            test.feature_name
            for test in self.feature_tests
            if test.status == GenesisFeatureStatus.WORKING
        ]

    def get_broken_features(self) -> List[str]:
        """Get list of broken features."""
        return [
            test.feature_name
            for test in self.feature_tests
            if test.status == GenesisFeatureStatus.BROKEN
        ]

    def get_compatibility_score(self) -> float:
        """Calculate compatibility score (0.0 to 1.0)."""
        if not self.feature_tests:
            return 0.0

        working_count = len(self.get_working_features())
        total_count = len(self.feature_tests)

        return working_count / total_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version_info": {
                "version": self.version_info.version,
                "build_date": self.version_info.build_date,
                "commit_hash": self.version_info.commit_hash,
                "api_level": self.version_info.api_level,
                "features": self.version_info.features,
            },
            "compatibility_level": self.compatibility_level.value,
            "compatibility_score": self.get_compatibility_score(),
            "feature_tests": [
                {
                    "feature_name": test.feature_name,
                    "status": test.status.value,
                    "execution_time": test.execution_time,
                    "error_message": test.error_message,
                    "details": test.details,
                    "timestamp": test.timestamp.isoformat(),
                }
                for test in self.feature_tests
            ],
            "performance_metrics": self.performance_metrics,
            "recommendations": self.recommendations,
            "tested_at": self.tested_at.isoformat(),
            "test_duration": self.test_duration,
            "working_features": self.get_working_features(),
            "broken_features": self.get_broken_features(),
        }


class GenesisAPIMonitor:
    """Monitors Genesis API compatibility and performance."""

    def __init__(self, report_dir: str = "genesis_reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._genesis_initialized = False

        # Feature test registry
        self.feature_tests = {
            "basic_import": self._test_basic_import,
            "scene_creation": self._test_scene_creation,
            "rigid_solver": self._test_rigid_solver,
            "entity_management": self._test_entity_management,
            "material_system": self._test_material_system,
            "simulation_step": self._test_simulation_step,
            "robot_loading": self._test_robot_loading,
            "visualization": self._test_visualization,
            "gpu_acceleration": self._test_gpu_acceleration,
            "performance_baseline": self._test_performance_baseline,
        }

    def _ensure_genesis_initialized(self) -> None:
        """Ensure Genesis is initialized (required for v0.2.1+)."""
        if not GENESIS_AVAILABLE:
            return

        if not self._genesis_initialized:
            try:
                if hasattr(gs, "init"):
                    gs.init()
                    self._genesis_initialized = True
                    self.logger.debug("Genesis initialized successfully")
            except Exception as e:
                # Genesis may already be initialized or initialization failed
                self.logger.debug(f"Genesis initialization attempt: {e}")
                self._genesis_initialized = True  # Assume it's already initialized

    async def monitor_compatibility(self) -> GenesisCompatibilityReport:
        """Run comprehensive Genesis compatibility monitoring."""
        start_time = time.time()

        # Get version information
        version_info = GenesisVersionInfo.from_genesis()
        self.logger.info(f"Testing Genesis version: {version_info.version}")

        # Run feature tests
        feature_results = []
        performance_metrics = {}

        for feature_name, test_func in self.feature_tests.items():
            self.logger.info(f"Testing feature: {feature_name}")

            try:
                test_start = time.time()
                result = await self._run_feature_test(test_func, feature_name)
                test_duration = time.time() - test_start

                result.execution_time = test_duration
                feature_results.append(result)

                # Collect performance metrics
                if result.details and "performance" in result.details:
                    performance_metrics[feature_name] = result.details["performance"]

            except Exception as e:
                self.logger.error(f"Feature test {feature_name} failed: {e}")
                feature_results.append(
                    GenesisFeatureTest(
                        feature_name=feature_name,
                        status=GenesisFeatureStatus.BROKEN,
                        execution_time=0.0,
                        error_message=str(e),
                    )
                )

        # Determine overall compatibility
        compatibility_level = self._assess_compatibility(feature_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(feature_results, version_info)

        # Create comprehensive report
        report = GenesisCompatibilityReport(
            version_info=version_info,
            compatibility_level=compatibility_level,
            feature_tests=feature_results,
            performance_metrics=performance_metrics,
            recommendations=recommendations,
            test_duration=time.time() - start_time,
        )

        # Save report
        await self._save_report(report)

        return report

    async def _run_feature_test(
        self, test_func, feature_name: str
    ) -> GenesisFeatureTest:
        """Run individual feature test with error handling."""
        try:
            if asyncio.iscoroutinefunction(test_func):
                return await test_func()
            else:
                return test_func()
        except Exception as e:
            return GenesisFeatureTest(
                feature_name=feature_name,
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_basic_import(self) -> GenesisFeatureTest:
        """Test basic Genesis import functionality."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="basic_import",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis module not available",
            )

        # Test basic module access
        try:
            # Check core classes
            core_classes = ["Scene", "RigidSolver", "materials"]
            missing_classes = []

            for cls_name in core_classes:
                if not hasattr(gs, cls_name):
                    missing_classes.append(cls_name)

            if missing_classes:
                return GenesisFeatureTest(
                    feature_name="basic_import",
                    status=GenesisFeatureStatus.PARTIALLY_COMPATIBLE,
                    execution_time=0.0,
                    details={"missing_classes": missing_classes},
                )

            return GenesisFeatureTest(
                feature_name="basic_import",
                status=GenesisFeatureStatus.WORKING,
                execution_time=0.0,
                details={"available_classes": core_classes},
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="basic_import",
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_scene_creation(self) -> GenesisFeatureTest:
        """Test Genesis scene creation."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="scene_creation",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis not available",
            )

        try:
            start_time = time.time()

            # Ensure Genesis is initialized (required for v0.2.1+)
            self._ensure_genesis_initialized()

            # Test scene creation with different configurations
            scene_configs = [
                {},  # Default config
                {"show_viewer": False},  # Headless mode
            ]

            successful_configs = []
            failed_configs = []

            for config in scene_configs:
                try:
                    scene = gs.Scene(**config)
                    successful_configs.append(config)
                    # Clean up
                    if hasattr(scene, "close"):
                        scene.close()
                except Exception as e:
                    failed_configs.append({"config": config, "error": str(e)})

            execution_time = time.time() - start_time

            if len(successful_configs) == len(scene_configs):
                status = GenesisFeatureStatus.WORKING
            elif successful_configs:
                status = GenesisFeatureStatus.PARTIALLY_COMPATIBLE
            else:
                status = GenesisFeatureStatus.BROKEN

            return GenesisFeatureTest(
                feature_name="scene_creation",
                status=status,
                execution_time=execution_time,
                details={
                    "successful_configs": successful_configs,
                    "failed_configs": failed_configs,
                    "genesis_initialized": hasattr(gs, "init"),
                },
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="scene_creation",
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_rigid_solver(self) -> GenesisFeatureTest:
        """Test Genesis rigid body solver."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="rigid_solver",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis not available",
            )

        try:
            start_time = time.time()

            # Ensure Genesis is initialized (required for v0.2.1+)
            self._ensure_genesis_initialized()

            # Create scene and test solver
            scene = gs.Scene(show_viewer=False)

            # Test solver configuration
            solver_params = {"dt": 0.01, "substeps": 10}

            # Check if solver can be configured
            solver_working = True
            solver_details = {}

            try:
                # Test basic solver functionality
                scene.build()
                solver_details["build_successful"] = True
            except Exception as e:
                solver_working = False
                solver_details["build_error"] = str(e)

            # Test step functionality
            try:
                scene.step()
                solver_details["step_successful"] = True
            except Exception as e:
                solver_working = False
                solver_details["step_error"] = str(e)

            execution_time = time.time() - start_time

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            status = (
                GenesisFeatureStatus.WORKING
                if solver_working
                else GenesisFeatureStatus.BROKEN
            )

            return GenesisFeatureTest(
                feature_name="rigid_solver",
                status=status,
                execution_time=execution_time,
                details=solver_details,
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="rigid_solver",
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_entity_management(self) -> GenesisFeatureTest:
        """Test Genesis entity management."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="entity_management",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis not available",
            )

        try:
            start_time = time.time()

            # Ensure Genesis is initialized (required for v0.2.1+)
            self._ensure_genesis_initialized()

            scene = gs.Scene(show_viewer=False)

            entity_tests = {}

            # Test basic entity creation
            try:
                # Test plane creation
                plane = gs.morphs.Plane()
                scene.add_entity(plane)
                entity_tests["plane_creation"] = True
            except Exception as e:
                entity_tests["plane_creation"] = False
                entity_tests["plane_error"] = str(e)

            # Test box creation
            try:
                box = gs.morphs.Box(size=(1, 1, 1))
                scene.add_entity(box)
                entity_tests["box_creation"] = True
            except Exception as e:
                entity_tests["box_creation"] = False
                entity_tests["box_error"] = str(e)

            execution_time = time.time() - start_time

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            # Determine status
            successful_tests = sum(
                1 for k, v in entity_tests.items() if k.endswith("_creation") and v
            )
            total_tests = sum(1 for k in entity_tests.keys() if k.endswith("_creation"))

            if successful_tests == total_tests:
                status = GenesisFeatureStatus.WORKING
            elif successful_tests > 0:
                status = GenesisFeatureStatus.PARTIALLY_COMPATIBLE
            else:
                status = GenesisFeatureStatus.BROKEN

            return GenesisFeatureTest(
                feature_name="entity_management",
                status=status,
                execution_time=execution_time,
                details=entity_tests,
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="entity_management",
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_material_system(self) -> GenesisFeatureTest:
        """Test Genesis material system."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="material_system",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis not available",
            )

        try:
            start_time = time.time()

            material_tests = {}

            # Test basic material access
            try:
                if hasattr(gs, "materials"):
                    materials_module = gs.materials
                    material_tests["materials_module"] = True

                    # Test specific materials if available
                    common_materials = ["Rigid", "Soft", "Liquid"]
                    available_materials = []

                    for mat_name in common_materials:
                        if hasattr(materials_module, mat_name):
                            available_materials.append(mat_name)

                    material_tests["available_materials"] = available_materials
                else:
                    material_tests["materials_module"] = False

            except Exception as e:
                material_tests["materials_module"] = False
                material_tests["materials_error"] = str(e)

            execution_time = time.time() - start_time

            # Determine status
            if material_tests.get("materials_module", False):
                status = GenesisFeatureStatus.WORKING
            else:
                status = GenesisFeatureStatus.BROKEN

            return GenesisFeatureTest(
                feature_name="material_system",
                status=status,
                execution_time=execution_time,
                details=material_tests,
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="material_system",
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_simulation_step(self) -> GenesisFeatureTest:
        """Test Genesis simulation stepping."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="simulation_step",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis not available",
            )

        try:
            start_time = time.time()

            # Ensure Genesis is initialized (required for v0.2.1+)
            self._ensure_genesis_initialized()

            scene = gs.Scene(show_viewer=False)

            # Add basic entities
            plane = gs.morphs.Plane()
            scene.add_entity(plane)

            # Build scene
            scene.build()

            # Test multiple simulation steps
            step_tests = {}
            step_count = 10
            step_times = []

            for i in range(step_count):
                step_start = time.time()
                try:
                    scene.step()
                    step_duration = time.time() - step_start
                    step_times.append(step_duration)
                except Exception as e:
                    step_tests["step_error"] = str(e)
                    break

            execution_time = time.time() - start_time

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            # Calculate performance metrics
            if step_times:
                avg_step_time = sum(step_times) / len(step_times)
                max_step_time = max(step_times)
                min_step_time = min(step_times)

                step_tests.update(
                    {
                        "successful_steps": len(step_times),
                        "avg_step_time": avg_step_time,
                        "max_step_time": max_step_time,
                        "min_step_time": min_step_time,
                        "performance": 1.0 / avg_step_time,  # Steps per second
                    }
                )

                status = GenesisFeatureStatus.WORKING
            else:
                status = GenesisFeatureStatus.BROKEN

            return GenesisFeatureTest(
                feature_name="simulation_step",
                status=status,
                execution_time=execution_time,
                details=step_tests,
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="simulation_step",
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_robot_loading(self) -> GenesisFeatureTest:
        """Test Genesis robot loading capabilities."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="robot_loading",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis not available",
            )

        try:
            start_time = time.time()

            # Ensure Genesis is initialized (required for v0.2.1+)
            self._ensure_genesis_initialized()

            scene = gs.Scene(show_viewer=False)
            robot_tests = {}

            # Test URDF loading capability
            try:
                # Check if URDF loading is available
                if hasattr(gs, "morphs") and hasattr(gs.morphs, "URDF"):
                    robot_tests["urdf_support"] = True

                    # Try to load a simple robot if available
                    # Note: This is a capability test, not actual robot loading
                    robot_tests["urdf_loader_available"] = True
                else:
                    robot_tests["urdf_support"] = False

            except Exception as e:
                robot_tests["urdf_support"] = False
                robot_tests["urdf_error"] = str(e)

            # Test articulated body support
            try:
                if hasattr(gs, "RigidEntity") or hasattr(gs, "ArticulatedEntity"):
                    robot_tests["articulated_support"] = True
                else:
                    robot_tests["articulated_support"] = False
            except Exception as e:
                robot_tests["articulated_support"] = False
                robot_tests["articulated_error"] = str(e)

            execution_time = time.time() - start_time

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            # Determine status
            if robot_tests.get("urdf_support", False):
                status = GenesisFeatureStatus.WORKING
            else:
                status = GenesisFeatureStatus.PARTIALLY_COMPATIBLE

            return GenesisFeatureTest(
                feature_name="robot_loading",
                status=status,
                execution_time=execution_time,
                details=robot_tests,
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="robot_loading",
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_visualization(self) -> GenesisFeatureTest:
        """Test Genesis visualization capabilities."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="visualization",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis not available",
            )

        try:
            start_time = time.time()

            # Ensure Genesis is initialized (required for v0.2.1+)
            self._ensure_genesis_initialized()

            viz_tests = {}

            # Test headless mode (most important for production)
            try:
                scene = gs.Scene(show_viewer=False)
                scene.build()
                viz_tests["headless_mode"] = True

                if hasattr(scene, "close"):
                    scene.close()

            except Exception as e:
                viz_tests["headless_mode"] = False
                viz_tests["headless_error"] = str(e)

            # Test visualization module availability
            try:
                if hasattr(gs, "visualizer") or hasattr(gs, "vis"):
                    viz_tests["visualizer_available"] = True
                else:
                    viz_tests["visualizer_available"] = False
            except Exception as e:
                viz_tests["visualizer_available"] = False
                viz_tests["visualizer_error"] = str(e)

            execution_time = time.time() - start_time

            # Determine status - headless mode is most critical
            if viz_tests.get("headless_mode", False):
                status = GenesisFeatureStatus.WORKING
            else:
                status = GenesisFeatureStatus.BROKEN

            return GenesisFeatureTest(
                feature_name="visualization",
                status=status,
                execution_time=execution_time,
                details=viz_tests,
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="visualization",
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_gpu_acceleration(self) -> GenesisFeatureTest:
        """Test Genesis GPU acceleration."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="gpu_acceleration",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis not available",
            )

        try:
            start_time = time.time()

            # Ensure Genesis is initialized (required for v0.2.1+)
            self._ensure_genesis_initialized()

            gpu_tests = {}

            # Test CUDA availability
            try:
                if hasattr(gs, "cuda") or hasattr(gs, "gpu"):
                    gpu_tests["gpu_module_available"] = True
                else:
                    gpu_tests["gpu_module_available"] = False
            except Exception as e:
                gpu_tests["gpu_module_available"] = False
                gpu_tests["gpu_error"] = str(e)

            # Test device detection
            try:
                # Try to create scene with GPU backend if available
                scene = gs.Scene(show_viewer=False)
                gpu_tests["scene_creation_success"] = True

                if hasattr(scene, "close"):
                    scene.close()

            except Exception as e:
                gpu_tests["scene_creation_success"] = False
                gpu_tests["scene_error"] = str(e)

            execution_time = time.time() - start_time

            # GPU is optional but beneficial
            if gpu_tests.get("gpu_module_available", False):
                status = GenesisFeatureStatus.WORKING
            else:
                status = GenesisFeatureStatus.PARTIALLY_COMPATIBLE  # CPU fallback

            return GenesisFeatureTest(
                feature_name="gpu_acceleration",
                status=status,
                execution_time=execution_time,
                details=gpu_tests,
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="gpu_acceleration",
                status=GenesisFeatureStatus.UNKNOWN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _test_performance_baseline(self) -> GenesisFeatureTest:
        """Test Genesis performance baseline."""
        if not GENESIS_AVAILABLE:
            return GenesisFeatureTest(
                feature_name="performance_baseline",
                status=GenesisFeatureStatus.MISSING,
                execution_time=0.0,
                error_message="Genesis not available",
            )

        try:
            start_time = time.time()

            # Ensure Genesis is initialized (required for v0.2.1+)
            self._ensure_genesis_initialized()

            # Create performance test scenario
            scene = gs.Scene(show_viewer=False)

            # Add multiple entities for realistic performance test
            for i in range(10):
                box = gs.morphs.Box(size=(0.1, 0.1, 0.1))
                scene.add_entity(box, pos=(i * 0.2, 0, 1.0))

            # Build scene
            build_start = time.time()
            scene.build()
            build_time = time.time() - build_start

            # Run simulation steps and measure performance
            step_times = []
            num_steps = 100

            for _ in range(num_steps):
                step_start = time.time()
                scene.step()
                step_duration = time.time() - step_start
                step_times.append(step_duration)

            execution_time = time.time() - start_time

            # Clean up
            if hasattr(scene, "close"):
                scene.close()

            # Calculate performance metrics
            avg_step_time = sum(step_times) / len(step_times)
            fps = 1.0 / avg_step_time
            total_sim_time = sum(step_times)

            performance_details = {
                "build_time": build_time,
                "avg_step_time": avg_step_time,
                "fps": fps,
                "total_simulation_time": total_sim_time,
                "num_entities": 10,
                "num_steps": num_steps,
                "performance": fps,  # Main performance metric
            }

            # Determine performance status
            if fps > 50:  # Good performance
                status = GenesisFeatureStatus.WORKING
            elif fps > 10:  # Acceptable performance
                status = GenesisFeatureStatus.PARTIALLY_COMPATIBLE
            else:  # Poor performance
                status = GenesisFeatureStatus.BROKEN

            return GenesisFeatureTest(
                feature_name="performance_baseline",
                status=status,
                execution_time=execution_time,
                details=performance_details,
            )

        except Exception as e:
            return GenesisFeatureTest(
                feature_name="performance_baseline",
                status=GenesisFeatureStatus.BROKEN,
                execution_time=0.0,
                error_message=str(e),
            )

    def _assess_compatibility(
        self, feature_tests: List[GenesisFeatureTest]
    ) -> GenesisCompatibilityLevel:
        """Assess overall Genesis compatibility level."""
        if not feature_tests:
            return GenesisCompatibilityLevel.UNKNOWN

        # Count feature statuses
        status_counts = {}
        for test in feature_tests:
            status = test.status
            status_counts[status] = status_counts.get(status, 0) + 1

        total_tests = len(feature_tests)
        working_tests = status_counts.get(GenesisFeatureStatus.WORKING, 0)
        broken_tests = status_counts.get(GenesisFeatureStatus.BROKEN, 0)
        missing_tests = status_counts.get(GenesisFeatureStatus.MISSING, 0)

        # Calculate compatibility percentage
        compatibility_ratio = working_tests / total_tests

        # Determine compatibility level
        if missing_tests > total_tests * 0.5:
            return GenesisCompatibilityLevel.INCOMPATIBLE
        elif compatibility_ratio >= 0.9:
            return GenesisCompatibilityLevel.FULLY_COMPATIBLE
        elif compatibility_ratio >= 0.7:
            return GenesisCompatibilityLevel.MOSTLY_COMPATIBLE
        elif compatibility_ratio >= 0.4:
            return GenesisCompatibilityLevel.PARTIALLY_COMPATIBLE
        else:
            return GenesisCompatibilityLevel.INCOMPATIBLE

    def _generate_recommendations(
        self, feature_tests: List[GenesisFeatureTest], version_info: GenesisVersionInfo
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Version-specific recommendations
        if version_info.version == "unavailable":
            recommendations.append(
                "Install Genesis physics engine: pip install genesis-world"
            )
            return recommendations

        # Feature-specific recommendations
        broken_features = [
            test.feature_name
            for test in feature_tests
            if test.status == GenesisFeatureStatus.BROKEN
        ]

        if "basic_import" in broken_features:
            recommendations.append(
                "Genesis installation appears corrupted - reinstall Genesis"
            )

        if "scene_creation" in broken_features:
            recommendations.append(
                "Scene creation failing - check Genesis configuration and dependencies"
            )

        if "rigid_solver" in broken_features:
            recommendations.append(
                "Rigid body solver issues - may need Genesis version update"
            )

        if "performance_baseline" in broken_features:
            recommendations.append(
                "Poor performance detected - consider GPU acceleration or system upgrade"
            )

        # Performance recommendations
        performance_tests = [
            test
            for test in feature_tests
            if test.feature_name == "performance_baseline"
            and test.status == GenesisFeatureStatus.WORKING
        ]

        if performance_tests:
            perf_test = performance_tests[0]
            fps = perf_test.details.get("fps", 0)

            if fps < 30:
                recommendations.append(
                    "Consider enabling GPU acceleration for better performance"
                )
            if fps < 10:
                recommendations.append(
                    "Performance is critically low - check system requirements"
                )

        # GPU recommendations
        gpu_tests = [
            test for test in feature_tests if test.feature_name == "gpu_acceleration"
        ]
        if gpu_tests and gpu_tests[0].status != GenesisFeatureStatus.WORKING:
            recommendations.append(
                "GPU acceleration not available - training will be slower"
            )

        # General recommendations
        working_count = sum(
            1 for test in feature_tests if test.status == GenesisFeatureStatus.WORKING
        )
        total_count = len(feature_tests)

        if working_count / total_count < 0.7:
            recommendations.append(
                "Multiple features failing - consider Genesis version downgrade/upgrade"
            )

        if not recommendations:
            recommendations.append(
                "Genesis appears fully compatible - ready for production use"
            )

        return recommendations

    async def _save_report(self, report: GenesisCompatibilityReport) -> None:
        """Save compatibility report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_dir / f"genesis_compatibility_{timestamp}.json"

        try:
            with open(report_file, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

            self.logger.info(f"Genesis compatibility report saved to {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to save compatibility report: {e}")

    def get_latest_report(self) -> Optional[GenesisCompatibilityReport]:
        """Get the most recent compatibility report."""
        try:
            report_files = list(self.report_dir.glob("genesis_compatibility_*.json"))
            if not report_files:
                return None

            # Get most recent report
            latest_file = max(report_files, key=lambda f: f.stat().st_mtime)

            with open(latest_file, "r") as f:
                report_data = json.load(f)

            # Convert back to report object (simplified)
            return report_data

        except Exception as e:
            self.logger.error(f"Failed to load latest report: {e}")
            return None


# Convenience functions for easy usage
async def monitor_genesis_compatibility(
    report_dir: str = "genesis_reports",
) -> GenesisCompatibilityReport:
    """Run Genesis compatibility monitoring and return report."""
    monitor = GenesisAPIMonitor(report_dir)
    return await monitor.monitor_compatibility()


def check_genesis_status() -> Dict[str, Any]:
    """Quick Genesis status check."""
    if not GENESIS_AVAILABLE:
        return {"available": False, "version": "unavailable", "status": "not_installed"}

    try:
        version_info = GenesisVersionInfo.from_genesis()
        return {
            "available": True,
            "version": version_info.version,
            "api_level": version_info.api_level,
            "features": version_info.features,
            "status": "available",
        }
    except Exception as e:
        return {
            "available": False,
            "version": "error",
            "status": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    # Run compatibility monitoring
    import asyncio

    async def main():
        print("Running Genesis API compatibility monitoring...")
        report = await monitor_genesis_compatibility()

        print(f"\nGenesis Version: {report.version_info.version}")
        print(f"Compatibility Level: {report.compatibility_level.value}")
        print(f"Compatibility Score: {report.get_compatibility_score():.2%}")
        print(f"Test Duration: {report.test_duration:.2f}s")

        print(f"\nWorking Features ({len(report.get_working_features())}):")
        for feature in report.get_working_features():
            print(f"  ✓ {feature}")

        if report.get_broken_features():
            print(f"\nBroken Features ({len(report.get_broken_features())}):")
            for feature in report.get_broken_features():
                print(f"  ✗ {feature}")

        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")

    asyncio.run(main())
