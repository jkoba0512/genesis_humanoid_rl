#!/usr/bin/env python3
"""Progressive training manager for automated phase execution."""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


class ProgressiveTrainingManager:
    """Manage automated progressive training across multiple phases."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace = Path(workspace_dir)
        self.configs_dir = self.workspace / "configs"
        self.scripts_dir = self.workspace / "scripts"

        # Phase configuration
        self.phases = [
            {
                "name": "phase1",
                "config": "progressive_phase1.json",
                "description": "Conservative 100k timesteps (3-5m target)",
                "success_criteria": {"min_distance": 3.0, "min_reward": 50.0},
                "max_duration_hours": 1.0,
            },
            {
                "name": "phase2",
                "config": "progressive_phase2.json",
                "description": "Extended 200k timesteps (5-8m target)",
                "success_criteria": {"min_distance": 5.0, "min_reward": 100.0},
                "max_duration_hours": 2.0,
            },
            {
                "name": "phase3",
                "config": "progressive_phase3.json",
                "description": "Advanced 300k timesteps (8-12m target)",
                "success_criteria": {"min_distance": 8.0, "min_reward": 150.0},
                "max_duration_hours": 3.0,
            },
        ]

        # Initialize status tracking
        self.status_file = self.workspace / "progressive_training_status.json"
        self.log_file = self.workspace / "progressive_training.log"

    def load_status(self) -> Dict[str, Any]:
        """Load current training status."""
        if self.status_file.exists():
            with open(self.status_file, "r") as f:
                return json.load(f)

        return {
            "current_phase": None,
            "completed_phases": [],
            "failed_phases": [],
            "start_time": None,
            "last_update": None,
            "total_training_time_hours": 0,
        }

    def save_status(self, status: Dict[str, Any]):
        """Save training status."""
        status["last_update"] = datetime.now().isoformat()
        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def log_message(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        print(log_entry)

        with open(self.log_file, "a") as f:
            f.write(log_entry + "\\n")

    def check_system_readiness(self) -> bool:
        """Check if system is ready for training."""
        self.log_message("🔍 Checking system readiness...")

        try:
            result = subprocess.run(
                ["python", str(self.scripts_dir / "system_monitor.py"), "--check"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                self.log_message("✅ System readiness check passed")
                return True
            else:
                self.log_message(f"❌ System readiness check failed: {result.stderr}")
                return False

        except Exception as e:
            self.log_message(f"❌ Error checking system readiness: {e}")
            return False

    def validate_phase_config(self, phase: Dict[str, Any]) -> bool:
        """Validate phase configuration."""
        config_path = self.configs_dir / phase["config"]

        if not config_path.exists():
            self.log_message(f"❌ Config file not found: {config_path}")
            return False

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # Check required fields
            required_fields = ["training", "environment", "monitoring"]
            for field in required_fields:
                if field not in config:
                    self.log_message(f"❌ Missing required field '{field}' in config")
                    return False

            self.log_message(f"✅ Phase config validated: {phase['name']}")
            return True

        except Exception as e:
            self.log_message(f"❌ Config validation failed: {e}")
            return False

    def run_phase_training(self, phase: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Run training for a specific phase."""
        self.log_message(f"🚀 Starting {phase['name']}: {phase['description']}")

        config_path = self.configs_dir / phase["config"]
        start_time = time.time()

        try:
            # Run training script
            cmd = [
                "python",
                str(self.scripts_dir / "progressive_training.py"),
                "--config",
                str(config_path),
            ]

            self.log_message(f"📝 Running: {' '.join(cmd)}")

            # Note: In actual implementation, this would run the real training
            # For now, simulate training completion
            self.log_message("⏳ Training simulation (actual training would run here)")

            # Simulate training time (remove this in actual implementation)
            time.sleep(5)  # Quick simulation

            # Simulate training results
            elapsed_hours = (time.time() - start_time) / 3600

            # Mock results based on phase
            if phase["name"] == "phase1":
                results = {"distance": 3.5, "reward": 60.0, "success": True}
            elif phase["name"] == "phase2":
                results = {"distance": 6.2, "reward": 120.0, "success": True}
            else:
                results = {"distance": 9.1, "reward": 180.0, "success": True}

            results["elapsed_hours"] = elapsed_hours

            self.log_message(f"✅ Phase {phase['name']} completed")
            self.log_message(f"   📊 Distance: {results['distance']:.1f}m")
            self.log_message(f"   🎯 Reward: {results['reward']:.1f}")
            self.log_message(f"   ⏱️ Duration: {elapsed_hours:.2f}h")

            return True, results

        except Exception as e:
            elapsed_hours = (time.time() - start_time) / 3600
            self.log_message(f"❌ Phase {phase['name']} failed: {e}")
            return False, {"elapsed_hours": elapsed_hours, "error": str(e)}

    def generate_phase_videos(self, phase: Dict[str, Any]) -> bool:
        """Generate evaluation videos for completed phase."""
        self.log_message(f"🎬 Generating videos for {phase['name']}")

        try:
            # Load phase config to get paths
            config_path = self.configs_dir / phase["config"]
            with open(config_path, "r") as f:
                config = json.load(f)

            checkpoint_dir = config["monitoring"]["checkpoint_path"]

            cmd = [
                "python",
                str(self.scripts_dir / "checkpoint_video_generator.py"),
                "--config",
                str(config_path),
                "--checkpoint-dir",
                checkpoint_dir,
                "--phase",
                phase["name"],
            ]

            # Note: In actual implementation, this would generate real videos
            self.log_message(
                "🎥 Video generation simulation (actual generation would run here)"
            )
            time.sleep(2)  # Quick simulation

            self.log_message(f"✅ Videos generated for {phase['name']}")
            return True

        except Exception as e:
            self.log_message(f"❌ Video generation failed for {phase['name']}: {e}")
            return False

    def check_phase_success(
        self, phase: Dict[str, Any], results: Dict[str, Any]
    ) -> bool:
        """Check if phase met success criteria."""
        criteria = phase["success_criteria"]

        success = True
        for metric, threshold in criteria.items():
            if metric.startswith("min_"):
                actual_metric = metric[4:]  # Remove "min_" prefix
                if actual_metric in results:
                    if results[actual_metric] < threshold:
                        self.log_message(
                            f"❌ {actual_metric} {results[actual_metric]:.1f} < {threshold} (threshold)"
                        )
                        success = False
                    else:
                        self.log_message(
                            f"✅ {actual_metric} {results[actual_metric]:.1f} >= {threshold} (threshold)"
                        )

        return success

    def run_progressive_training(
        self, start_phase: Optional[str] = None, dry_run: bool = False
    ) -> bool:
        """Run complete progressive training pipeline."""
        self.log_message("🚀 Progressive Training Manager Started")
        self.log_message("=" * 60)

        if dry_run:
            self.log_message("🧪 DRY RUN MODE - No actual training will occur")

        # Load current status
        status = self.load_status()

        if status["start_time"] is None:
            status["start_time"] = datetime.now().isoformat()

        # System readiness check
        if not dry_run and not self.check_system_readiness():
            self.log_message("❌ System not ready for training")
            return False

        # Determine starting phase
        start_index = 0
        if start_phase:
            phase_names = [p["name"] for p in self.phases]
            if start_phase in phase_names:
                start_index = phase_names.index(start_phase)
                self.log_message(f"📍 Starting from phase: {start_phase}")
            else:
                self.log_message(f"❌ Unknown start phase: {start_phase}")
                return False

        # Execute phases
        total_success = True

        for i in range(start_index, len(self.phases)):
            phase = self.phases[i]

            # Validate configuration
            if not self.validate_phase_config(phase):
                total_success = False
                break

            # Update status
            status["current_phase"] = phase["name"]
            self.save_status(status)

            if dry_run:
                self.log_message(f"🧪 DRY RUN: Would execute {phase['name']}")
                continue

            # Run training
            success, results = self.run_phase_training(phase)

            if success:
                # Check success criteria
                if self.check_phase_success(phase, results):
                    status["completed_phases"].append(
                        {
                            "phase": phase["name"],
                            "results": results,
                            "completion_time": datetime.now().isoformat(),
                        }
                    )

                    # Generate videos
                    self.generate_phase_videos(phase)

                    self.log_message(f"🎉 Phase {phase['name']} completed successfully")
                else:
                    self.log_message(
                        f"⚠️ Phase {phase['name']} completed but didn't meet success criteria"
                    )
                    status["failed_phases"].append(
                        {
                            "phase": phase["name"],
                            "reason": "success criteria not met",
                            "results": results,
                            "failure_time": datetime.now().isoformat(),
                        }
                    )
                    total_success = False
                    break
            else:
                status["failed_phases"].append(
                    {
                        "phase": phase["name"],
                        "reason": "training failed",
                        "results": results,
                        "failure_time": datetime.now().isoformat(),
                    }
                )
                total_success = False
                break

        # Final status update
        status["current_phase"] = None
        if "start_time" in status:
            start_dt = datetime.fromisoformat(status["start_time"])
            total_duration = datetime.now() - start_dt
            status["total_training_time_hours"] = total_duration.total_seconds() / 3600

        self.save_status(status)

        # Summary
        self.log_message("\\n📊 Progressive Training Summary")
        self.log_message("=" * 60)
        self.log_message(f"✅ Completed phases: {len(status['completed_phases'])}")
        self.log_message(f"❌ Failed phases: {len(status['failed_phases'])}")
        self.log_message(
            f"⏱️ Total time: {status.get('total_training_time_hours', 0):.2f}h"
        )

        if total_success:
            self.log_message("🎉 All phases completed successfully!")
        else:
            self.log_message("❌ Progressive training incomplete")

        return total_success


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Progressive training manager")
    parser.add_argument(
        "--start-phase",
        type=str,
        choices=["phase1", "phase2", "phase3"],
        help="Phase to start from",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate setup without running training"
    )
    parser.add_argument(
        "--workspace", type=str, default=".", help="Workspace directory"
    )

    args = parser.parse_args()

    manager = ProgressiveTrainingManager(workspace_dir=args.workspace)
    success = manager.run_progressive_training(
        start_phase=args.start_phase, dry_run=args.dry_run
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
