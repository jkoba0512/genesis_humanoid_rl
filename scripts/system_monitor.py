#!/usr/bin/env python3
"""System monitoring and resource management for progressive training."""

import psutil
import GPUtil
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class SystemMonitor:
    """Monitor system resources during training."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else None
        self.start_time = time.time()

        # Resource thresholds
        self.memory_warning_gb = 6.0
        self.memory_critical_gb = 7.0
        self.gpu_memory_warning_percent = 80
        self.gpu_memory_critical_percent = 90
        self.cpu_warning_percent = 85

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_used_gb = memory.used / (1024**3)
        memory_available_gb = memory.available / (1024**3)

        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # GPU info
        gpu_info = {}
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                gpu_info = {
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_free_mb": gpu.memoryFree,
                    "memory_usage_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_usage_percent": gpu.load * 100,
                    "temperature": gpu.temperature,
                }
        except Exception as e:
            gpu_info = {"error": str(e)}

        # Disk info
        disk = psutil.disk_usage("/")
        disk_info = {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "usage_percent": (disk.used / disk.total) * 100,
        }

        # Process info for current Python process
        process = psutil.Process()
        process_info = {
            "memory_mb": process.memory_info().rss / (1024**2),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "connections": len(process.connections()),
        }

        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime_minutes": (time.time() - self.start_time) / 60,
            "memory": {
                "total_gb": memory_gb,
                "used_gb": memory_used_gb,
                "available_gb": memory_available_gb,
                "usage_percent": memory.percent,
            },
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "load_avg": (
                    list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else None
                ),
            },
            "gpu": gpu_info,
            "disk": disk_info,
            "process": process_info,
        }

        return status

    def check_resource_warnings(self, status: Dict[str, Any]) -> Dict[str, str]:
        """Check for resource warnings and return alerts."""
        alerts = {}

        # Memory warnings
        memory_used_gb = status["memory"]["used_gb"]
        if memory_used_gb > self.memory_critical_gb:
            alerts["memory"] = "CRITICAL"
        elif memory_used_gb > self.memory_warning_gb:
            alerts["memory"] = "WARNING"

        # CPU warnings
        cpu_percent = status["cpu"]["usage_percent"]
        if cpu_percent > self.cpu_warning_percent:
            alerts["cpu"] = "WARNING"

        # GPU warnings
        if "memory_usage_percent" in status["gpu"]:
            gpu_memory_percent = status["gpu"]["memory_usage_percent"]
            if gpu_memory_percent > self.gpu_memory_critical_percent:
                alerts["gpu_memory"] = "CRITICAL"
            elif gpu_memory_percent > self.gpu_memory_warning_percent:
                alerts["gpu_memory"] = "WARNING"

        # Disk warnings
        if status["disk"]["usage_percent"] > 90:
            alerts["disk"] = "WARNING"

        return alerts

    def log_status(self, status: Dict[str, Any]):
        """Log system status to file."""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing log or create new
            log_data = []
            if self.log_file.exists():
                try:
                    with open(self.log_file, "r") as f:
                        log_data = json.load(f)
                except:
                    log_data = []

            # Add current status
            log_data.append(status)

            # Keep only last 1000 entries
            if len(log_data) > 1000:
                log_data = log_data[-1000:]

            # Save log
            with open(self.log_file, "w") as f:
                json.dump(log_data, f, indent=2)

    def print_status(self, status: Dict[str, Any], alerts: Dict[str, str]):
        """Print formatted status to console."""
        print(f"\n🔍 System Status - {status['timestamp']}")
        print("=" * 60)

        # Memory
        mem = status["memory"]
        mem_status = (
            "🔴"
            if "memory" in alerts and alerts["memory"] == "CRITICAL"
            else "🟡" if "memory" in alerts else "🟢"
        )
        print(
            f"{mem_status} Memory: {mem['used_gb']:.1f}/{mem['total_gb']:.1f}GB "
            f"({mem['usage_percent']:.1f}%) | Available: {mem['available_gb']:.1f}GB"
        )

        # CPU
        cpu = status["cpu"]
        cpu_status = "🟡" if "cpu" in alerts else "🟢"
        print(f"{cpu_status} CPU: {cpu['usage_percent']:.1f}% ({cpu['count']} cores)")

        # GPU
        gpu = status["gpu"]
        if "error" not in gpu:
            gpu_status = (
                "🔴"
                if "gpu_memory" in alerts and alerts["gpu_memory"] == "CRITICAL"
                else "🟡" if "gpu_memory" in alerts else "🟢"
            )
            print(
                f"{gpu_status} GPU: {gpu['name']} | "
                f"Memory: {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f}MB "
                f"({gpu['memory_usage_percent']:.1f}%) | "
                f"Usage: {gpu['gpu_usage_percent']:.1f}%"
            )
        else:
            print(f"🔴 GPU: Error - {gpu['error']}")

        # Process
        proc = status["process"]
        print(
            f"🔧 Process: {proc['memory_mb']:.0f}MB | "
            f"CPU: {proc['cpu_percent']:.1f}% | "
            f"Threads: {proc['num_threads']}"
        )

        # Alerts
        if alerts:
            print(f"\n⚠️ ALERTS:")
            for resource, level in alerts.items():
                print(f"   {level}: {resource.upper()}")

    def monitor_continuously(
        self, interval_seconds: int = 30, max_duration_minutes: Optional[int] = None
    ):
        """Monitor system continuously."""
        print(f"🔍 Starting continuous monitoring (interval: {interval_seconds}s)")
        if max_duration_minutes:
            print(f"   Max duration: {max_duration_minutes} minutes")

        start_time = time.time()

        try:
            while True:
                status = self.get_system_status()
                alerts = self.check_resource_warnings(status)

                self.print_status(status, alerts)
                self.log_status(status)

                # Check duration limit
                if max_duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= max_duration_minutes:
                        print(
                            f"\n⏰ Monitoring duration limit reached ({max_duration_minutes} minutes)"
                        )
                        break

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n⚠️ Monitoring stopped by user")


def check_training_readiness() -> bool:
    """Check if system is ready for training."""
    print("🔍 Checking training readiness...")

    monitor = SystemMonitor()
    status = monitor.get_system_status()
    alerts = monitor.check_resource_warnings(status)

    monitor.print_status(status, alerts)

    # Check minimum requirements
    requirements_met = True

    # Memory requirement: at least 4GB available
    if status["memory"]["available_gb"] < 4.0:
        print("❌ Insufficient available memory (need 4GB+)")
        requirements_met = False

    # GPU requirement: at least 2GB free
    if "memory_free_mb" in status["gpu"]:
        if status["gpu"]["memory_free_mb"] < 2000:
            print("❌ Insufficient GPU memory (need 2GB+ free)")
            requirements_met = False
    else:
        print("❌ GPU not available or accessible")
        requirements_met = False

    # Critical alerts check
    if any(level == "CRITICAL" for level in alerts.values()):
        print("❌ Critical resource alerts detected")
        requirements_met = False

    if requirements_met:
        print("✅ System ready for training")
    else:
        print("❌ System not ready for training")

    return requirements_met


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="System monitoring for progressive training"
    )
    parser.add_argument("--check", action="store_true", help="Check training readiness")
    parser.add_argument(
        "--monitor", action="store_true", help="Start continuous monitoring"
    )
    parser.add_argument(
        "--interval", type=int, default=30, help="Monitoring interval in seconds"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Maximum monitoring duration in minutes",
    )
    parser.add_argument(
        "--log-file", type=str, default="./system_monitor.log", help="Log file path"
    )

    args = parser.parse_args()

    if args.check:
        check_training_readiness()
    elif args.monitor:
        monitor = SystemMonitor(log_file=args.log_file)
        monitor.monitor_continuously(
            interval_seconds=args.interval, max_duration_minutes=args.duration
        )
    else:
        # Show current status
        monitor = SystemMonitor()
        status = monitor.get_system_status()
        alerts = monitor.check_resource_warnings(status)
        monitor.print_status(status, alerts)


if __name__ == "__main__":
    main()
