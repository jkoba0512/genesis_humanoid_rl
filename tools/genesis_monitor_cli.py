#!/usr/bin/env python3
"""
Genesis API Monitoring CLI Tool.

Command-line interface for monitoring Genesis API compatibility,
running version tests, and generating compatibility reports.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis_humanoid_rl.infrastructure.monitoring.genesis_monitor import (
    GenesisAPIMonitor,
    monitor_genesis_compatibility,
    check_genesis_status,
)
from genesis_humanoid_rl.infrastructure.monitoring.version_tester import (
    GenesisVersionTester,
    test_genesis_version,
    run_all_version_tests,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)

    # Reduce noise from other modules
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


async def cmd_status(args) -> int:
    """Check Genesis status command."""
    print("Checking Genesis status...")

    status = check_genesis_status()

    print(f"\nGenesis Status Report")
    print("=" * 40)
    print(f"Available: {status['available']}")
    print(f"Status: {status['status']}")
    print(f"Version: {status['version']}")

    if status.get("api_level"):
        print(f"API Level: {status['api_level']}")

    if status.get("features"):
        print(f"\nFeatures:")
        for feature, available in status["features"].items():
            status_icon = "✓" if available else "✗"
            print(f"  {status_icon} {feature}")

    if status.get("error"):
        print(f"\nError: {status['error']}")
        return 1

    return 0


async def cmd_monitor(args) -> int:
    """Run Genesis compatibility monitoring."""
    print("Running Genesis API compatibility monitoring...")

    try:
        report = await monitor_genesis_compatibility(args.output_dir)

        print(f"\nGenesis Compatibility Report")
        print("=" * 50)
        print(f"Version: {report.version_info.version}")
        print(f"Compatibility Level: {report.compatibility_level.value}")
        print(f"Compatibility Score: {report.get_compatibility_score():.1%}")
        print(f"Test Duration: {report.test_duration:.2f}s")

        # Show working features
        working_features = report.get_working_features()
        if working_features:
            print(f"\nWorking Features ({len(working_features)}):")
            for feature in working_features:
                print(f"  ✓ {feature}")

        # Show broken features
        broken_features = report.get_broken_features()
        if broken_features:
            print(f"\nBroken Features ({len(broken_features)}):")
            for feature in broken_features:
                print(f"  ✗ {feature}")

        # Show performance metrics
        if report.performance_metrics:
            print(f"\nPerformance Metrics:")
            for metric, value in report.performance_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")

        # Show recommendations
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  • {rec}")

        # Save detailed report if requested
        if args.save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = (
                Path(args.output_dir) / f"genesis_compatibility_{timestamp}.json"
            )

            with open(report_file, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

            print(f"\nDetailed report saved to: {report_file}")

        # Return non-zero if compatibility issues found
        if report.compatibility_level.value in ["incompatible", "partially_compatible"]:
            return 1

        return 0

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        return 1


async def cmd_test_version(args) -> int:
    """Run version compatibility tests."""
    print(f"Running Genesis version compatibility tests...")

    if args.test_suites:
        print(f"Test suites: {', '.join(args.test_suites)}")
    else:
        print("Test suites: all")

    try:
        results = await test_genesis_version(
            version=args.version, test_suites=args.test_suites
        )

        print(f"\nVersion Test Results")
        print("=" * 50)

        total_tests = 0
        total_passed = 0
        total_failed = 0

        for suite_name, suite in results.items():
            summary = suite.get_summary()
            total_tests += summary["total_tests"]
            total_passed += summary["results"].get("pass", 0)
            total_failed += summary["results"].get("fail", 0)

            print(f"\n{suite.name.upper()}:")
            print(f"  Description: {suite.description}")
            print(f"  Total Tests: {summary['total_tests']}")
            print(f"  Pass Rate: {summary['pass_rate']:.1%}")
            print(f"  Execution Time: {summary['execution_time']:.2f}s")

            # Show result breakdown
            for result_type in ["pass", "fail", "skip", "error"]:
                count = summary["results"].get(result_type, 0)
                if count > 0:
                    if result_type == "pass":
                        icon = "✓"
                    elif result_type == "fail":
                        icon = "✗"
                    elif result_type == "skip":
                        icon = "○"
                    else:  # error
                        icon = "⚠"
                    print(f"    {icon} {result_type.upper()}: {count}")

            # Show failed tests if verbose
            if args.verbose:
                failed_tests = [
                    test
                    for test in suite.tests
                    if test.actual_result
                    and test.actual_result.value in ["fail", "error"]
                ]
                if failed_tests:
                    print(f"  Failed Tests:")
                    for test in failed_tests:
                        print(
                            f"    - {test.test_name}: {test.error_message or 'Unknown error'}"
                        )

        # Overall summary
        print(f"\nOVERALL SUMMARY:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")

        if total_tests > 0:
            overall_pass_rate = total_passed / total_tests
            print(f"Overall Pass Rate: {overall_pass_rate:.1%}")

            if overall_pass_rate < 0.8:
                print("\n⚠ WARNING: Low pass rate indicates compatibility issues")
                return 1

        return 0

    except Exception as e:
        logger.error(f"Version testing failed: {e}")
        return 1


async def cmd_test_all(args) -> int:
    """Run all compatibility tests."""
    print("Running comprehensive Genesis compatibility testing...")

    # First run basic monitoring
    print("\n1. Running API monitoring...")
    args.save_report = True  # Force save for comprehensive test
    monitor_result = await cmd_monitor(args)

    # Then run version tests
    print("\n2. Running version compatibility tests...")
    version_result = await cmd_test_version(args)

    print(f"\nComprehensive Testing Complete")
    print("=" * 40)

    if monitor_result == 0 and version_result == 0:
        print("✓ All tests passed - Genesis fully compatible")
        return 0
    elif monitor_result != 0:
        print("✗ API monitoring detected issues")
        return 1
    else:
        print("✗ Version testing detected issues")
        return 1


async def cmd_benchmark(args) -> int:
    """Run Genesis performance benchmarks."""
    print("Running Genesis performance benchmarks...")

    try:
        # Use version tester for performance benchmarks
        tester = GenesisVersionTester()
        results = await tester.test_version_compatibility(
            test_suites=["performance_benchmarks"]
        )

        if "performance_benchmarks" not in results:
            print("Performance benchmark suite not available")
            return 1

        suite = results["performance_benchmarks"]
        summary = suite.get_summary()

        print(f"\nPerformance Benchmark Results")
        print("=" * 50)
        print(f"Total Benchmarks: {summary['total_tests']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Total Execution Time: {summary['execution_time']:.2f}s")

        # Show individual benchmark results
        for test in suite.tests:
            if test.actual_result:
                status_icon = "✓" if test.actual_result.value == "pass" else "✗"
                print(f"\n{status_icon} {test.test_name}:")
                print(f"    Status: {test.actual_result.value}")
                print(f"    Time: {test.execution_time:.3f}s")
                if test.output:
                    print(f"    Result: {test.output}")
                if test.error_message:
                    print(f"    Error: {test.error_message}")

        return 0 if summary["pass_rate"] > 0.8 else 1

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Genesis API Monitoring and Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                          # Check Genesis status
  %(prog)s monitor                         # Run compatibility monitoring
  %(prog)s test-version                    # Run version tests
  %(prog)s test-version --suites core_functionality humanoid_robotics
  %(prog)s test-all                        # Run comprehensive tests
  %(prog)s benchmark                       # Run performance benchmarks
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--output-dir",
        default="genesis_reports",
        help="Output directory for reports (default: genesis_reports)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Check Genesis installation status"
    )

    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Run Genesis API compatibility monitoring"
    )
    monitor_parser.add_argument(
        "--save-report", action="store_true", help="Save detailed compatibility report"
    )

    # Test version command
    test_parser = subparsers.add_parser(
        "test-version", help="Run Genesis version compatibility tests"
    )
    test_parser.add_argument(
        "--version", help="Genesis version to test (default: current)"
    )
    test_parser.add_argument(
        "--suites",
        dest="test_suites",
        nargs="+",
        choices=["core_functionality", "humanoid_robotics", "performance_benchmarks"],
        help="Test suites to run (default: all)",
    )

    # Test all command
    test_all_parser = subparsers.add_parser(
        "test-all", help="Run comprehensive Genesis compatibility testing"
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run Genesis performance benchmarks"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    setup_logging(args.verbose)

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)

    # Route to appropriate command
    command_map = {
        "status": cmd_status,
        "monitor": cmd_monitor,
        "test-version": cmd_test_version,
        "test-all": cmd_test_all,
        "benchmark": cmd_benchmark,
    }

    if args.command in command_map:
        try:
            return asyncio.run(command_map[args.command](args))
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            return 1
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            return 1
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
