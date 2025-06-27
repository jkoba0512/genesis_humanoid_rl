#!/usr/bin/env python3
"""
Performance test runner and report generator.

Executes all performance tests and generates comprehensive performance reports
with metrics, charts, and recommendations for optimization.
"""

import argparse
import json
import logging
import os
import psutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceTestRunner:
    """Runs performance tests and generates reports."""
    
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.system_info = self._collect_system_info()
    
    def _collect_system_info(self) -> Dict:
        """Collect system information for the performance report."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage': {
                'total_gb': psutil.disk_usage('/').total / (1024**3),
                'free_gb': psutil.disk_usage('/').free / (1024**3)
            },
            'python_version': sys.version,
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_test_suite(self, test_pattern: str = "test_*.py", verbose: bool = True) -> Dict:
        """Run all performance tests matching the pattern."""
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob(test_pattern))
        
        if not test_files:
            logger.error(f"No test files found matching pattern: {test_pattern}")
            return {}
        
        logger.info(f"Found {len(test_files)} test files to execute")
        
        all_results = {}
        
        for test_file in test_files:
            if test_file.name == "run_performance_tests.py":
                continue  # Skip this runner script
            
            logger.info(f"Running performance tests in {test_file.name}")
            
            try:
                result = self._run_single_test_file(test_file, verbose)
                all_results[test_file.name] = result
                
            except Exception as e:
                logger.error(f"Failed to run {test_file.name}: {e}")
                all_results[test_file.name] = {
                    'status': 'failed',
                    'error': str(e),
                    'duration': 0,
                    'tests': []
                }
        
        self.results = all_results
        return all_results
    
    def _run_single_test_file(self, test_file: Path, verbose: bool) -> Dict:
        """Run a single test file and capture results."""
        start_time = time.time()
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        # Run the test
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=test_file.parent
        )
        
        duration = time.time() - start_time
        
        # Parse test results from stdout
        test_results = self._parse_pytest_output(process.stdout)
        
        return {
            'status': 'passed' if process.returncode == 0 else 'failed',
            'duration': duration,
            'return_code': process.returncode,
            'stdout': process.stdout,
            'stderr': process.stderr,
            'tests': test_results
        }
    
    def _parse_pytest_output(self, stdout: str) -> List[Dict]:
        """Parse pytest stdout to extract test results."""
        tests = []
        
        # Simple parsing of pytest verbose output
        lines = stdout.split('\n')
        for line in lines:
            line = line.strip()
            if '::' in line and ('PASSED' in line or 'FAILED' in line):
                parts = line.split()
                if len(parts) >= 2:
                    test_name = parts[0]
                    if 'PASSED' in line:
                        outcome = 'passed'
                    elif 'FAILED' in line:
                        outcome = 'failed'
                    else:
                        outcome = 'unknown'
                    
                    # Extract duration if available (e.g., [100%] in 0.15s)
                    duration = 0.0
                    for part in parts:
                        if part.endswith('s'):
                            try:
                                duration = float(part[:-1])
                                break
                            except ValueError:
                                pass
                    
                    tests.append({
                        'name': test_name,
                        'outcome': outcome,
                        'duration': duration
                    })
        
        return tests
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.results:
            logger.error("No test results available for report generation")
            return ""
        
        report_content = self._generate_markdown_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"performance_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Performance report saved to {report_file}")
        return str(report_file)
    
    def _generate_markdown_report(self) -> str:
        """Generate a Markdown formatted performance report."""
        report_lines = [
            "# Performance Test Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Information",
            "",
            f"- **CPU Cores:** {self.system_info['cpu_count']}",
            f"- **Memory:** {self.system_info['memory_total_gb']:.1f} GB total, {self.system_info['memory_available_gb']:.1f} GB available",
            f"- **Python Version:** {self.system_info['python_version']}",
            f"- **Platform:** {self.system_info['platform']}",
            "",
            "## Test Suite Summary",
            ""
        ]
        
        # Overall statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        total_duration = 0
        
        for test_file, result in self.results.items():
            total_duration += result.get('duration', 0)
            if result.get('status') == 'passed':
                passed_tests += len([t for t in result.get('tests', []) if t.get('outcome') == 'passed'])
                failed_tests += len([t for t in result.get('tests', []) if t.get('outcome') == 'failed'])
            total_tests = passed_tests + failed_tests
        
        report_lines.extend([
            f"- **Total Test Files:** {len(self.results)}",
            f"- **Total Tests:** {total_tests}",
            f"- **Passed:** {passed_tests}",
            f"- **Failed:** {failed_tests}",
            f"- **Success Rate:** {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "- **Success Rate:** N/A",
            f"- **Total Duration:** {total_duration:.2f} seconds",
            "",
            "## Detailed Results",
            ""
        ])
        
        # Detailed results for each test file
        for test_file, result in self.results.items():
            report_lines.extend([
                f"### {test_file}",
                "",
                f"- **Status:** {result.get('status', 'unknown')}",
                f"- **Duration:** {result.get('duration', 0):.2f} seconds",
                f"- **Return Code:** {result.get('return_code', 'N/A')}",
                ""
            ])
            
            # Individual test results
            if result.get('tests'):
                report_lines.extend([
                    "#### Individual Tests",
                    "",
                    "| Test Name | Outcome | Duration (s) |",
                    "|-----------|---------|-------------|"
                ])
                
                for test in result['tests']:
                    test_name = test.get('name', '').split("::")[-1]  # Get just the test method name
                    outcome = test.get('outcome', 'unknown')
                    duration = test.get('duration', 0)
                    
                    report_lines.append(f"| {test_name} | {outcome} | {duration:.3f} |")
                
                report_lines.append("")
            
            # Error details if any
            if result.get('status') == 'failed' and result.get('stderr'):
                report_lines.extend([
                    "#### Error Details",
                    "",
                    "```",
                    result.get('stderr', ''),
                    "```",
                    ""
                ])
        
        # Performance insights and recommendations
        report_lines.extend([
            "## Performance Insights",
            "",
            self._generate_performance_insights(),
            "",
            "## Recommendations",
            "",
            self._generate_recommendations(),
            ""
        ])
        
        return "\n".join(report_lines)
    
    def _generate_performance_insights(self) -> str:
        """Generate performance insights based on test results."""
        insights = []
        
        # Analyze test durations
        all_durations = []
        for result in self.results.values():
            for test in result.get('tests', []):
                all_durations.append(test.get('duration', 0))
        
        if all_durations:
            avg_duration = sum(all_durations) / len(all_durations)
            max_duration = max(all_durations)
            min_duration = min(all_durations)
            
            insights.extend([
                f"- **Average test duration:** {avg_duration:.3f} seconds",
                f"- **Slowest test duration:** {max_duration:.3f} seconds",
                f"- **Fastest test duration:** {min_duration:.3f} seconds",
            ])
            
            # Identify slow tests
            slow_threshold = avg_duration * 2
            slow_tests = [d for d in all_durations if d > slow_threshold]
            if slow_tests:
                insights.append(f"- **Tests exceeding 2x average duration:** {len(slow_tests)}")
        
        # Memory usage insights (if available in system info)
        memory_usage_percent = (
            (self.system_info['memory_total_gb'] - self.system_info['memory_available_gb']) / 
            self.system_info['memory_total_gb'] * 100
        )
        insights.append(f"- **System memory usage during tests:** {memory_usage_percent:.1f}%")
        
        return "\n".join(insights) if insights else "No specific performance insights available."
    
    def _generate_recommendations(self) -> str:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze failure rates
        total_tests = 0
        failed_tests = 0
        for result in self.results.values():
            for test in result.get('tests', []):
                total_tests += 1
                if test.get('outcome') == 'failed':
                    failed_tests += 1
        
        if total_tests > 0:
            failure_rate = failed_tests / total_tests
            
            if failure_rate > 0.1:  # More than 10% failure rate
                recommendations.append(
                    "- **High failure rate detected:** Consider reviewing test stability and "
                    "database connection handling."
                )
            
            if failure_rate > 0.3:  # More than 30% failure rate
                recommendations.append(
                    "- **Critical failure rate:** Review transaction deadlock handling and "
                    "concurrent access patterns."
                )
        
        # Check for long-running tests
        all_durations = []
        for result in self.results.values():
            for test in result.get('tests', []):
                all_durations.append(test.get('duration', 0))
        
        if all_durations and max(all_durations) > 60:  # Tests taking more than 1 minute
            recommendations.append(
                "- **Long-running tests detected:** Consider optimizing database operations "
                "or reducing test data volume."
            )
        
        # Memory recommendations
        memory_usage_percent = (
            (self.system_info['memory_total_gb'] - self.system_info['memory_available_gb']) / 
            self.system_info['memory_total_gb'] * 100
        )
        
        if memory_usage_percent > 80:
            recommendations.append(
                "- **High memory usage:** Consider implementing connection pooling and "
                "optimizing object lifecycle management."
            )
        
        # General recommendations
        recommendations.extend([
            "- **Database optimization:** Consider adding indexes for frequently queried fields.",
            "- **Connection pooling:** Implement database connection pooling for better scalability.",
            "- **Batch operations:** Use batch insert/update operations for better performance.",
            "- **Monitoring:** Implement performance monitoring in production environments."
        ])
        
        return "\n".join(recommendations)
    
    def generate_json_report(self) -> str:
        """Generate a JSON formatted report for programmatic analysis."""
        json_report = {
            'system_info': self.system_info,
            'test_results': self.results,
            'summary': {
                'total_test_files': len(self.results),
                'total_duration': sum(r.get('duration', 0) for r in self.results.values()),
                'passed_files': len([r for r in self.results.values() if r.get('status') == 'passed']),
                'failed_files': len([r for r in self.results.values() if r.get('status') == 'failed'])
            }
        }
        
        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f"performance_report_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        logger.info(f"JSON report saved to {json_file}")
        return str(json_file)


def main():
    """Main entry point for performance test runner."""
    parser = argparse.ArgumentParser(description="Run performance tests and generate reports")
    parser.add_argument(
        "--pattern", 
        default="test_*.py",
        help="Test file pattern to match (default: test_*.py)"
    )
    parser.add_argument(
        "--output-dir",
        default="performance_reports",
        help="Output directory for reports (default: performance_reports)"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="both",
        help="Report format (default: both)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose test output"
    )
    
    args = parser.parse_args()
    
    # Create performance test runner
    runner = PerformanceTestRunner(args.output_dir)
    
    logger.info("Starting performance test execution...")
    logger.info(f"System: {runner.system_info['cpu_count']} cores, "
                f"{runner.system_info['memory_total_gb']:.1f}GB RAM")
    
    # Run tests
    start_time = time.time()
    results = runner.run_test_suite(args.pattern, args.verbose)
    total_time = time.time() - start_time
    
    if not results:
        logger.error("No test results to process")
        return 1
    
    # Generate reports
    logger.info(f"Test execution completed in {total_time:.2f} seconds")
    logger.info("Generating performance reports...")
    
    if args.format in ["markdown", "both"]:
        markdown_report = runner.generate_performance_report()
        print(f"Markdown report: {markdown_report}")
    
    if args.format in ["json", "both"]:
        json_report = runner.generate_json_report()
        print(f"JSON report: {json_report}")
    
    # Summary
    total_files = len(results)
    passed_files = len([r for r in results.values() if r.get('status') == 'passed'])
    failed_files = total_files - passed_files
    
    print(f"\n=== Performance Test Summary ===")
    print(f"Test files executed: {total_files}")
    print(f"Passed: {passed_files}")
    print(f"Failed: {failed_files}")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if failed_files > 0:
        print(f"\nFailed test files:")
        for file_name, result in results.items():
            if result.get('status') == 'failed':
                print(f"  - {file_name}: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())