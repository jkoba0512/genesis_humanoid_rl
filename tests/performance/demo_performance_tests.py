#!/usr/bin/env python3
"""
Performance test demonstration script.

Runs selected performance tests to showcase the capabilities of the performance
testing suite for the Unit of Work implementation.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    duration = time.time() - start_time
    
    print(f"\n⏱️ Completed in {duration:.2f} seconds")
    print(f"📊 Return code: {result.returncode}")
    
    return result.returncode == 0


def main():
    """Run performance test demonstrations."""
    print("🚀 Genesis Humanoid RL - Performance Testing Suite Demo")
    print("=" * 60)
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent.parent
    original_cwd = Path.cwd()
    
    try:
        import os
        os.chdir(project_root)
        
        tests_to_run = [
            {
                'cmd': ['uv', 'run', 'python', '-m', 'pytest', 
                       'tests/performance/test_simple_uow_performance.py::TestSimpleUnitOfWorkPerformance::test_basic_transaction_performance', 
                       '-v', '-s'],
                'description': 'Basic Unit of Work Transaction Performance'
            },
            {
                'cmd': ['uv', 'run', 'python', '-m', 'pytest', 
                       'tests/performance/test_simple_bottlenecks.py::TestSimpleBottleneckScenarios::test_rapid_transaction_cycling', 
                       '-v', '-s'],
                'description': 'Rapid Transaction Cycling Bottleneck Test'
            },
            {
                'cmd': ['uv', 'run', 'python', '-m', 'pytest', 
                       'tests/performance/test_simple_uow_performance.py::TestSimpleUnitOfWorkPerformance::test_memory_usage_under_load', 
                       '-v', '-s'],
                'description': 'Memory Usage Under Load Test'
            },
            {
                'cmd': ['uv', 'run', 'python', 'tests/performance/run_performance_tests.py', 
                       '--pattern', 'test_simple_uow_performance.py', 
                       '--format', 'markdown',
                       '--output-dir', 'tests/performance/demo_reports'],
                'description': 'Generate Performance Report with Test Runner'
            }
        ]
        
        successful_tests = 0
        total_tests = len(tests_to_run)
        
        for i, test in enumerate(tests_to_run, 1):
            print(f"\n📋 Running test {i}/{total_tests}")
            success = run_command(test['cmd'], test['description'])
            
            if success:
                print("✅ Test completed successfully")
                successful_tests += 1
            else:
                print("❌ Test failed or had issues")
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 PERFORMANCE TESTING DEMO SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful tests: {successful_tests}/{total_tests}")
        print(f"❌ Failed tests: {total_tests - successful_tests}/{total_tests}")
        print(f"📈 Success rate: {successful_tests/total_tests*100:.1f}%")
        
        # Show available performance tests
        print(f"\n📁 Available Performance Test Files:")
        perf_dir = Path("tests/performance")
        for test_file in sorted(perf_dir.glob("test_*.py")):
            print(f"   🧪 {test_file.name}")
        
        # Show generated reports
        report_dirs = [
            Path("tests/performance/reports"),
            Path("tests/performance/demo_reports")
        ]
        
        for report_dir in report_dirs:
            if report_dir.exists():
                reports = list(report_dir.glob("*.md"))
                if reports:
                    print(f"\n📄 Generated Reports in {report_dir}:")
                    for report in sorted(reports)[-3:]:  # Show last 3 reports
                        print(f"   📋 {report.name}")
        
        print(f"\n🎯 Performance Testing Features Demonstrated:")
        print("   • Unit of Work transaction performance measurement")
        print("   • Memory usage tracking and analysis")
        print("   • Bottleneck identification (rapid cycling)")
        print("   • Automated report generation")
        print("   • Resource utilization monitoring")
        print("   • Concurrent operation stress testing")
        
        print(f"\n📚 For more information, see:")
        print("   • tests/performance/README.md - Comprehensive documentation")
        print("   • tests/performance/performance_config.yaml - Configuration options")
        print("   • Generated reports - Detailed performance analysis")
        
        if successful_tests == total_tests:
            print(f"\n🎉 All performance tests completed successfully!")
            return 0
        else:
            print(f"\n⚠️ Some performance tests had issues. Check logs for details.")
            return 1
    
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    sys.exit(main())