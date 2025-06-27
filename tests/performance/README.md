# Performance Testing Suite

This directory contains comprehensive performance tests for the Unit of Work implementation and related infrastructure components in the Genesis Humanoid RL project.

## Overview

The performance testing suite is designed to identify bottlenecks, scalability issues, and resource utilization patterns in the Domain-Driven Design (DDD) architecture, particularly focusing on:

- **Unit of Work pattern implementation**
- **Repository pattern performance**
- **Database transaction handling**
- **Memory usage patterns**
- **Concurrent operation handling**
- **JSON serialization/deserialization performance**

## Test Files

### Core Performance Tests

1. **`test_simple_uow_performance.py`** - ✅ **Working**
   - Basic Unit of Work transaction performance
   - Concurrent transaction stress testing
   - Memory usage analysis under load
   - Database connection handling patterns
   - **Focus**: Core UoW functionality without complex domain dependencies

2. **`test_unit_of_work_performance.py`** - 🔧 **In Development**
   - High-volume transaction processing (100+ concurrent operations)
   - Large dataset operations (1000+ records)
   - Memory usage patterns under load
   - Transaction deadlock scenarios
   - Repository query performance
   - JSON serialization/deserialization performance
   - **Focus**: Comprehensive domain model integration testing

3. **`test_training_scenarios.py`** - 🔧 **In Development**
   - Multi-robot concurrent training scenarios
   - High-frequency skill assessment patterns
   - Curriculum progression under load
   - Mixed workload simulation
   - **Focus**: Realistic training environment performance

4. **`test_bottleneck_scenarios.py`** - 🔧 **In Development**
   - Memory exhaustion scenarios
   - Connection pool exhaustion testing
   - Serialization bottleneck identification
   - Concurrent write operation stress testing
   - **Focus**: Identifying specific performance bottlenecks

### Utilities and Configuration

5. **`run_performance_tests.py`** - ✅ **Working**
   - Automated performance test execution
   - Comprehensive report generation (Markdown & JSON)
   - System information collection
   - Performance metrics analysis
   - Recommendations generation

6. **`performance_config.yaml`**
   - Configurable test parameters
   - Performance thresholds and limits
   - Resource usage constraints
   - Test data configuration

## Quick Start

### Running Basic Performance Tests

```bash
# Run working performance tests
uv run python -m pytest tests/performance/test_simple_uow_performance.py -v

# Run specific test
uv run python -m pytest tests/performance/test_simple_uow_performance.py::TestSimpleUnitOfWorkPerformance::test_basic_transaction_performance -v -s
```

### Using the Performance Test Runner

```bash
# Run all working performance tests with reports
uv run python tests/performance/run_performance_tests.py --pattern "test_simple_*.py" --output-dir tests/performance/reports

# Run with specific format
uv run python tests/performance/run_performance_tests.py --pattern "test_simple_*.py" --format markdown --verbose

# Run all tests (when fixed)
uv run python tests/performance/run_performance_tests.py --pattern "test_*.py" --output-dir tests/performance/reports
```

### Performance Test Runner Options

```bash
Options:
  --pattern TEXT          Test file pattern to match (default: test_*.py)
  --output-dir TEXT       Output directory for reports (default: performance_reports)
  --format [markdown|json|both]  Report format (default: both)
  --verbose              Enable verbose test output
  --help                 Show help message
```

## Test Scenarios

### 1. Basic Transaction Performance
- **Test**: `test_basic_transaction_performance`
- **Scenario**: 100 sequential Unit of Work transactions
- **Metrics**: Transaction time, success rate, operations/second, memory usage
- **Thresholds**: <1s avg, >80% success, >10 ops/sec

### 2. Concurrent Transaction Stress
- **Test**: `test_concurrent_transaction_stress`
- **Scenario**: 10 threads × 20 transactions with deliberate conflicts
- **Metrics**: Success rate, deadlock frequency, transaction time distribution
- **Thresholds**: >50% success, <30% deadlocks, <5s avg

### 3. Memory Usage Under Load
- **Test**: `test_memory_usage_under_load`
- **Scenario**: 500 operations with temporary object creation
- **Metrics**: Memory growth, peak usage, garbage collection effectiveness
- **Thresholds**: <200MB growth, <1GB peak, >90% success

### 4. Database Connection Handling
- **Test**: `test_database_connection_handling`
- **Scenario**: Normal ops, exception handling, rapid open/close cycles
- **Metrics**: Connection reliability, exception handling, resource cleanup
- **Thresholds**: >85% success, <500ms avg, proper cleanup

## Performance Metrics

### Transaction Performance
- **Average Transaction Time**: Time to complete a full UoW transaction
- **P95/P99 Transaction Time**: 95th/99th percentile transaction times
- **Operations per Second**: Throughput under normal conditions
- **Success Rate**: Percentage of successful transactions

### Concurrency Performance
- **Deadlock Rate**: Frequency of database locking conflicts
- **Error Rate**: Percentage of failed operations under stress
- **Concurrent Success Rate**: Success rate under concurrent load
- **Thread Scalability**: Performance scaling with thread count

### Resource Usage
- **Peak Memory Usage**: Maximum memory consumption during tests
- **Memory Growth**: Net memory increase during test execution
- **CPU Utilization**: Processor usage patterns
- **File Descriptor Usage**: Open file handle consumption

### Database Performance
- **Query Response Time**: Individual query execution time
- **Connection Pool Efficiency**: Connection reuse and availability
- **Transaction Isolation**: Proper ACID compliance under load
- **Index Utilization**: Query optimization effectiveness

## Report Generation

### Markdown Reports
Generated reports include:
- **System Information**: Hardware specs, Python version, platform
- **Test Suite Summary**: Overall pass/fail rates and execution time
- **Detailed Results**: Per-test performance metrics and outcomes
- **Performance Insights**: Statistical analysis and trend identification
- **Recommendations**: Optimization suggestions based on results

### JSON Reports
Machine-readable format containing:
- Raw performance data
- Test execution metadata
- System configuration snapshot
- Structured metrics for automated analysis

## Performance Thresholds

### Critical Thresholds (Test Failure)
- Memory usage > 4GB
- Average transaction time > 5 seconds
- Success rate < 50%
- Deadlock rate > 50%

### Warning Thresholds
- Memory usage > 1GB
- Average transaction time > 1 second
- Success rate < 80%
- Deadlock rate > 20%

### Optimal Targets
- Memory usage < 500MB
- Average transaction time < 0.1 seconds
- Success rate > 95%
- Deadlock rate < 5%

## Troubleshooting

### Common Issues

1. **Import Errors in Domain Tests**
   - **Cause**: Missing or incorrect domain model imports
   - **Solution**: Use `test_simple_uow_performance.py` for basic testing
   - **Status**: Domain integration tests under development

2. **Database Lock Errors**
   - **Cause**: SQLite concurrency limitations
   - **Solution**: Reduce concurrent thread count or add retry logic
   - **Mitigation**: Tests include deadlock detection and handling

3. **Memory Growth Issues**
   - **Cause**: Object lifecycle management or circular references
   - **Solution**: Review object cleanup and garbage collection
   - **Monitoring**: Memory tests track growth patterns

4. **Test Runner JSON Report Errors**
   - **Cause**: Missing pytest-json-report plugin
   - **Solution**: Test runner automatically falls back to stdout parsing
   - **Status**: Fixed in current implementation

### Performance Investigation

1. **Identify Slow Tests**
   ```bash
   uv run python -m pytest tests/performance/ -v --durations=10
   ```

2. **Profile Memory Usage**
   ```bash
   uv run python -m memory_profiler tests/performance/test_simple_uow_performance.py
   ```

3. **Analyze Database Operations**
   - Check SQLite WAL mode configuration
   - Review transaction isolation levels
   - Monitor connection pool usage

## Development Status

### ✅ Completed Components
- Basic Unit of Work performance testing
- Performance test runner infrastructure
- Report generation (Markdown & JSON)
- System resource monitoring
- Concurrent transaction stress testing

### 🔧 In Development
- Domain model integration tests
- Complex training scenario simulations
- Bottleneck identification suite
- Advanced performance visualization

### 📋 Planned Enhancements
- Performance regression tracking
- Automated benchmark comparisons
- Integration with CI/CD pipelines
- Performance alert thresholds
- Database query optimization analysis

## Integration with Project

### CI/CD Integration
```yaml
# Example GitHub Actions integration
- name: Run Performance Tests
  run: |
    uv run python tests/performance/run_performance_tests.py --pattern "test_simple_*.py"
    
- name: Archive Performance Reports
  uses: actions/upload-artifact@v3
  with:
    name: performance-reports
    path: tests/performance/reports/
```

### Development Workflow
1. **Before Major Changes**: Run baseline performance tests
2. **After Implementation**: Compare performance metrics
3. **Regular Monitoring**: Weekly performance regression checks
4. **Optimization Cycles**: Use reports to guide performance improvements

## Contributing

### Adding New Performance Tests
1. Follow the existing test structure pattern
2. Include comprehensive performance assertions
3. Add proper error handling and cleanup
4. Update this README with new test documentation

### Performance Test Guidelines
- **Isolate Concerns**: Test one performance aspect per test method
- **Use Realistic Data**: Base test data on actual usage patterns
- **Include Cleanup**: Ensure proper resource cleanup after tests
- **Document Thresholds**: Explain performance assertion rationale
- **Monitor Resources**: Track memory, CPU, and I/O usage

## References

- **Domain-Driven Design**: Eric Evans - Domain model performance considerations
- **Unit of Work Pattern**: Martin Fowler - Transaction boundary management
- **SQLite Performance**: Official SQLite optimization guide
- **Python Performance**: Python performance profiling and optimization