# Performance Test Report

**Generated:** 2025-06-27 09:39:00

## System Information

- **CPU Cores:** 16
- **Memory:** 31.2 GB total, 29.0 GB available
- **Python Version:** 3.10.18 (main, Jun 12 2025, 12:39:39) [Clang 20.1.4 ]
- **Platform:** linux

## Test Suite Summary

- **Total Test Files:** 1
- **Total Tests:** 0
- **Passed:** 0
- **Failed:** 0
- **Success Rate:** N/A
- **Total Duration:** 1.47 seconds

## Detailed Results

### test_simple_uow_performance.py

- **Status:** failed
- **Duration:** 1.47 seconds
- **Return Code:** 1

#### Individual Tests

| Test Name | Outcome | Duration (s) |
|-----------|---------|-------------|
| FAILED | failed | 0.000 |

## Performance Insights

- **Average test duration:** 0.000 seconds
- **Slowest test duration:** 0.000 seconds
- **Fastest test duration:** 0.000 seconds
- **System memory usage during tests:** 6.9%

## Recommendations

- **High failure rate detected:** Consider reviewing test stability and database connection handling.
- **Critical failure rate:** Review transaction deadlock handling and concurrent access patterns.
- **Database optimization:** Consider adding indexes for frequently queried fields.
- **Connection pooling:** Implement database connection pooling for better scalability.
- **Batch operations:** Use batch insert/update operations for better performance.
- **Monitoring:** Implement performance monitoring in production environments.
