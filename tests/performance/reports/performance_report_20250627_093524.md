# Performance Test Report

**Generated:** 2025-06-27 09:35:24

## System Information

- **CPU Cores:** 16
- **Memory:** 31.2 GB total, 29.1 GB available
- **Python Version:** 3.10.18 (main, Jun 12 2025, 12:39:39) [Clang 20.1.4 ]
- **Platform:** linux

## Test Suite Summary

- **Total Test Files:** 1
- **Total Tests:** 0
- **Passed:** 0
- **Failed:** 0
- **Success Rate:** N/A
- **Total Duration:** 1.06 seconds

## Detailed Results

### test_simple_uow_performance.py

- **Status:** failed
- **Duration:** 1.06 seconds
- **Return Code:** 4

#### Error Details

```
ERROR: usage: __main__.py [options] [file_or_dir] [file_or_dir] [...]
__main__.py: error: unrecognized arguments: --json-report --json-report-file=test_simple_uow_performance_report.json
  inifile: /home/jkoba/SynologyDrive/genesis_humanoid_rl/pyproject.toml
  rootdir: /home/jkoba/SynologyDrive/genesis_humanoid_rl


```

## Performance Insights

- **System memory usage during tests:** 6.6%

## Recommendations

- **Database optimization:** Consider adding indexes for frequently queried fields.
- **Connection pooling:** Implement database connection pooling for better scalability.
- **Batch operations:** Use batch insert/update operations for better performance.
- **Monitoring:** Implement performance monitoring in production environments.
