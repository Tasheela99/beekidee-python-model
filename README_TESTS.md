# Test Documentation

## Overview
This document describes the test suite for the Flask API that handles real-time attention analysis. The tests use pytest with custom stubs to mock Firebase and attention analysis dependencies.

## Test Setup

### Dependencies
- pytest
- Flask test client
- Custom Firebase/Firestore mocks
- Attention analyzer stubs
- webbrowser (for HTML report opening)

### Test Configuration
Tests are configured to:
- Mock Firebase Admin SDK and Firestore
- Stub the RealTimeAttentionAnalyzer
- Log all test results to timestamped text files in `test-results/` folder
- Provide detailed console output with beautiful formatting and colors
- Add proper spacing and structure to log files
- Generate HTML reports and automatically open them in browser
- Include commented failure lines for testing purposes

## Test Cases

### 1. test_analyze_requires_uid
**Purpose**: Validates that the `/analyze` endpoint requires a UID parameter.

**Test Steps**:
1. Send POST request to `/analyze` with empty JSON body
2. Verify response status is 400
3. Verify error message contains "uid is required"

**Expected Results**:
- Status Code: 400
- Response: `{"error": "uid is required in request body"}`

**Console Output Example**:
```
RUNNING TEST: test_analyze_requires_uid
================================================================================
Sending POST request to /analyze with empty JSON body
Response Status:
   400
Response Data:
   {'error': 'uid is required in request body'}
UID validation working correctly
```

**Log File Example**:
```
============================================================
STARTING TEST: test_analyze_requires_uid
============================================================
[test_analyze_requires_uid] Response status: 400
[test_analyze_requires_uid] Response data: {'error': 'uid is required in request body'}
[test_analyze_requires_uid] TEST PASSED: UID validation working correctly

TEST COMPLETED: test_analyze_requires_uid - PASS
------------------------------------------------------------
```

### 2. test_analyze_starts_and_returns_status
**Purpose**: Tests that the `/analyze` endpoint successfully starts analysis and returns proper status.

**Test Steps**:
1. Send POST request to `/analyze` with valid UID and sessionId
2. Verify response status is 200
3. Verify response contains status, studentId, and latest fields
4. Verify status is either "running" or "paused"

**Expected Results**:
- Status Code: 200
- Response includes: `status`, `studentId`, `latest`, `sessionId`
- Status: "running" or "paused"

**Console Output Example**:
```
RUNNING TEST: test_analyze_starts_and_returns_status
================================================================================
Sending POST request to /analyze with valid UID and sessionId
Response Status:
   200
Response Data:
   {'latest': 'No data yet...', 'sessionId': 'sessA', 'status': 'running', 'studentId': 'user123'}
Analyze endpoint working correctly
```

**Log File Example**:
```
============================================================
STARTING TEST: test_analyze_starts_and_returns_status
============================================================
[test_analyze_starts_and_returns_status] Response status: 200
[test_analyze_starts_and_returns_status] Response data: {'latest': 'No data yet...', 'sessionId': 'sessA', 'status': 'running', 'studentId': 'user123'}
[test_analyze_starts_and_returns_status] TEST PASSED: Analyze endpoint working correctly

TEST COMPLETED: test_analyze_starts_and_returns_status - PASS
------------------------------------------------------------
```

### 3. test_overall_no_docs_returns_empty
**Purpose**: Verifies that `/overall/{uid}` returns empty array when no data exists.

**Test Steps**:
1. Initialize analyzer with a UID
2. Send GET request to `/overall/{uid}`
3. Verify response status is 200
4. Verify response is empty array

**Expected Results**:
- Status Code: 200
- Response: `[]`

**Console Output Example**:
```
RUNNING TEST: test_overall_no_docs_returns_empty
================================================================================
Initializing analyzer with UID
Sending GET request to /overall/u1
Response Status:
   200
Overall Data:
   []
Empty overall data returned correctly
```

**Log File Example**:
```
============================================================
STARTING TEST: test_overall_no_docs_returns_empty
============================================================
[test_overall_no_docs_returns_empty] Response status: 200
[test_overall_no_docs_returns_empty] Overall data for u1: []
[test_overall_no_docs_returns_empty] TEST PASSED: Empty overall data returned correctly

TEST COMPLETED: test_overall_no_docs_returns_empty - PASS
------------------------------------------------------------
```

### 4. test_overall_returns_latest_value
**Purpose**: Tests that `/overall/{uid}` returns the most recent overall attention value.

**Test Steps**:
1. Create session document in mock Firestore
2. Add two samples with different timestamps
3. Send GET request to `/overall/{uid}`
4. Verify response contains latest value rounded to 2 decimals

**Test Data**:
- Sample 1: overall=71.678, savedAt=now
- Sample 2: overall=65.4, savedAt=now-10seconds

**Expected Results**:
- Status Code: 200
- Response: `[{"overall": 71.68}]` (latest value, rounded)

**Console Output Example**:
```
RUNNING TEST: test_overall_returns_latest_value
================================================================================
Creating session document in mock Firestore
Adding sample data with different timestamps
Sample 1 (Latest):
   {'studentId': 'u2', 'sessionId': 'S1', 'overall': 71.678, 'savedAt': 1755001557.491424}
Sample 2 (Older):
   {'studentId': 'u2', 'sessionId': 'S1', 'overall': 65.4, 'savedAt': 1755001547.491424}
Sending GET request to /overall/u2
Response Status:
   200
Overall Data:
   [{'overall': 71.68}]
Latest overall value returned correctly (71.68)
```

**Log File Example**:
```
============================================================
STARTING TEST: test_overall_returns_latest_value
============================================================
[test_overall_returns_latest_value] Adding sample1: {'studentId': 'u2', 'sessionId': 'S1', 'overall': 71.678, 'savedAt': 1755001557.491424}
[test_overall_returns_latest_value] Adding sample2: {'studentId': 'u2', 'sessionId': 'S1', 'overall': 65.4, 'savedAt': 1755001547.491424}
[test_overall_returns_latest_value] Response status: 200
[test_overall_returns_latest_value] Overall data for u2: [{'overall': 71.68}]
[test_overall_returns_latest_value] TEST PASSED: Latest overall value returned correctly

TEST COMPLETED: test_overall_returns_latest_value - PASS
------------------------------------------------------------
```

### 5. test_all_analysis_requires_init
**Purpose**: Tests the `/all_analysis` endpoint behavior with and without analyzer initialization.

**Test Steps**:
1. Send GET request to `/all_analysis`
2. Handle two possible scenarios:
   - Analyzer not initialized (400 response)
   - Analyzer already initialized (200 response)
3. For uninitialized case: initialize analyzer and test again
4. Verify response structure and data types

**Expected Results**:
**Case 1 - Not Initialized**:
- Initial Status: 400
- Message: "Analyzer not initialized"
- After initialization: 200 with proper data structure

**Case 2 - Already Initialized**:
- Status: 200
- Response includes: `data`, `studentId`, `sessionId`
- Data is dictionary with arrays

**Console Output Example**:
```
RUNNING TEST: test_all_analysis_requires_init
================================================================================
Sending GET request to /all_analysis
Response Status:
   200
Response Data:
   {'data': {'overall': [], 'timestamp': []}, 'sessionId': None, 'studentId': 'u1'}
Analyzer already initialized - testing with existing analyzer
Testing UID update with existing analyzer
All analysis working with pre-initialized analyzer
```

**Log File Example**:
```
============================================================
STARTING TEST: test_all_analysis_requires_init
============================================================
[test_all_analysis_requires_init] Initial all_analysis response status: 200
[test_all_analysis_requires_init] Initial all_analysis response: {'data': {'overall': [], 'timestamp': []}, 'sessionId': None, 'studentId': 'u1'}
[test_all_analysis_requires_init] Analyzer already initialized with data: {'data': {'overall': [], 'timestamp': []}, 'sessionId': None, 'studentId': 'u1'}
[test_all_analysis_requires_init] TEST PASSED: All analysis working with pre-initialized analyzer

TEST COMPLETED: test_all_analysis_requires_init - PASS
------------------------------------------------------------
```

### 6. test_analyzer_start_stop_functionality
**Purpose**: Tests analyzer control endpoints (pause, resume, stop) if available.

**Test Steps**:
1. Start analyzer with `/analyze`
2. Test `/pause` endpoint
3. If pause exists: test `/resume` and `/stop`
4. If pause doesn't exist: verify analyzer is running and log notice

**Expected Results**:
**If Control Endpoints Exist**:
- `/pause`: Status 200, status="paused"
- `/resume`: Status 200, status="running"  
- `/stop`: Status 200, status="stopped"

**If Control Endpoints Don't Exist**:
- `/pause`: Status 404
- Log notice about missing endpoints
- Verify analyzer is initialized

**Console Output Example**:
```
RUNNING TEST: test_analyzer_start_stop_functionality
================================================================================
Starting analyzer with /analyze endpoint
Analyze Response Status:
   200
Analyze Response Data:
   {'latest': 'No data yet...', 'sessionId': 'sessB', 'status': 'running', 'studentId': 'u4'}
Testing /pause endpoint
Pause Response Status:
   404
Pause Response Data:
   None
/pause endpoint not implemented
Checking analyzer status directly
Analyzer tracking status: True
Analyzer initialized and accessible
```

**Log File Example**:
```
============================================================
STARTING TEST: test_analyzer_start_stop_functionality
============================================================
[test_analyzer_start_stop_functionality] Analyze response status: 200
[test_analyzer_start_stop_functionality] Analyze response: {'latest': 'No data yet...', 'sessionId': 'sessB', 'status': 'running', 'studentId': 'u4'}
[test_analyzer_start_stop_functionality] Pause response status: 404
[test_analyzer_start_stop_functionality] Pause response: None
[test_analyzer_start_stop_functionality] NOTICE: /pause endpoint not implemented
[test_analyzer_start_stop_functionality] TEST PASSED: Analyze endpoint working, pause/resume/stop endpoints not implemented

TEST COMPLETED: test_analyzer_start_stop_functionality - PASS
------------------------------------------------------------
```

## Testing Features

### Failure Testing
Each test includes commented lines that will cause the test to fail when uncommented. This allows for:
- Testing the failure reporting system
- Demonstrating how failures are handled and logged
- Validating that the HTML report correctly shows failed tests

Example failure lines in tests:
```python
# Uncomment the line below to force test failure
# assert resp.status_code == 200  # This will fail - expecting 400 but forcing 200
```

### HTML Report Generation
After all tests complete, an HTML report is automatically generated and opened in your default browser. The report includes:

- **Summary Dashboard**: Total tests, passed, and failed counts with color-coded cards
- **Overall Status**: Visual indicator showing if all tests passed or some failed
- **Detailed Test Results**: Individual test status with descriptions
- **Responsive Design**: Works well on desktop and mobile devices
- **Professional Styling**: Clean, modern interface with gradient headers

**HTML Report Features**:
- Automatically saved to `test-results/test_report_YYYYMMDD_HHMMSS.html`
- Opens automatically in default browser after test completion
- Color-coded test results (green for pass, red for fail)
- Test details and error messages displayed
- Timestamp and execution information

### Console Output Features

### Color-Coded Messages
- **Purple Header**: Test names and section headers
- **Cyan**: Test steps and actions
- **Cyan**: Response data and values
- **Green**: Successful test results
- **Red**: Failed test results
- **Yellow**: Warnings and notices
- **Blue**: Informational messages

### Final Summary
After all tests complete, a beautiful summary is displayed in console, followed by HTML report generation:
```
TEST EXECUTION SUMMARY
================================================================================
Total Tests: 6
Passed: 6
Failed: 0

ALL TESTS PASSED!

Detailed Results:
  test_analyze_requires_uid: PASS
    UID validation enforced properly
  test_analyze_starts_and_returns_status: PASS
    Analyzer started with status: running
  test_overall_no_docs_returns_empty: PASS
    Empty array returned for no data
  test_overall_returns_latest_value: PASS
    Latest value: 71.68
  test_all_analysis_requires_init: PASS
    Pre-initialized analyzer works correctly
  test_analyzer_start_stop_functionality: PASS
    Analyzer running, control endpoints not implemented
================================================================================

📊 HTML report generated and opened: test-results/test_report_20250812_235959.html
```

## Test Infrastructure

### Mock Components

#### FakeDB
- Simulates Firestore database
- Supports collections, documents, subcollections
- Implements basic querying (where, order_by, limit)

#### FakeDocSnapshot
- Mimics Firestore document snapshots
- Provides `to_dict()` method

#### Stub Modules
- **firebase_admin**: Mocked Firebase Admin SDK
- **firestore**: Mocked Firestore client
- **attention_count**: Stubbed RealTimeAttentionAnalyzer

### Logging System
- All test results logged to `test-results/test_results_YYYYMMDD_HHMMSS.txt`
- Structured log format with clear test boundaries
- Includes timestamps, test names, and detailed output
- Both console and file logging with different formatting
- Automatic cleanup after test session

### Log File Structure
```
Test Results - 2025-08-12 23:36:05
==================================================

============================================================
STARTING TEST: test_analyze_requires_uid
============================================================
[test_analyze_requires_uid] Response status: 400
[test_analyze_requires_uid] Response data: {'error': 'uid is required in request body'}
[test_analyze_requires_uid] TEST PASSED: UID validation working correctly

TEST COMPLETED: test_analyze_requires_uid - PASS
------------------------------------------------------------

============================================================
STARTING TEST: test_analyze_starts_and_returns_status
============================================================
[test_analyze_starts_and_returns_status] Response status: 200
[test_analyze_starts_and_returns_status] Response data: {'latest': 'No data yet...', 'sessionId': 'sessA', 'status': 'running', 'studentId': 'user123'}
[test_analyze_starts_and_returns_status] TEST PASSED: Analyze endpoint working correctly

TEST COMPLETED: test_analyze_starts_and_returns_status - PASS
------------------------------------------------------------

==================================================
Tests completed at 2025-08-12 23:36:05
```

### Running Tests

```bash
# Run all tests with quiet output (recommended for colorized output)
pytest -q

# Run with verbose output
pytest -v

# Run specific test
pytest test_app.py::test_analyze_requires_uid

# Run with stdout capture disabled (see all print statements)
pytest -s

# Run tests and save output to file (preserves colors in terminal)
pytest -q | tee console_output.txt
```

### Test File Structure
```
test_app.py
├── Console Formatting (Colors class, print functions)
├── Logging Functions (log_test_result, log_test_start, log_test_end)
├── Mock Classes (FakeDB, FakeDocSnapshot, etc.)
├── Fixtures (stub_modules, app_and_client)
├── Test Cases (6 test functions with beautiful output)
└── Cleanup (automatic log file closure and summary)
```

### Expected Artifacts
- `test-results/test_results_YYYYMMDD_HHMMSS.txt`: Detailed test execution log with structured formatting
- `test-results/test_report_YYYYMMDD_HHMMSS.html`: Professional HTML report with test results
- Console output: Real-time test progress with colors and emojis
- pytest summary: Pass/fail status for each test

## Advanced Usage

### Testing Failure Scenarios
To test how the system handles failures:

1. Uncomment any of the failure lines in the test functions
2. Run the tests to see failure reporting in action
3. Check both console output and HTML report for failure details

Example:
```python
# In test_analyze_requires_uid function, uncomment:
assert resp.status_code == 200  # This will fail
```

### HTML Report Customization
The HTML report styling can be customized by modifying the CSS in the `generate_html_report()` function. Key styling elements:
- `.header`: Main title section with gradient background
- `.summary-card`: Individual metric cards (total, passed, failed)
- `.test-item`: Individual test result containers
- `.overall-status`: Main pass/fail indicator

### Multiple Test Runs
Each test execution creates separate files with timestamps, allowing you to:
- Compare results across different runs
- Track test performance over time
- Maintain history of test executions

## Troubleshooting

### Common Issues
1. **ImportError**: Ensure stubs are loaded before importing app
2. **404 Endpoints**: Some control endpoints may not be implemented (handled gracefully)
3. **Firebase Errors**: Check that mock modules are properly injected
4. **Color Issues**: If colors don't display properly, your terminal may not support ANSI colors

### Test Failures
Tests are designed to be resilient and handle missing endpoints gracefully. Check:
1. Console output for immediate feedback with color-coded results
2. Log files in `test-results/` folder for detailed information
3. HTML report for visual overview of all test results
4. Final summary section for overall test status

### Performance Notes
- Each test runs independently with fresh mocks
- Log files are written incrementally during test execution
- Console output is optimized for readability with proper spacing
- Test results are preserved across multiple test runs in separate files
- HTML reports are generated quickly and open automatically
- Browser opening can be disabled by modifying the `generate_html_report()` function
