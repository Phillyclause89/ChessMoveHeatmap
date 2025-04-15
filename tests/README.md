# Tests Overview

This directory contains the test suite for the **ChessMoveHeatmap** project. These tests ensure that the various 
components of the system function correctly and validate the reliability of the chess move heatmap generation tool.

## Types of Tests

The test suite is divided into the following categories:

1. **Core Functionality Tests**:  
   These validate the primary logic behind generating chess move heatmaps, ensuring accurate visualization for any given chess position.

2. **Engine-Specific Tests**:  
   Focused on individual chess engines implemented in the project, these tests ensure that each engine behaves as expected.

3. **Integration Tests**:  
   These tests verify that different components of the system collaborate seamlessly to produce the desired outputs.

4. **Utility Tests**:  
   Cover helper functions and utilities that support the main application logic.

## Dependency Analysis in Pipelines

The repository uses a custom dependency analysis script, located at `.github/scripts/scripts_analyze_dependencies.py`, 
to determine which tests are relevant to run during CI/CD workflows. This script identifies the dependencies of each 
test and matches them against the files modified in a commit to decide the subset of tests to execute.

This approach optimizes the testing pipeline by running only the necessary tests, 
reducing testing time while maintaining confidence in the changes introduced.

## **Important Note on Imports**

The way test scripts import modules or classes significantly impacts the dependency analysis. 
Improper imports can lead to undesired tests being executed, potentially causing inefficiencies 
or misleading test results.

### Example Scenario

If a test script imports `chmengine.engines.cmhmey1` indirectly through its parent module (e.g., `chmengine`), 
changes to unrelated parts of `chmengine`, such as `chmengine.play`, may trigger the execution of tests for `cmhmey1`. 
This issue was demonstrated in [commit 6088510](https://github.com/Phillyclause89/ChessMoveHeatmap/commit/6088510b963770bb52a6f7978c069cbc6a823c3e).

### Best Practices for Imports

- **Direct Imports**: Always import the specific module or class required for the test. For example:
  ```python
  from chmengine.engines import cmhmey1
  ```
  instead of:
  ```python
  import chmengine
  ```
- **Targeted Tests**: Ensure that each test script is tailored to test only the functionality directly related to the module or component being modified.

By adhering to these practices, you ensure that the pipeline runs only the necessary tests, 
making the CI/CD process efficient and accurate.

## Running Tests Locally

To execute the tests locally, navigate to the root of the repository and run:
```bash
python -m unittest discover
```

This command will automatically discover and execute all tests in the `/tests` directory.

---
