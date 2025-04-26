# Cython Benchmarking Results

## Purpose
This directory contains the results of benchmarking tests performed on the `ChessMoveHeatmap` project to evaluate the performance impact of Cython compilation under different configurations. The tests aim to identify the fastest and most efficient setup for running the project's unit tests and focus benchmarks.

---

## Benchmarking Scenarios
The benchmarking was conducted under the following four scenarios, each involving targeted compilation of specific files:

1. **Everything Compiled**:
   - All specified `.py` files in the repository and the `python-chess` library were compiled with Cython.
   - **From the `python-chess` library**:
     - Every `.py` file in the library's directory, except for `_interactive.py`, was compiled.
   - **From the repository**:
     - The following files were compiled:
       - `heatmaps\\__init__.py`
       - `chmutils\\__init__.py`
       - `chmengine\\utils\\pick.py`
       - `chmengine\\utils\\__init__.py`
       - `chmengine\\engines\\cmhmey1.py`
       - `chmengine\\engines\\quartney.py`
       - `chmengine\\engines\\cmhmey2.py`
       - `chmengine\\engines\\__init__.py`
       - `chmengine\\__init__.py`

2. **Chess-lib Compiled**:
   - Only the targeted `.py` files in the `python-chess` library were compiled, excluding `_interactive.py`.

3. **Repo Compiled**:
   - Only the targeted `.py` files in the repository (listed above) were compiled.

4. **No Compilation**:
   - No Cython compilation was performed; all `.py` files ran as standard Python code.

---

## Benchmarking Results

### Key Metrics
- **Setup Time**: Total time taken to compile the specified `.py` files in the given scenario.
- **Full Test Suite Runtime**: Total execution time for all 91 unit tests in the `tests` directory.
- **Focus Test Runtime (`test_pick_move`)**: Total runtime for the `test_pick_move` function.
- **`s/branch`**: Time taken per first-level game tree branch analyzed by the `pick_move` method.
- **90th Percentile Time (`s/branch`)**: Time under which 90% of `s/branch` measurements fall, representing upper-normal performance.

---

### Analysis of `s/branch` Metric
The `s/branch` metric represents the time taken by the `pick_move` method for each first-level game tree branch from the current board position. Here's a breakdown:

1. **Notation**:
   - `s` refers to the elapsed time in seconds.
   - `branch` refers to the number of first-level game tree branches analyzed.

2. **Game Tree Exploration**:
   - The `pick_move` method evaluates potential moves from a given position, exploring deeper branches in case of checks or captures.
   - The `s/branch` metric does not account for deeper game tree branches beyond the first level due to limitations in the unittest's scope.

3. **Example**:
   - For the initial chess position (`rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`), the following applies:
     - White has **20 possible moves** (first-level branches).
     - Black's follow-up results in **20x20 = 400 boards** analyzed.
   - Since there are no checks or captures in this simplified example, the `pick_move` method exits after analyzing the first two plies.
   - The reported `s/branch` time for this position is equivalent to `s/20`.

---

### Detailed Results
| **Scenario**          | **Setup Time** | **Full Test Suite Runtime** | **Focus Test Runtime (`test_pick_move`)** | **Mean Pick Time (`s/branch`)** | **90th Percentile Pick Time (`s/branch`)** | **Mean Response Time (`s/branch`)** | **90th Percentile Response Time (`s/branch`)** | **Mean Revisit Time (`s/branch`)** | **90th Percentile Revisit Time (`s/branch`)** |
|------------------------|----------------|-----------------------------|-------------------------------------------|----------------------------------|--------------------------------------------|-------------------------------------|-----------------------------------------------|------------------------------------|-----------------------------------------------|
| Everything Compiled    | 5 m 20 s       | 11 m 49 s                  | 3 m 44 s                                  | 0.0025                           | 0.0027                                     | 0.02545                             | 0.02656                                        | 0.0012                             | 0.0013                                        |
| Chess-lib Compiled     | 5 m 5 s        | 12 m 20 s (+0:31)          | 3 m 52 s (+0:08)                          | 0.00255 (+0.00005 s/branch)      | 0.0028 (+0.00010 s/branch)                 | 0.0264 (+0.00095 s/branch)          | 0.02762 (+0.00106 s/branch)                   | 0.00125 (+0.00005 s/branch)        | 0.00135 (+0.00005 s/branch)                   |
| Repo Compiled          | 16 s           | 12 m 29 s (+0:40)          | 3 m 52 s (+0:08)                          | 0.0026 (+0.00010 s/branch)       | 0.00285 (+0.00015 s/branch)                | 0.02635 (+0.00090 s/branch)         | 0.02762 (+0.00106 s/branch)                   | 0.00125 (+0.00005 s/branch)        | 0.00135 (+0.00005 s/branch)                   |
| No Compilation         | 0 s            | 12 m 50 s (+1:01)          | 3 m 59 s (+0:15)                          | 0.0026 (+0.00010 s/branch)       | 0.0029 (+0.00020 s/branch)                 | 0.02725 (+0.00180 s/branch)         | 0.02868 (+0.00212 s/branch)                   | 0.00125 (+0.00005 s/branch)        | 0.00135 (+0.00005 s/branch)                   |

---

## Conclusion

The primary objective of this benchmarking experiment was to determine whether the speed gains from various compiled states justify their setup costs, particularly in the context of a continuous integration (CI) pipeline.

The results clearly show that the **"Everything Compiled" setup** offers the best overall performance, with significant runtime savings across all benchmarks. However, its high setup time of **5 minutes and 20 seconds** makes it impractical for short-term or one-off executions, like those in a CI pipeline. On the other hand, the **No Compilation** setup, while slower in raw runtime, avoids any setup overhead and remains efficient for one-off tasks.

### Final Recommendations:
1. **For Long-Term, Repeated Use**:
   - The **"Everything Compiled" setup** is the optimal choice for scenarios where the same code is run repeatedly over an extended period. The initial setup cost is easily amortized by the significant runtime savings across repeated executions.

2. **For Short-Term, One-Off Use (e.g., CI Pipelines)**:
   - The **No Compilation** setup is the best option for short-term or one-off tasks, such as CI pipeline actions, where minimizing total execution time (including setup) is critical.

These findings reinforce the importance of balancing setup costs against runtime savings when optimizing for different use cases, ensuring that the chosen setup is tailored to the specific needs of the task or workflow.

---

## Reproducing the Tests
Follow these steps to reproduce the benchmarking tests:

1. **Install Dependencies**:
   - Ensure you have Python 3.7 or higher and Cython installed in your environment.

2. **Set Up a Virtual Environment** (Highly Recommended):
   - To avoid modifying your global Python environment, create and activate a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - This ensures that any compilation or changes are restricted to the virtual environment.

3. **Compile the Code**:
   - Use the following commands to compile the code under different scenarios:
     - Everything Compiled: `python setup.py --all`
     - Chess-lib Compiled: `python setup.py --chess`
     - Repo Compiled: `python setup.py --main`

   **Important Note**: The `setup.py` script dynamically imports the `chess` library to parse its module location in the active Python environment. If you do not use a virtual environment, your global site-packages version of `chess` may be compiled, which could have unintended consequences.

4. **Run the Tests**:
   - Execute the tests using the `pytest` framework:
     ```bash
     pytest tests/
     ```

5. **Focus Benchmark**:
   - Run the `test_pick_move` benchmark separately:
     ```bash
     pytest tests/test_pick_move.py
     ```
---

## Notes on Dynamic Test Runner
The repository includes a dynamic GitHub Actions runner that executes only the tests relevant to the most recent code changes. This ensures efficient CI/CD pipelines without rerunning unnecessary tests. While the benchmarking tests were run manually for consistency and thoroughness, the dynamic runner complements ongoing development by keeping the testing process targeted and efficient.