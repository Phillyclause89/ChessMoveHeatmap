[![Pytests](https://github.com/Phillyclause89/ChessMoveHeatmap/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Phillyclause89/ChessMoveHeatmap/actions/workflows/python-app.yml)

# **ChessMoveHeatmap**

A **visual heatmap generator for chess PGN games**, built with Python, `tkinter`, `numpy`, and `python-chess`. This application analyzes possible move activity throughout a chess game and generates a dynamic heatmap visualization.

<p align="center">
  <a href="https://youtu.be/hRwwK6vnPzs?si=hTRCUvtmDCgvCDna" target="_blank">
    <img src="docs/images/ChessMoveHeatmap-depth3.gif" alt="ChessMoveHeatmap Screenshot" width="800">
  </a>
</p>

## **An Introduction by Phillyclause89 (Not ChatGPT or GitHub Copilot)**

The idea for this project got into my head over 2 years ago. And I even got as far as [a version that ran ok upto depth 1](https://youtu.be/tV9pxEQnRHU?si=SSc_HT5Mu8XeKaOa). But sadly, I never put that project on GitHub and only have that video to remember it by. Anyway, someone on [r/learnpython](https://www.reddit.com/r/learnpython/) made a post asking for project ideas and I mentioned this one as a fun one that I remember doing once upon a time. I felt bad that I had lost the code to show for it though, thus I have decided to restart the project from scratch. Though I admit to using chatgpt for doc generation and rubberduck debugging this go around as there is no way I'm typing the rest of this ReadMe out on my own...

## **Features**

- ✅ **PGN File Support** – Load chess games from PGN files for analysis.
- ✅ **Move-by-Move Navigation** – Step through the game and observe heatmap changes dynamically.
- ✅ **Parallelized Heatmap Calculation** – Uses `ProcessPoolExecutor` for efficient computation.
- ✅ **Customizable Board Colors & Fonts** – Adjust square colors and piece fonts directly in the UI.
- ✅ **Real-time Heatmap Updates** – Background processing ensures smooth rendering.  
- ✅ **Real-time Heatmap Updates** – Background processing ensures smooth rendering.

## **Installation**

### **Prerequisites**

Ensure you have Python **3.7 - 3.10** installed. Python **3.11+** is not supported at this stage and will require significant refactoring to imports for compatibility. See [issue #16](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/16) for details.

### **Steps to Set Up the Project**

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Phillyclause89/ChessMoveHeatmap.git
   cd ChessMoveHeatmap
   ```

2. **Set Up a Virtual Environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   - **Preferred Method** (Using `requirements.txt`):

     ```bash
     pip install -r requirements.txt
     ```

   - **Alternative (Untested) Method** (Using `.toml`):
     Use this method if you prefer `.toml` files. However, note that this approach is not officially supported and may not work as expected:

     ```bash
     pip install .[all]
     ```

### **Run the Application**

```bash
python main.py
```

This will open the Chess Heatmap UI, allowing you to load a PGN file for analysis.

### **Optional: Compile with Cython for Performance**

For optimal performance, it is recommended to compile the project and the `python-chess` library with Cython. This can be done using the following no-argument command:

```bash
python setup.py
```

This will compile all relevant components automatically, improving the speed of recursive calculations and other performance-critical operations.

> **Note**: For details on partial compile options or reproducing specific benchmarking tests, see the [Cython Benchmarking Results README](tests/CythonBenchmarkingResults/README.md).

## **Usage**

### **Loading a PGN File**

1. Click **File > Open PGN** and select a `.pgn` chess game file.
2. The heatmap will compute in the background and display when ready.

### **Move Navigation**

- Press the **Right Arrow (`→`)** to advance to the next move.
- Press the **Left Arrow (`←`)** to go back to the previous move.

### **Customization Options**

- **Change Default Board Colors:** `Options > Change Board Colors`
- **Change Default Font:** `Options > Font`
- **Change Depth:** `Options > Change Depth`

### **Standalone Color Legend**

The standalone `standalone_color_legend.py` script can be used to understand what the heatmap colors represent. Integration of a color legend into the main application is planned for a future release. See [issue #17](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/17) for details.

```bash
python standalone_color_legend.py
```

## **Project Structure**

```plaintext
ChessMoveHeatmap/
├── main.py                        # Main GUI Application
├── standalone_color_legend.py     # Standalone app to visualize heatmap colors (legend)
├── setup.py                       # Optional setup script for compiling with Cython (recommended for performance)
├── chmengine/                     # Chess Engine Module
├── chmutils/                      # Heatmap Calculation Utilities
├── heatmaps/                      # Core Heatmap objects, foundational to the entire project
├── tooltips/                      # GUI sub-package for tooltip popups on board hover
├── docs/                          # Documentation and Images
├── tests/                         # Unit Tests and Benchmarking Results (incl. Cython benchmarks)
│   └── CythonBenchmarkingResults/ # Benchmarking results of using Cython for performance
├── pgns/                          # Example PGN Files and game outputs
│   └── trainings/                 # Output directory for engine training games
├── SQLite3Caches/                 # Cached Heatmaps and Q-Tables (Auto-Generated)
│   ├── Better/                    # Cached Heatmaps with the updated algorithm (created during main app usage)
│   ├── QTables/                   # Q-Tables generated by the engine during console interactions
│   └── Faster/                    # Cached Heatmaps with the deprecated, faster discounting algorithm (rarely used)
```

## **Performance Considerations**

- **Recursive Depth & Complexity:** Heatmap calculations have an estimated **O(35^d)** complexity (where `d` is the recursion depth). Higher `depth` values may lead to performance degradation.
- **Parallel Processing:** The app utilizes parallel processing for efficiency, but large depth values can still be computationally intensive.

## **Future Plans**

- 🎨 **Integrate Color Legend** – Adapt `standalone_color_legend.py` into the main UI. See [issue #17](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/17).
- 🚀 **Optimize Performance** – Explore better recursion and caching strategies. See [issue #4](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/4), [issue #8](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/8) and [issue #16](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/16)
- 📈 **Enhanced Visualizations** – Provide more customization for scaling heatmap intensities. See [issue #6](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/6), [issue #7](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/7), [issue #10](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/10) and [issue #11](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/11)
- 👾 **AI Improvements** – Refine logic for `CMHMEngine2` based on game outcomes. See [issue #13](https://github.com/Phillyclause89/ChessMoveHeatmap/issues/13)

## **ChessMoveHeatmap Engines (`chmengine`)**

The `chmengine` module is an experimental chess engine component that leverages heatmaps to inform move decisions. While still in development and not yet integrated into the main application, it provides features such as:

- Playing games against the engine.
- Training the engine using reinforcement learning.

For detailed usage examples and instructions, see the [chmengine README](chmengine/README.md).

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## **Contributors**

- **Phillyclause89** – Project Creator & Lead Developer
- **ChatGPT (OpenAI)** – Development Assistance, Documentation, Debugging, and README Updates
- **GitHub Copilot (OpenAI)** – Code Suggestions, Inline Completions, and Documentation Drafting
