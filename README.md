# **ChessMoveHeatmap**

A **visual heatmap generator for chess PGN games**, built with Python, `tkinter`, and `chess`. This application does a
discounted count
of possible move activity throughout a chess game and generates **gradient-based heatmaps** that highlight move
intensity for each
square at each position in the loaded game.

<p align="center">
  <a href="https://youtu.be/hRwwK6vnPzs?si=hTRCUvtmDCgvCDna" target="_blank">
    <img src="docs/images/ChessMoveHeatmap-depth3.gif" alt="ChessMoveHeatmap Screenshot" width="800">
  </a>
</p>

As you can see in the GIF above, the app assigns a Light Blue color range to the white player and a Yellow range to the
black player. Squares that have no possible moves from either player within the search depth (this case depth=3)
will get the default colors applied to them (which in the example above are set to `"#ffffff"` for light squares
and `"#c0c0c0"` for dark.) If both players have possible moves detected to a square then both players' assigned colors
will merge into a purple square color. For a detailed breakdown of the algorith see the
[docs/COLORALGO.md](docs/COLORALGO.md) file that took chatgpt 30 seconds to write and me an hour to fix the `.md` syntax
of so that it would look nice in GitHub...

## **An Introduction by Phillyclause89 (Not ChatGPT)

The idea for this project got into my head over 2 years ago. And I even got as far
as [a version that ran ok upto depth 1](https://youtu.be/tV9pxEQnRHU?si=SSc_HT5Mu8XeKaOa).
But sadly, I never put that project on GitHub and only have that video to remember it by.
Anyway, someone on [r/learnpython](https://www.reddit.com/r/learnpython/) made a post asking for project ideas and I
mentioned this one as a fun one that I remember doing once upon a time. I felt bad that I had lost the code to show for
it though, thus I have decided to restart the project from scratch. Though I admit to using chatgpt for doc generation
and rubberduck debugging this go around as there is no way I'm typing the rest of this ReadMe out on my own...

## **Features**

✅ **PGN File Support** – Load chess games from PGN files to analyze move activity.  
✅ **Move-by-Move Navigation** – Step through the game and observe heatmap changes dynamically.  
✅ **Parallelized Heatmap Calculation** – Uses `ProcessPoolExecutor` to compute heatmaps efficiently.  
✅ **Configurable Board Colors & Fonts** – Customize square colors and piece fonts in the UI.  
✅ **Real-time Heatmap Updates** – Background processing ensures smooth heatmap rendering.

---

## **Installation**

### **Prerequisites**

Ensure you have Python **3.7+** installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## **Running the Application**

### **Run the main application:**

```bash
python main.py
```

This will open the Chess Heatmap UI, prompting you to **load a PGN file** for analysis.

## **Usage**

### **Loading a PGN File**

1. Click **File > Open PGN** and select a `.pgn` chess game file.
2. The heatmap will be computed in the background (and hopefully load in when ready).

### **Navigating Moves**

- Press the **Right Arrow (`→`)** to advance to the next move
- Press the **Left Arrow (`←`)** to go back to the previous move.

### **Customization**

- **Change Default Board Colors:** `Options > Change Board Colors`
- **Change Default Font:** `Options > Font`

## **Project Structure**

```graphql
ChessMoveHeatmap/
|   .gitattributes
|   .gitignore
|   LICENSE
|   main.py
|   main_basic.py
|   main_basic_piece_counts.py
|   README.md
|   requirements.txt
|   standalone_color_legend.py
|   
+---.github
|   \---workflows
|           python-app.yml
|               
+---chmutils
|   |   __init__.py
|           
+---docs
|   |   COLORALGO.md
|   |           
|   +---images
|   |       ChessMoveHeatmap-depth3-Basic.gif
|   |       ChessMoveHeatmap-depth3.gif
|
+---heatmaps
|   |   __init__.py
|           
+---pgns
|   |   [Assortment of valid and invalid .pgn files for testing...]
|       
+---SQLite3Caches
|   |   [Assortment of sqlite3 db files will be saved here when running main.py]
|       
+---tests
|   |   test_chmutils.py
|   |   test_heatmaps.py
|   |   __init__.py
|           
+---tooltips
|   |   __init__.py
```

### **Key Components**

- `main.py` – The main GUI application, handles PGN loading, move navigation, and heatmap visualization.
- `chmutils.calculate_heatmap` – Recursively calculates the heatmap for a given board state.
- `heatmaps.GradientHeatmap` – Manages heatmap intensity-to-color mapping.
- `heatmaps.GradientHeatmapT` – Base class for type-safe heatmap operations.
- `standalone_color_legend.py` – Standalone Tkinter window for prototyping the color legend (not integrated yet).

## **Performance Considerations**

- **Recursive Depth & Complexity:** The heatmap calculation has an estimated **O(35^d)** complexity, where `d` is the
  recursion
  `depth`.
- **High** `depth` **values** can cause **significant calculation times and/or performance degradation** and may hit
  Python's recursion depth limits.
- The application uses `parallel processing` to optimize calculations across full games, but the underlying function it
  uses does not impose hard limits on recursion depth.
- Right now, the application is hard coded to run at `depth=3` which runs with acceptable load times on my intel i7.

## **License**

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## **Future Plans**

- 🎨 **Integrate Color Legend** – Adapt `standalone_color_legend.py` into the main UI.
- 🚀 **Improve Performance** – Optimize recursion and caching strategies.
- 🏗️ **Database Storage** – Store precomputed heatmaps for faster access.
- 📈 **Enhanced Visualizations** – Provide more customization for heatmap intensity scaling.

### Contributors

- Phillyclause89 - Project Creator & Lead Developer
- ChatGPT (OpenAI) - Developer, Documentation Assistance, Type Hinting Refinements, Complexity Analysis
