# `chmutils`

The `chmutils` module provides utilities for calculating and caching chess move heatmaps.
These heatmaps visualize the intensity of possible future moves for any chess position,
helping to analyze and understand chess strategies.

## Functions

### `calculate_heatmap`

```python
from chmutils import calculate_heatmap
from chess import Board

hmap = calculate_heatmap(board=Board(), depth=3)
```

Recursively computes a gradient heatmap for a given chess board position.

**Parameters:**

- `board` (chess.Board): The chess board position for which to calculate the heatmap.
- `depth` (int, optional): The recursion depth to explore legal moves. Default is 1.
- _`heatmap` Parameter is used internally during recursive calls of itself and should not be specified._
- _`discount` Parameter is used internally during recursive calls of itself and should not be specified._

**Returns:**

- `GradientHeatmap`: The computed heatmap with accumulated move intensities for each square.

### `calculate_chess_move_heatmap`

```python
from chmutils import calculate_chess_move_heatmap
from chess import Board

cmhmap = calculate_chess_move_heatmap(board=Board(), depth=3)
```

Recursively computes a chess move heatmap that tracks both move intensities and piece counts.

**Parameters:**

- `board` (chess.Board): The chess board position for which to calculate the move heatmap.
- `depth` (int, optional): The recursion depth to explore legal moves. Default is 1.
- _`heatmap` Parameter is used internally during recursive calls of itself and should not be specified._
- _`discount` Parameter is used internally during recursive calls of itself and should not be specified._

**Returns:**

- `ChessMoveHeatmap`: The computed heatmap containing move intensities and piece counts.

### `calculate_chess_move_heatmap_with_better_discount`

```python
from chmutils import calculate_chess_move_heatmap_with_better_discount
from chess import Board

cmhmap_better = calculate_chess_move_heatmap_with_better_discount(board=Board(), depth=3)
```

Recursively computes a chess move heatmap with a discount based on the branching factor at each level.

**Parameters:**

- `board` (chess.Board): The current chess board state.
- `depth` (int, optional): The depth to which moves are recursively explored. Default is 1.

**Returns:**

- `ChessMoveHeatmap`: A composite heatmap reflecting the discounted cumulative influence of moves.

### `flatten_heatmap`

```python
from chmutils import calculate_chess_move_heatmap_with_better_discount, flatten_heatmap
from chess import Board

cmhmap_better = calculate_chess_move_heatmap_with_better_discount(board=Board(), depth=3)
cmhmap_better_flat = flatten_heatmap(heatmap=cmhmap_better)
```

Flattens a `ChessMoveHeatmap` into a dictionary of primitive values.

**Parameters:**

- `heatmap` (ChessMoveHeatmap): The heatmap to be flattened.

**Returns:**

- `Dict[str, float64]`: A dictionary with keys for each square's move intensities and piece counts.

### `inflate_heatmap`

```python
from chmutils import calculate_chess_move_heatmap_with_better_discount, flatten_heatmap, inflate_heatmap
from chess import Board

cmhmap_better = calculate_chess_move_heatmap_with_better_discount(board=Board(), depth=3)
cmhmap_better_flat = flatten_heatmap(heatmap=cmhmap_better)
cmhmap_better_again = inflate_heatmap(data=cmhmap_better_flat)
```

Inflates a flat dictionary of primitive values back into a `ChessMoveHeatmap`.

**Parameters:**

- `data` (Dict[str, float]): A flat dictionary representing a `ChessMoveHeatmap`.

**Returns:**

- `ChessMoveHeatmap`: The reconstituted heatmap.

### `HeatmapCache`

Class for caching `ChessMoveHeatmap` objects using SQLite.

**Attributes:**

- `depth` (int): The recursion depth for heatmap calculations.
- `board` (chess.Board): The chess board whose heatmap is being cached.
- `db_path` (str): Path to the SQLite database used for caching.

### `get_or_compute_heatmap`

```python
from chmutils import get_or_compute_heatmap
from chess import Board

board = Board()
computed_hmap = get_or_compute_heatmap(board=board, depth=3)
fetched_hmap = get_or_compute_heatmap(board=board, depth=3)
```

Retrieves a cached `ChessMoveHeatmap` or computes and caches it if not available.

**Parameters:**

- `board` (chess.Board): The chess board position.
- `depth` (int): The recursion depth for calculating the heatmap.

**Returns:**

- `ChessMoveHeatmap`: The corresponding heatmap.

### `BetterHeatmapCache`

Subclass of `HeatmapCache` with an overridden cache directory.

### `get_or_compute_heatmap_with_better_discounts`

```python
from chmutils import get_or_compute_heatmap_with_better_discounts
from chess import Board

board = Board()
computed_cmhmap = get_or_compute_heatmap_with_better_discounts(board=board, depth=3)
fetched_cmhmap = get_or_compute_heatmap_with_better_discounts(board=board, depth=3)
```

Retrieves a cached `ChessMoveHeatmap` with better discounts or computes and caches it if not available.

**Parameters:**

- `board` (chess.Board): The chess board position.
- `depth` (int): The recursion depth for calculating the heatmap.

**Returns:**

- `ChessMoveHeatmap`: The corresponding heatmap.