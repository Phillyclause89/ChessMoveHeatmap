# `heatmaps`

The `heatmaps` module provides classes and methods for creating and manipulating heatmaps representing
chessboard positions. These heatmaps can track the intensity of possible future moves
and maintain counts of chess pieces on each square.

## Classes

### `GradientHeatmapT`

A base class representing a gradient heatmap for a chessboard position.

**Attributes:**

- `shape`: Tuple[int, int] - The fixed shape of the heatmap data array (default is (64, 2)).
- `data`: numpy.ndarray - A NumPy array holding the heatmap data, initialized to zeros.

**Methods:**

- `__getitem__(self, square: Union[int, Iterable]) -> NDArray[float64]`: Retrieve the heatmap data for a given square.
- `__setitem__(self, square: int, value: Union[NDArray[float64], ArrayLike]) -> None`: Set the heatmap data for a given
  square.
- `__add__(self, other: Union["GradientHeatmapT", NDArray[float64], ArrayLike]) -> "GradientHeatmap"`: Perform
  element-wise addition with another heatmap or compatible array.

### `GradientHeatmap`

A subclass of `GradientHeatmapT` that supports additional operations like normalization and color conversion.

**Methods:**

- `__init__(self, data: Union[Optional[NDArray[float64]], GradientHeatmapT, ArrayLike] = None) -> None`: Initialize a
  gradient heatmap.
- `colors`: Generate an array of color strings for each square.
- `_repr_html_(self) -> str`: Render the heatmap's data as an HTML table with colorized rows representing move
  intensities.

### `ChessMoveHeatmapT`

A gradient heatmap that additionally tracks chess piece counts per square.

**Attributes:**

- `piece_counts`: numpy.ndarray of dict - A NumPy array of shape (64,) where each element is a dictionary mapping
  chess `Piece` objects to a `float` count.

**Methods:**

- `__truediv__(self, divisor: Real) -> "ChessMoveHeatmap"`: Returns a new ChessMoveHeatmap where each value is divided
  by the given divisor.
- `__add__(self, other: Union["ChessMoveHeatmapT", Tuple[NDArray[float64], NDArray[Dict[Piece, float64]]], ArrayLike, Dict[str, Union[NDArray[float64], NDArray[Dict[Piece, float64]]]]]) -> "ChessMoveHeatmap"`:
  Perform element-wise addition with another heatmap or compatible array.
- `add_piece_counts(self, other_piece_counts: NDArray[Dict[Piece, float64]]) -> NDArray[Dict[Piece, float64]]`: Add
  piece counts from another heatmap.

### `ChessMoveHeatmap`

A concrete extension of `ChessMoveHeatmapT` for tracking move-related piece counts.

**Methods:**

- `__init__(self, piece_counts: Optional[NDArray[Dict[Piece, float64]]] = None, **kwargs) -> None`: Initialize the
  ChessMoveHeatmap instance.