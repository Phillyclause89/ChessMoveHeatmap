"""Heatmaps"""
from typing import Dict, Optional, Tuple, Union
from copy import deepcopy
from chess import PIECE_TYPES, Piece, COLORS
from numpy.typing import NDArray, ArrayLike
import numpy as np

PIECES: Tuple[Piece] = tuple(Piece(p, c) for c in COLORS for p in PIECE_TYPES)


class GradientHeatmapT:
    """A base class representing a gradient heatmap for a chessboard position.

    Attributes
    ----------
    shape : Tuple[int, int]
        The fixed shape of the heatmap data array. Default is (64, 2), representing
        64 squares with two values each (one for White and one for Black).
    data : numpy.ndarray
        A NumPy array holding the heatmap data, initialized to zeros.
    """
    _data: NDArray[np.float64]
    _shape: Tuple[int, int] = (64, 2)

    def __init__(self) -> None:
        """Initialize a heatmap with zeros."""
        self._data = np.zeros(self._shape, dtype=np.float64)

    def __getitem__(self, square: int) -> NDArray[np.float64]:
        """Retrieve the heatmap data for a given square.

        Parameters
        ----------
        square : int
            The index of the chessboard square (0 to 63).

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (2,) representing the heatmap data for the square.
        """
        try:
            return self.data[square]
        except IndexError as i:
            raise IndexError(f"square must be in range({self.shape[0]}) got {square}") from i

    def __setitem__(self, square: int, value: Union[NDArray[np.float64], ArrayLike]) -> None:
        """Set the heatmap data for a given square.

        Parameters
        ----------
        square : int
            The index of the chessboard square (0 to 63).
        value : Union[NDArray[numpy.float64], ArrayLike]
            A NumPy array or ArrayLike of shape (2,) representing the new heatmap data for the square.

        Raises
        ------
        ValueError
            If the provided value does not have shape (2,).
        IndexError
            If the provided square is not in range(64)
        """
        expected_shape = (self.shape[1],)
        try:
            if value.shape == expected_shape and value.dtype == np.float64:
                self.data[square] = value
                return None
            raise ValueError(f"Value must have shape {expected_shape}, got {value.shape}.")
        except AttributeError:
            try:
                return self.__setitem__(square, np.asarray(value, dtype=np.float64))
            except ValueError as value_error:
                raise value_error
        except IndexError as i:
            raise IndexError(f"square must be in range({self.shape[0]}) got {square}") from i

    def __add__(self, other: Union["GradientHeatmapT", NDArray[np.float64], ArrayLike]) -> "GradientHeatmap":
        """Perform element-wise addition with another heatmap or compatible array.

        Parameters
        ----------
        other : Union[GradientHeatmapT, numpy.ndarray, ArrayLike]
            The other object to be added, which can be:
            - Another `GradientHeatmapT` instance.
            - A NumPy array or array-like structure that can be cast to shape (64, 2).

        Raises
        ------
        TypeError
            If `other` is neither a `GradientHeatmapT` nor a compatible array.
        ValueError
            If `other` does not have the correct shape after conversion.

        Returns
        -------
        GradientHeatmap
            A new `GradientHeatmap` instance with element-wise summed data.
        """
        try:
            if other.shape == self.shape:
                return GradientHeatmap(self.data + other.data)
            raise ValueError(f"Other {type(other)} must have shape {self.shape}, got {other.shape}.")
        except AttributeError:
            try:
                return self.__add__(GradientHeatmap(np.asarray(other, dtype=np.float64)))
            except Exception as exception:
                text: str = "Other must be a GradientHeatmapT "
                text += f"or a shape {self.shape} NDArray[np.float64] like, got {type(other)}"
                raise TypeError(text) from exception

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the heatmap."""
        return self._shape

    @shape.setter
    def shape(self, value: Tuple[int, int]) -> None:
        """Prevent modification of the shape."""
        raise AttributeError("Shape is immutable and cannot be changed.")

    @property
    def data(self) -> NDArray[np.float64]:
        """Return the heatmap data."""
        return self._data

    @data.setter
    def data(self, value: Union[NDArray[np.float64], ArrayLike]) -> None:
        """Set the heatmap data, ensuring correct type and shape.

        The setter first checks if `value` has the correct shape and dtype.
        If it does not, an `AttributeError` is expected to occur, triggering
        an attempt to convert `value` into a NumPy array. The setter is then
        recursively called with the converted array.

        Parameters
        ----------
        value : Union[numpy.ndarray, ArrayLike]
            The new data to be assigned. If not already a NumPy array, an attempt
            will be made to convert it.

        Raises
        ------
        ValueError
            If the value does not have the expected shape after conversion.
        TypeError
            If the value cannot be converted to a NumPy array of shape (64, 2).
        """
        try:
            if value.shape == self.shape and value.dtype == np.float64:
                self._data = value
                return
            raise ValueError(f"Other {type(value)} must have shape {self.shape}, got {value.shape}.")
        except AttributeError:
            try:
                self.data = np.asarray(value, dtype=np.float64)
                return
            except Exception as exception:
                raise TypeError(f"Other must be a shape {self.shape} ArrayLike, got {type(value)}") from exception


class GradientHeatmap(GradientHeatmapT):
    """A subclass of `GradientHeatmapT` that supports additional operations like normalization and color conversion."""

    def __init__(self, data: Union[Optional[NDArray[np.float64]], GradientHeatmapT, ArrayLike] = None) -> None:
        """Initialize a gradient heatmap.

        Parameters
        ----------
        data : Optional[Union[numpy.ndarray, GradientHeatmapT, ArrayLike]], optional
            The heatmap data. If `None`, initializes an empty heatmap.
            - If a NumPy array of shape (64,2) with dtype numpy.float64 is provided, it is copied.
            - If a `GradientHeatmapT` instance is provided, its data is copied.
            - If an unsupported type is provided, a `TypeError` is raised.
        """
        super().__init__()
        if data is None:
            return
        if isinstance(data, np.ndarray) and data.dtype == np.float64:
            self.data = deepcopy(data)
        elif isinstance(data, (
                GradientHeatmapT, GradientHeatmap, ChessMoveHeatmapT, ChessMoveHeatmap,
        )) or str(type(self)).replace('__main__', 'heatmaps') == str(type(data)):
            self.data = deepcopy(data.data)
        else:
            raise TypeError(
                f"Data must be either a NumPy array of float64 or a GradientHeatmapT instance, got: {type(data)}"
            )

    @property
    def _normalize_(self) -> NDArray[np.float64]:
        """Return the normalized heatmap data.

        Normalization scales intensity values between 0 and 1.

        Returns
        -------
        numpy.ndarray
            The normalized heatmap data.
        """
        max_value: np.float64 = self.data.max(initial=None)
        return self.data / max_value if max_value > 0 else self.data

    @staticmethod
    def _intensity_to_color_(red64: np.float64, blue64: np.float64) -> str:
        """Convert intensity values into a hexadecimal color string.

        The function computes the green channel based on the absolute difference
        between the red and blue intensities. Each channel is scaled and offset,
        and then formatted as a two-digit hexadecimal value.

        Parameters
        ----------
        red64 : numpy.float64
            The normalized intensity value for the red channel.
        blue64 : numpy.float64
            The normalized intensity value for the blue channel.

        Returns
        -------
        str
            A hexadecimal color string in the format '#rrggbb'.
        """
        delta: np.float64 = np.abs(red64 - blue64)
        green: int = 175 + int(80 * delta) if (red64 or blue64) else 175
        red: int = 175 + int(80 * red64) if red64 else 175
        blue: int = 175 + int(80 * blue64) if blue64 else 175
        return f"#{red:02x}{green:02x}{blue:02x}"

    @property
    def colors(self) -> NDArray[np.str_]:
        """Generate an array of color strings for each square.

        Each square's color is determined by converting its normalized red and blue
        intensity values (extracted from the normalized data array) using the
        intensity_to_color function.

        Returns
        -------
        numpy.ndarray
            NumPy array of hexadecimal color codes.
        """
        return np.array([self._intensity_to_color_(s[1], s[0]) for s in self._normalize_], dtype=np.str_)

    def _repr_html_(self) -> str:
        """Render the heatmap's data as an HTML table with colorized rows representing move intensities.

        This method generates an HTML representation of the heatmap, where each square is displayed
        with its corresponding move intensities for White and Black, along with the calculated color
        based on the heatmap data. The row background color is determined by the heatmap intensity for
        each square, and the text color is chosen for contrast to ensure readability.

        The table has the following columns:
        - **Square**: The index of the square (0-63).
        - **White Intensity**: The intensity value for White's moves.
        - **Black Intensity**: The intensity value for Black's moves.
        - **Heat Map Color**: The background color corresponding to the move intensity.

        Returns
        -------
        str
            An HTML string representing the heatmap as a table. The rows are colorized based on the
            heatmap's intensity values, with readable text color determined dynamically based on the
            background color.

        Notes
        -----
        - The `text_color` (black or white) is determined based on the brightness of the background
            color for each square to ensure contrast and legibility.
        - The `colors` array is expected to contain hexadecimal color strings generated based on the
            heatmap's intensity data for White and Black.

        Examples
        --------
        # >>> heatmap = GradientHeatmap()  # Assuming this is an initialized heatmap object
        # >>> heatmap._repr_html_()  # This will generate an HTML table representation of the heatmap.
        """
        colors: NDArray[np.str_] = self.colors  # Get final color values
        sep: str = "</td><td>"

        html: str = f"<h4>{repr(self)}</h4>"
        html += "<table border='1' style='border-collapse: collapse; text-align: center;'>"
        html += "<tr><th>Square</th><th>White Intensity</th><th>Black Intensity</th><th>Heat Map Color</th></tr>"

        i: int
        for i in range(64):
            bg_color: str = colors[i]  # Background color based on heatmap intensity
            text_color: str = "#FFFFFF" if int(bg_color[1:3], 16) < 128 else "#000000"  # Ensure readable text

            html += f"<tr style='background-color:{bg_color}; color:{text_color};'>"
            html += f"<td>{i}{sep}{self.data[i, 0]:.2f}{sep}{self.data[i, 1]:.2f}{sep}{bg_color}</td></tr>"

        html += "</table>"
        return html


class ChessMoveHeatmapT(GradientHeatmap):
    """A gradient heatmap that additionally tracks chess piece counts per square.

    This class extends `GradientHeatmap` by maintaining, for each square, a dictionary
    mapping each chess piece to a floating point value. This value can represent an intensity
    or count of moves involving that piece on the square.

    Attributes
    ----------
    piece_counts : numpy.ndarray of dict
        A NumPy array of shape (64,) where each element is a dictionary mapping
        chess `Piece` objects to a `float` count.
    """
    _piece_counts: NDArray[Dict[Piece, np.float64]]

    def __init__(self, **kwargs) -> None:
        """Initialize the ChessMoveHeatmapT.

        All inherited data is initialized via `GradientHeatmap.__init__()`, and a new
        `piece_counts` array is created with zero values for all pieces on each square.
        """
        super().__init__(**kwargs)
        self._piece_counts = np.array(
            [{k: np.float64(0) for k in PIECES} for _ in range(64)],
            dtype=dict
        )

    @property
    def piece_counts(self) -> NDArray[Dict[Piece, np.float64]]:
        """
        Get the piece count array.

        Returns
        -------
        NDArray[Dict[Piece, np.float64]]
            A NumPy array of length 64 where each element is a dictionary mapping chess pieces
            to their associated count or intensity.
        """
        return self._piece_counts

    @piece_counts.setter
    def piece_counts(self, value: NDArray[Dict[Piece, np.float64]]) -> None:
        """
        Set the piece count array.

        Parameters
        ----------
        value : NDArray[Dict[Piece, np.float64]]
            A NumPy array of shape (64,) where each element is a dictionary mapping chess pieces
            to a float. The array should have dtype `object`.

        Raises
        ------
        ValueError
            If the provided value does not have shape (64,).
        TypeError
            If the value cannot be converted to a NumPy array of dictionaries.
        """
        try:
            if value.shape == (64,) and value.dtype == object:
                self._piece_counts = value
                return
            raise ValueError(f"Other {type(value)} must have shape (64,), got {value.shape}.")
        except AttributeError:
            try:
                self.piece_counts = np.asarray(value, dtype=dict)
                return
            except Exception as exception:
                raise TypeError(f"Other must be a shape (64,) ArrayLike, got {type(value)}") from exception


class ChessMoveHeatmap(ChessMoveHeatmapT):
    """A concrete extension of `ChessMoveHeatmapT` for tracking move-related piece counts.

    This class refines `ChessMoveHeatmapT` by optionally initializing the piece counts from an external
    source. If neither `piece_counts` nor an existing heatmap data object is provided, the default
    initialization from `ChessMoveHeatmapT` is used.
    """

    def __init__(
            self,
            piece_counts: Optional[NDArray[Dict[Piece, np.float64]]] = None,
            **kwargs
    ) -> None:
        """Initialize the ChessMoveHeatmap instance.

        Parameters
        ----------
        piece_counts : Optional[NDArray[Dict[Piece, np.float64]]], optional
            A NumPy array of shape (64,) where each element is a dictionary mapping chess pieces
            to a float representing move intensity. If not provided, the default initialization
            from the base class is used.
        **kwargs
            Additional keyword arguments passed to the base class.

        Raises
        ------
        TypeError
            If `piece_counts` is provided but is not a NumPy array with the expected dtype.
        """
        super().__init__(**kwargs)
        data: Optional[object] = kwargs.get("data")
        if piece_counts is None and data is None:
            return
        if isinstance(piece_counts, np.ndarray) and piece_counts.dtype == object:
            self.piece_counts = deepcopy(piece_counts)
        elif isinstance(
                data, (ChessMoveHeatmapT, ChessMoveHeatmap)
        ) or str(type(self)).replace('__main__', 'heatmaps') == str(type(data)):
            self.piece_counts = deepcopy(data.piece_counts)
        elif piece_counts is not None:
            raise TypeError(f"piece_counts must be a NumPy array of object, got {type(piece_counts)}")
