"""Tests Heatmaps"""
from re import fullmatch
from typing import Dict, List, Optional
from unittest import TestCase, main

from chess import Piece
from numpy import array, float64, isnan, ndarray, random as np_random, testing as np_testing, zeros
from numpy.typing import NDArray

from heatmaps import ChessMoveHeatmap, ChessMoveHeatmapT, GradientHeatmap, GradientHeatmapT
from tests.utils import INVALID_OBJ_STR, PIECES, SHAPE, validate_data_types


class TestGradientHeatmapT(TestCase):
    """Unit tests for the GradientHeatmapT class."""
    heatmap: Optional[GradientHeatmapT]

    def setUp(self) -> None:
        """Set up a fresh heatmap for each test."""
        self.heatmap = GradientHeatmapT()

    def test_initialization(self):
        """Test if heatmap initializes with correct shape and zeros."""
        validate_data_types((self.heatmap,), self)
        np_testing.assert_array_equal(self.heatmap.data, zeros(SHAPE, dtype=float64))

    def test_getitem_setitem(self) -> None:
        """Test setting and getting heatmap values."""
        value0: List[float] = [1.5, 2.5]
        self.assertNotIsInstance(value0, ndarray)
        value: NDArray[float64] = array([x * 2 for x in value0], dtype=float64)
        self.assertIsInstance(value, ndarray)
        self.heatmap[63] = value
        np_testing.assert_array_equal(self.heatmap[-1], value)
        self.heatmap[-64] = value0
        np_testing.assert_array_equal(self.heatmap[0], value0)
        self.assertIsInstance(self.heatmap[-64], ndarray)
        validate_data_types((self.heatmap,), self)

    # noinspection PyTypeChecker,PydanticTypeChecker
    def test_invalid_getitem_setitem(self) -> None:
        """Ensure Error is raised when setting incorrectly per the object model."""
        with self.assertRaises(ValueError):
            self.heatmap[0] = INVALID_OBJ_STR
        with self.assertRaises(ValueError):
            self.heatmap[1] = [0.0, INVALID_OBJ_STR]
        with self.assertRaises(ValueError):
            self.heatmap[2] = None
        with self.assertRaises(ValueError):
            self.heatmap[3] = array([1.5, 2.5, 3.0], dtype=float64)
        with self.assertRaises(IndexError):
            self.heatmap[64] = array([2.5, 3.0], dtype=float64)
        with self.assertRaises(IndexError):
            self.heatmap[-65] = array([2.5, 3.0], dtype=float64)
        validate_data_types((self.heatmap,), self)

    def test_addition(self) -> None:
        """Test addition of two GradientHeatmapT instances and a valid np.array."""
        other_heatmap: GradientHeatmapT = GradientHeatmapT()
        self.heatmap[5] = array([1.0, -1], dtype=float64)
        other_heatmap[5] = [2.0, -3]
        result1: GradientHeatmap = self.heatmap + other_heatmap
        self.assertIsInstance(result1, GradientHeatmap)  # Validate type
        np_testing.assert_array_equal(result1[5], other_heatmap[5] + self.heatmap[5])
        # Technically, we are testing GradientHeatmap(GradientHeatmapT).__add__ here, but I'm ok with that
        result2: GradientHeatmap = result1 + self.heatmap.data
        self.assertIsInstance(result2, GradientHeatmap)
        np_testing.assert_array_equal(result2[5], result1[5] + self.heatmap[5])
        validate_data_types((self.heatmap, other_heatmap, result1, result2), self)

    def test_invalid_addition_type(self) -> None:
        """Ensure TypeError is raised when adding an incompatible type."""
        with self.assertRaises(TypeError):
            _ = self.heatmap + INVALID_OBJ_STR
        result1: GradientHeatmap = self.heatmap + self.heatmap
        np_testing.assert_array_equal(result1.data, self.heatmap.data)
        with self.assertRaises(TypeError):
            _ = result1 + array([[INVALID_OBJ_STR, INVALID_OBJ_STR]] * 64, dtype=object)
        result2: GradientHeatmap = result1 + zeros(SHAPE, dtype=float64)
        np_testing.assert_array_equal(result2.data, self.heatmap.data)
        with self.assertRaises(TypeError):
            _ = result2 + [1.5]
        validate_data_types((self.heatmap, result1, result2), self)

    def test_invalid_addition_shape(self) -> None:
        """Ensure ValueError is raised when setting incorrect shape."""
        with self.assertRaises(ValueError):
            _ = self.heatmap + array([1.5, 2.5, 3.0], dtype=float64)
        with self.assertRaises(ValueError):
            _ = self.heatmap + array([3.0], dtype=float64)
        validate_data_types((self.heatmap,), self)


class TestGradientHeatmap(TestCase):
    """Unit tests for the GradientHeatmap class."""
    heatmap: GradientHeatmap
    data: NDArray[float64]

    def setUp(self) -> None:
        """Set up a GradientHeatmap instance."""
        self.data = np_random.rand(64, 2).astype(float64)
        self.heatmap = GradientHeatmap(self.data)

    def test_initialization(self) -> None:
        """Test if heatmap initializes correctly from valid data."""
        validate_data_types((self.heatmap,), self)
        np_testing.assert_array_equal(self.heatmap.data, self.data)

    def test_normalization(self) -> None:
        """Test normalization functionality."""
        # pylint: disable=protected-access
        normalized: NDArray[float64] = self.heatmap._normalize_
        max_val: float64 = self.data.max(initial=None)
        expected: NDArray[float64] = self.data / max_val if max_val > 0 else self.data
        np_testing.assert_array_almost_equal(normalized, expected)
        validate_data_types((self.heatmap,), self)

    def test_intensity_to_color(self) -> None:
        """Test if color conversion produces valid hex RGB values."""
        # pylint: disable=protected-access
        color: str = self.heatmap._intensity_to_color_(float64(0.5), float64(0.5))
        self.assertTrue(fullmatch(r"#[0-9A-Fa-f]{6}", color), f"Invalid hex color format: {color}")
        validate_data_types((self.heatmap,), self)

    def test_colors(self) -> None:
        """Test if colors property returns valid hex color codes for all squares."""
        colors: NDArray[str] = self.heatmap.colors
        self.assertEqual(colors.shape, (SHAPE[0],))

        color: str
        for color in colors:
            self.assertTrue(fullmatch(r"#[0-9A-Fa-f]{6}", color), f"Invalid hex color format: {color}")
        validate_data_types((self.heatmap,), self)

    def test_html_representation(self) -> None:
        """Ensure HTML representation is properly formatted."""
        # pylint: disable=protected-access
        html: str = self.heatmap._repr_html_()
        self.assertIn("<table", html)
        self.assertIn("</table>", html)
        validate_data_types((self.heatmap,), self)


class TestChessMoveHeatmapT(TestCase):
    """Unit tests for ChessMoveHeatmapT."""
    heatmap: ChessMoveHeatmapT

    def setUp(self) -> None:
        """Set up a ChessMoveHeatmapT instance."""
        self.heatmap = ChessMoveHeatmapT()

    def test_initialization(self) -> None:
        """Ensure piece counts initialize correctly."""
        validate_data_types((self.heatmap,), self, True)
        self.assertEqual(self.heatmap.piece_counts.shape, (SHAPE[0],))
        counts: Dict[Piece, float64]
        for counts in self.heatmap.piece_counts:
            self.assertEqual(len(counts), len(PIECES))

    def test_piece_counts_setter(self) -> None:
        """Test setting piece count data."""
        new_counts: NDArray[Dict[Piece, float64]] = array(
            [{p: float64(1) for p in PIECES} for _ in range(SHAPE[0])], dtype=object
        )
        self.heatmap.piece_counts = new_counts
        np_testing.assert_array_equal(self.heatmap.piece_counts, new_counts)
        validate_data_types((self.heatmap,), self, True)

    def test_invalid_piece_counts(self) -> None:
        """Ensure TypeError is raised for invalid piece_counts assignment."""
        with self.assertRaises(TypeError):
            self.heatmap.piece_counts = "invalid"

        with self.assertRaises(ValueError):
            self.heatmap.piece_counts = zeros((32,), dtype=object)
        validate_data_types((self.heatmap,), self, True)

    def test_addition(self) -> None:
        """Test addition of two ChessMoveHeatmapT instances and a valid constructor args"""
        other_heatmap = ChessMoveHeatmapT()
        self.heatmap[5] = array([1.0, -1], dtype=float64)
        piece: Piece
        self.heatmap.piece_counts[5] = {piece: float64(1) for piece in PIECES}
        other_heatmap[5] = [2.0, -3]
        other_heatmap.piece_counts[5] = {piece: float64(2) for piece in PIECES}
        # ChessMoveHeatmapT addition
        result1: ChessMoveHeatmap = self.heatmap + other_heatmap
        self.assertIsInstance(result1, ChessMoveHeatmap)  # Validate type
        np_testing.assert_array_equal(result1[5], other_heatmap[5] + self.heatmap[5])
        self.assertEqual(
            sum(result1.piece_counts[5].values()),
            sum(other_heatmap.piece_counts[5].values()) + sum(self.heatmap.piece_counts[5].values())
        )
        # Tuple addition
        i: int
        pieces_to_add: NDArray[Dict[Piece, float64]] = array(
            [{piece: float64(i) for piece in PIECES} for i in range(64)],
            dtype=dict
        )
        result2: ChessMoveHeatmap = result1 + (
            self.heatmap.data,
            pieces_to_add
        )
        self.assertIsInstance(result2, ChessMoveHeatmap)
        np_testing.assert_array_equal(result2[5], result1[5] + self.heatmap[5])
        # Dict addition
        result3: ChessMoveHeatmap = result2 + {"data": self.heatmap.data, "piece_counts": self.heatmap.piece_counts}
        self.assertIsInstance(result3, ChessMoveHeatmap)
        np_testing.assert_array_equal(result3[5], result2[5] + self.heatmap[5])
        self.assertEqual(
            sum(result3.piece_counts[5].values()),
            sum(self.heatmap.piece_counts[5].values()) + sum(result2.piece_counts[5].values())
        )
        for piece in PIECES:
            square: int
            for square in range(64):
                self.assertEqual(
                    result1.piece_counts[square][piece],
                    self.heatmap.piece_counts[square][piece] + other_heatmap.piece_counts[square][piece]
                )
                self.assertEqual(
                    result2.piece_counts[square][piece],
                    result1.piece_counts[square][piece] + pieces_to_add[square][piece]
                )
                self.assertEqual(
                    result3.piece_counts[square][piece],
                    self.heatmap.piece_counts[square][piece] + result2.piece_counts[square][piece]
                )
        validate_data_types((self.heatmap, other_heatmap, result1, result2, result3), self, True)

    def test_invalid_addition_type(self) -> None:
        """Ensure TypeError is raised when adding an incompatible type."""
        with self.assertRaises(TypeError):
            _ = self.heatmap + INVALID_OBJ_STR
        result1: ChessMoveHeatmap = self.heatmap + self.heatmap
        np_testing.assert_array_equal(result1.data, self.heatmap.data)
        with self.assertRaises(TypeError):
            _ = result1 + array([[INVALID_OBJ_STR, INVALID_OBJ_STR]] * 64, dtype=object)
        result2: ChessMoveHeatmap = result1 + (zeros(SHAPE, dtype=float64), self.heatmap.piece_counts)
        np_testing.assert_array_equal(result2.data, self.heatmap.data)
        with self.assertRaises(TypeError):
            _ = result2 + array([[INVALID_OBJ_STR, INVALID_OBJ_STR]], dtype=object)
        with self.assertRaises(TypeError):
            _ = self.heatmap + (self.heatmap.data, INVALID_OBJ_STR)
        with self.assertRaises(TypeError):
            _ = self.heatmap + [INVALID_OBJ_STR, self.heatmap.piece_counts]
        validate_data_types((self.heatmap, result1, result2), self, True)

    def test_invalid_addition_shape(self) -> None:
        """Ensure ValueError is raised when setting incorrect shape."""
        with self.assertRaises(ValueError):
            _ = self.heatmap + array([1.5, 2.5, 3.0], dtype=float64)
        with self.assertRaises(ValueError):
            _ = self.heatmap + array([3.0], dtype=float64)
        with self.assertRaises(ValueError):
            _ = self.heatmap + []
        with self.assertRaises(ValueError):
            _ = self.heatmap + (self.heatmap.data, array([{0: 0}], dtype=dict))
        validate_data_types((self.heatmap,), self, True)

    def test_division(self) -> None:
        """Test division of ChessMoveHeatmapT"""
        """Test division of ChessMoveHeatmapT."""
        # Golden path: valid divisor
        divisor = 2
        result = self.heatmap / divisor
        validate_data_types((self.heatmap, result,), self, True)
        np_testing.assert_array_equal(result.data, self.heatmap.data / divisor)
        for square in range(64):
            for piece, count in self.heatmap.piece_counts[square].items():
                self.assertAlmostEqual(
                    result.piece_counts[square][piece],
                    count / divisor,
                    msg=f"Failed for square {square}, piece {piece}"
                )

        # Edge case: floating-point divisor
        divisor = 2.5
        result = self.heatmap / divisor
        validate_data_types((self.heatmap, result,), self, True)
        np_testing.assert_array_equal(result.data, self.heatmap.data / divisor)
        for square in range(64):
            for piece, count in self.heatmap.piece_counts[square].items():
                self.assertAlmostEqual(
                    result.piece_counts[square][piece],
                    count / divisor,
                    msg=f"Failed for square {square}, piece {piece}"
                )

        # Non-golden path: ZeroDivisionError
        with self.assertRaises(ZeroDivisionError):
            _ = self.heatmap / 0

        # Allow NaN to propagate without raising an error
        divisor = float('nan')
        result = self.heatmap / divisor
        validate_data_types((self.heatmap, result,), self, True)
        # Check that all resulting data values are NaN
        self.assertTrue(isnan(result.data).all(), "Data values should be NaN when dividing by NaN.")
        for square in range(64):
            for piece, count in self.heatmap.piece_counts[square].items():
                self.assertTrue(
                    isnan(result.piece_counts[square][piece]),
                    f"Piece counts should be NaN for square {square}, piece {piece}."
                )

        # Validate data integrity post-failure
        validate_data_types((self.heatmap, result,), self, True)


class TestChessMoveHeatmap(TestCase):
    """Unit tests for ChessMoveHeatmap."""
    heatmap: ChessMoveHeatmap
    piece_counts: NDArray[Dict[Piece, float64]]

    def setUp(self) -> None:
        """Set up a ChessMoveHeatmap instance."""
        self.piece_counts = array([{p: float64(1) for p in PIECES} for _ in range(SHAPE[0])], dtype=object)
        self.heatmap = ChessMoveHeatmap(piece_counts=self.piece_counts)

    def test_initialization(self) -> None:
        """Ensure ChessMoveHeatmap initializes correctly."""
        validate_data_types((self.heatmap,), self, True)
        np_testing.assert_array_equal(self.heatmap.piece_counts, self.piece_counts)

    def test_copy_behavior(self) -> None:
        """Ensure copying behavior works correctly."""
        other_heatmap = ChessMoveHeatmap(piece_counts=self.heatmap.piece_counts)
        self.assertIsNot(self.heatmap.piece_counts, other_heatmap.piece_counts)  # Ensure deep copy
        validate_data_types((self.heatmap, other_heatmap), self, True)

    # noinspection PyTypeChecker,PydanticTypeChecker
    def test_invalid_piece_counts(self) -> None:
        """Ensure TypeError is raised when setting invalid piece_counts."""
        with self.assertRaises(TypeError):
            _ = ChessMoveHeatmap(piece_counts="invalid")
        validate_data_types((self.heatmap,), self, True)


if __name__ == "__main__":
    main()
