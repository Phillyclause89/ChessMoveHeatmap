import unittest
import numpy as np
from chess import Piece, COLORS, PIECE_TYPES
from heatmaps import GradientHeatmapT, GradientHeatmap, ChessMoveHeatmapT, ChessMoveHeatmap
import re

# Constants
PIECES = tuple(Piece(p, c) for c in COLORS for p in PIECE_TYPES)


class TestGradientHeatmapT(unittest.TestCase):
    """Unit tests for the GradientHeatmapT class."""

    def setUp(self):
        """Set up a fresh heatmap for each test."""
        self.heatmap = GradientHeatmapT()

    def test_initialization(self):
        """Test if heatmap initializes with correct shape and zeros."""
        self.assertEqual(self.heatmap.shape, (64, 2))
        np.testing.assert_array_equal(self.heatmap.data, np.zeros((64, 2), dtype=np.float64))

    def test_getitem_setitem(self):
        """Test setting and getting heatmap values."""
        value = np.array([1.5, 2.5], dtype=np.float64)
        self.heatmap[10] = value
        np.testing.assert_array_equal(self.heatmap[10], value)

    # noinspection PyTypeChecker,PydanticTypeChecker
    def test_invalid_setitem_type(self):
        """Ensure TypeError is raised when setting invalid data type."""
        with self.assertRaises(TypeError):
            self.heatmap[5] = [1, 2]  # Not a NumPy array

    def test_invalid_setitem_shape(self):
        """Ensure ValueError is raised when setting incorrect shape."""
        with self.assertRaises(ValueError):
            self.heatmap[3] = np.array([1.5, 2.5, 3.0], dtype=np.float64)

    def test_addition(self):
        """Test addition of two GradientHeatmapT instances and a valid np.array."""
        h2 = GradientHeatmapT()
        self.heatmap[5] = np.array([1.0, 1.0], dtype=np.float64)
        h2[5] = np.array([2.0, 3.0], dtype=np.float64)

        result1 = self.heatmap + h2
        self.assertIsInstance(result1, GradientHeatmap)  # Validate type
        np.testing.assert_array_equal(result1[5], np.array([3.0, 4.0], dtype=np.float64))
        # Technically, we are testing GradientHeatmap(GradientHeatmapT).__add__ here, but I'm ok with that
        result2 = result1 + self.heatmap.data
        self.assertIsInstance(result2, GradientHeatmap)  # Validate type after ndarray addition
        np.testing.assert_array_equal(result2[5], np.array([4.0, 5.0], dtype=np.float64))

    def test_invalid_addition_type(self):
        """Ensure TypeError is raised when adding an incompatible type."""
        with self.assertRaises(TypeError):
            _ = self.heatmap + "invalid"

    def test_invalid_addition_shape(self):
        """Ensure ValueError is raised when setting incorrect shape."""
        with self.assertRaises(ValueError):
            self.heatmap + np.array([1.5, 2.5, 3.0], dtype=np.float64)


class TestGradientHeatmap(unittest.TestCase):
    """Unit tests for the GradientHeatmap class."""

    def setUp(self):
        """Set up a GradientHeatmap instance."""
        self.data = np.random.rand(64, 2).astype(np.float64)
        self.heatmap = GradientHeatmap(self.data)

    def test_initialization(self):
        """Test if heatmap initializes correctly from valid data."""
        np.testing.assert_array_equal(self.heatmap.data, self.data)

    def test_normalization(self):
        """Test normalization functionality."""
        normalized = self.heatmap._normalize_
        max_val = self.data.max(initial=None)
        expected = self.data / max_val if max_val > 0 else self.data
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_intensity_to_color(self):
        """Test if color conversion produces valid hex RGB values."""
        color = self.heatmap._intensity_to_color_(np.float64(0.5), np.float64(0.5))
        self.assertTrue(re.fullmatch(r"#[0-9A-Fa-f]{6}", color), f"Invalid hex color format: {color}")

    def test_colors(self):
        """Test if colors property returns valid hex color codes for all squares."""
        colors = self.heatmap.colors
        self.assertEqual(colors.shape, (64,))

        for color in colors:
            self.assertTrue(re.fullmatch(r"#[0-9A-Fa-f]{6}", color), f"Invalid hex color format: {color}")

    def test_html_representation(self):
        """Ensure HTML representation is properly formatted."""
        html = self.heatmap._repr_html_()
        self.assertIn("<table", html)
        self.assertIn("</table>", html)


class TestChessMoveHeatmapT(unittest.TestCase):
    """Unit tests for ChessMoveHeatmapT."""

    def setUp(self):
        """Set up a ChessMoveHeatmapT instance."""
        self.heatmap = ChessMoveHeatmapT()

    def test_initialization(self):
        """Ensure piece counts initialize correctly."""
        self.assertEqual(self.heatmap.piece_counts.shape, (64,))
        for counts in self.heatmap.piece_counts:
            self.assertEqual(len(counts), len(PIECES))

    def test_piece_counts_setter(self):
        """Test setting piece count data."""
        new_counts = np.array([{p: np.float64(1) for p in PIECES} for _ in range(64)], dtype=object)
        self.heatmap.piece_counts = new_counts
        np.testing.assert_array_equal(self.heatmap.piece_counts, new_counts)

    def test_invalid_piece_counts(self):
        """Ensure TypeError is raised for invalid piece_counts assignment."""
        with self.assertRaises(TypeError):
            self.heatmap.piece_counts = "invalid"

        with self.assertRaises(ValueError):
            self.heatmap.piece_counts = np.zeros((32,), dtype=object)


class TestChessMoveHeatmap(unittest.TestCase):
    """Unit tests for ChessMoveHeatmap."""

    def setUp(self):
        """Set up a ChessMoveHeatmap instance."""
        self.piece_counts = np.array([{p: np.float64(1) for p in PIECES} for _ in range(64)], dtype=object)
        self.heatmap = ChessMoveHeatmap(piece_counts=self.piece_counts)

    def test_initialization(self):
        """Ensure ChessMoveHeatmap initializes correctly."""
        np.testing.assert_array_equal(self.heatmap.piece_counts, self.piece_counts)

    def test_copy_behavior(self):
        """Ensure copying behavior works correctly."""
        cmh2 = ChessMoveHeatmap(piece_counts=self.heatmap.piece_counts)
        self.assertIsNot(self.heatmap.piece_counts, cmh2.piece_counts)  # Ensure deep copy

    # noinspection PyTypeChecker,PydanticTypeChecker
    def test_invalid_piece_counts(self):
        """Ensure TypeError is raised when setting invalid piece_counts."""
        with self.assertRaises(TypeError):
            ChessMoveHeatmap(piece_counts="invalid")


if __name__ == "__main__":
    unittest.main()
