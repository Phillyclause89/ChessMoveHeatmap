import unittest

import numpy as np
from heatmaps import GradientHeatmapT, GradientHeatmap, ChessMoveHeatmapT, ChessMoveHeatmap
import re

from tests.utils import INVALID_OBJ_STR, PIECES, SHAPE, validate_data_types


class TestGradientHeatmapT(unittest.TestCase):
    """Unit tests for the GradientHeatmapT class."""

    def setUp(self) -> None:
        """Set up a fresh heatmap for each test."""
        self.heatmap = GradientHeatmapT()

    def test_initialization(self):
        """Test if heatmap initializes with correct shape and zeros."""
        validate_data_types((self.heatmap,), self)
        np.testing.assert_array_equal(self.heatmap.data, np.zeros(SHAPE, dtype=np.float64))

    def test_getitem_setitem(self) -> None:
        """Test setting and getting heatmap values."""
        value0 = [1.5, 2.5]
        self.assertNotIsInstance(value0, np.ndarray)
        value = np.array([x * 2 for x in value0], dtype=np.float64)
        self.assertIsInstance(value, np.ndarray)
        self.heatmap[63] = value
        np.testing.assert_array_equal(self.heatmap[-1], value)
        self.heatmap[-64] = value0
        np.testing.assert_array_equal(self.heatmap[0], value0)
        self.assertIsInstance(self.heatmap[-64], np.ndarray)
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
            self.heatmap[3] = np.array([1.5, 2.5, 3.0], dtype=np.float64)
        with self.assertRaises(IndexError):
            self.heatmap[64] = np.array([2.5, 3.0], dtype=np.float64)
        with self.assertRaises(IndexError):
            self.heatmap[-65] = np.array([2.5, 3.0], dtype=np.float64)
        validate_data_types((self.heatmap,), self)

    def test_addition(self) -> None:
        """Test addition of two GradientHeatmapT instances and a valid np.array."""
        h2 = GradientHeatmapT()
        self.heatmap[5] = np.array([1.0, -1], dtype=np.float64)
        h2[5] = [2.0, -3]
        result1 = self.heatmap + h2
        self.assertIsInstance(result1, GradientHeatmap)  # Validate type
        np.testing.assert_array_equal(result1[5], h2[5] + self.heatmap[5])
        # Technically, we are testing GradientHeatmap(GradientHeatmapT).__add__ here, but I'm ok with that
        result2 = result1 + self.heatmap.data
        self.assertIsInstance(result2, GradientHeatmap)
        np.testing.assert_array_equal(result2[5], result1[5] + self.heatmap[5])
        validate_data_types((self.heatmap, h2, result1, result2), self)

    def test_invalid_addition_type(self) -> None:
        """Ensure TypeError is raised when adding an incompatible type."""
        with self.assertRaises(TypeError):
            self.heatmap + INVALID_OBJ_STR
        hmap0 = self.heatmap + self.heatmap
        np.testing.assert_array_equal(hmap0.data, self.heatmap.data)
        with self.assertRaises(TypeError):
            hmap0 + np.array([[INVALID_OBJ_STR, INVALID_OBJ_STR]] * 64, dtype=object)
        hmap1 = hmap0 + np.zeros(SHAPE, dtype=np.float64)
        np.testing.assert_array_equal(hmap1.data, self.heatmap.data)
        with self.assertRaises(TypeError):
            hmap1 + [1.5]
        validate_data_types((self.heatmap, hmap0, hmap1), self)

    def test_invalid_addition_shape(self) -> None:
        """Ensure ValueError is raised when setting incorrect shape."""
        with self.assertRaises(ValueError):
            self.heatmap + np.array([1.5, 2.5, 3.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            self.heatmap + np.array([3.0], dtype=np.float64)
        validate_data_types((self.heatmap,), self)


class TestGradientHeatmap(unittest.TestCase):
    """Unit tests for the GradientHeatmap class."""

    def setUp(self) -> None:
        """Set up a GradientHeatmap instance."""
        self.data = np.random.rand(64, 2).astype(np.float64)
        self.heatmap = GradientHeatmap(self.data)

    def test_initialization(self) -> None:
        """Test if heatmap initializes correctly from valid data."""
        validate_data_types((self.heatmap,), self)
        np.testing.assert_array_equal(self.heatmap.data, self.data)

    def test_normalization(self) -> None:
        """Test normalization functionality."""
        normalized = self.heatmap._normalize_
        max_val = self.data.max(initial=None)
        expected = self.data / max_val if max_val > 0 else self.data
        np.testing.assert_array_almost_equal(normalized, expected)
        validate_data_types((self.heatmap,), self)

    def test_intensity_to_color(self) -> None:
        """Test if color conversion produces valid hex RGB values."""
        color = self.heatmap._intensity_to_color_(np.float64(0.5), np.float64(0.5))
        self.assertTrue(re.fullmatch(r"#[0-9A-Fa-f]{6}", color), f"Invalid hex color format: {color}")
        validate_data_types((self.heatmap,), self)

    def test_colors(self) -> None:
        """Test if colors property returns valid hex color codes for all squares."""
        colors = self.heatmap.colors
        self.assertEqual(colors.shape, (SHAPE[0],))

        for color in colors:
            self.assertTrue(re.fullmatch(r"#[0-9A-Fa-f]{6}", color), f"Invalid hex color format: {color}")
        validate_data_types((self.heatmap,), self)

    def test_html_representation(self) -> None:
        """Ensure HTML representation is properly formatted."""
        html = self.heatmap._repr_html_()
        self.assertIn("<table", html)
        self.assertIn("</table>", html)
        validate_data_types((self.heatmap,), self)


class TestChessMoveHeatmapT(unittest.TestCase):
    """Unit tests for ChessMoveHeatmapT."""

    def setUp(self) -> None:
        """Set up a ChessMoveHeatmapT instance."""
        self.heatmap = ChessMoveHeatmapT()

    def test_initialization(self) -> None:
        """Ensure piece counts initialize correctly."""
        validate_data_types((self.heatmap,), self)
        self.assertEqual(self.heatmap.piece_counts.shape, (SHAPE[0],))
        for counts in self.heatmap.piece_counts:
            self.assertEqual(len(counts), len(PIECES))

    def test_piece_counts_setter(self) -> None:
        """Test setting piece count data."""
        new_counts = np.array([{p: np.float64(1) for p in PIECES} for _ in range(SHAPE[0])], dtype=object)
        self.heatmap.piece_counts = new_counts
        np.testing.assert_array_equal(self.heatmap.piece_counts, new_counts)
        validate_data_types((self.heatmap,), self)

    def test_invalid_piece_counts(self) -> None:
        """Ensure TypeError is raised for invalid piece_counts assignment."""
        with self.assertRaises(TypeError):
            self.heatmap.piece_counts = "invalid"

        with self.assertRaises(ValueError):
            self.heatmap.piece_counts = np.zeros((32,), dtype=object)
        validate_data_types((self.heatmap,), self)


class TestChessMoveHeatmap(unittest.TestCase):
    """Unit tests for ChessMoveHeatmap."""

    def setUp(self) -> None:
        """Set up a ChessMoveHeatmap instance."""
        self.piece_counts = np.array([{p: np.float64(1) for p in PIECES} for _ in range(SHAPE[0])], dtype=object)
        self.heatmap = ChessMoveHeatmap(piece_counts=self.piece_counts)

    def test_initialization(self) -> None:
        """Ensure ChessMoveHeatmap initializes correctly."""
        validate_data_types((self.heatmap,), self)
        np.testing.assert_array_equal(self.heatmap.piece_counts, self.piece_counts)

    def test_copy_behavior(self) -> None:
        """Ensure copying behavior works correctly."""
        cmh2 = ChessMoveHeatmap(piece_counts=self.heatmap.piece_counts)
        self.assertIsNot(self.heatmap.piece_counts, cmh2.piece_counts)  # Ensure deep copy
        validate_data_types((self.heatmap, cmh2), self)

    # noinspection PyTypeChecker,PydanticTypeChecker
    def test_invalid_piece_counts(self) -> None:
        """Ensure TypeError is raised when setting invalid piece_counts."""
        with self.assertRaises(TypeError):
            ChessMoveHeatmap(piece_counts="invalid")
        validate_data_types((self.heatmap,), self)


if __name__ == "__main__":
    unittest.main()
