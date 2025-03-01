"""Tests chmutils"""
from unittest import TestCase, main
from os import path
from shutil import rmtree
from typing import Dict, List, Optional, Tuple

from chess import Board, Piece
from numpy import float64, float_, array, testing

from chmutils import flatten_heatmap, calculate_chess_move_heatmap_with_better_discount, calculate_heatmap
from chmutils import HeatmapCache, calculate_chess_move_heatmap, inflate_heatmap, get_or_compute_heatmap
from heatmaps import ChessMoveHeatmap, GradientHeatmap

from tests.utils import validate_data_types

CACHE_DIR = "SQLite3TestCaches"
HeatmapCache.cache_dir = CACHE_DIR


class TestCalculateHeatmap(TestCase):
    """Tests Calculate Heatmap Function"""
    board: Board
    expected_values: Dict[int, Tuple[List[float], Dict[Piece, float]]] = {
        16: (
            [2.0, 0.0],  # expected heatmap values for a square
            {
                Piece.from_symbol('P'): 1.0,
                Piece.from_symbol('N'): 1.0,
                Piece.from_symbol('B'): 0.0,
                Piece.from_symbol('R'): 0.0,
                Piece.from_symbol('k'): 0.0,
                Piece.from_symbol('b'): 0.0,
                Piece.from_symbol('n'): 0.0,
                Piece.from_symbol('p'): 0.0,
                Piece.from_symbol('q'): 0.0,
                Piece.from_symbol('r'): 0.0,
                Piece.from_symbol('K'): 0.0,
                Piece.from_symbol('Q'): 0.0}),  # expected piece counts for that square
        17: (
            [1.0, 0.0],
            {
                Piece.from_symbol('P'): 1.0,
                Piece.from_symbol('N'): 0.0,
                Piece.from_symbol('B'): 0.0,
                Piece.from_symbol('R'): 0.0,
                Piece.from_symbol('k'): 0.0,
                Piece.from_symbol('b'): 0.0,
                Piece.from_symbol('n'): 0.0,
                Piece.from_symbol('p'): 0.0,
                Piece.from_symbol('q'): 0.0,
                Piece.from_symbol('r'): 0.0,
                Piece.from_symbol('K'): 0.0,
                Piece.from_symbol('Q'): 0.0}),
    }

    def setUp(self):
        """Set up test board"""
        self.board = Board()

    def test_calculate_heatmap_depth_0(self):
        """Test calculate_heatmap at depth 0 for accuracy of values."""
        # Calculate the heatmap for depth 0
        heatmap: GradientHeatmap = calculate_heatmap(self.board, depth=0)

        # Test expected values against the heatmap at depth 0
        square: int
        expected: Tuple[List[float], Dict[Piece, float]]
        for square, expected in self.expected_values.items():
            testing.assert_array_equal(heatmap[square], expected[0])
        validate_data_types([heatmap], self)

    def test_calculate_heatmap_depth_1(self):
        """Test calculate_heatmap at depth 1 for performance and type."""
        heatmap: GradientHeatmap = calculate_heatmap(self.board)
        # Ensure that the result is a GradientHeatmap object
        validate_data_types([heatmap], self)

    def test_calculate_heatmap_depth_2(self):
        """Test calculate_heatmap at depth 2 for performance and type."""
        heatmap: GradientHeatmap = calculate_heatmap(self.board, depth=2)
        validate_data_types([heatmap], self)

    def test_calculate_heatmap_depth_3(self):
        """Test calculate_heatmap at depth 3 for performance and type."""
        heatmap: GradientHeatmap = calculate_heatmap(self.board, depth=3)
        validate_data_types([heatmap], self)


class TestCalculateChessMoveHeatmap(TestCase):
    """Tests Calculate Chess Move Heatmap Function"""
    board: Board
    expected_values: Dict[int, Tuple[List[float], Dict[Piece, float]]] = TestCalculateHeatmap.expected_values

    def setUp(self):
        """Set up test board"""
        self.board = Board()

    def test_calculate_chess_move_heatmap_depth_0(self):
        """Test calculate_chess_move_heatmap at depth 0 for accuracy of values."""
        heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap(self.board, depth=0)

        square: int
        expected_heatmap: List[float]
        expected_pieces: Dict[Piece, float]
        for square, (expected_heatmap, expected_pieces) in self.expected_values.items():
            testing.assert_array_equal(heatmap[square], expected_heatmap)
            self.assertEqual(heatmap.piece_counts[square], expected_pieces)
            # pylint: disable=no-member
            self.assertEqual(heatmap[square][0], sum(expected_pieces.values()))
        validate_data_types([heatmap], self, True)

    def test_calculate_chess_move_heatmap_depth_1(self):
        """Test calculate_chess_move_heatmap at depth 1 for performance and type."""
        heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap(self.board)
        validate_data_types([heatmap], self, True)

    def test_calculate_chess_move_heatmap_depth_2(self):
        """Test calculate_chess_move_heatmap at depth 2 for performance and type."""
        heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap(self.board, depth=2)
        validate_data_types([heatmap], self, True)

    def test_calculate_chess_move_heatmap_depth_3(self):
        """Test calculate_chess_move_heatmap at depth 3 for performance and type."""
        heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap(self.board, depth=3)
        validate_data_types([heatmap], self, True)


def clear_test_cache(cache_dir: str = CACHE_DIR) -> None:
    """Ensure a clean test environment by removing any existing cache file."""
    if path.exists(cache_dir):
        rmtree(cache_dir)


class TestHeatmapCacheAndFunctions(TestCase):
    """Test Heatmap Cache And related Functions"""
    pawn: Piece
    board: Board

    def setUp(self):
        """Set up test board, piece and clear any leftover test cache"""
        self.board = Board()
        self.pawn = Piece.from_symbol('P')
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def tearDown(self) -> None:
        """clear any leftover test cache"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def test_flatten_and_inflate_heatmap(self):
        """Verify that a ChessMoveHeatmap can be flattened and then re-inflated correctly."""
        # Create a ChessMoveHeatmap and modify some values at square 0.
        cmhm: ChessMoveHeatmap = ChessMoveHeatmap()
        cmhm.data[0][0], cmhm.data[0][1], cmhm.piece_counts[0][self.pawn] = float64(1.0), float64(2.0), float64(3.0)

        # Flatten the heatmap.
        flat: Dict[str, float_] = flatten_heatmap(cmhm)
        self.assertEqual(flat["sq0_white"], 1.0)
        self.assertEqual(flat["sq0_black"], 2.0)
        self.assertEqual(flat[f"sq0_piece_{self.pawn.unicode_symbol()}"], 3.0)

        # Inflate the flat dict back into a ChessMoveHeatmap.
        inflated: ChessMoveHeatmap = inflate_heatmap(flat)
        testing.assert_array_equal(inflated.data[0], array([1.0, 2.0], dtype=float64))
        self.assertEqual(inflated.piece_counts[0][self.pawn], 3.0)
        validate_data_types([cmhm, inflated], self, True)

    def test_heatmap_cache_store_and_retrieve(self):
        """Test that HeatmapCache stores a heatmap and later retrieves an equivalent object."""
        cache: HeatmapCache = HeatmapCache(self.board, 0)
        self.assertTrue(path.exists(CACHE_DIR))
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        # pylint: disable=protected-access
        cache._initialize_db()
        self.assertTrue(path.exists(CACHE_DIR))

        # Initially, no cache should be available.
        self.assertIsNone(cache.get_cached_heatmap())

        # Compute a ChessMoveHeatmap (depth=0 for predictability) and modify a value.
        cmhm: ChessMoveHeatmap = calculate_chess_move_heatmap(self.board, depth=0)
        cmhm.data[0][0] = 5.0  # Modify a value for testing.

        # Store the heatmap.
        cache.store_heatmap(cmhm)

        # Retrieve from cache.
        cached: Optional[ChessMoveHeatmap] = cache.get_cached_heatmap()
        self.assertIsNotNone(cached)
        testing.assert_array_equal(cached.data, cmhm.data)

        self.assertEqual(cached.piece_counts[0][self.pawn], cmhm.piece_counts[0][self.pawn])
        validate_data_types([cmhm, cached], self, True)

    def test_get_or_compute_heatmap(self) -> None:
        """Test that get_or_compute_heatmap computes the heatmap on the first call and then retrieves it from cache."""

        # First call should compute and store the heatmap.
        cmhm1: ChessMoveHeatmap = get_or_compute_heatmap(self.board, 0)
        self.assertIsInstance(cmhm1, ChessMoveHeatmap)

        # Second call should retrieve the heatmap from the cache.
        cmhm2: ChessMoveHeatmap = get_or_compute_heatmap(self.board, 0)
        self.assertIsInstance(cmhm2, ChessMoveHeatmap)

        # Verify that the data arrays match.
        testing.assert_array_equal(cmhm1.data, cmhm2.data)

        # Also verify that at least one piece count (e.g., for a pawn on square 0) is identical.
        self.assertEqual(cmhm1.piece_counts[0][self.pawn], cmhm2.piece_counts[0][self.pawn])
        validate_data_types([cmhm1, cmhm2], self, True)


class TestCalculateChessMoveHeatmapWithBetterDiscount(TestCase):
    """Test Calculate ChessMoveHeatmap With Better Discount functions"""
    board: Board
    expected_values: Dict[int, Tuple[List[float], Dict[Piece, float]]] = TestCalculateHeatmap.expected_values

    def setUp(self):
        """Set up test board"""
        self.board = Board()

    def test_calculate_chess_move_heatmap_depth_0(self):
        """Test calculate_chess_move_heatmap at depth 0 for accuracy of values."""
        heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap_with_better_discount(self.board, depth=0)

        square: int
        expected_heatmap: List[float]
        expected_pieces: Dict[Piece, float]
        for square, (expected_heatmap, expected_pieces) in self.expected_values.items():
            testing.assert_array_equal(heatmap[square], expected_heatmap)
            self.assertEqual(heatmap.piece_counts[square], expected_pieces)
            # pylint: disable=no-member
            self.assertEqual(heatmap[square][0], sum(expected_pieces.values()))
        validate_data_types([heatmap], self, True)

    def test_calculate_chess_move_heatmap_depth_1(self):
        """Test calculate_chess_move_heatmap at depth 1 for performance and type."""
        heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap_with_better_discount(self.board)
        validate_data_types([heatmap], self, True)

    def test_calculate_chess_move_heatmap_depth_2(self):
        """Test calculate_chess_move_heatmap at depth 2 for performance and type."""
        heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap_with_better_discount(self.board, depth=2)
        validate_data_types([heatmap], self, True)

    def test_calculate_chess_move_heatmap_depth_3(self):
        """Test calculate_chess_move_heatmap at depth 3 for performance and type."""
        heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap_with_better_discount(self.board, depth=3)
        validate_data_types([heatmap], self, True)


if __name__ == '__main__':
    main()
