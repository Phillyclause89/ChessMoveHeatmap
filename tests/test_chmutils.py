import os
import shutil
from typing import Dict, List, Optional, Tuple

import chess
from chess import Board, Piece
from numpy import float_

import heatmaps
import chmutils
import unittest
import numpy as np

from chmutils import HeatmapCache
from heatmaps import ChessMoveHeatmap, GradientHeatmap

CACHE_DIR = "SQLite3Caches"


class TestCalculateHeatmap(unittest.TestCase):
    board: Board
    expected_values: Dict[int, Tuple[List[float], Dict[Piece, float]]] = {
        16: ([2.0, 0.0],  # expected heatmap values for a square
             {chess.Piece.from_symbol('P'): 1.0,
              chess.Piece.from_symbol('N'): 1.0,
              chess.Piece.from_symbol('B'): 0.0,
              chess.Piece.from_symbol('R'): 0.0,
              chess.Piece.from_symbol('k'): 0.0,
              chess.Piece.from_symbol('b'): 0.0,
              chess.Piece.from_symbol('n'): 0.0,
              chess.Piece.from_symbol('p'): 0.0,
              chess.Piece.from_symbol('q'): 0.0,
              chess.Piece.from_symbol('r'): 0.0,
              chess.Piece.from_symbol('K'): 0.0,
              chess.Piece.from_symbol('Q'): 0.0}),  # expected piece counts for that square
        17: ([1.0, 0.0],
             {chess.Piece.from_symbol('P'): 1.0,
              chess.Piece.from_symbol('N'): 0.0,
              chess.Piece.from_symbol('B'): 0.0,
              chess.Piece.from_symbol('R'): 0.0,
              chess.Piece.from_symbol('k'): 0.0,
              chess.Piece.from_symbol('b'): 0.0,
              chess.Piece.from_symbol('n'): 0.0,
              chess.Piece.from_symbol('p'): 0.0,
              chess.Piece.from_symbol('q'): 0.0,
              chess.Piece.from_symbol('r'): 0.0,
              chess.Piece.from_symbol('K'): 0.0,
              chess.Piece.from_symbol('Q'): 0.0}),
    }

    def setUp(self):
        self.board = chess.Board()

    def test_calculate_heatmap_depth_0(self):
        """Test calculate_heatmap at depth 0 for accuracy of values."""
        # Calculate the heatmap for depth 0
        heatmap: GradientHeatmap = chmutils.calculate_heatmap(self.board, depth=0)

        # Test expected values against the heatmap at depth 0
        square: int
        expected: Tuple[List[float], Dict[Piece, float]]
        for square, expected in self.expected_values.items():
            np.testing.assert_array_equal(heatmap[square], expected[0])

    def test_calculate_heatmap_depth_1(self):
        """Test calculate_heatmap at depth 1 for performance and type."""
        heatmap: GradientHeatmap = chmutils.calculate_heatmap(self.board)
        # Ensure that the result is a GradientHeatmap object
        self.assertIsInstance(heatmap, heatmaps.GradientHeatmap)

    def test_calculate_heatmap_depth_2(self):
        """Test calculate_heatmap at depth 2 for performance and type."""
        heatmap: GradientHeatmap = chmutils.calculate_heatmap(self.board, depth=2)

        self.assertIsInstance(heatmap, heatmaps.GradientHeatmap)

    def test_calculate_heatmap_depth_3(self):
        """Test calculate_heatmap at depth 3 for performance and type."""
        heatmap: GradientHeatmap = chmutils.calculate_heatmap(self.board, depth=3)

        self.assertIsInstance(heatmap, heatmaps.GradientHeatmap)


class TestCalculateChessMoveHeatmap(unittest.TestCase):
    board: Board
    expected_values: Dict[int, Tuple[List[float], Dict[Piece, float]]] = TestCalculateHeatmap.expected_values

    def setUp(self):
        self.board = chess.Board()

    def test_calculate_chess_move_heatmap_depth_0(self):
        """Test calculate_chess_move_heatmap at depth 0 for accuracy of values."""
        heatmap: ChessMoveHeatmap = chmutils.calculate_chess_move_heatmap(self.board, depth=0)

        square: int
        expected_heatmap: List[float]
        expected_pieces: Dict[Piece, float]
        for square, (expected_heatmap, expected_pieces) in self.expected_values.items():
            np.testing.assert_array_equal(heatmap[square], expected_heatmap)
            self.assertEqual(heatmap.piece_counts[square], expected_pieces)
            self.assertEqual(heatmap[square][0], sum(expected_pieces.values()))

    def test_calculate_chess_move_heatmap_depth_1(self):
        """Test calculate_chess_move_heatmap at depth 1 for performance and type."""
        heatmap: ChessMoveHeatmap = chmutils.calculate_chess_move_heatmap(self.board)
        self.assertIsInstance(heatmap, heatmaps.ChessMoveHeatmap)

    def test_calculate_chess_move_heatmap_depth_2(self):
        """Test calculate_chess_move_heatmap at depth 2 for performance and type."""
        heatmap: ChessMoveHeatmap = chmutils.calculate_chess_move_heatmap(self.board, depth=2)
        self.assertIsInstance(heatmap, heatmaps.ChessMoveHeatmap)

    def test_calculate_chess_move_heatmap_depth_3(self):
        """Test calculate_chess_move_heatmap at depth 3 for performance and type."""
        heatmap: ChessMoveHeatmap = chmutils.calculate_chess_move_heatmap(self.board, depth=3)
        self.assertIsInstance(heatmap, heatmaps.ChessMoveHeatmap)


def clear_test_cache(cache_dir: str = CACHE_DIR) -> None:
    """Ensure a clean test environment by removing any existing cache file."""
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


class TestHeatmapCacheAndFunctions(unittest.TestCase):
    pawn: Piece
    board: Board

    def setUp(self):
        self.board = chess.Board()
        self.pawn = chess.Piece.from_symbol('P')
        clear_test_cache()
        self.assertFalse(os.path.exists(CACHE_DIR))

    def tearDown(self) -> None:
        clear_test_cache()
        self.assertFalse(os.path.exists(CACHE_DIR))

    def test_flatten_and_inflate_heatmap(self):
        """Verify that a ChessMoveHeatmap can be flattened and then re-inflated correctly."""
        # Create a ChessMoveHeatmap and modify some values at square 0.
        cmhm: ChessMoveHeatmap = heatmaps.ChessMoveHeatmap()
        cmhm.data[0][0], cmhm.data[0][1], cmhm.piece_counts[0][self.pawn] = 1.0, 2.0, 3.0

        # Flatten the heatmap.
        flat: Dict[str, float_] = chmutils.flatten_heatmap(cmhm)
        self.assertEqual(flat["sq0_white"], 1.0)
        self.assertEqual(flat["sq0_black"], 2.0)
        self.assertEqual(flat[f"sq0_piece_{self.pawn.unicode_symbol()}"], 3.0)

        # Inflate the flat dict back into a ChessMoveHeatmap.
        inflated: ChessMoveHeatmap = chmutils.inflate_heatmap(flat)
        np.testing.assert_array_equal(inflated.data[0], np.array([1.0, 2.0]))
        self.assertEqual(inflated.piece_counts[0][self.pawn], 3.0)

    def test_heatmap_cache_store_and_retrieve(self):
        """Test that HeatmapCache stores a heatmap and later retrieves an equivalent object."""
        cache: HeatmapCache = chmutils.HeatmapCache(self.board, 0)
        self.assertTrue(os.path.exists(CACHE_DIR))
        clear_test_cache()
        self.assertFalse(os.path.exists(CACHE_DIR))
        cache._initialize_db()
        self.assertTrue(os.path.exists(CACHE_DIR))

        # Initially, no cache should be available.
        self.assertIsNone(cache.get_cached_heatmap())

        # Compute a ChessMoveHeatmap (depth=0 for predictability) and modify a value.
        cmhm: ChessMoveHeatmap = chmutils.calculate_chess_move_heatmap(self.board, depth=0)
        cmhm.data[0][0] = 5.0  # Modify a value for testing.

        # Store the heatmap.
        cache.store_heatmap(cmhm)

        # Retrieve from cache.
        cached: Optional[ChessMoveHeatmap] = cache.get_cached_heatmap()
        self.assertIsNotNone(cached)
        np.testing.assert_array_equal(cached.data, cmhm.data)

        self.assertEqual(cached.piece_counts[0][self.pawn], cmhm.piece_counts[0][self.pawn])

    def test_get_or_compute_heatmap(self):
        """Test that get_or_compute_heatmap computes the heatmap on the first call and then retrieves it from cache."""

        # First call should compute and store the heatmap.
        cmhm1: ChessMoveHeatmap = chmutils.get_or_compute_heatmap(self.board, 0)
        self.assertIsInstance(cmhm1, heatmaps.ChessMoveHeatmap)

        # Second call should retrieve the heatmap from the cache.
        cmhm2: ChessMoveHeatmap = chmutils.get_or_compute_heatmap(self.board, 0)
        self.assertIsInstance(cmhm2, heatmaps.ChessMoveHeatmap)

        # Verify that the data arrays match.
        np.testing.assert_array_equal(cmhm1.data, cmhm2.data)

        # Also verify that at least one piece count (e.g., for a pawn on square 0) is identical.
        self.assertEqual(cmhm1.piece_counts[0][self.pawn], cmhm2.piece_counts[0][self.pawn])


if __name__ == '__main__':
    unittest.main()
