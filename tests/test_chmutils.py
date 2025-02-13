import chess
import heatmaps
import chmutils
import unittest
import numpy as np


class TestCalculateHeatmap(unittest.TestCase):
    expected_values = {
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
        heatmap = chmutils.calculate_heatmap(self.board, depth=0)

        # Test expected values against the heatmap at depth 0
        for square, expected in self.expected_values.items():
            np.testing.assert_array_equal(heatmap[square], expected[0])

    def test_calculate_heatmap_depth_1(self):
        """Test calculate_heatmap at depth 1 for performance and type."""
        heatmap = chmutils.calculate_heatmap(self.board)
        # Ensure that the result is a GradientHeatmap object
        self.assertIsInstance(heatmap, heatmaps.GradientHeatmap)

    def test_calculate_heatmap_depth_2(self):
        """Test calculate_heatmap at depth 2 for performance and type."""
        heatmap = chmutils.calculate_heatmap(self.board, depth=2)

        self.assertIsInstance(heatmap, heatmaps.GradientHeatmap)

    def test_calculate_heatmap_depth_3(self):
        """Test calculate_heatmap at depth 3 for performance and type."""
        heatmap = chmutils.calculate_heatmap(self.board, depth=3)

        self.assertIsInstance(heatmap, heatmaps.GradientHeatmap)


class TestCalculateChessMoveHeatmap(unittest.TestCase):
    expected_values = TestCalculateHeatmap.expected_values

    def setUp(self):
        self.board = chess.Board()

    def test_calculate_chess_move_heatmap_depth_0(self):
        """Test calculate_chess_move_heatmap at depth 0 for accuracy of values."""
        heatmap = chmutils.calculate_chess_move_heatmap(self.board, depth=0)

        for square, (expected_heatmap, expected_pieces) in self.expected_values.items():
            np.testing.assert_array_equal(heatmap[square], expected_heatmap)
            self.assertEqual(heatmap.piece_counts[square], expected_pieces)
            self.assertEqual(heatmap[square][0], sum(expected_pieces.values()))

    def test_calculate_chess_move_heatmap_depth_1(self):
        """Test calculate_chess_move_heatmap at depth 1 for performance and type."""
        heatmap = chmutils.calculate_chess_move_heatmap(self.board)
        self.assertIsInstance(heatmap, heatmaps.ChessMoveHeatmap)

    def test_calculate_chess_move_heatmap_depth_2(self):
        """Test calculate_chess_move_heatmap at depth 2 for performance and type."""
        heatmap = chmutils.calculate_chess_move_heatmap(self.board, depth=2)
        self.assertIsInstance(heatmap, heatmaps.ChessMoveHeatmap)

    def test_calculate_chess_move_heatmap_depth_3(self):
        """Test calculate_chess_move_heatmap at depth 3 for performance and type."""
        heatmap = chmutils.calculate_chess_move_heatmap(self.board, depth=3)
        self.assertIsInstance(heatmap, heatmaps.ChessMoveHeatmap)


if __name__ == '__main__':
    unittest.main()
