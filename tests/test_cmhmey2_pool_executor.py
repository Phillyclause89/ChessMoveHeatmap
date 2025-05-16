"""Test Cmhmey Jr.'s Uncle Poole"""
from io import StringIO
from os import path
from time import perf_counter
from typing import Iterable, Optional, Union
from unittest import TestCase
from unittest.mock import MagicMock, patch

from chess import Board, Move, pgn
from numpy import float64, isnan, mean, percentile

from chmengine import Pick, CMHMEngine2, Quartney
from chmengine.engines.cmhmey2_pool_executor import CMHMEngine2PoolExecutor
from chmutils import BetterHeatmapCache, HeatmapCache
from tests.utils import CACHE_DIR, clear_test_cache

MATE_IN_ONE_4 = '2k5/Q1p5/2K5/8/8/8/8/8 w - - 0 1'

MATE_IN_ONE_3 = 'kb6/p7/5p2/2RPpp2/2RK4/2BPP3/6B1/8 w - e6 0 2'

MATE_IN_ONE_2 = 'kb6/p3p3/5p2/2RP1p2/2RK4/2BPP3/6B1/8 b - - 0 1'

MATE_IN_ONE_1 = '4k2r/ppp4p/4pBp1/2Q5/4B3/4p1PK/PP1r3P/5R2 w - - 2 32'

HeatmapCache.cache_dir = CACHE_DIR
BetterHeatmapCache.cache_dir = CACHE_DIR
CMHMEngine2.cache_dir = CACHE_DIR
Quartney.cache_dir = CACHE_DIR
CMHMEngine2PoolExecutor.cache_dir = CACHE_DIR


class TestCMHMEngine2PoolExecutor(TestCase):
    """Tests Cmhmey Jr."""
    starting_board: Board
    executor: CMHMEngine2PoolExecutor

    def setUp(self) -> None:
        """Sets ups the engine instance to be tested with"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        self.executor = CMHMEngine2PoolExecutor()
        self.assertIsInstance(self.executor, CMHMEngine2PoolExecutor)
        self.assertEqual(self.executor.cache_dir, CACHE_DIR)
        self.assertEqual(self.executor.engine.cache_dir, CACHE_DIR)
        self.starting_board = Board()

    def tearDown(self) -> None:
        """clear any leftover test cache"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        self.assertIsInstance(self.executor.board, Board)
        self.assertEqual(self.executor.board.fen(), self.starting_board.fen())
        self.assertEqual(self.executor.depth, 1)
        self.assertIsNone(self.executor.max_workers)
        self.assertIsInstance(self.executor.engine, CMHMEngine2)

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_board = Board("8/8/8/8/8/8/8/8 w - - 0 1")
        with self.assertRaises(ValueError):
            _ = CMHMEngine2PoolExecutor(board=custom_board, depth=3, max_workers=4)
        custom_board = Board(MATE_IN_ONE_4)
        executor = CMHMEngine2PoolExecutor(board=custom_board, depth=0, max_workers=0)
        self.assertEqual(executor.board, custom_board)
        self.assertEqual(executor.depth, 0)
        self.assertEqual(executor.max_workers, 0)

    @patch("chmengine.engines.cmhmey2_pool_executor.CMHMEngine2.pick_move")
    def test_pick_move_valid(self, mock_pick_move):
        """Test pick_move with a valid board state."""
        mock_pick_move.return_value = Pick(move=next(iter(self.starting_board.legal_moves)), score=10.5)
        pick = self.executor.pick_move()
        self.assertIsInstance(pick, Pick)
        self.assertGreaterEqual(pick.score, -float("inf"))
        self.assertLessEqual(pick.score, float("inf"))

    def test_no_legal_moves(self):
        """Test pick_move when no legal moves are available (stalemate or checkmate)."""
        checkmate_board = Board("7K/8/8/8/8/8/7Q/k6Q b - - 15 99")  # Black king is checkmated
        print(checkmate_board)
        self.assertTrue(checkmate_board.is_checkmate())
        self.assertEqual(checkmate_board.legal_moves.count(), 0)
        # CMHMEngine2.@board.setter won't let us set an invalid board, so we must cheat...
        self.executor.engine._board = checkmate_board
        # Really need to make this a property that keeps the executor.engine._board inline with executor.board
        self.executor.board = checkmate_board
        with self.assertRaises(ValueError):
            self.executor.pick_move()

    # TODO: These tests seems too slow for a GitHub action test. I had to terminate every local attempt to run
    # @patch("chmengine.engines.cmhmey2_pool_executor.evaluate_move")
    # def test_uncached_moves_evaluation(self, mock_evaluate_move):
    #     """Test that uncached moves are evaluated in parallel."""
    #     mock_evaluate_move.return_value = Pick(move=next(iter(self.starting_board.legal_moves)), score=5.0)
    #     pick = self.executor.pick_move()
    #     self.assertIsInstance(pick, Pick)
    #     self.assertGreaterEqual(pick.score, -float("inf"))
    #     self.assertLessEqual(pick.score, float("inf"))
    #     mock_evaluate_move.assert_called()

    # @patch("chmengine.engines.cmhmey2_pool_executor.ProcessPoolExecutor")
    # def test_max_workers(self, mock_executor):
    #     """Test that the ProcessPoolExecutor respects the max_workers parameter."""
    #     self.executor.max_workers = 2
    #     self.executor.pick_move()
    #     mock_executor.assert_called_with(max_workers=2)

    def test_debug_output(self):
        """Test debug output when debug=True."""
        with patch("sys.stdout") as mock_stdout:
            self.executor.pick_move(debug=True)
            self.assertTrue(mock_stdout.write.called)
