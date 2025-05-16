"""Test Cmhmey Jr.'s Uncle Poole"""
from os import path
from time import perf_counter
from typing import List, Optional, Tuple
from unittest import TestCase
from unittest.mock import patch

from chess import Board, Move
from numpy import mean, percentile

from chmengine import CMHMEngine2, Pick, Quartney
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
    """Tests Cmhmey Jr.'s Uncle Poole"""
    starting_board: Board
    executor: CMHMEngine2PoolExecutor

    def setUp(self) -> None:
        """Sets up the engine instance and test environment."""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        self.executor = CMHMEngine2PoolExecutor()
        self.assertIsInstance(self.executor, CMHMEngine2PoolExecutor)
        self.assertEqual(self.executor.cache_dir, CACHE_DIR)
        self.assertEqual(self.executor.engine.cache_dir, CACHE_DIR)
        self.starting_board = Board()

    def tearDown(self) -> None:
        """Cleans up the test environment."""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        self.assertIsInstance(self.executor.engine.board, Board)
        self.assertEqual(self.executor.engine.board.fen(), self.starting_board.fen())
        self.assertEqual(self.executor.engine.depth, 1)
        self.assertIsInstance(self.executor.engine, CMHMEngine2)

    def test_initialization_custom(self):
        """Test initialization with custom parameters, good and bad."""
        custom_board = Board("8/8/8/8/8/8/8/8 w - - 0 1")
        with self.assertRaises(ValueError):
            _ = CMHMEngine2PoolExecutor(board=custom_board, depth=3, max_workers=4)
        custom_board = Board(MATE_IN_ONE_4)
        executor = CMHMEngine2PoolExecutor(board=custom_board, depth=0, max_workers=1)
        self.assertEqual(executor.engine.board, custom_board)
        self.assertEqual(executor.engine.depth, 0)

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
        self.assertTrue(checkmate_board.is_checkmate())
        self.assertEqual(checkmate_board.legal_moves.count(), 0)
        # CMHMEngine2.@board.setter won't let us set an invalid board, so we must cheat...
        self.executor.engine._board = checkmate_board
        # Really need to make this a property that keeps the executor.engine._board inline with executor.board
        self.executor.board = checkmate_board
        with self.assertRaises(ValueError):
            self.executor.pick_move()

    def test_debug_output(self):
        """Test debug output when debug=True."""
        with patch("sys.stdout") as mock_stdout:
            self.executor.pick_move(debug=True)
            self.assertTrue(mock_stdout.write.called)

    def test_pick_move(self) -> None:
        """Tests pick_move method."""
        pick: Pick
        duration_per_branch: float
        duration: float
        branch_count: int
        pick, duration_per_branch, duration, branch_count = self.measure_pick(message=1)
        init_w_moves: List[Move] = list(self.executor.engine.board.legal_moves)
        move: Move
        first_time_pick_times = [duration_per_branch]  # list of all times a position was seen for the first time.
        init_board_pick_times = [duration_per_branch]  # list of all times the first board position was seen.
        revisit_pick_times = []  # lists all the times the init board position was revisited.
        new_duration = 999999.99
        for i, move in enumerate(init_w_moves[:len(init_w_moves) // 6], 1):
            self.executor.push(move)
            response_pick, duration_rep_pick, duration, branch_count = self.measure_pick(move=move, message=2)
            first_time_pick_times.append(duration_rep_pick)
            self.executor.engine.board.pop()
            new_pick, new_duration, duration, branch_count = self.measure_pick(i=i, message=3)
            init_board_pick_times.append(new_duration)
            revisit_pick_times.append(new_duration)
        self.assertLess(new_duration, duration_per_branch)
        avg_duration = mean(init_board_pick_times)
        avg_response = mean(first_time_pick_times)
        avg_revisit = mean(revisit_pick_times)
        self.assertLess(avg_duration, avg_response)
        pre_durations = percentile(init_board_pick_times, [0, 1, 10, 25, 50, 75, 90, 99, 100])
        pre_response = percentile(first_time_pick_times, [0, 1, 10, 25, 50, 75, 90, 99, 100])
        pre_revisit = percentile(revisit_pick_times, [0, 1, 10, 25, 50, 75, 90, 99, 100])
        print(f"mean pick time: {avg_duration:.3f}s\npercentiles (0, 1, 10, 25, 50, 75, 90, 99, 100):\n{pre_durations}")
        print(
            f"mean response time: {avg_response:.3f}s\npercentiles (0, 1, 10, 25, 50, 75, 90, 99, 100):\n{pre_response}"
        )
        print(
            f"mean revisit time: {avg_revisit:.3f}s\npercentiles (0, 1, 10, 25, 50, 75, 90, 99, 100):\n{pre_revisit}"
        )

    def measure_pick(
            self,
            i: Optional[int] = None,
            move: Optional[Move] = None,
            message: int = 0
    ) -> Tuple[Pick, float, float, int]:
        """

        Parameters
        ----------
        i : Optional[int]
        move : Optional[Move]
        message : int

        Returns
        -------
        Tuple[Pick, float]
        """
        branch_count: int = self.executor.engine.board.legal_moves.count()
        start: float = perf_counter()
        pick: Pick = self.executor.pick_move()
        end: float = perf_counter()
        duration: float = (end - start)
        duration_per_branch: float = duration / branch_count
        if message == 1:
            print(
                f"{self.executor.engine.fen()} initial pick_move call: "
                f"({pick:+.2f}) {duration_per_branch:.3f}s/branch"
                f" (branch_count={branch_count}, duration={duration:.3f}s)"
            )
        elif message == 2:
            if move is not None:
                print(
                    f"{move} -> {self.executor.engine.fen()} initial pick_move call: "
                    f"({pick:+.2f}) {duration_per_branch:.3f}s/branch"
                    f" (branch_count={branch_count}, duration={duration:.3f}s)"
                )
        elif message == 3:
            if i is not None:
                print(
                    f"{self.executor.engine.fen()} revisit {i} pick_move call: "
                    f"({pick:+.2f}) {duration_per_branch:.3f}s/branch"
                    f" (branch_count={branch_count}, duration={duration:.3f}s)"
                )
        return pick, duration_per_branch, duration, branch_count
