"""Test Cmhmey Jr."""
from unittest import TestCase, main
from os import path
import chess
from numpy import float64
from numpy import testing

import heatmaps
from chmutils import HeatmapCache, BetterHeatmapCache
from tests.utils import clear_test_cache, CACHE_DIR

HeatmapCache.cache_dir = CACHE_DIR
BetterHeatmapCache.cache_dir = CACHE_DIR


class TestCMHMEngine2(TestCase):
    """Tests Cmhmey Jr."""
    filename_32 = "qtable_depth_1_piece_count_32.db"
    filename_2 = "qtable_depth_1_piece_count_2.db"
    filename_3 = "qtable_depth_1_piece_count_3.db"
    filename_17 = "qtable_depth_1_piece_count_17.db"
    fen_2 = "8/8/4k3/8/8/3K4/8/8 w - - 0 1"
    fen_3 = "8/8/4k3/8/8/p2K4/8/8 w - - 0 1"

    def setUp(self) -> None:
        """Sets ups the engine instance to be tested with"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        # pylint: disable=import-outside-toplevel
        from chmengine import CMHMEngine2
        CMHMEngine2.cache_dir = CACHE_DIR
        self.engine = CMHMEngine2()
        self.assertIsInstance(self.engine, CMHMEngine2)

    def tearDown(self) -> None:
        """clear any leftover test cache"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def test_qtable_filename(self) -> None:
        """Tests q-table filename builder"""
        filename_32 = self.engine.qtable_filename()
        self.assertEqual(filename_32, self.filename_32)
        filename_2 = self.engine.qtable_filename(fen=self.fen_2)
        self.assertEqual(filename_2, self.filename_2)
        filename_3 = self.engine.qtable_filename(board=chess.Board(fen=self.fen_3))
        self.assertEqual(filename_3, filename_3)
        filename_17 = self.engine.qtable_filename(piece_count=17)
        self.assertEqual(filename_17, self.filename_17)

    def test_qdb_path(self) -> None:
        """Tests q-table path builder"""
        q_path_32 = self.engine.qdb_path()
        self.assertEqual(q_path_32, path.join(CACHE_DIR, self.filename_32))
        q_path_2 = self.engine.qdb_path(fen=self.fen_2)
        self.assertEqual(q_path_2, path.join(CACHE_DIR, self.filename_2))
        q_path_3 = self.engine.qdb_path(board=chess.Board(fen=self.fen_3))
        self.assertEqual(q_path_3, path.join(CACHE_DIR, self.filename_3))
        q_path_17 = self.engine.qdb_path(piece_count=17)
        self.assertEqual(q_path_17, path.join(CACHE_DIR, self.filename_17))

    def test__init_qdb(self) -> None:
        """Tests method that inits DB files"""
        self.assertTrue(path.exists(CACHE_DIR))
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        # pylint: disable=protected-access
        self.engine._init_qdb()
        self.assertTrue(path.exists(CACHE_DIR))

    def test_state_fen(self) -> None:
        """Tests state_fen method"""
        board = chess.Board()
        fen = self.engine.state_fen()
        self.assertEqual(fen, board.fen())
        move = tuple(self.engine.board.legal_moves)[0]
        self.engine.board.push(move)
        other_fen = self.engine.state_fen(board)
        self.assertEqual(other_fen, fen)
        fen = self.engine.state_fen()
        board.push(move)
        self.assertEqual(fen, board.fen())

    def test_get_q_value(self) -> None:
        """Tests q-value getter, somewhat... see test_set_q_value."""
        nothing = self.engine.get_q_value()
        self.assertIs(nothing, None)
        more_nothing = self.engine.get_q_value(state_fen=self.fen_2)
        self.assertIs(more_nothing, None)
        even_more_nothing = self.engine.get_q_value(board=chess.Board(fen=self.fen_3))
        self.assertIs(even_more_nothing, None)

    def test_set_q_value(self) -> None:
        """Test q-value setter, as well as the getter"""
        value = float64(3.0)
        self.engine.set_q_value(value=value)
        saved_value = self.engine.get_q_value()
        self.assertEqual(saved_value, value)
        value_2 = float64(-16)
        self.engine.set_q_value(value=value_2, state_fen=self.fen_2)
        saved_value = self.engine.get_q_value(state_fen=self.fen_2)
        self.assertEqual(saved_value, value_2)
        value_3 = float64(99)
        self.engine.set_q_value(value_3, board=chess.Board(self.fen_3))
        saved_value = self.engine.get_q_value(state_fen=self.fen_3)
        self.assertEqual(saved_value, value_3)

    def test_update_q_values(self):
        pass

    def test_pick_move(self):
        pass

    def test_update_current_move_choices(self):
        pass

    def test_get_or_calculate_responses(self):
        pass

    def test_get_or_calc_next_move_score(self):
        pass

    def test_calculate_next_move_score(self):
        pass

    def test_update_heatmap_transposed_with_mate_values(self):
        pass

    def test_insert_ordered_best_to_worst(self):
        pass

    def test_insert_ordered_worst_to_best(self):
        pass

    def test_calculate_score(self):
        pass

    def test_formatted_moves(self):
        pass
