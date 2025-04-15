"""Tests engine utilities"""
from os import path
from unittest import TestCase

from chess import Move
from numpy import float64, testing

from chmengine.utils import (
    calculate_score,
    format_moves,
    insert_ordered_best_to_worst,
    insert_ordered_worst_to_best,
    pieces_count_from_fen
)
from chmutils import BetterHeatmapCache, HeatmapCache, calculate_chess_move_heatmap_with_better_discount
from heatmaps import ChessMoveHeatmap
from tests.utils import CACHE_DIR, clear_test_cache

HeatmapCache.cache_dir = CACHE_DIR
BetterHeatmapCache.cache_dir = CACHE_DIR


class TestEngineUtils(TestCase):
    """Tests engine utilities"""
    filename_32 = "qtable_depth_1_piece_count_32.db"
    filename_2 = "qtable_depth_1_piece_count_2.db"
    filename_3 = "qtable_depth_1_piece_count_3.db"
    filename_17 = "qtable_depth_1_piece_count_17.db"
    fen_2 = "8/8/4k3/8/8/3K4/8/8 w - - 0 1"
    fen_3 = "8/8/4k3/8/8/p2K4/8/8 w - - 0 1"
    E3 = Move.from_uci('e2e3')
    E4 = Move.from_uci('e2e4')
    E5 = Move.from_uci('e7e5')

    def setUp(self) -> None:
        """Sets ups the engine instance to be tested with"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        # pylint: disable=import-outside-toplevel
        from chmengine.engines.cmhmey2 import CMHMEngine2

        CMHMEngine2.cache_dir = CACHE_DIR
        self.engine = CMHMEngine2()
        self.assertIsInstance(self.engine, CMHMEngine2)

    def tearDown(self) -> None:
        """clear any leftover test cache"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def test_format_moves(self):
        """Tests internal format method"""
        # pylint: disable=protected-access
        null_formatted_moves = format_moves([(None, None)])
        testing.assert_array_equal(null_formatted_moves, [])
        formatted_moves = format_moves([(self.E4, float64(10.0))])
        testing.assert_array_equal(formatted_moves, [('e2e4', '10.00')])

    def test_calculate_score(self):
        """tests internal calculate_score method"""
        # pylint: disable=protected-access
        null_score = calculate_score(0, ChessMoveHeatmap().data.transpose(), [4], [60])
        self.assertEqual(null_score, 0)
        self.assertIsInstance(null_score, float64)
        hmap_data_transposed = calculate_chess_move_heatmap_with_better_discount(
            self.engine.board
        ).data.transpose()
        score = calculate_score(
            self.engine.current_player_heatmap_index(),
            hmap_data_transposed,
            *self.engine.get_king_boxes()
        )
        self.assertEqual(score, 0)
        self.assertIsInstance(score, float64)
        self.engine.board.push(self.E4)
        e4_hmap_data_transposed = calculate_chess_move_heatmap_with_better_discount(
            self.engine.board
        ).data.transpose()
        e4_score = calculate_score(
            self.engine.current_player_heatmap_index(),
            e4_hmap_data_transposed,
            *self.engine.get_king_boxes()
        )
        self.assertEqual(e4_score, -10.0)
        self.assertIsInstance(e4_score, float64)
        self.engine.board.push(self.E5)
        e5_hmap_data_transposed = calculate_chess_move_heatmap_with_better_discount(
            self.engine.board
        ).data.transpose()
        e5_score = calculate_score(
            self.engine.current_player_heatmap_index(),
            e5_hmap_data_transposed,
            *self.engine.get_king_boxes()
        )
        self.assertEqual(e5_score, 0.20689655172414234)
        self.assertIsInstance(e4_score, float64)

    def test_insert_ordered_worst_to_best(self):
        """Tests internal insert_ordered_worst_to_best method"""
        all_moves = [
            (Move.from_uci('a2a4'), float64(-100)),
            (Move.from_uci('d2d4'), float64(80)),
            (self.E4, float64(100))
        ]
        # pylint: disable=protected-access
        moves = [all_moves[0]]
        insert_ordered_worst_to_best(moves, *all_moves[2])
        testing.assert_array_equal(moves, [all_moves[0]] + [all_moves[2]])
        insert_ordered_worst_to_best(moves, *all_moves[1])
        testing.assert_array_equal(moves, all_moves)

    def test_insert_ordered_best_to_worst(self):
        """Tests internal insert_ordered_best_to_worst method"""
        all_moves = [
            (self.E4, float64(100)),
            (Move.from_uci('d2d4'), float64(80)),
            (Move.from_uci('a2a4'), float64(-100))
        ]
        # pylint: disable=protected-access
        moves = [all_moves[0]]
        insert_ordered_best_to_worst(moves, *all_moves[2])
        testing.assert_array_equal(moves, [all_moves[0]] + [all_moves[2]])
        insert_ordered_best_to_worst(moves, *all_moves[1])
        testing.assert_array_equal(moves, all_moves)

    def test_pieces_count_from_fen(self):
        """Tests internal pieces_count_from_fen method"""
        pieces_count = pieces_count_from_fen(self.engine.fen())
        self.assertEqual(pieces_count, 32)
