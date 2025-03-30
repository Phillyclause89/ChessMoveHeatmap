"""Test Cmhmey Sr."""
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

class TestCMHMEngine(TestCase):
    """Test Cmhmey Sr."""
    e2e4 = chess.Move.from_uci('e2e4')
    e7e5 = chess.Move.from_uci('e7e5')

    def setUp(self) -> None:
        """Sets ups the engine instance to be tested with"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        # pylint: disable=import-outside-toplevel
        from chmengine import CMHMEngine
        self.engine = CMHMEngine()
        self.assertIsInstance(self.engine, CMHMEngine)

    def tearDown(self) -> None:
        """clear any leftover test cache"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def test_depth(self) -> None:
        """test depth property"""
        self.assertEqual(self.engine.depth, 1)
        self.engine.depth = 3
        self.assertEqual(self.engine.depth, 3)
        self.engine.depth = 69.7
        self.assertEqual(self.engine.depth, 69)
        with self.assertRaises(TypeError):
            self.engine.depth = "not a number"
        with self.assertRaises(ValueError):
            self.engine.depth = -1.9

    def test_board(self) -> None:
        """tests board property"""
        self.assertIsInstance(self.engine.board, chess.Board)
        board = chess.Board()
        self.assertEqual(board.fen(), self.engine.board.fen())
        board.push(tuple(board.legal_moves)[0])
        self.assertNotEqual(board.fen(), self.engine.board.fen())
        self.engine.board = board.copy()
        self.assertEqual(board.fen(), self.engine.board.fen())
        self.engine.board.push(tuple(board.legal_moves)[0])
        self.assertNotEqual(board.fen(), self.engine.board.fen())
        self.engine.board = board
        self.assertEqual(board.fen(), self.engine.board.fen())
        self.engine.board.push(tuple(board.legal_moves)[1])
        self.assertNotEqual(board.fen(), self.engine.board.fen())
        bad_board = chess.Board("rnbqkbnp/pppppppr/8/8/8/7N/PPPPPPPR/RNBQKB1P b KQkq - 1 1")
        with self.assertRaises(TypeError):
            self.engine.board = []
        with self.assertRaises(ValueError):
            self.engine.board = bad_board

    def test_board_copy_pushed(self) -> None:
        """tests board_copy_pushed method"""
        board_copy = self.engine.board_copy_pushed(tuple(self.engine.board.legal_moves)[4])
        self.assertNotEqual(board_copy.fen(), self.engine.board.fen())
        board_copy2 = self.engine.board_copy_pushed(tuple(board_copy.legal_moves)[5])
        self.assertNotEqual(board_copy2.fen(), board_copy.fen())

    def test_pick_move(self) -> None:
        """tests pick_move method"""
        move_response = (move, _) = self.engine.pick_move()
        self.assertEqual(move_response, (self.e2e4, float64(10.0)))
        same_move_response = self.engine.pick_move()
        self.assertEqual(move_response, same_move_response)
        self.engine.board.push(move)
        new_move_response = self.engine.pick_move()
        self.assertEqual(new_move_response, (self.e7e5, float64(-0.20689655172414234)))
        self.engine.board = chess.Board()
        move_response = (move, _) = self.engine.pick_move(pick_by='all-max')
        self.assertEqual(move_response, (self.e2e4, float64(30.0)))
        self.engine.board.push(move)
        new_move_response = self.engine.pick_move(pick_by='all-min')
        self.assertEqual(new_move_response[1], float64(29.0))

    def test_null_target_moves(self) -> None:
        """test null_target_moves method"""
        ntm = self.engine.null_target_moves(1)
        self.assertEqual(ntm, ([(None, None)],))
        ntm1, ntm2 = self.engine.null_target_moves(2)
        self.assertEqual((ntm1, ntm2), ([(None, None)], [(None, None)]))
        ntm0 = self.engine.null_target_moves(0)
        self.assertEqual(ntm0, ())

    def test_other_player_heatmap_index(self) -> None:
        """Tests other_player_heatmap_index property"""
        other_index = self.engine.other_player_heatmap_index
        self.assertEqual(other_index, 1)
        with self.assertRaises(AttributeError):
            self.engine.other_player_heatmap_index = 5
        self.engine.board.push(self.e2e4)
        other_index = self.engine.other_player_heatmap_index
        self.assertEqual(other_index, 0)

    def test_current_player_heatmap_index(self) -> None:
        """Tests current_player_heatmap_index property"""
        current_index = self.engine.current_player_heatmap_index
        self.assertEqual(current_index, 0)
        with self.assertRaises(AttributeError):
            self.engine.other_player_heatmap_index = 5
        self.engine.board.push(self.e2e4)
        current_index = self.engine.current_player_heatmap_index
        self.assertEqual(current_index, 1)

    def test_update_target_moves_by_delta(self) -> None:
        """Tests the update_target_moves_by_delta method"""
        null_target_moves, = self.engine.null_target_moves(1)
        target_moves_by_delta = self.engine.update_target_moves_by_delta(
            null_target_moves,
            float64(1.0),
            float64(2.0),
            self.e2e4
        )
        target_moves_by_delta2 = self.engine.update_target_moves_by_delta(
            target_moves_by_delta,
            float64(1.0),
            float64(1.0),
            self.e7e5
        )
        target_moves_by_delta3 = self.engine.update_target_moves_by_delta(
            target_moves_by_delta2,
            float64(-1.0),
            float64(1.0),
            self.e2e4
        )
        self.assertEqual(null_target_moves, [(None, None)])
        self.assertEqual(target_moves_by_delta, [(self.e2e4, float64(-1.0))])
        self.assertEqual(target_moves_by_delta2, [(self.e7e5, float64(0.0))])
        self.assertEqual(target_moves_by_delta3, target_moves_by_delta2)

    def test_update_target_moves_by_min_other(self) -> None:
        null_target_moves, = self.engine.null_target_moves(1)
        heatmap_t = heatmaps.ChessMoveHeatmap().data.transpose()
        score, target_moves_by_min = self.engine.update_target_moves_by_min_other(
            null_target_moves,
            heatmap_t,
            self.e2e4,
            self.engine.other_player_heatmap_index
        )
        self.assertEqual(null_target_moves, [(None, None)])
        self.assertEqual(target_moves_by_min, [(self.e2e4, float64(0))])
        self.assertEqual(score, float64(0.0))

    def test_update_target_moves_by_max_current(self) -> None:
        pass

    def test_update_target_moves_by_king_delta(self) -> None:
        pass

    def test_update_target_moves_by_min_current_king(self) -> None:
        pass

    def test_update_target_moves_by_max_other_king(self) -> None:
        pass

    def test_heatmap_data_is_zeros(self) -> None:
        pass

    def test_get_king_boxes(self) -> None:
        pass

    def test_is_valid_king_box_square(self) -> None:
        pass

    def test_get_or_calc_move_maps_list(self) -> None:
        pass

    def test_get_or_calc_move_maps(self) -> None:
        pass

    def test_current_moves_list(self) -> None:
        pass


if __name__ == "__main__":
    main()
