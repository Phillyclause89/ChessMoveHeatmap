"""Test Cmhmey Sr."""
from os import path
from unittest import TestCase, main

from chess import Board, Move
from numpy import float64, testing

from chmengine import Pick
from chmengine.utils import is_valid_king_box_square, null_target_moves
from chmutils import BetterHeatmapCache, HeatmapCache
from heatmaps import ChessMoveHeatmap, ChessMoveHeatmapT, GradientHeatmap, GradientHeatmapT
from tests.utils import CACHE_DIR, clear_test_cache

HeatmapCache.cache_dir = CACHE_DIR
BetterHeatmapCache.cache_dir = CACHE_DIR


class TestCMHMEngine(TestCase):
    """Test Cmhmey Sr."""
    e2e4 = Move.from_uci('e2e4')
    e7e5 = Move.from_uci('e7e5')

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
        with self.assertRaises(TypeError):
            self.engine.depth = "not a number"
        with self.assertRaises(ValueError):
            self.engine.depth = -1

    def test_board(self) -> None:
        """tests board property"""
        self.assertIsInstance(self.engine.board, Board)
        board = Board()
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
        bad_board = Board("rnbqkbnp/pppppppr/8/8/8/7N/PPPPPPPR/RNBQKB1P b KQkq - 1 1")
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
        self.assertEqual(move_response, Pick(self.e2e4, float64(10.0)))
        same_move_response = self.engine.pick_move()
        self.assertEqual(move_response, same_move_response)
        self.engine.board.push(move)
        new_move_response = self.engine.pick_move()
        self.assertEqual(new_move_response, Pick(self.e7e5, float64(-0.20689655172414234)))
        self.engine.board = Board()
        move_response = (move, _) = self.engine.pick_move(pick_by='all-max')
        self.assertEqual(move_response, Pick(self.e2e4, float64(30.0)))
        self.engine.board.push(move)
        new_move_response = self.engine.pick_move(pick_by='all-min')
        self.assertEqual(new_move_response[1], float64(29.0))

    def test_null_target_moves(self) -> None:
        """test null_target_moves method"""
        ntm = null_target_moves(1)
        self.assertEqual(ntm, ([(None, None)],))
        ntm1, ntm2 = null_target_moves(2)
        self.assertEqual((ntm1, ntm2), ([(None, None)], [(None, None)]))
        ntm0 = null_target_moves(0)
        self.assertEqual(ntm0, ())

    def test_other_player_heatmap_index(self) -> None:
        """Tests other_player_heatmap_index property"""
        other_index = self.engine.other_player_heatmap_index()
        self.assertEqual(other_index, 1)
        self.engine.board.push(self.e2e4)
        other_index = self.engine.other_player_heatmap_index()
        self.assertEqual(other_index, 0)

    def test_current_player_heatmap_index(self) -> None:
        """Tests current_player_heatmap_index property"""
        current_index = self.engine.current_player_heatmap_index()
        self.assertEqual(current_index, 0)
        self.engine.board.push(self.e2e4)
        current_index = self.engine.current_player_heatmap_index()
        self.assertEqual(current_index, 1)

    def test_update_target_moves_by_delta(self) -> None:
        """Tests the update_target_moves_by_delta method"""
        target_moves, = null_target_moves(1)
        target_moves_by_delta = self.engine.update_target_moves_by_delta(
            target_moves,
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
        self.assertEqual(target_moves, [(None, None)])
        self.assertEqual(target_moves_by_delta, [Pick(self.e2e4, float64(-1.0))])
        self.assertEqual(target_moves_by_delta2, [Pick(self.e7e5, float64(0.0))])
        self.assertEqual(target_moves_by_delta3, target_moves_by_delta2)

    def test_update_target_moves_by_min_other(self) -> None:
        target_moves, = null_target_moves(1)
        heatmap_t = ChessMoveHeatmap().data.transpose()
        score, target_moves_by_min = self.engine.update_target_moves_by_min_other(
            target_moves,
            heatmap_t,
            self.e2e4,
            self.engine.other_player_heatmap_index()
        )
        self.assertEqual(target_moves, [(None, None)])
        self.assertEqual(target_moves_by_min, [Pick(self.e2e4, float64(0))])
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
        """Tests heatmap_data_is_zeros method"""
        heatmap_t = GradientHeatmapT()
        heatmap = GradientHeatmap()
        move_heatmap_t = ChessMoveHeatmapT()
        move_heatmap = ChessMoveHeatmap()
        for hmap in (heatmap_t, heatmap, move_heatmap_t, move_heatmap):
            self.assertTrue(self.engine.heatmap_data_is_zeros(hmap))
            hmap[0][0] = float64(0.1)
            self.assertFalse(self.engine.heatmap_data_is_zeros(hmap))

    def test_get_king_boxes(self) -> None:
        king_box_current, king_box_other = self.engine.get_king_boxes()
        testing.assert_array_equal(king_box_current, [4, 3, 11, 12, 5, 13])
        testing.assert_array_equal(king_box_other, [60, 51, 59, 52, 53, 61])
        board = Board("8/4n3/3kB3/2n5/5N2/3bK3/3N4/8 w - - 0 1")
        king_box_current, king_box_other = self.engine.get_king_boxes(board)
        testing.assert_array_equal(king_box_current, [20, 11, 19, 27, 12, 28, 13, 21, 29])
        testing.assert_array_equal(king_box_other, [43, 34, 42, 50, 35, 51, 36, 44, 52])

    def test_is_valid_king_box_square(self) -> None:
        self.assertFalse(is_valid_king_box_square(4, 4))
        self.assertFalse(is_valid_king_box_square(60, 60))

    def test_get_or_calc_move_maps_list(self) -> None:
        pass

    def test_get_or_calc_move_maps(self) -> None:
        pass

    def test_current_moves_list(self) -> None:
        pass


if __name__ == "__main__":
    main()
