"""Test Cmhmey Jr."""
import time
from io import StringIO
from unittest import TestCase, main
from os import path
import chess
from chess import pgn
import numpy

from numpy import float64, testing

import chmutils
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
    E3 = chess.Move.from_uci('e2e3')
    E4 = chess.Move.from_uci('e2e4')
    E5 = chess.Move.from_uci('e7e5')

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
        filename_17 = self.engine.qtable_filename(pieces_count=17)
        self.assertEqual(filename_17, self.filename_17)

    def test_qdb_path(self) -> None:
        """Tests q-table path builder"""
        q_path_32 = self.engine.qdb_path()
        self.assertEqual(q_path_32, path.join(CACHE_DIR, self.filename_32))
        q_path_2 = self.engine.qdb_path(fen=self.fen_2)
        self.assertEqual(q_path_2, path.join(CACHE_DIR, self.filename_2))
        q_path_3 = self.engine.qdb_path(board=chess.Board(fen=self.fen_3))
        self.assertEqual(q_path_3, path.join(CACHE_DIR, self.filename_3))
        q_path_17 = self.engine.qdb_path(pieces_count=17)
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
        """Tests fen method"""
        board = chess.Board()
        fen = self.engine.fen()
        self.assertEqual(fen, board.fen())
        move = tuple(self.engine.board.legal_moves)[0]
        self.engine.board.push(move)
        other_fen = self.engine.fen(board)
        self.assertEqual(other_fen, fen)
        fen = self.engine.fen()
        board.push(move)
        self.assertEqual(fen, board.fen())

    def test_get_q_value(self) -> None:
        """Tests q-value getter, somewhat... see test_set_q_value."""
        nothing = self.engine.get_q_value()
        self.assertIs(nothing, None)
        more_nothing = self.engine.get_q_value(fen=self.fen_2)
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
        self.engine.set_q_value(value=value_2, fen=self.fen_2)
        saved_value = self.engine.get_q_value(fen=self.fen_2)
        self.assertEqual(saved_value, value_2)
        value_3 = float64(99)
        self.engine.set_q_value(value_3, board=chess.Board(self.fen_3))
        saved_value = self.engine.get_q_value(fen=self.fen_3)
        self.assertEqual(saved_value, value_3)

    def test_update_q_values(self) -> None:
        """Tests update_q_values method (rewards function)"""
        pgn_buffer = StringIO(
            """
            1. f3 e5 2. g4 Qh4# 0-1
            
            
            """
        )
        game = pgn.read_game(pgn_buffer)
        for move in game.mainline_moves():
            print(self.engine.board.fen())
            pick = self.engine.pick_move(debug=True)
            move_board = self.engine.board_copy_pushed(move)
            move_score = self.engine.get_q_value(fen=move_board.fen(), board=move_board)
            if self.engine.board.turn:
                self.assertGreater(pick[1], move_score)
            else:
                self.assertEqual(pick[1], move_score)
            print(' game line move:', (move, move_score), '\n', 'engine line move:', pick)
            self.engine.board.push(move)
        print(self.engine.board.fen())
        self.engine.update_q_values(debug=True)
        for move in game.mainline_moves():
            print(self.engine.board.fen())
            pick = self.engine.pick_move(debug=True)
            move_board = self.engine.board_copy_pushed(move)
            move_score = self.engine.get_q_value(fen=move_board.fen(), board=move_board)
            if self.engine.board.turn:
                self.assertGreater(pick[1], move_score)
            else:
                self.assertEqual(pick[1], move_score)
            print(' game line move:', (move, move_score), '\n', 'engine line move:', pick)
            self.engine.board.push(move)
        print(self.engine.board.fen())

    def test_pick_move(self) -> None:
        """Tests pick_move method."""
        start = time.perf_counter()
        pick = self.engine.pick_move()
        duration_first = time.perf_counter() - start
        print(f"{self.engine.fen()} pick_move call: ({pick[0].uci()}, {pick[1]:.2f}) {duration_first:.3f}s")
        testing.assert_array_equal(pick, (self.E3, 0.23333333333333428))
        init_w_moves = list(self.engine.board.legal_moves)
        move: chess.Move
        first_time_pick_times = [duration_first]
        init_board_pick_times = [duration_first]
        revisit_pick_times = []
        new_duration = 999999.99
        for i, move in enumerate(init_w_moves, 2):
            self.engine.board.push(move)
            start = time.perf_counter()
            response_pick = self.engine.pick_move()
            duration_rep_pick = time.perf_counter() - start
            first_time_pick_times.append(duration_rep_pick)
            print(
                f"{self.engine.fen()} pick_move call: "
                f"({response_pick[0].uci()}, {response_pick[1]:.2f}) {duration_rep_pick:.3f}s"
            )
            self.engine.board.pop()
            start = time.perf_counter()
            new_pick = self.engine.pick_move()
            new_duration = time.perf_counter() - start
            init_board_pick_times.append(new_duration)
            revisit_pick_times.append(new_duration)
            print(f"{self.engine.fen()} pick_move call {i}: ({new_pick[0].uci()}, {new_pick[1]:.2f}) {new_duration:.3f}s")
        self.assertLess(new_duration, duration_first)
        avg_duration = numpy.mean(init_board_pick_times)
        avg_response = numpy.mean(first_time_pick_times)
        avg_revisit = numpy.mean(revisit_pick_times)
        self.assertLess(avg_duration, avg_response)
        pre_durations = numpy.percentile(init_board_pick_times, [0, 1, 10, 25, 50, 75, 90, 99, 100])
        pre_response = numpy.percentile(first_time_pick_times, [0, 1, 10, 25, 50, 75, 90, 99, 100])
        pre_revisit = numpy.percentile(revisit_pick_times, [0, 1, 10, 25, 50, 75, 90, 99, 100])
        print(f"mean pick time: {avg_duration:.3f}s\npercentiles (0, 1, 10, 25, 50, 75, 90, 99, 100):\n{pre_durations}")
        print(
            f"mean response time: {avg_response:.3f}s\npercentiles (0, 1, 10, 25, 50, 75, 90, 99, 100):\n{pre_response}"
        )
        print(
            f"mean revisit time: {avg_revisit:.3f}s\npercentiles (0, 1, 10, 25, 50, 75, 90, 99, 100):\n{pre_revisit}"
        )
        false_positive_fen = "r1b1kb1r/1p1pqppp/5n2/pp3Q2/3p4/1P1PP3/PB1PNPPP/2RK3R b kq - 1 12"
        self.engine.board = chess.Board(fen=false_positive_fen)
        print(self.engine.board)
        start = time.perf_counter()
        pick = self.engine.pick_move(debug=True)
        duration_ = time.perf_counter() - start
        print(f"{self.engine.board.fen()} pick: ({pick[0].uci()}, {pick[1]:.1f}) {duration_:.3f}s")
        self.assertNotEqual(pick[0].uci(), 'e7c5')
        self.engine.board.push(pick[0])
        print(self.engine.board)

    def test__update_current_move_choices_(self) -> None:
        """Tests internal _update_current_move_choices_ method."""
        e4_board = self.engine.board_copy_pushed(self.E4)
        # pylint: disable=protected-access
        move_choices = self.engine._update_current_move_choices_(
            [(None, None)],
            e4_board,
            self.E4,
            chmutils.calculate_chess_move_heatmap_with_better_discount(e4_board).data.transpose(),
            self.engine.current_player_heatmap_index(),
            self.engine.other_player_heatmap_index(),
            *self.engine.get_king_boxes(e4_board),
            new_fen=e4_board.fen()
        )
        testing.assert_array_equal(move_choices, [(self.E4, 0.20689655172414234)])
        e3_board = self.engine.board_copy_pushed(self.E3)
        move_choices = self.engine._update_current_move_choices_(
            move_choices,
            e3_board,
            self.E3,
            chmutils.calculate_chess_move_heatmap_with_better_discount(e3_board).data.transpose(),
            self.engine.current_player_heatmap_index(),
            self.engine.other_player_heatmap_index(),
            *self.engine.get_king_boxes(e3_board),
            new_fen=e3_board.fen()
        )
        testing.assert_array_equal(move_choices, [(self.E3, 0.23333333333333428), (self.E4, 0.20689655172414234)])

    def test__get_or_calculate_responses_(self) -> None:
        """Tests internal _get_or_calculate_responses_ method."""
        # pylint: disable=protected-access
        responses = self.engine._get_or_calculate_responses_(
            self.engine.board,
            self.engine.other_player_heatmap_index(),
            self.engine.current_player_heatmap_index(),
            False
        )
        current_moves = self.engine.current_moves_list()
        self.assertEqual(len(responses), len(current_moves))
        last_response_score = min(s for _, s in responses)
        for response_move, response_score in responses:
            self.assertIsInstance(response_move, chess.Move)
            self.assertIsInstance(response_score, float64)
            self.assertIn(response_move, current_moves)
            self.assertGreaterEqual(response_score, last_response_score)
            last_response_score = response_score

    def test__get_or_calc_next_move_score_(self) -> None:
        """Tests internal _get_or_calc_next_move_score_ method."""
        # pylint: disable=protected-access
        next_move_scores = self.engine._get_or_calc_next_move_score_(
            self.E4,
            [(None, None)],
            self.engine.board,
            self.engine.current_player_heatmap_index(),
            self.engine.other_player_heatmap_index(),
            False
        )
        testing.assert_array_equal(next_move_scores, [(self.E4, 14.0)])
        next_move_scores = self.engine._get_or_calc_next_move_score_(
            self.E3,
            next_move_scores,
            self.engine.board,
            self.engine.current_player_heatmap_index(),
            self.engine.other_player_heatmap_index(),
            False
        )
        testing.assert_array_equal(next_move_scores, [(self.E3, 13.95), (self.E4, 14.0)])

    def test__calculate_next_move_score_(self) -> None:
        """Tests internal _calculate_next_move_score_ method."""
        # pylint: disable=protected-access
        e4_next_move_score = self.engine._calculate_next_move_score_(
            self.engine.board_copy_pushed(self.E4),
            self.engine.current_player_heatmap_index(),
            self.engine.other_player_heatmap_index(),
        )
        self.assertEqual(e4_next_move_score, 14.0)
        self.assertIsInstance(e4_next_move_score, float64)

    def test__update_heatmap_transposed_with_mate_values_(self) -> None:
        """Tests internal _update_heatmap_transposed_with_mate_values_ method."""
        hmap_data_transposed = heatmaps.ChessMoveHeatmap().data.transpose()
        # pylint: disable=protected-access
        self.engine._update_heatmap_transposed_with_mate_values_(
            hmap_data_transposed,
            self.engine.current_player_heatmap_index(),
            self.engine.board
        )
        self.assertEqual(sum(hmap_data_transposed[self.engine.current_player_heatmap_index()]), 4096.0)
        self.assertEqual(sum(hmap_data_transposed[self.engine.other_player_heatmap_index()]), 0)

    def test__insert_ordered_best_to_worst_(self) -> None:
        """Tests internal _insert_ordered_best_to_worst_ method"""
        all_moves = [
            (self.E4, float64(100)),
            (chess.Move.from_uci('d2d4'), float64(80)),
            (chess.Move.from_uci('a2a4'), float64(-100))
        ]
        # pylint: disable=protected-access
        moves = [all_moves[0]]
        self.engine._insert_ordered_best_to_worst_(moves, *all_moves[2])
        testing.assert_array_equal(moves, [all_moves[0]] + [all_moves[2]])
        self.engine._insert_ordered_best_to_worst_(moves, *all_moves[1])
        testing.assert_array_equal(moves, all_moves)

    def test__insert_ordered_worst_to_best_(self) -> None:
        """Tests internal _insert_ordered_worst_to_best_ method"""
        all_moves = [
            (chess.Move.from_uci('a2a4'), float64(-100)),
            (chess.Move.from_uci('d2d4'), float64(80)),
            (self.E4, float64(100))
        ]
        # pylint: disable=protected-access
        moves = [all_moves[0]]
        self.engine._insert_ordered_worst_to_best_(moves, *all_moves[2])
        testing.assert_array_equal(moves, [all_moves[0]] + [all_moves[2]])
        self.engine._insert_ordered_worst_to_best_(moves, *all_moves[1])
        testing.assert_array_equal(moves, all_moves)

    def test__calculate_score_(self) -> None:
        """tests internal _calculate_score_ method"""
        # pylint: disable=protected-access
        null_score = self.engine._calculate_score_(0, 1, heatmaps.ChessMoveHeatmap().data.transpose(), [4], [60])
        self.assertEqual(null_score, 0)
        self.assertIsInstance(null_score, float64)
        hmap_data_transposed = chmutils.calculate_chess_move_heatmap_with_better_discount(
            self.engine.board
        ).data.transpose()
        score = self.engine._calculate_score_(
            self.engine.current_player_heatmap_index(),
            self.engine.other_player_heatmap_index(),
            hmap_data_transposed,
            *self.engine.get_king_boxes()
        )
        self.assertEqual(score, 0)
        self.assertIsInstance(score, float64)
        self.engine.board.push(self.E4)
        e4_hmap_data_transposed = chmutils.calculate_chess_move_heatmap_with_better_discount(
            self.engine.board
        ).data.transpose()
        e4_score = self.engine._calculate_score_(
            self.engine.current_player_heatmap_index(),
            self.engine.other_player_heatmap_index(),
            e4_hmap_data_transposed,
            *self.engine.get_king_boxes()
        )
        self.assertEqual(e4_score, -10.0)
        self.assertIsInstance(e4_score, float64)
        self.engine.board.push(self.E5)
        e5_hmap_data_transposed = chmutils.calculate_chess_move_heatmap_with_better_discount(
            self.engine.board
        ).data.transpose()
        e5_score = self.engine._calculate_score_(
            self.engine.current_player_heatmap_index(),
            self.engine.other_player_heatmap_index(),
            e5_hmap_data_transposed,
            *self.engine.get_king_boxes()
        )
        self.assertEqual(e5_score, 0.20689655172414234)
        self.assertIsInstance(e4_score, float64)

    def test__formatted_moves_(self) -> None:
        """Tests internal format method"""
        # pylint: disable=protected-access
        null_formatted_moves = self.engine._formatted_moves_([(None, None)])
        testing.assert_array_equal(null_formatted_moves, [])
        formatted_moves = self.engine._formatted_moves_([(self.E4, float64(10.0))])
        testing.assert_array_equal(formatted_moves, [('e2e4', '10.00')])
