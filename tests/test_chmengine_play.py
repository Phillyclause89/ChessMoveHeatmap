"""Test the engine handler"""
from datetime import datetime
from io import StringIO
from os import path
from unittest import TestCase, mock

from chess import pgn
from numpy import testing

from chmutils import BetterHeatmapCache, HeatmapCache
from tests.utils import CACHE_DIR, YHVH, YHWH, clear_test_cache

HeatmapCache.cache_dir = CACHE_DIR
BetterHeatmapCache.cache_dir = CACHE_DIR


class TestPlayCMHMEngine(TestCase):
    """Tests the PlayCMHMEngine Class"""

    def setUp(self) -> None:
        """Sets ups the engine instance to be tested with"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        # pylint: disable=import-outside-toplevel
        from chmengine import PlayCMHMEngine, CMHMEngine2, CMHMEngine
        CMHMEngine2.cache_dir = CACHE_DIR
        PlayCMHMEngine.pgn_dir = CACHE_DIR
        self.engine1_handler = PlayCMHMEngine(engine=CMHMEngine, player_name=YHWH, site='Kingdom of Phil')
        self.engine1 = self.engine1_handler.engine
        self.engine2_handler = PlayCMHMEngine(engine=CMHMEngine2, player_name=YHVH, site='Kingdom of Phil')
        self.handlers = (self.engine1_handler, self.engine2_handler)
        self.engine2 = self.engine2_handler.engine
        self.assertIsInstance(self.engine1, CMHMEngine)
        self.assertIsInstance(self.engine2, CMHMEngine2)

    def tearDown(self) -> None:
        """clear any leftover test cache"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def test_play(self) -> None:
        """tests play method. (TBD)"""
        moves_1 = iter(list(m for m in ('f2f3', 'g2g4')))
        moves_2 = iter(list(m for m in ('f2f3', 'g2g4')))

        def next_move(prompt: str):
            """Gets the next user move to mock."""
            print(prompt)
            self.assertIn('white', prompt)
            try:
                return next(moves_1)
            except StopIteration:
                return next(moves_2)

        for handler in self.handlers:
            self.assertEqual(len(handler.round_results), 0)
            with mock.patch('builtins.input', next_move):
                handler.play()
            self.assertEqual(len(handler.round_results), 1)

    def test_save_to_pgn(self) -> None:
        """Tests save_to_pgn method. (TBD)"""
        pgn_buffer = StringIO(
            """
            1. f3 e5 2. g4 Qh4# 0-1


            """
        )
        game = pgn.read_game(pgn_buffer)
        for handler in self.handlers:
            pgn_path = path.join(handler.pgn_dir, 'test.pgn')
            handler.save_to_pgn(pgn_path, game)
            with open(pgn_path, encoding='UTF8') as pgn_file:
                same_game = pgn.read_game(pgn_file)
                testing.assert_array_equal(
                    list(m.uci() for m in same_game.mainline()),
                    list(m.uci() for m in game.mainline())
                )
        self.assertEqual(same_game.headers['Result'], '0-1')

    def test_train_cmhmey_jr(self) -> None:
        """Tests train_cmhmey_jr method. (TBD)"""
        with self.assertRaises(TypeError):
            self.engine1_handler.train_cmhmey_jr()

    def test_set_all_datetime_headers(self) -> None:
        """Tests set_all_datetime_headers method. (TBD)"""
        default_header = pgn.Headers(
            Event='?', Site='?',
            Date='????.??.??',
            Round='?', White='?',
            Black='?', Result='*'
        )
        for handler in self.handlers:
            headers = pgn.Headers()
            testing.assert_array_equal(
                default_header.items(),
                headers.items()
            )
            self.assertNotIn('UTCDate', headers)
            self.assertNotIn('UTCTime', headers)
            self.assertNotIn('Timezone', headers)
            handler.set_all_datetime_headers(game_heads=headers, local_time=handler.get_local_time())
            self.assertIn('UTCDate', headers)
            self.assertIn('UTCTime', headers)
            self.assertIn('Timezone', headers)
            self.assertNotEqual(default_header['Date'], headers['Date'])

    def test_get_local_time(self) -> None:
        """Tests get_local_time method."""
        for handler in self.handlers:
            local_time = handler.get_local_time()
            self.assertIsInstance(local_time, datetime)

    def test_set_utc_headers(self) -> None:
        """Tests set_utc_headers method (TBD)"""
        for handler in self.handlers:
            headers = pgn.Headers()
            self.assertNotIn('UTCDate', headers)
            self.assertNotIn('UTCTime', headers)
            handler.set_utc_headers(game_heads=headers, local_time=handler.get_local_time())
            self.assertIn('UTCDate', headers)
            self.assertIn('UTCTime', headers)
