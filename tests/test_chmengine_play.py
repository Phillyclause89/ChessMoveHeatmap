"""Test the engine handler"""
import datetime
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
from tests.utils import clear_test_cache, CACHE_DIR, YHWH, YHVH

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

    def test_save_to_pgn(self) -> None:
        """Tests save_to_pgn method. (TBD)"""

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
            self.assertIsInstance(local_time, datetime.datetime)

    def test_set_utc_headers(self) -> None:
        """Tests set_utc_headers method (TBD)"""
        for handler in self.handlers:
            headers = pgn.Headers()
            self.assertNotIn('UTCDate', headers)
            self.assertNotIn('UTCTime', headers)
            handler.set_utc_headers(game_heads=headers, local_time=handler.get_local_time())
            self.assertIn('UTCDate', headers)
            self.assertIn('UTCTime', headers)
