"""Tests Quartney"""
from typing import Callable
from unittest import TestCase
from os import path

from chmutils import BetterHeatmapCache, HeatmapCache
from tests.utils import CACHE_DIR, clear_test_cache

HeatmapCache.cache_dir = CACHE_DIR
BetterHeatmapCache.cache_dir = CACHE_DIR


class TestQuartney(TestCase):
    """Tests Quartney"""
    mother_class: Callable

    def setUp(self) -> None:
        """Sets ups the engine instance to be tested with"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        # pylint: disable=import-outside-toplevel
        from chmengine.engines.quartney import Quartney
        self.mother_class = Quartney

    def tearDown(self) -> None:
        """clear any leftover test cache"""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def test_instantiate_quartney(self) -> None:
        """Test that trying to instantiate Quartney raises TypeError"""
        with self.assertRaises(TypeError):
            # pylint: disable=abstract-class-instantiated
            self.mother_class()
