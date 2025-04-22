"""Tests chmengine packaging"""

from typing import Callable
from unittest import TestCase, main

from numpy import testing


class TestCMHEngineImports(TestCase):
    """Tests chmengine package imports"""
    _dir = ['self']
    _name = ['chmengine']
    _engine1 = ['CMHMEngine']
    _engine2 = ['CMHMEngine2']
    _engine_manager = ['PlayCMHMEngine']
    _engines = _engine1 + _engine2
    _all = _engines + _engine_manager

    def setUp(self) -> None:
        """Ensure dir() is fresh each TC"""
        _dir_0 = dir()
        self.assertEqual(_dir_0, self._dir)

    def test_mod_level_imports(self) -> None:
        """tests imports"""
        # pylint: disable=import-outside-toplevel
        import chmengine
        _dir_1 = dir()
        self.assertEqual(_dir_1, self._name + self._dir)
        testing.assert_array_equal(chmengine.__all__, self._all)
        testing.assert_array_equal(chmengine.engines.__all__, self._engines)
        testing.assert_array_equal(chmengine.play.__all__, self._engine_manager)
        self.assertTrue(callable(getattr(chmengine, "CMHMEngine")))

    def test_from_imports(self) -> None:
        """test from chmengine imports"""
        # pylint: disable=import-outside-toplevel
        from chmengine import CMHMEngine, CMHMEngine2, PlayCMHMEngine
        _dir_1 = dir()
        self.assertEqual(_dir_1, self._all + self._dir)
        self.assertIsInstance(CMHMEngine, Callable)
        self.assertIsInstance(CMHMEngine2, Callable)
        self.assertIsInstance(PlayCMHMEngine, Callable)

    def test_star_imports(self) -> None:
        """test from chmengine import *"""
        # pylint: disable=exec-used
        exec("from chmengine import *")
        _dir_1 = dir()
        self.assertEqual(_dir_1, self._all + self._dir)


if __name__ == '__main__':
    main()
