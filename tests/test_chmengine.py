"""Tests chmengine packaging"""

from typing import Callable
from unittest import TestCase, main


class TestCMHEngineImports(TestCase):
    """Tests chmengine package imports"""
    _dir = ['self']
    _name = ['chmengine']
    _engine1 = ['CMHMEngine']
    _engine2 = ['CMHMEngine2']
    _engine3 = ['Quartney']
    _engine_manager = ['PlayCMHMEngine']
    _pick = ['Pick']
    _modules = ['engines', 'play', 'utils']
    _functions = [
        'format_moves',
        'calculate_white_minus_black_score',
        'checkmate_score',
        'is_draw',
        'get_white_and_black_king_boxes',
        'insert_ordered_worst_to_best',
        'insert_ordered_best_to_worst',
        'pieces_count_from_fen',
        'pieces_count_from_board',
        'set_all_datetime_headers',
        'set_utc_headers',
        'better_checkmate_score',
        'calculate_better_white_minus_black_score',
        'get_static_delta_score',
        'get_static_value',
        'max_moves_map',
    ]
    _engines = _engine1 + _engine2 + _engine3
    _all = _engines + _engine_manager + _pick + _functions + _modules

    def setUp(self) -> None:
        """Ensure dir() is fresh each TC"""
        _dir_0 = dir()
        self.assertEqual(_dir_0, self._dir)

    def test_mod_level_imports(self) -> None:
        """tests imports"""
        # pylint: disable=import-outside-toplevel
        import chmengine
        _dir_1 = dir()
        self.assertEqual(sorted(_dir_1), sorted(self._name + self._dir))
        self.assertEqual(sorted(chmengine.__all__), sorted(self._all))
        self.assertEqual(sorted(chmengine.engines.__all__), sorted(self._engines + ['cmhmey1', 'cmhmey2', 'quartney']))
        self.assertEqual(sorted(chmengine.play.__all__), sorted(self._engine_manager))
        self.assertTrue(callable(getattr(chmengine, "CMHMEngine")))

    def test_from_imports(self) -> None:
        """test from chmengine imports"""
        # pylint: disable=import-outside-toplevel
        from chmengine import CMHMEngine, CMHMEngine2, PlayCMHMEngine, Pick, Quartney
        _dir_1 = dir()
        self.assertEqual(sorted(_dir_1), sorted(self._engines + self._pick + self._engine_manager + self._dir))
        self.assertIsInstance(CMHMEngine, Callable)
        self.assertIsInstance(CMHMEngine2, Callable)
        self.assertIsInstance(PlayCMHMEngine, Callable)
        self.assertIsInstance(Pick, Callable)
        self.assertIsInstance(Quartney, Callable)

    def test_star_imports(self) -> None:
        """test from chmengine import *"""
        # pylint: disable=exec-used
        exec("from chmengine import *")
        _dir_1 = dir()
        self.assertEqual(sorted(_dir_1), sorted(self._all + self._dir))


if __name__ == '__main__':
    main()
