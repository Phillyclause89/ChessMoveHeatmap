"""Tests the Pick class"""
from unittest import TestCase
from chess import Move
from chmengine.utils.pick import Pick

F3: Move = Move.from_uci('f2f3')
E4: Move = Move.from_uci('e2e4')


class TestPick(TestCase):
    """Tests Pick"""
    pick_f3: Pick
    pick_e4: Pick
    e4: Move = E4
    f3: Move = F3
    score_abs: int = 10

    def setUp(self) -> None:
        """Sets ups the test obj(s)"""
        self.pick_e4 = Pick(self.e4, self.score_abs)
        self.pick_f3 = Pick(self.f3, -self.score_abs)

    def test_init_failure(self) -> None:
        """Tests init failures"""
        with self.assertRaises(TypeError):
            Pick('e2e4', 10)

    def test_score(self) -> None:
        """test_score"""

    def test_move(self) -> None:
        """test_move"""

    def test_data(self) -> None:
        """test_data"""
