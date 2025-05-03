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
            # noinspection PyTypeChecker,PydanticTypeChecker
            Pick('e2e4', 10)
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker,PydanticTypeChecker
            Pick(self.e4, '')

    def test_score(self) -> None:
        """test_score"""
        self.assertEqual(
            (self.pick_e4.score, self.pick_f3.score),
            (self.score_abs, -self.score_abs)
        )
        self.assertGetItemsEqual()
        self.pick_f3.score, self.pick_e4.score = self.pick_e4.score, self.pick_f3.score
        self.assertEqual(
            (self.pick_e4.score, self.pick_f3.score),
            (-self.score_abs, self.score_abs)
        )
        self.assertGetItemsEqual()

    def test_bad_score(self):
        with self.assertRaises(ValueError):
            self.pick_e4.score = ''
        for key in self.pick_e4.score_getter_keys:
            with self.assertRaises(ValueError):
                # noinspection PyTypeChecker,PydanticTypeChecker
                self.pick_e4[key] = ''

    def assertGetItemsEqual(self) -> None:
        """Asserts __getitem__ calls match property getter."""
        self.assertEqual(
            (self.pick_e4[1], self.pick_f3[1]),
            (self.pick_e4.score, self.pick_f3.score)
        )
        self.assertEqual(
            (self.pick_e4[-1], self.pick_f3[-1]),
            (self.pick_e4[1], self.pick_f3[1])
        )
        self.assertEqual(
            (self.pick_e4['score'], self.pick_f3['score']),
            (self.pick_e4[-1], self.pick_f3[-1])
        )

    def test_move(self) -> None:
        """test_move"""
        self.assertEqual(
            (self.pick_e4.move, self.pick_f3.move),
            (self.e4, self.f3)
        )
        self.assertEqual(
            (self.pick_e4[0], self.pick_f3[0]),
            (self.pick_e4.move, self.pick_f3.move)
        )
        self.assertEqual(
            (self.pick_e4[-2], self.pick_f3[-2]),
            (self.pick_e4[0], self.pick_f3[0])
        )
        self.assertEqual(
            (self.pick_e4['move'], self.pick_f3['move']),
            (self.pick_e4[-2], self.pick_f3[-2])
        )
        with self.assertRaises(AttributeError):
            exec('self.pick_e4.move = self.f3')

    def test_data(self) -> None:
        """test_data"""
        self.assertEqual(
            (self.pick_e4.data, self.pick_f3.data),
            ((self.e4, self.score_abs), (self.f3, -self.score_abs))
        )
        with self.assertRaises(AttributeError):
            exec('self.pick_e4.data = self.f3, -self.score_abs')

    def test_other_magic_methods(self) -> None:
        """Tests other methods"""
        move, score = self.pick_e4
        self.assertEqual(
            (move, score),
            (self.e4, self.score_abs)
        )
        self.assertEqual(len(self.pick_e4), 2)
        self.assertTrue(self.pick_e4 and self.pick_f3)
        self.pick_e4.score = 0
        self.assertFalse(self.pick_e4)
        self.assertGreater(self.pick_e4, self.pick_f3)
        # noinspection PyTypeChecker,PydanticTypeChecker
        self.assertListEqual(
            sorted([self.pick_f3, self.pick_e4], reverse=True),
            [self.pick_e4, self.pick_f3]
        )
        self.pick_f3.score = 0
        self.assertEqual(self.pick_e4, self.pick_f3)
