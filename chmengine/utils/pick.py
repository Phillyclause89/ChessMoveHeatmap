"""Pick object"""
from dataclasses import dataclass
from numbers import Number
from typing import Tuple, Union

from chess import Move


@dataclass(order=True)
class Pick:
    """Container for a (move, score) pair where move is immutable and score is mutable."""
    _score: Number
    _move: Move

    __slots__ = ('_score', '_move')

    def __init__(self, move: Move, score: Number) -> None:
        """Initializes a Pick instance

        Parameters
        ----------
        move : chess.Move
        score : Number

        Examples
        --------
        >>> from

        """
        if not isinstance(move, Move):
            raise TypeError(f"the argument for move must be type 'chess.Move', got: {type(move)}")
        self._move, self.score = move, score

    def __getitem__(self, index: Union[str, int]) -> Union[Move, Number]:
        """Allows tuple-like access with indexing."""
        if index in (0, 'move'):
            return self._move
        if index in (1, 'score'):
            return self._score
        raise IndexError("Index out of range. Use 0 for 'move', 1 for 'score'.")

    def __setitem__(self, key: Union[str, int], value: Number) -> None:
        if key in (1, 'score'):
            self.score = value
        else:
            raise IndexError("Index out of range. Only 1 for 'score' can be set.")

    def __iter__(self):
        """Allows unpacking like a tuple."""
        yield self._move
        yield self._score

    def __len__(self) -> int:
        """Length is always 2: (move, score)."""
        return 2

    def __repr__(self) -> str:
        return f"Pick(move={self._move}, score={self._score})"

    def __bool__(self) -> bool:
        return bool(self._score)

    @property
    def score(self) -> Number:
        """Read access to the score."""
        return self._score

    @score.setter
    def score(self, score: Number) -> None:
        """Write access to the score."""
        if not isinstance(score, Number):
            raise TypeError(f'Value must be Number like, got {type(score)}')
        self._score = score

    @property
    def move(self) -> Move:
        """Read-only access to the move."""
        return self._move

    @property
    def data(self) -> Tuple[Move, Number]:
        """Read-only access to the pick data."""
        return self._move, self._score
