"""Pick object representing a (move, score) evaluation pair."""
from dataclasses import dataclass, field
from numbers import Number
from typing import Iterator, Tuple, Union

from chess import Move
from numpy import float64

__all__ = ['Pick']


@dataclass(order=True)
class PickT:
    # noinspection PyUnresolvedReferences
    """Base class for Pick containing a (move, score) pair.

    Attributes
    ----------
    _score : float64
        The numeric evaluation score associated with the move.
    _move : chess.Move
        The move object from python-chess.

    Notes
    -----
    Comparison is based solely on `score`, while `move` is excluded from comparison.
    """
    _score: float64 = field()
    _move: Move = field(compare=False)

    @property
    def score(self) -> float64:
        """Read access to the score.

        Returns
        -------
        float64
            The numeric evaluation of the move.
        """
        return self._score

    @score.setter
    def score(self, score: Number) -> None:
        """Write access to the score. Converts the value to float64.

        Parameters
        ----------
        score : Number
            A numeric value representing the evaluation to set.
        """
        self._score = float64(score)

    @property
    def move(self) -> Move:
        """Read-only access to the move.

        Returns
        -------
        Move
            The chess move associated with this Pick.
        """
        return self._move

    @property
    def data(self) -> Tuple[Move, float64]:
        """Read-only access to the pick data as a tuple.

        Returns
        -------
        Tuple[Move, float64]
            The (move, score) pair represented by this object.
        """
        return self._move, self._score


class Pick(PickT):
    # noinspection PyUnresolvedReferences
    """Fully featured container for a (move, score) pair.

    This class provides tuple-like access, type casting, and rich formatting.
    It extends `PickT` and is the public-facing API for move-score pairs.

    Attributes
    ----------
    move_getter_keys : Tuple[int, str, int]
        Allowed keys to access the move.
    score_getter_keys : Tuple[int, str, int]
        Allowed keys to access the score.
    """
    move_getter_keys: Tuple[int, str, int] = (0, 'move', -2)
    score_getter_keys: Tuple[int, str, int] = (1, 'score', -1)

    def __init__(self, move: Move, score: Number) -> None:
        """Initializes a Pick instance.

        Parameters
        ----------
        move : chess.Move
            The chess move to associate with this pick.
        score : Number
            The numeric evaluation of the move.
        """
        if not isinstance(move, Move):
            raise TypeError(f"the argument for move must be type 'chess.Move', got: {type(move)}")
        self._move, self.score = move, float64(score)

    def __getitem__(self, index: Union[str, int]) -> Union[Move, float64]:
        """Allows tuple-like access to the move and score using string or integer keys.

        Parameters
        ----------
        index : Union[str, int]
            Key to access ('move', 0, -2) or ('score', 1, -1).

        Returns
        -------
        Union[Move, float64]
            The requested component of the Pick.

        Raises
        ------
        IndexError
            If the key is not recognized.
        """
        if index in self.move_getter_keys:
            return self._move
        if index in self.score_getter_keys:
            return self._score
        raise IndexError("Index out of range. Use 0 (or -2) for 'move', 1 (or -1) for 'score'.")

    def __setitem__(self, key: Union[str, int], value: Number) -> None:
        """Allows updating of score via tuple-style access.

        Parameters
        ----------
        key : Union[str, int]
            Key that matches score_getter_keys.
        value : Number
            The new score value to assign.

        Raises
        ------
        IndexError
            If trying to assign to a read-only component (i.e., move).
        """
        if key in self.score_getter_keys:
            # Go through the prop setter here to convert to float64
            self.score = value
        else:
            raise IndexError("Index out of range. Only 1 (or -1) for 'score' can be set.")

    def __iter__(self) -> Iterator[Union[Move, float64]]:
        """Allows unpacking like a tuple.

        Returns
        -------
        Iterator[Union[Move, float64]]
            An iterator over (move, score).
        """
        yield self._move
        yield self._score

    def __len__(self) -> int:
        """Length of the object, always 2: (move, score).

        Returns
        -------
        int
        """
        return 2

    def __repr__(self) -> str:
        """Developer-friendly string representation.

        Returns
        -------
        str
        """
        return f"Pick(move={self._move.__repr__()}, score={self._score})"

    def __bool__(self) -> bool:
        """Truthiness of the Pick is based on score.

        Returns
        -------
        bool
        """
        return bool(self._score)

    def __int__(self) -> int:
        """Integer conversion of the score.

        Returns
        -------
        int
        """
        return int(self._score)

    def __float__(self) -> float64:
        """Float conversion of the score.

        Returns
        -------
        float64
        """
        return self._score

    def __format__(self, format_spec) -> str:
        """Formatted string output using the score's format spec.

        Parameters
        ----------
        format_spec : Any
            Format specifier passed to the float64 formatter.

        Returns
        -------
        str
        """
        return f"Pick(move={self._move.__repr__()}, score={format(self._score, format_spec)})"
