"""Player class"""
from dataclasses import dataclass
from typing import Optional, Tuple

from chess import COLOR_NAMES


@dataclass
class Player:
    """Player dataclass for engine manager"""
    _name: str = 'Unknown'
    _index: int = 0
    _color: str = COLOR_NAMES[1]
    _COLORS: Tuple[str, str] = (_color, COLOR_NAMES[0])

    def __init__(self, name: Optional[str] = None, index: Optional[int] = None, color: Optional[str] = None):
        """Initializes the Player object.

        Parameters
        ----------
        name : Optional[str]
        index : Optional[int]
        color : Optional[str]
        """
        if name is not None:
            self.name = name
        if index is not None:
            self.index = index
        if color is not None:
            self.color = color

    @property
    def name(self) -> str:
        """

        Returns
        -------
        str

        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        """

        Parameters
        ----------
        new_name : str
        """
        self._name = str(new_name).strip()

    @property
    def index(self) -> int:
        """

        Returns
        -------
        int

        """
        return self._index

    @index.setter
    def index(self, new_index: int):
        """

        Parameters
        ----------
        new_index : int
        """
        try:
            new_index = int(new_index)
            self._color = self._COLORS[new_index]
            self._index = new_index
        except TypeError as error:
            raise TypeError(f'Object type of new_index is not covertable to int: {type(new_index)}') from error
        except IndexError as error:
            raise ValueError(f"new_index value must be 0 (for white) or 1 (for black), got {new_index}") from error

    @property
    def color(self) -> str:
        """

        Returns
        -------
        str

        """
        return self._color

    @color.setter
    def color(self, new_color: str):
        """

        Parameters
        ----------
        new_color : str
        """
        try:
            new_color = str(new_color).strip().lower()
            self._index = self._COLORS.index(new_color)
            self._color = new_color
        except ValueError as error:
            raise ValueError(f"new_color must be 'black' or 'white', got `{new_color}`") from error
