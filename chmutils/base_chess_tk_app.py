"""BaseChessTkApp"""
from abc import ABCMeta, abstractmethod
from tkinter import Canvas, Event, Menu, Tk, colorchooser, font as tk_font, simpledialog
from typing import Callable, List, Optional, Set, Tuple

__all__ = [
    # Classes
    'BaseChessTkApp',
    # Constants
    'DARK_SQUARE_COLOR_PROMPT',
    'LIGHT_SQUARE_COLOR_PROMPT',
    'DEFAULT_COLORS',
    'DEFAULT_FONT'
]

DARK_SQUARE_COLOR_PROMPT: str = "Pick Dark Square Color"
LIGHT_SQUARE_COLOR_PROMPT: str = "Pick Light Square Color"
DEFAULT_COLORS: Tuple[str, str] = ("#ffffff", "#c0c0c0")
DEFAULT_FONT: str = "Arial"


class BaseChessTkApp(metaclass=ABCMeta):
    """Base class for a chess app"""
    updating: bool
    canvas: Canvas
    square_size: int
    depth: int
    font: str
    colors: List[str]
    highlight_squares: Set[int]
    current_move_index: int

    @abstractmethod
    def on_closing(self):
        """Clean up resources before closing the application."""
        pass

    def on_resize(self, event: Event) -> None:
        """Handle window resize events.

        This method is called whenever the user resizes the window, adjusting the canvas and square size
        to fit the new dimensions of the window. It triggers a board update after resizing.
        """
        if self.updating:
            return
        self.updating = True
        try:
            self.canvas.config(width=event.width, height=event.height)
            self.square_size = min(event.width, event.height) // 8
            self.update_board()
        finally:
            self.updating = False

    @abstractmethod
    def set_depth(self):
        """Prompt the user to set a new recursion depth for heatmap calculations.

        The user is asked to input an integer value. If a valid value is provided,
        the depth is updated and the window title is refreshed to reflect the change.
        """
        pass

    @abstractmethod
    def create_menu(self):
        """Constructs the app menu during initialization.

        This method creates a menu bar with options to load a PGN file, change the board colors,
        and modify the font. It also allows navigation through the moves in the game (next/previous move).
        """
        pass

    @abstractmethod
    def set_title(self):
        """Sets the App window title"""
        pass

    def ask_depth(self) -> Optional[int]:
        """
        Prompt the user to set a new recursion depth for heatmap calculations.

        This method displays a dialog box warning the user about the exponential
        complexity of increasing depth values. It recommends using odd values for
        unbiased heatmaps, as even depths favor the current player.

        Returns
        -------
        Union[int, None]
            The user-provided depth value if valid, otherwise None.
        """
        depth_warning: str = "Every increment to this value increases calculation times exponentially!\n\n"
        depth_warning += "Note: Odd values are recommended for least biased heatmaps\nas each depth starting from 0 "
        depth_warning += "only counts one half turn of possible moves."
        return simpledialog.askinteger(
            "Set Depth",
            f"\nWARNING: {depth_warning}\n\nEnter new recursion depth:",
            initialvalue=self.depth,
            minvalue=0,
            maxvalue=100
        )  # Adjust maxvalue as needed.

    def change_font(self, new_font: str) -> None:
        """Handle font option updates.

        This method updates the font used for displaying chess pieces on the board.
        After updating, the board is re-rendered with the new font.
        """
        if not self.updating:
            self.updating = True
            self.font = new_font
            self.update_board()
            self.updating = False
        else:
            self.after(100, lambda: self.change_font(new_font))

    def choose_square_color(self, title: str, index: int) -> None:
        """Allow the user to change a specific square color.

        This method opens a tkinter color chooser dialog, allowing the user to select a new color
        for either the light or dark squares on the chessboard.

        Parameters
        ----------
        title : str
            The title of the color picker dialog (e.g., "Pick Light Square Color").
        index : int
            The index (0 for light squares, 1 for dark squares) of the square color to be changed.
        """
        if not self.updating:
            self.updating = True
            color: Optional[str] = colorchooser.askcolor(title=title)[1]
            if isinstance(color, str):
                self.colors[index] = color
                self.update_board()
            self.updating = False
        else:
            self.after(100, lambda: self.choose_square_color(title, index))

    def change_board_colors(self) -> None:
        """Invoke both light and dark square color option prompts.

        This method opens color pickers for both the light and dark squares, allowing the user
        to select custom colors for the chessboard squares.
        """
        self.choose_square_color(title=LIGHT_SQUARE_COLOR_PROMPT, index=0)
        self.choose_square_color(title=DARK_SQUARE_COLOR_PROMPT, index=1)

    @abstractmethod
    def update_board(self):
        """Update the chessboard display based on the current position.

        This method redraws the entire chessboard based on the current state of the `Board` object,
        including updating the colors of the squares and displaying the chess pieces. It uses the
        precomputed heatmap colors if available, or calculates them if necessary.
        """
        pass

    @abstractmethod
    def clear_board(self):
        """Clears the canvas of any drawn board objects"""
        pass

    @staticmethod
    def get_xys(col: int, flipped_row: int, square_size: int) -> Tuple[int, int, int, int]:
        r"""Get x_0, x_1, y_0, y_1 of a square.

        Parameters
        ----------
        col : int
        flipped_row : int
        square_size : int

        Returns
        -------
        Tuple[int, int, int, int]

        """
        x_0: int = col * square_size
        y_0: int = flipped_row * square_size
        x_1: int = x_0 + square_size
        y_1: int = y_0 + square_size
        return x_0, x_1, y_0, y_1

    def after(self, ms: int, function: Optional[Callable] = None, *args) -> None:
        """Placeholder for `Tk.after(self, ms, function=None, *args)` method.

        Notes
        -----
        Tk should be listed before BaseChessTkApp to default to the `Tk.after()` method:
            ```
            class ChessHeatMapApp(Tk, BaseChessTkApp):
                ...
            ```

        """
        # noinspection PyTypeChecker,PydanticTypeChecker
        Tk.after(self, ms, function, *args)

    def add_options(self, menu_bar: Menu) -> None:
        """Adds options menu to menu bar.

        Parameters
        ----------
        menu_bar : tkinter.Menu
        """
        fonts_menu: Menu = Menu(menu_bar, tearoff=0)
        font: str
        for font in tk_font.families():
            fonts_menu.add_command(label=font.title(), command=lambda f=font: self.change_font(new_font=f))
        options_menu: Menu = Menu(menu_bar, tearoff=0)
        options_menu.add_cascade(label="Font", menu=fonts_menu)
        options_menu.add_command(label="Change Board Colors", command=self.change_board_colors)
        options_menu.add_command(
            label=LIGHT_SQUARE_COLOR_PROMPT,
            command=lambda: self.choose_square_color(title=LIGHT_SQUARE_COLOR_PROMPT, index=0))
        options_menu.add_command(
            label=DARK_SQUARE_COLOR_PROMPT,
            command=lambda: self.choose_square_color(title=DARK_SQUARE_COLOR_PROMPT, index=1))
        options_menu.add_command(label="Set Depth", command=self.set_depth)
        menu_bar.add_cascade(label="Options", menu=options_menu)
