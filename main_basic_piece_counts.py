"""Chess Heat Map App"""
import chess
import tkinter as tk
from tkinter import Canvas, Menu, filedialog, colorchooser, messagebox, font as tk_font, Event, simpledialog
from tooltips import CanvasTooltip
from numpy import float64
from numpy.typing import NDArray
from chess import Board, Piece, Move
from chess.pgn import GameBuilder, Game, Headers
from typing import Dict, List, Optional, Set, TextIO, Tuple
from concurrent.futures import ProcessPoolExecutor, Future
from multiprocessing.context import SpawnProcess
import os
import signal
from chmutils import calculate_chess_move_heatmap
from heatmaps import ChessMoveHeatmap

DARK_SQUARE_COLOR_PROMPT: str = "Pick Dark Square Color"
LIGHT_SQUARE_COLOR_PROMPT: str = "Pick Light Square Color"
DEFAULT_COLORS: Tuple[str, str] = ("#ffffff", "#c0c0c0")
DEFAULT_FONT: str = "Arial"


class Builder(GameBuilder):
    """Overrides GameBuilder.handle_error to raise exception."""

    def handle_error(self, error: Exception) -> None:
        """Override of GameBuilder.handle_error method to raise errors.

        """
        raise error


class ChessHeatMapApp(tk.Tk):
    """Main application window for the Chess Heat Map app.

    This class is responsible for rendering the chessboard, handling user inputs,
    and managing heatmap calculations for each move in the loaded PGN game.
    It extends the `tk.Tk` class to create a GUI with various interactive features.

    Attributes
    ----------
    depth : int
        The depth of recursion for calculating the heatmap.
    heatmap_futures : Dict[Optional[int], Optional[Future]]
        A list of `Future` objects tracking the status of background heatmap calculations.
    heatmaps : Dict[Optional[int], Optional[NDArray[str]]]
        A dictionary mapping move indices to corresponding heatmap color arrays.
    executor : Optional[ProcessPoolExecutor]
        A process pool executor used to run heatmap calculations in parallel.
    highlight_squares : Set[Optional[int]]
        A set of square indices to highlight (e.g., the squares involved in the current move).
    current_move_index : int
        The index of the current move in the loaded PGN game.
    moves : Optional[List[Optional[Move]]]
        A list of moves from the loaded game.
    game : Optional[Game]
        The current chess game being played, parsed from the PGN file.
    canvas : Canvas
        The tkinter canvas widget used to render the chessboard and heatmap.
    font : str
        The font used for displaying chess pieces on the board.
    colors : List[str]
        A list of colors for the light and dark squares on the board.
    square_size : int
        The size of each square on the chessboard.
    board : Board
        The current state of the chessboard.
    updating : bool
        A flag indicating whether the board is being updated (e.g., during a move change).

    Methods
    -------
    __init__ : Initializes the main application window and prompts the user to load a PGN file.
    on_resize : Handles window resizing events and updates the board display accordingly.
    create_menu : Constructs the application menu with options for loading a PGN, changing board colors, etc.
    change_font : Changes the font used for displaying chess pieces.
    change_board_colors : Prompts the user to change the light and dark square colors.
    choose_square_color : Opens a color picker dialog to allow the user to select a square color.
    open_pgn : Prompts the user to load and parse a PGN file and starts heatmap calculations for the game.
    check_heatmap_futures : Periodically checks for completed heatmap calculations and updates the board.
    next_move : Updates the board to show the next move in the loaded game.
    prev_move : Updates the board to show the previous move in the loaded game.
    update_board : Updates the chessboard display based on the current board state and heatmap.
    """
    tooltips: List[Optional[CanvasTooltip]]
    pieces_maps: Dict[Optional[int], Optional[NDArray[Dict[Piece, float64]]]]
    depth: int
    heatmap_futures: Dict[Optional[int], Optional[Future]]  # Refactored to dict so -1 can be a key
    heatmaps: Dict[Optional[int], Optional[NDArray[str]]]
    executor: Optional[ProcessPoolExecutor]
    highlight_squares: Set[Optional[int]]
    current_move_index: int
    moves: Optional[List[Optional[Move]]]
    game: Optional[Game]
    canvas: Canvas
    font: str
    colors: List[str]
    square_size: int
    board: Board
    updating: bool

    def __init__(self) -> None:
        """Initialize the ChessHeatMapApp application.

        This method sets up the window size, board dimensions, and various attributes like colors,
        font, and heatmap futures. It also prompts the user to load a PGN file and begins the process
        of calculating heatmaps for the game.
        """
        self.updating = True
        super().__init__()
        self.depth = 3
        self.title(f"Chess Move Heatmap | Depth = {self.depth}")
        self.board = chess.Board()
        screen_height: int = self.winfo_screenheight()
        screen_width: int = int(screen_height * 0.75)
        self.geometry(f"{screen_width}x{screen_height}")
        self.square_size = screen_width // 8
        self.colors = list(DEFAULT_COLORS)
        self.font = DEFAULT_FONT
        self.create_menu()
        self.canvas = tk.Canvas(self, )
        self.canvas.pack(fill="both", expand=True)
        self.game = None
        self.moves = None
        self.current_move_index = -1
        self.highlight_squares = set()  # Store the squares to highlight
        # Parallel processing setup

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.executor = ProcessPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.9)))
        self.heatmap_futures = {}  # Track running futures
        self.heatmaps = {}  # Store completed heatmaps
        self.pieces_maps = {}
        self.tooltips = []
        self.open_pgn()

        self.bind("<Configure>", self.on_resize)
        self.bind("<Left>", lambda event: self.prev_move())
        self.bind("<Right>", lambda event: self.next_move())
        self.focus_force()
        self.updating = False

    def on_closing(self) -> None:
        """Clean up resources before closing the application."""
        self.updating = True
        if self.executor is not None:
            process: SpawnProcess
            for process in self.executor._processes.values():
                os.kill(process.pid, signal.SIGTERM)
            self.executor.shutdown()
        self.destroy()

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

    def create_menu(self) -> None:
        """Constructs the app menu during initialization.

        This method creates a menu bar with options to load a PGN file, change the board colors,
        and modify the font. It also allows navigation through the moves in the game (next/previous move).
        """
        menu_bar: Menu = tk.Menu(self)
        file_menu: Menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open PGN", command=self.open_pgn)
        menu_bar.add_cascade(label="File", menu=file_menu)
        fonts_menu: Menu = tk.Menu(menu_bar, tearoff=0)
        font: str
        for font in tk_font.families():
            fonts_menu.add_command(label=font.title(), command=lambda f=font: self.change_font(new_font=f))
        options_menu: Menu = tk.Menu(menu_bar, tearoff=0)
        options_menu.add_cascade(label="Font", menu=fonts_menu)
        options_menu.add_command(label="Change Board Colors", command=self.change_board_colors)
        options_menu.add_command(label=LIGHT_SQUARE_COLOR_PROMPT,
                                 command=lambda: self.choose_square_color(title=LIGHT_SQUARE_COLOR_PROMPT, index=0))
        options_menu.add_command(label=DARK_SQUARE_COLOR_PROMPT,
                                 command=lambda: self.choose_square_color(title=DARK_SQUARE_COLOR_PROMPT, index=1))
        options_menu.add_command(label="Set Depth", command=self.set_depth)
        menu_bar.add_cascade(label="Options", menu=options_menu)
        menu_bar.add_command(label="Prev Move", command=self.prev_move)
        menu_bar.add_command(label="Next Move", command=self.next_move)
        self.config(menu=menu_bar)

    def set_depth(self) -> None:
        """Prompt the user to set a new recursion depth for heatmap calculations.

        The user is asked to input an integer value. If a valid value is provided,
        the depth is updated and the window title is refreshed to reflect the change.
        """
        new_depth: Optional[int] = self.ask_depth()
        if new_depth is not None and new_depth != self.depth:
            self.depth = new_depth
            # Update the window title to reflect the new depth.
            self.title(f"Chess Move Heatmap | Depth = {self.depth} | {self.format_game_headers}")
            self.ensure_executor()
            self.clear_heatmaps()
            if self.game is not None:
                board: Board = self.game.board()
                moves: List = [None] + self.moves
            else:
                board = Board()
                moves = [None]
            for i, move in enumerate(moves):  # Include initial board state
                new_board: Board = board.copy()
                if move:
                    j: int
                    for j in range(i):
                        new_board.push(self.moves[j])
                future = self.executor.submit(calculate_chess_move_heatmap, new_board, depth=self.depth)
                self.heatmap_futures[i - 1] = future
            self.after(100, self.check_heatmap_futures)
            self.update_board()

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
        return simpledialog.askinteger("Set Depth",
                                       f"\nWARNING: {depth_warning}\n\nEnter new recursion depth:",
                                       initialvalue=self.depth,
                                       minvalue=0,
                                       maxvalue=100)  # Adjust maxvalue as needed.

    def clear_heatmaps(self) -> None:
        """
        Cancel any running heatmap calculations and clear completed ones.

        This method ensures that all heatmap-related processes are halted, preventing
        conflicts when a new PGN file is loaded or when the user adjusts the depth setting.
        """
        future: Future
        for future in self.heatmap_futures.values():
            future.cancel()
        self.heatmap_futures.clear()
        self.heatmaps.clear()
        self.pieces_maps.clear()

    def change_font(self, new_font: str) -> None:
        """Handle font option updates.

        This method updates the font used for displaying chess pieces on the board.
        After updating, the board is re-rendered with the new font.
        """
        self.font = new_font
        print(f"Font updated to: {self.font}")
        self.update_board()

    def change_board_colors(self) -> None:
        """Invoke both light and dark square color option prompts.

        This method opens color pickers for both the light and dark squares, allowing the user
        to select custom colors for the chessboard squares.
        """
        self.choose_square_color(title=LIGHT_SQUARE_COLOR_PROMPT, index=0)
        self.choose_square_color(title=DARK_SQUARE_COLOR_PROMPT, index=1)

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
        color: Optional[str] = colorchooser.askcolor(title=title)[1]
        print(color)
        if isinstance(color, str):
            self.colors[index] = color
            self.update_board()

    def open_pgn(self) -> None:
        """Prompts the user to open and parse a PGN file.

        This method opens a file dialog to allow the user to select a PGN file, parses the game,
        and starts the background process for heatmap calculation. It also clears previous heatmaps
        and resets the game state.
        """
        self.ensure_executor()
        file_path: str = filedialog.askopenfilename(filetypes=[("PGN Files", "*.pgn")])
        try:
            file: TextIO
            with open(file_path, "r") as file:
                game: Optional[Game] = chess.pgn.read_game(file, Visitor=Builder)
                board: Board = game.board()
                moves: Optional[List[Optional[Move]]] = list(game.mainline_moves())
                self.clear_heatmaps()
                self.game = game
                self.moves = moves
                self.board = board
                self.current_move_index = -1
                self.title(f"Chess Move Heatmap | Depth = {self.depth}{self.format_game_headers}")
                # Start background heatmap calculations
                move: Optional[Move]
                i: int
                for i, move in enumerate([None] + moves):  # Include initial board state
                    new_board: Board = self.board.copy()
                    if move:
                        j: int
                        for j in range(i):
                            new_board.push(self.moves[j])
                    future = self.executor.submit(calculate_chess_move_heatmap, new_board, depth=self.depth)
                    self.heatmap_futures[i - 1] = future
                self.after(100, self.check_heatmap_futures)
        except Exception as e:
            if isinstance(e, AttributeError):
                e = "The file does not contain a valid PGN game."
            elif not file_path:
                e = "No PGN file was selected."
            messagebox.showerror("Error", f"Failed to load PGN file: {e}")
        if not self.heatmap_futures and not self.heatmaps:
            new_board: Board = self.board.copy()
            future = self.executor.submit(calculate_chess_move_heatmap, new_board, depth=self.depth)
            self.heatmap_futures[self.current_move_index] = future
            self.after(100, self.check_heatmap_futures)
        self.update_board()

    def ensure_executor(self) -> None:
        """Ensure that the process pool executor is initialized.

        This method is called before submitting new heatmap calculations to ensure
        that parallel processing resources are available.
        """
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.9)))

    @property
    def format_game_headers(self) -> str:
        """Generate a formatted string containing game details from PGN headers.

        Returns
        -------
        str
            A formatted string displaying White/Black player names, result, date, and site.
        """
        if self.game is None:
            return ""
        headers: Headers = self.game.headers
        f_head: str = f" | White (Blue): {headers.get('White', '?')} | Black (Yellow): {headers.get('Black', '?')}"
        f_head += f" | Result: {headers.get('Result', '?')} | Date: {headers.get('Date', '?')}"
        return f"{f_head} | Site: {headers.get('Site', '?')}"

    def check_heatmap_futures(self) -> None:
        """Periodically check for completed heatmap calculations.

        This method periodically checks if any heatmap calculation futures have completed and updates
        the board display with the new heatmap if available. If no updates are available, it continues
        checking periodically.
        """
        updated = False

        for i, future in self.heatmap_futures.items():  # Start at -1 to align with self.current_move_index
            if i not in self.heatmaps and future.done():
                heatmap: ChessMoveHeatmap = future.result()
                self.heatmaps[i] = heatmap.colors  # Store only colors
                self.pieces_maps[i] = heatmap.piece_counts
                if self.current_move_index == i:
                    updated = True

        if updated and not self.updating:
            self.updating = True
            self.update_board()
            self.updating = False
        if len(self.heatmaps) != len(self.heatmap_futures):
            self.after(100, self.check_heatmap_futures)
        elif self.executor is not None:
            self.executor.shutdown()
            self.executor = None

    def next_move(self) -> None:
        """Display the next move in the game.

        This method updates the board to reflect the next move in the loaded game. It also highlights
        the squares involved in the move and updates the chessboard display accordingly.
        """
        if self.game and (self.current_move_index < len(self.moves) - 1) and not self.updating:
            self.updating = True
            self.current_move_index += 1
            move: Move = self.moves[self.current_move_index]
            self.highlight_squares = {move.from_square, move.to_square}
            self.board.push(move)
            self.update_board()
            self.updating = False

    def prev_move(self) -> None:
        """Display the previous move in the game.

        This method updates the board to reflect the previous move in the loaded game. It also highlights
        the squares involved in the move and updates the chessboard display accordingly.
        """
        if self.game and (self.current_move_index >= 0) and not self.updating:
            self.updating = True
            self.current_move_index -= 1
            if self.current_move_index >= 0:
                move: Move = self.moves[self.current_move_index]
                self.highlight_squares = {move.from_square, move.to_square}
            else:
                self.highlight_squares = set()
            self.board.pop()
            self.update_board()
            self.updating = False

    def update_board(self) -> None:
        """Update the chessboard display based on the current position.

        This method redraws the entire chessboard based on the current state of the `Board` object,
        including updating the colors of the squares and displaying the chess pieces. It uses the
        precomputed heatmap colors if available, or calculates them if necessary.
        """
        self.canvas.delete("all")
        if len(self.tooltips) >> 0:
            for t_tip in self.tooltips:
                t_tip.hide()
                del t_tip
            self.tooltips.clear()
        square_size: int = self.square_size
        colors: List[str] = self.colors
        font_size: int = int(square_size * 0.4)
        map_index: int = self.current_move_index
        future: Future = self.heatmap_futures[map_index]

        if map_index in self.heatmaps:
            heatmap: ChessMoveHeatmap = future.result()
            heatmap_colors: NDArray[str] = self.heatmaps[map_index]  # Use precomputed color list
            pieces_map: NDArray[Dict[Piece, float64]] = self.pieces_maps[map_index]
        else:
            if future.done():
                heatmap = future.result()
                heatmap_colors = heatmap.colors  # Extract colors immediately
                pieces_map = heatmap.piece_counts
                self.heatmaps[map_index] = heatmap_colors  # Store only colors
                self.pieces_maps[map_index] = pieces_map
            else:
                heatmap = ChessMoveHeatmap()
                heatmap_colors = heatmap.colors
                pieces_map = heatmap.piece_counts

        for square in chess.SQUARES:
            square_pieces: Dict[Piece, float64] = pieces_map[square]
            black_hint_text: str = ' '.join([
                f"{k.unicode_symbol()}:{square_pieces[k]:.2f}" for k in square_pieces if
                square_pieces[k] > 0 and not k.color
            ])
            if not black_hint_text.replace(" ", ""):
                black_hint_text = "None"
            white_hint_text: str = ' '.join([
                f"{k.unicode_symbol()}:{square_pieces[k]:.2f}" for k in square_pieces if
                square_pieces[k] > 0 and k.color
            ])
            if not white_hint_text.replace(" ", ""):
                white_hint_text = "None"
            wb_text = f"White: {white_hint_text}\nBlack: {black_hint_text}"
            turn_depth = f"{((self.depth + 1) / 2):.1f}"
            tip: str = f"Possible moves to {chess.square_name(square)} within {turn_depth} turns:\n{wb_text}"
            row: int
            col: int
            row, col = divmod(square, 8)
            flipped_row: int = 7 - row
            x0: int = col * square_size
            y0: int = flipped_row * square_size
            x1: int = x0 + square_size
            y1: int = y0 + square_size
            color: str = colors[(flipped_row + col) % 2]
            heatmap_color: str = heatmap_colors[square]
            width: int = 1
            if black_hint_text != "None" or white_hint_text != "None":
                color = heatmap_color
                make_tip = True
            else:
                make_tip = False
            outline_color: str = "black"

            if square in self.highlight_squares:
                if self.board.turn:
                    outline_color = "yellow"
                else:
                    outline_color = "blue"
                width = 3
            offset: int = width // 2
            sq_id = self.canvas.create_rectangle(
                x0 + offset, y0 + offset, x1 - offset, y1 - offset,
                fill=color, outline=outline_color, width=width
            )
            if make_tip:
                self.tooltips.append(CanvasTooltip(self, self.canvas, sq_id, text=tip, bg_color=color))
            piece: Optional[Piece] = self.board.piece_at(square)
            self.create_piece(font_size, piece, square_size, x0, y0, tip, make_tip, color)
            self.create_count_labels(
                font_size, heatmap, offset, square, square_size, x0, y0, y1,
                black_hint_text, white_hint_text, color
            )

    def create_count_labels(
            self, font_size: int, heatmap: ChessMoveHeatmap,
            offset: int, square: int,
            square_size: int, x0: int, y0: int, y1: int,
            black_hint_text: str, white_hint_text: str, color: str
    ) -> None:
        """Display move count intensity labels on the board.

        This method creates small text labels on each square representing
        the number of times that square was moved to in the analyzed game.

        Parameters
        ----------
        color
        white_hint_text
        black_hint_text
        font_size : int
            The font size for the count labels.
        heatmap : GradientHeatmap
            The heatmap object containing move count data.
        offset : int
            Offset for label positioning within the square.
        square : int
            The square index (0-63) corresponding to the chessboard layout.
        square_size : int
            The size of each square in pixels.
        x0 : int
            The top-left x-coordinate of the square.
        y0 : int
            The top-left y-coordinate of the square.
        y1 : int
            The bottom-right y-coordinate of the square.
        """
        black_total: float64 = heatmap[square][1]
        bbg_id = self.canvas.create_rectangle(
            x0 + offset + 2, y0 + offset + 2, x0 + (square_size / 9) * 2, y0 + (square_size / 10) * 1.8,
            fill="black"
        )
        btx_id = self.canvas.create_text(
            x0 + square_size / 9, y0 + square_size / 10,
            text=f"{black_total:.1f}", font=(self.font, font_size // 5), fill="yellow"
        )

        white_total: float64 = heatmap[square][0]
        wbg_id = self.canvas.create_rectangle(
            x0 + offset + 2, y1 - (square_size / 10) * 1.8, x0 + (square_size / 9) * 2, y1 - offset - 2,
            fill="white",
        )
        wtx_id = self.canvas.create_text(
            x0 + square_size / 9, y1 - square_size / 10,
            text=f"{white_total:.1f}", font=(self.font, font_size // 5), fill="blue"
        )

        if white_total > 0:
            CanvasTooltip(self, self.canvas, wbg_id, text=white_hint_text, bg_color=color)
            CanvasTooltip(self, self.canvas, wtx_id, text=white_hint_text, bg_color=color)
        if black_total > 0:
            CanvasTooltip(self, self.canvas, bbg_id, text=black_hint_text, bg_color=color)
            CanvasTooltip(self, self.canvas, btx_id, text=black_hint_text, bg_color=color)

    def create_piece(
            self, font_size: int, piece: Optional[Piece],
            square_size: int, x0: int, y0: int,
            tip: str, make_tip: bool, color: str
    ) -> None:
        """Render a chess piece on the board.

        This method displays a chess piece symbol at the correct square location.
        It also includes a circular background for better visibility.

        Parameters
        ----------
        color
        make_tip
        tip
        font_size : int
            The font size for the piece symbol.
        piece : Optional[Piece]
            The chess piece object to render, or None if the square is empty.
        square_size : int
            The size of each square in pixels.
        x0 : int
            The top-left x-coordinate of the square.
        y0 : int
            The top-left y-coordinate of the square.
        """
        if piece:
            piece_bg: str = "⬤"
            bg_id = self.canvas.create_text(
                x0 + square_size / 2, y0 + square_size / 2,
                text=piece_bg, font=(self.font, font_size + 25), fill="white" if piece.color else "black"
            )
            piece_symbol: str = piece.unicode_symbol()
            pc_id = self.canvas.create_text(
                x0 + square_size / 2, y0 + square_size / 2,
                text=piece_symbol, font=(self.font, font_size), fill="blue" if piece.color else "yellow"
            )
            if make_tip:
                self.tooltips.append(CanvasTooltip(self, self.canvas, bg_id, text=tip, bg_color=color))
                self.tooltips.append(CanvasTooltip(self, self.canvas, pc_id, text=tip, bg_color=color))


if __name__ == "__main__":
    app = ChessHeatMapApp()
    app.mainloop()
