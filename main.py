"""Chess Heat Map App main module.

Defines the ChessHeatMapApp GUI class and related helpers for rendering
chess move heatmaps in a Tkinter application.
"""
from concurrent.futures import Future
# noinspection PyProtectedMember
from multiprocessing.context import Process
from os import cpu_count, kill
from signal import SIGTERM
from tkinter import Canvas, Menu, Tk, filedialog, messagebox
from typing import Dict, List, Optional, TextIO, Tuple

from chess import Board, Move, Piece, SQUARES, pgn, square_name
from chess.pgn import Game, Headers
from numpy import float64
from numpy.typing import NDArray

from chmutils import (
    PPExecutor,
    get_or_compute_heatmap_with_better_discounts,
    GBuilder,
    BaseChessTkApp,
    DEFAULT_COLORS,
    DEFAULT_FONT
)
from heatmaps import ChessMoveHeatmap
from tooltips import CanvasTooltip

__all__ = [
    'ChessHeatMapApp'
]


# pylint: disable=too-many-instance-attributes,too-many-public-methods


class ChessHeatMapApp(Tk, BaseChessTkApp):
    """Main application window for the Chess Heat Map app.

    This class is responsible for rendering the chessboard, handling
    user input, and managing heatmap calculations for each move in
    the loaded PGN game. It extends `tk.Tk` to provide a GUI with
    interactive features.
    """
    tooltips: List[Optional[CanvasTooltip]]
    pieces_maps: Dict[Optional[int], Optional[NDArray[Dict[Piece, float64]]]]
    heatmap_futures: Dict[Optional[int], Optional[Future]]  # Refactored to dict so -1 can be a key
    heatmaps: Dict[Optional[int], Optional[NDArray[str]]]
    executor: Optional[PPExecutor]
    moves: Optional[List[Optional[Move]]]
    game: Optional[Game]
    board: Board

    def __init__(self) -> None:
        """Initialize the ChessHeatMapApp application.

        Sets up window size, board dimensions, colors, fonts, and
        heatmap futures. Prompts the user to load a PGN file and
        begins background heatmap computation.
        """
        self.updating = True
        super().__init__()
        self.depth = 3
        self.title(f"Chess Move Heatmap | Depth = {self.depth}")
        self.board = Board()
        screen_height: int = self.winfo_screenheight()
        screen_width: int = int(screen_height * 0.75)
        self.geometry(f"{screen_width}x{screen_height}")
        self.square_size = screen_width // 8
        self.colors = list(DEFAULT_COLORS)
        self.font = DEFAULT_FONT
        self.create_menu()
        self.canvas = Canvas(self, )
        self.canvas.pack(fill="both", expand=True)
        self.game = None
        self.moves = None
        self.current_move_index = -1
        self.highlight_squares = set()  # Store the squares to highlight
        # Parallel processing setup
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.executor = PPExecutor(max_workers=max(1, int(cpu_count() * 0.9)))
        self.heatmap_futures = {}  # Track running futures
        self.heatmaps = {}  # Store completed heatmaps
        self.pieces_maps = {}
        self.tooltips = []
        self.open_pgn()
        self.set_bindings()
        self.focus_force()
        self.updating = False

    def set_bindings(self) -> None:
        """Sets app bindings"""
        self.bind("<Configure>", self.on_resize)
        self.bind("<Left>", lambda event: self.prev_move())
        self.bind("<a>", lambda event: self.prev_move())
        self.bind("<Right>", lambda event: self.next_move())
        self.bind("<d>", lambda event: self.next_move())

    def on_closing(self) -> None:
        """Clean up resources before closing the application."""
        self.updating = True
        if self.executor is not None:
            process: Process
            for process in self.executor.processes:
                kill(process.pid, SIGTERM)
            self.executor.shutdown()
        self.destroy()

    def create_menu(self) -> None:
        """Constructs the app menu during initialization.

        This method creates a menu bar with options to load a PGN file, change the board colors,
        and modify the font. It also allows navigation through the moves in the game (next/previous move).
        """
        menu_bar: Menu = Menu(self)
        file_menu: Menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open PGN", command=self.open_pgn)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.add_options(menu_bar)
        menu_bar.add_command(label="Prev Move", command=self.prev_move)
        menu_bar.add_command(label="Next Move", command=self.next_move)
        self.config(menu=menu_bar)

    def set_depth(self) -> None:
        """Prompt the user to set a new recursion depth for heatmap calculations.

        The user is asked to input an integer value. If a valid value is provided,
        the depth is updated and the window title is refreshed to reflect the change.
        """
        if not self.updating:
            self.updating = True
            new_depth: Optional[int] = self.ask_depth()
            if new_depth is not None and new_depth != self.depth:
                self.depth = new_depth
                # Update the window title to reflect the new depth.
                self.set_title()
                self.ensure_executor()
                self.clear_heatmaps()
                if self.game is not None:
                    board: Board = self.game.board()
                    moves: List[Optional[Move]] = [None] + self.moves
                else:
                    board = Board()
                    moves = [None]
                i: int
                move: Optional[Move]
                for i, move in enumerate(moves):  # Include initial board state
                    new_board = self.new_board_pushed_upto(board, i, move)
                    future = self.executor.submit(
                        get_or_compute_heatmap_with_better_discounts,
                        new_board,
                        depth=self.depth)
                    self.heatmap_futures[i - 1] = future
                self.after(100, self.check_heatmap_futures)
                self.update_board()
            self.updating = False
        else:
            self.after(100, self.set_depth)

    def new_board_pushed_upto(self, board: Board, i: int, move: Move) -> Board:
        """Gets copy of board pushed upto move

        Parameters
        ----------
        board : chess.Board
        i : int
        move : chess.Move

        Returns
        -------
        chess.Board

        """
        new_board: Board = board.copy()
        if move:
            j: int
            for j in range(i):
                new_board.push(self.moves[j])
        return new_board

    def set_title(self) -> None:
        """Sets the App window title"""
        target: int = len(self.heatmap_futures.items())
        completed: int = len(self.heatmaps.items())
        status: str = f" | Calculating Heatmaps: {completed}/{target} loaded" if not completed == target else ""
        self.title(f"Chess Move Heatmap | Depth = {self.depth}{status}{self.format_game_headers}")

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
            with open(file_path, mode="r", encoding="utf-8") as file:
                game: Optional[Game] = pgn.read_game(file, Visitor=GBuilder)
                board: Board = game.board()
                moves: List[Optional[Move]] = list(game.mainline_moves())
                self.clear_heatmaps()
                self.game = game
                self.moves = moves
                self.board = board
                self.current_move_index = -1
                self.set_title()
                self.submit_heatmap_futures(moves)
        # pylint: disable=broad-exception-caught
        except Exception as exception:
            self.handle_bad_pgn_selection(exception, file_path)
        self.ensure_board_has_heatmap()
        self.update_board()

    def submit_heatmap_futures(self, moves: List[Optional[Move]]) -> None:
        """Submits heatmap futures for all positions in the game.

        Parameters
        ----------
        moves : List[Optional[Move]]
        """
        # Start background heatmap calculations
        move: Optional[Move]
        i: int
        for i, move in enumerate([None] + moves):  # Include initial board state
            new_board = self.new_board_pushed_upto(self.board, i, move)
            future: Future = self.executor.submit(
                get_or_compute_heatmap_with_better_discounts, new_board, depth=self.depth
            )
            self.heatmap_futures[i - 1] = future
        self.after(100, self.check_heatmap_futures)

    def ensure_board_has_heatmap(self) -> None:
        """Ensures a default heatmap is calculated/retrieved if none exists in app."""
        if not self.heatmap_futures and not self.heatmaps:
            new_board: Board = self.board.copy()
            future = self.executor.submit(
                get_or_compute_heatmap_with_better_discounts, new_board, depth=self.depth
            )
            self.heatmap_futures[self.current_move_index] = future
            self.after(100, self.check_heatmap_futures)

    @staticmethod
    def handle_bad_pgn_selection(exception: Exception, file_path: str) -> None:
        """Presents the user with an error when an Exception is encountered in open_pgn.

        Parameters
        ----------
        exception : Exception
        file_path : str
        """
        if isinstance(exception, AttributeError):
            exception = "The file does not contain a valid PGN game."
        elif not file_path:
            exception = "No PGN file was selected."
        messagebox.showerror("Error", f"Failed to load PGN file: {exception}")

    def ensure_executor(self) -> None:
        """Ensure that the process pool executor is initialized.

        This method is called before submitting new heatmap calculations to ensure
        that parallel processing resources are available.
        """
        if self.executor is None:
            self.executor = PPExecutor(max_workers=max(1, int(cpu_count() * 0.9)))

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
        self.set_title()
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
        if not self.updating:
            self.updating = True
            if self.game and (self.current_move_index < len(self.moves) - 1):
                self.current_move_index += 1
                move: Move = self.moves[self.current_move_index]
                self.highlight_squares = {move.from_square, move.to_square}
                self.board.push(move)
                self.update_board()
            self.updating = False
        else:
            self.after(100, self.next_move)

    def prev_move(self) -> None:
        """Display the previous move in the game.

        This method updates the board to reflect the previous move in the loaded game. It also highlights
        the squares involved in the move and updates the chessboard display accordingly.
        """
        if not self.updating:
            self.updating = True
            if self.game and (self.current_move_index >= 0):
                self.current_move_index -= 1
                if self.current_move_index >= 0:
                    move: Move = self.moves[self.current_move_index]
                    self.highlight_squares = {move.from_square, move.to_square}
                else:
                    self.highlight_squares = set()
                self.board.pop()
                self.update_board()
            self.updating = False
        else:
            self.after(100, self.prev_move)

    def update_board(self) -> None:
        """Update the chessboard display based on the current position.

        This method redraws the entire chessboard based on the current state of the `Board` object,
        including updating the colors of the squares and displaying the chess pieces. It uses the
        precomputed heatmap colors if available, or calculates them if necessary.
        """
        self.clear_board()
        colors: List[str]
        font_size: int
        heatmap: ChessMoveHeatmap
        heatmap_colors: NDArray[str]
        pieces_map: NDArray[Dict[Piece, float64]]
        square_size: int
        colors, font_size, heatmap, heatmap_colors, pieces_map, square_size = self.get_board_drawing_properties()
        self.draw_board(heatmap, heatmap_colors, pieces_map, colors, square_size, font_size)

    def clear_board(self) -> None:
        """Clears the canvas of any drawn board objects"""
        self.canvas.delete("all")
        self.clear_tooltips()

    # pylint: disable=too-many-arguments, too-many-locals
    def draw_board(
            self, heatmap: ChessMoveHeatmap, heatmap_colors: NDArray[str],
            pieces_map: NDArray[Dict[Piece, float64]],
            colors: List[str], square_size: int, font_size: int
    ) -> None:
        """Draws the board to the canvas.

        Parameters
        ----------
        heatmap : ChessMoveHeatmap
        heatmap_colors : NDArray[str]
        pieces_map : NDArray[Dict[Piece, float64]]
        colors : List[str]
        square_size : int
        font_size : int
        """
        square: int
        for square in SQUARES:
            black_hint_text: str
            color: str
            make_tip: bool
            offset: int
            outline_color: str
            tip: str
            white_hint_text: str
            width: int
            x_0: int
            x_1: int
            y_0: int
            y_1: int
            (
                black_hint_text, color, make_tip, offset, outline_color, tip, white_hint_text, width, x_0, x_1, y_0, y_1
            ) = self.get_all_draw_properties_for(square, square_size, colors, heatmap_colors, pieces_map)
            self.draw_square_to_canvas(
                heatmap, square, x_0, x_1, y_0, y_1, square_size, width, color, outline_color,
                offset, font_size, make_tip, tip, white_hint_text, black_hint_text
            )

    def get_board_drawing_properties(self) -> Tuple[
        List[str], int, ChessMoveHeatmap, NDArray[str], NDArray[Dict[Piece, float64]], int
    ]:
        """Gets properties needed for drawing the board to the canvas.

        Returns
        -------
        Tuple[List[str], int, ChessMoveHeatmap, NDArray[str], NDArray[Dict[Piece, float64]], int]

        """
        square_size: int = self.square_size
        colors: List[str] = self.colors
        font_size: int = int(square_size * 0.4)
        map_index: int = self.current_move_index
        future: Future = self.heatmap_futures[map_index]
        heatmap: ChessMoveHeatmap
        heatmap_colors: NDArray[str]
        pieces_map: NDArray[Dict[Piece, float64]]
        heatmap, heatmap_colors, pieces_map = self.get_active_maps(future, map_index)
        return colors, font_size, heatmap, heatmap_colors, pieces_map, square_size

    def get_all_draw_properties_for(
            self, square: int, square_size: int, colors: List[str],
            heatmap_colors: NDArray[str], pieces_map: NDArray[Dict[Piece, float64]]
    ) -> Tuple[str, str, bool, int, str, str, str, int, int, int, int, int]:
        """Gets properties needed for drawing a board square to the canvas.

        Parameters
        ----------
        square : int
        square_size : int
        colors : List[str]
        heatmap_colors : numpy.NDArray[str]
        pieces_map : NDArray[Dict[Piece, float64]]

        Returns
        -------
        Tuple[str, str, bool, int, str, str, str, int, int, int, int, int]

        """
        square_pieces: Dict[Piece, float64] = pieces_map[square]
        black_hint_text: str
        white_hint_text: str
        tip: str
        black_hint_text, tip, white_hint_text = self.get_tooltip_texts(square, square_pieces)
        row: int
        col: int
        row, col = divmod(square, 8)
        flipped_row: int = 7 - row
        y_1: int
        y_0: int
        x_1: int
        x_0: int
        x_0, x_1, y_0, y_1 = self.get_xys(col, flipped_row, square_size)
        color: str
        make_tip: bool
        offset: int
        outline_color: str
        width: int
        color, make_tip, offset, outline_color, width = self.get_square_properties(
            black_hint_text, col, colors,
            flipped_row, heatmap_colors,
            square, white_hint_text
        )
        return black_hint_text, color, make_tip, offset, outline_color, tip, white_hint_text, width, x_0, x_1, y_0, y_1

    def draw_square_to_canvas(
            self, heatmap: ChessMoveHeatmap, square: int, x_0: int, x_1: int, y_0: int, y_1: int,
            square_size: int, width: int, color: str, outline_color: str,
            offset: int, font_size: int, make_tip: bool, tip: str, white_hint_text: str,
            black_hint_text: str
    ) -> None:
        """Draws a complete Square to the canvas.

        Parameters
        ----------
        heatmap : ChessMoveHeatmap
        square : int
        x_0 : int
        x_1 : int
        y_0 : int
        y_1 : int
        square_size : int
        width : int
        color : str
        outline_color : str
        offset : int
        font_size : int
        make_tip : bool
        tip : str
        white_hint_text : str
        black_hint_text : str
        """
        sq_id: int = self.canvas.create_rectangle(
            x_0 + offset, y_0 + offset, x_1 - offset, y_1 - offset,
            fill=color, outline=outline_color, width=width
        )
        if make_tip:
            self.tooltips.append(CanvasTooltip(self, self.canvas, sq_id, text=tip, bg_color=color))
        piece: Optional[Piece] = self.board.piece_at(square)
        self.create_piece(font_size, piece, square_size, x_0, y_0, tip, make_tip, color)
        self.create_count_labels(
            font_size, heatmap, offset, square, square_size, x_0, y_0, y_1,
            black_hint_text, white_hint_text, color
        )

    def get_active_maps(
            self, future: Future, map_index: int
    ) -> Tuple[ChessMoveHeatmap, NDArray[str], NDArray[Dict[Piece, float64]]]:
        """Gets maps for rendering active positon.

        Parameters
        ----------
        future : Future
        map_index : int

        Returns
        -------
        Tuple[ChessMoveHeatmap, numpy.NDArray[str], numpy.NDArray[Dict[Piece, numpy.float64]]]

        """
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
        return heatmap, heatmap_colors, pieces_map

    def clear_tooltips(self) -> None:
        """Clears any existing tooltips."""
        if len(self.tooltips) > 0:
            for t_tip in self.tooltips:
                t_tip.onLeave()
            self.tooltips.clear()

    def get_tooltip_texts(self, square: int, square_pieces: Dict[Piece, float64]) -> Tuple[str, str, str]:
        """Get texts for tooltips

        Parameters
        ----------
        square : int
        square_pieces : Dict[Piece, float64]

        Returns
        -------
        Tuple[str, str, str]

        """
        black_hint_text: str
        white_hint_text: str
        black_hint_text, white_hint_text = self.get_black_white_hints(square_pieces)
        wb_text: str = f"White: {white_hint_text}\nBlack: {black_hint_text}"
        turn_depth: str = f"{((self.depth + 1) / 2):.1f}"
        tip: str = f"Possible moves to {square_name(square)} within {turn_depth} turns:\n{wb_text}"
        return black_hint_text, tip, white_hint_text

    def get_square_properties(
            self,
            black_hint_text: str, col: int, colors: List[str],
            flipped_row: int, heatmap_colors: NDArray[str],
            square: int, white_hint_text: str) -> Tuple[str, bool, int, str, int]:
        """Gets border and fill properties of a square.

        Parameters
        ----------
        black_hint_text : str
        col : int
        colors : List[str]
        flipped_row : int
        heatmap_colors : numpy.NDArray[str]
        square : int
        white_hint_text : str

        Returns
        -------
        Tuple[str, bool, int, str, int]

        """
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
        return color, make_tip, offset, outline_color, width

    @staticmethod
    def get_black_white_hints(square_pieces: Dict[Piece, float64]) -> Tuple[str, str]:
        """Gets hint text for black and white piece counts

        Parameters
        ----------
        square_pieces : Dict[Piece, float64]

        Returns
        -------
        Tuple[str, str]

        """
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
        return black_hint_text, white_hint_text

    def create_count_labels(
            self, font_size: int, heatmap: ChessMoveHeatmap,
            offset: int, square: int,
            square_size: int, x_0: int, y_0: int, y_1: int,
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
        x_0 : int
            The top-left x-coordinate of the square.
        y_0 : int
            The top-left y-coordinate of the square.
        y_1 : int
            The bottom-right y-coordinate of the square.
        """
        black_total: float64 = heatmap[square][1]
        bbg_id = self.canvas.create_rectangle(
            x_0 + offset + 2, y_0 + offset + 2, x_0 + (square_size / 9) * 2, y_0 + (square_size / 10) * 1.8,
            fill="black"
        )
        btx_id = self.canvas.create_text(
            x_0 + square_size / 9, y_0 + square_size / 10,
            text=f"{black_total:.1f}", font=(self.font, font_size // 5), fill="yellow"
        )

        white_total: float64 = heatmap[square][0]
        wbg_id = self.canvas.create_rectangle(
            x_0 + offset + 2, y_1 - (square_size / 10) * 1.8, x_0 + (square_size / 9) * 2, y_1 - offset - 2,
            fill="white",
        )
        wtx_id = self.canvas.create_text(
            x_0 + square_size / 9, y_1 - square_size / 10,
            text=f"{white_total:.1f}", font=(self.font, font_size // 5), fill="blue"
        )

        if white_total > 0:
            self.tooltips.append(CanvasTooltip(self, self.canvas, wbg_id, text=white_hint_text, bg_color=color))
            self.tooltips.append(CanvasTooltip(self, self.canvas, wtx_id, text=white_hint_text, bg_color=color))
        if black_total > 0:
            self.tooltips.append(CanvasTooltip(self, self.canvas, bbg_id, text=black_hint_text, bg_color=color))
            self.tooltips.append(CanvasTooltip(self, self.canvas, btx_id, text=black_hint_text, bg_color=color))

    def create_piece(
            self, font_size: int, piece: Optional[Piece],
            square_size: int, x_0: int, y_0: int,
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
        x_0 : int
            The top-left x-coordinate of the square.
        y_0 : int
            The top-left y-coordinate of the square.
        """
        if piece:
            piece_bg: str = "⬤"
            bg_id = self.canvas.create_text(
                x_0 + square_size / 2, y_0 + square_size / 2,
                text=piece_bg, font=(self.font, font_size + 25), fill="white" if piece.color else "black"
            )
            piece_symbol: str = piece.unicode_symbol()
            pc_id = self.canvas.create_text(
                x_0 + square_size / 2, y_0 + square_size / 2,
                text=piece_symbol, font=(self.font, font_size), fill="blue" if piece.color else "yellow"
            )
            if make_tip:
                self.tooltips.append(CanvasTooltip(self, self.canvas, bg_id, text=tip, bg_color=color))
                self.tooltips.append(CanvasTooltip(self, self.canvas, pc_id, text=tip, bg_color=color))


if __name__ == "__main__":
    app = ChessHeatMapApp()
    app.mainloop()
