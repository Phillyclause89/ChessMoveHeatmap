"""Play against the CMHMEngine in a GUI."""
from datetime import datetime
from os import makedirs, path
from tkinter import Canvas, Menu, Tk, messagebox
from typing import Callable, Dict, List, Optional, Union
from concurrent.futures import Future, ThreadPoolExecutor

from chess import Board, Move, Outcome, Piece, SQUARES
from chess.pgn import Game
from numpy import float64

from chmengine import CMHMEngine, CMHMEngine2, Pick, set_all_datetime_headers
from chmutils import BaseChessTkApp, DEFAULT_COLORS, DEFAULT_FONT, get_local_time


class PlayChessApp(Tk, BaseChessTkApp):
    """Play against the CMHMEngine in a GUI."""
    training: bool
    site: str
    # TODO: Refactor these into their own Player and Engines dataclasses
    player: Dict[str, Union[str, int]] = dict(
        name="Unknown",
        index=0,  # 0 is white in our mapping (inverse from python-chess lib)
        color='white'
    )
    # PlayChessApp.engines indexes align with python-chess
    engines: List[Dict[str, Union[str, int, Optional[CMHMEngine]]]] = [
        dict(
            engine=None,
            name='None',
            index=1,  # 1 is black in our mapping (inverse from python-chess lib)
            color='black'
        ),
        dict(
            engine=None,
            name='None',
            index=0,  # 0 is white in our mapping (inverse from python-chess lib)
            color='white'
        ),
    ]
    training_dir: str = "trainings"
    pgn_dir: str = "pgns"
    depth = 1

    def __init__(
            self,
            engine_type: Callable = CMHMEngine,
            depth: int = depth,
            player_name: str = player['name'],
            player_color_is_black: bool = False,
            site: Optional[str] = None,
            engine_type_2: Optional[Callable] = None,
            depth_2: Optional[int] = None,

    ) -> None:
        """Initialize the PlayChessApp GUI and configure engines.

        Parameters
        ----------
        engine_type : Callable, optional
            Class or factory for the primary engine (default: CMHMEngine).
        depth : int, optional
            Initial search depth for the primary engine (default: class‐level `depth`).
        player_name : str, optional
            Human player’s name (default: "Unknown").
        player_color_is_black : bool, optional
            If True, human plays black; otherwise white.
        site : str, optional
            Site string for PGN headers (default: "{player_name}’s place").
        engine_type_2 : Callable, optional
            Class or factory for the secondary engine. If None, the secondary engine
            is set to the *same* instance created by `engine_type`, enabling that
            single engine instance to play against itself in training mode.
        depth_2 : int, optional
            Search depth for the secondary engine (default: same as `depth` if not provided).
        """
        self.updating = True
        self.training = False
        super().__init__()
        self.player['name'] = player_name
        self.site = f"{self.player['name']}'s place" if site is None else site
        if player_color_is_black:
            self.player['index'], self.player['color'] = 1, 'black'
        self.engines[1]['engine'] = engine_type(depth=depth)
        self.engines[0]['engine'] = self.engines[1]['engine'] if engine_type_2 is None else engine_type_2(
            depth=depth if depth_2 is None else depth_2
        )
        # noinspection PyProtectedMember
        self.engines[0]['engine']._board = self.engines[1]['engine'].board
        (
            self.engines[1]['name'],
            self.engines[0]['name']
        ) = (
            self.engines[1]['engine'].__class__.__name__,
            self.engines[0]['engine'].__class__.__name__
        )
        self.depth = self.engines[1]['engine'].depth
        self.set_title()
        screen_height: int = self.winfo_screenheight()
        screen_width: int = int(screen_height * 0.75)
        self.geometry(f"{screen_width}x{screen_height}")
        self.square_size = screen_width // 8
        self.colors = list(DEFAULT_COLORS)
        self.font = DEFAULT_FONT
        self.create_menu()
        self.canvas = Canvas(self)
        self.canvas.pack(fill="both", expand=True)
        self.current_move_index = -1
        self.highlight_squares = set()
        # TODO: Prompt user to start a new (or load an incomplete) game.
        self.update_board()
        self._move_executor = ThreadPoolExecutor(max_workers=1)
        self.bind("<Configure>", self.on_resize)
        self.focus_force()
        self.updating = False

    def on_closing(self) -> None:
        """Clean up and confirm exit when the window is closed.

        If the app is idle, prompts the user to confirm quitting, then destroys the window.
        If an update is in progress, retries after a short delay.
        """
        if not self.updating:
            self.updating = True
            if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
                # TODO Add save game method call here.
                self.destroy()
            self.updating = False
        else:
            self.after(100, self.on_closing)

    def set_depth(self) -> None:
        """Prompt the user to change the engine search depth.

        If idle and not training, calls `ask_depth()`, updates both engines’ `.depth`,
        and updates the window title. Shows an error if a training session is active.
        """
        if not self.updating and not self.training:
            self.updating = True
            new_depth: Optional[int] = self.ask_depth()
            if new_depth is not None and new_depth != self.depth:
                for engine_dict in self.engines:
                    engine_dict['engine'].depth = new_depth
                self.depth = new_depth
                self.set_title()
            self.updating = False
        elif self.training:
            messagebox.showerror("Error", "Changing depth is not permitted while the engine is training.")
        else:
            self.after(100, self.set_depth)

    def create_menu(self) -> None:
        """Construct and attach the main menu bar.

        - Adds a “File” menu with “New Game” and “New Training Session”.
        - Invokes `add_options()` from BaseChessTkApp for additional entries.
        """
        menu_bar: Menu = Menu(self)
        file_menu: Menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Game", command=self.new_game)
        file_menu.add_command(label="New Training Session", command=self.train_engine)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.add_options(menu_bar)
        self.config(menu=menu_bar)

    def set_title(self) -> None:
        """Update the window’s title to reflect the current mode and search depth.

        Title format: "<mode> | Depth = <depth>", where <mode> comes from `get_mode()`.
        """
        self.title(f"{self.get_mode()} | Depth = {self.depth}")

    def get_mode(self) -> str:
        """Get a descriptive string of the current play configuration.

        Returns
        -------
        mode : str
            One of:
            - "<EngineName> vs <EngineName>" (if training)
            - "<EngineName> vs <PlayerName>" (if player_index==0)
            - "<PlayerName> vs <EngineName>" (if player_index==1)
        """
        if self.training:
            return f"{self.engines[0]['name']} vs {self.engines[1]['name']}"
        if self.player['index']:
            return f"{self.engines[0]['name']} vs {self.player['name']}"
        return f"{self.player['name']} vs {self.engines[1]['name']}"

    def new_game(self) -> None:
        """Start a brand‐new game after user confirmation.

        If idle, asks “Are you sure?”, resets engines’ boards, and updates the display.
        Displays an error if a training session is running.
        """
        if not self.updating and not self.training:
            self.updating = True
            if messagebox.askyesno("New Game", "Are you sure you want to start a new game?"):
                self.reset_engines_board()
                self.update_board()
            self.updating = False
        elif self.training:
            messagebox.showerror("Error", "Starting a new game is not permitted while the engine is training.")
        else:
            self.after(100, self.new_game)

    def reset_engines_board(self, new_board: Optional[Board] = None):
        """Reset both engines to share the same board instance.

        Parameters
        ----------
        new_board : chess.Board, optional
            Board to assign to both engines. If None, creates a new fresh Board.

        Notes
        -----
        Ensures both `.engine.board` references point to the same object.
        """
        new_board = Board() if new_board is None else new_board
        for engine_dict in self.engines:
            # noinspection PyProtectedMember
            engine_dict['engine']._board = new_board

    def train_engine(self):
        """Run a training loop where two engines play multiple games and update Q-values.

        Notes
        -----
        1. Iterates over a predefined range of game IDs.
        2. Plays moves alternating engines until game ends.
        3. Saves each game to a PGN with proper headers.
        4. If an engine is CMHMEngine2, submits `update_q_values()` and waits.
        5. Swaps engine colors between games.
            - If two **distinct** engine instances were provided via `engine_type_2`, then after **N** games:
                - `self.engines` will be in its original order if **N** is even.
                - `self.engines` will be swapped if **N** is odd.
            - This ensures engines alternate colors across the training set but means the final list order depends
            on the (user‐configurable) number of games.
        6. Restores training flag when complete.

        Errors
        ------
        Raises an error dialog if called while already training.
        """
        if not self.updating and not self.training:
            self.training = True
            # TODO: Prompt user for training game start and end number
            illegal_move: Move = Move.from_uci('a1b8')
            start_game_index: int = 0
            end_game_index: int = 1
            game_index: int
            for game_id in range(start_game_index + 1, end_game_index + 1):
                local_time: datetime = get_local_time()
                self.set_title()
                engine_white = self.engines[1]['engine']
                engine_black = self.engines[0]['engine']
                # We go out of our way to make sure both engines point to the same board object
                board: Board = engine_white.board
                future: Future = Future()
                # Kick off loop with a done future containing an illegal pick result
                future.set_result(Pick(illegal_move, float64(None)))
                while board.outcome(claim_draw=True) is None:
                    # TODO: Use these unused variables
                    all_moves: List[Move] = engine_white.current_moves_list()
                    move_number: int = board.fullmove_number
                    if future.done():
                        # TODO surface eval score in pick to GUI
                        pick: Pick = future.result()
                        move: Move = pick.move
                        if move != illegal_move:
                            board.push(move)
                        if board.turn:
                            future: Future = self._move_executor.submit(engine_white.pick_move)
                        else:
                            future: Future = self._move_executor.submit(engine_black.pick_move)
                    self.updating = True
                    self.update_board()
                    self.updating = False
                outcome: Outcome = board.outcome(claim_draw=True)
                game: Game = Game.from_board(board)
                game_heads = game.headers
                game_heads["Event"] = self.get_mode()
                game_heads["Site"] = self.site
                game_heads["Round"] = str(game_id)
                set_all_datetime_headers(game_heads=game_heads, local_time=local_time)
                game_heads["White"] = self.engines[1]['name']
                game_heads["Black"] = self.engines[0]['name']
                game_heads["Termination"] = outcome.termination.name
                game_heads["CMHMEngineDepth"] = str(self.depth)
                file_name: str = path.join(
                    self.pgn_dir,
                    self.training_dir,
                    f"{game_heads['Date']}_{game_heads['Event'].replace(' ', '_')}_{game_heads['Round']}.pgn"
                )
                self.save_to_pgn(file_name=file_name, game=game)
                for engine in {engine_white, engine_black}:
                    if isinstance(engine, CMHMEngine2):
                        # This is going to pop all the moves out of the shared board...
                        update_future: Future = self._move_executor.submit(engine.update_q_values)
                        while not update_future.done():
                            self.updating = True
                            self.update_board()
                            self.updating = False
                self.reset_engines_board()
                self.updating = True
                self.update_board()
                self.updating = False
                (
                    self.engines[0]['engine'], self.engines[1]['engine'],
                    self.engines[0]['name'], self.engines[1]['name'],
                ) = (
                    self.engines[1]['engine'], self.engines[0]['engine'],
                    self.engines[1]['name'], self.engines[0]['name'],
                )
            self.training = False
        elif self.training:
            messagebox.showerror("Error", "The engine is already training.")
        else:
            self.after(100, self.train_engine())

    def save_to_pgn(self, file_name: str, game: Game) -> None:
        """Write a completed Game object to a PGN file on disk.

        Parameters
        ----------
        file_name : str
            Full file path where the PGN will be written.
        game : chess.pgn.Game
            The Game instance to serialize.

        Notes
        -----
        Creates `pgn_dir` and `training_dir` if they do not exist.
        """
        if not path.isdir(self.pgn_dir):
            makedirs(self.pgn_dir)
        if not path.isdir(self.training_dir):
            makedirs(self.training_dir)
        with open(file_name, "w", encoding="utf-8") as file:
            print(game, file=file, end="\n\n")

    def show_game_over(self, outcome: Optional[Outcome] = None) -> None:
        """Display an informational dialog with the game outcome.

        Parameters
        ----------
        outcome : chess.Outcome, optional
            Outcome to display. If None, computes from the current engine board.
        """
        outcome: Optional[Outcome] = self.engines[0]['engine'].board.outcome(
            claim_draw=True
        ) if outcome is None else outcome
        messagebox.showinfo("Game Over", f"Game Over: {outcome}")

    def update_board(self) -> None:
        """Refresh the chessboard GUI.

        Clears the current canvas, redraws the board and pieces, then forces an update.
        """
        self.clear_board()
        self.draw_board()
        self.update()

    def clear_board(self) -> None:
        """Clear all drawings from the canvas.

        Deletes every item so the board can be fully redrawn.
        """
        self.canvas.delete("all")

    def draw_board(self) -> None:
        """Draw the 8×8 chessboard grid and overlay piece symbols.

        Iterates over all `chess.SQUARES`, draws each square’s background color,
        and if a piece is present, centers its unicode symbol with a background marker.

        Notes
        -----
        - Uses `self.square_size`, `self.colors`, and `self.font` for layout.
        - Flips rank ordering so that white’s back rank appears at the bottom.
        """
        square: int
        half_square_size: int = self.square_size // 2
        piece_bg: str = "⬤"
        font_size = int(self.square_size * 0.6)
        for square in SQUARES:
            row: int
            col: int
            row, col = divmod(square, 8)
            row_flipped: int = 7 - row
            x0: int
            y0: int
            x1: int
            y1: int
            x0, x1, y0, y1 = self.get_xys(col=col, flipped_row=row_flipped, square_size=self.square_size)
            color: str = self.colors[int(not (row + col) % 2)]
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
            piece: Optional[Piece] = self.engines[0]['engine'].board.piece_at(square)
            if piece is not None:
                # TODO: Refactor this into a `PieceTk` (or `CanvasPiece`) class that can support drag and drop
                # CanvasPiece aligns with CanvasTooltip better...
                # TODO: Consider adapting such a `CanvasPiece` in `main.ChessHeatMapApp.create_piece`.
                piece_x = x0 + half_square_size
                piece_y = y0 + half_square_size
                self.canvas.create_text(
                    piece_x,
                    piece_y,
                    text=piece_bg,
                    font=(self.font, font_size + 25),
                    fill="white" if piece.color else "black"
                )
                self.canvas.create_text(
                    piece_x,
                    piece_y,
                    text=piece.unicode_symbol(),
                    font=(self.font, font_size),
                    fill="blue" if piece.color else "yellow"
                )


if __name__ == "__main__":
    app = PlayChessApp(engine_type=CMHMEngine2)
    app.mainloop()
