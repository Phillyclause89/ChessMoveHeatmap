"""Play against many clones of CMHMEngine2 via CMHMEngine2PoolExecutor in a GUI."""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import cycle
from os import makedirs, path
from pathlib import Path
from random import choice
from tkinter import (
    Button, Canvas, Entry, Event, Frame, IntVar, Label, Menu, Radiobutton, StringVar, Tk, Toplevel, messagebox,
    simpledialog
)
from typing import Callable, Dict, Generator, Iterator, List, Optional, Set, Tuple, Union

from chess import Board, Move, Outcome, Piece, SQUARES
from chess.pgn import ChildNode, Game
from numpy import float64, isnan

from chmengine import CMHMEngine2PoolExecutor, Pick, set_all_datetime_headers
from chmutils import (
    BaseChessTkApp, DEFAULT_COLORS, DEFAULT_FONT, Player, get_local_time, get_promotion_choice, state_faces_within_bmp
)

__all__ = [
    'PlayDeeperChessApp',
    'EnginePoolContainer'
]


@dataclass
class EnginePoolContainer:
    """Container for the engine pool(s)."""
    _white: CMHMEngine2PoolExecutor
    _black: CMHMEngine2PoolExecutor

    def __init__(
            self,
            engine_type: Callable = CMHMEngine2PoolExecutor,
            depth: int = 1,
            time_limit: Optional[float64] = None,
            engine_type_2: Optional[Callable] = None,
            depth_2: Optional[int] = None,
            time_limit_2: Optional[float64] = None,
    ) -> None:
        """Initialize the EnginePoolContainer

        Parameters
        ----------
        engine_type : Callable
        depth : int
        engine_type_2 : Optional[Callable]
        depth_2 : Optional[int]
        """
        self._white = engine_type(depth=depth, time_limit=time_limit)
        # Allows for same Engine instance or different based on engine_type_2 being passed
        self._black = self._white if engine_type_2 is None else engine_type_2(
            depth=depth if depth_2 is None else depth_2, time_limit=time_limit if time_limit_2 is None else time_limit_2
        )
        # CMHMEngine.board.setter sets a copy thus we need to bypass the method to set the same object instance
        self._black.engine._board = self._white.engine.board

    @property
    def board(self) -> Board:
        """Gets the shared board object between the engines

        Returns
        -------
        Board
        """
        return self._white.engine.board

    @board.setter
    def board(self, new_board: Board) -> None:
        """Sets the shared board object to a copy of the new board

        Parameters
        ----------
        new_board : Board
        """
        try:
            if not new_board.is_valid():
                raise ValueError(f"new_board is not valid: {new_board}")
            self._white._board = new_board
            self._black._board = self._white.engine.board
        except AttributeError as error:
            raise TypeError(f"new_board must be type `chess.Board`, got `{type(new_board)}`") from error

    @property
    def white(self) -> CMHMEngine2PoolExecutor:
        """Gets the engine playing as white

        Returns
        -------
        CMHMEngine2PoolExecutor
        """
        return self._white

    @white.setter
    def white(self, new_engine: CMHMEngine2PoolExecutor) -> None:
        """Sets the engine playing as white

        Parameters
        ----------
        new_engine : CMHMEngine2PoolExecutor
        """
        if isinstance(new_engine, CMHMEngine2PoolExecutor):
            self._white = new_engine
            self._white._board = self._black.engine.board
        else:
            raise TypeError(f"new_engine must be type CMHMEngine, got {type(new_engine)}")

    @property
    def black(self) -> CMHMEngine2PoolExecutor:
        """Gets the engine playing as white

        Returns
        -------
        CMHMEngine2PoolExecutor
        """
        return self._black

    @black.setter
    def black(self, new_engine: CMHMEngine2PoolExecutor) -> None:
        """Sets the engine playing as white

        Parameters
        ----------
        new_engine : CMHMEngine2PoolExecutor
        """
        if isinstance(new_engine, CMHMEngine2PoolExecutor):
            self._black = new_engine
            self._black._board = self._white.engine.board
        else:
            raise TypeError(f"new_engine must be type CMHMEngine, got {type(new_engine)}")

    @property
    def white_name(self) -> str:
        """Gets class name of white engine.

        Returns
        -------
        str
        """
        return self._white.__class__.__name__

    @white_name.setter
    def white_name(self, new_name: Callable) -> None:
        """

        Parameters
        ----------
        new_name : Callable
        """
        self.white = new_name(depth=self._white.engine.depth)

    @property
    def black_name(self) -> str:
        """Gets class name of white engine.

        Returns
        -------
        str
        """
        return self._black.__class__.__name__

    @black_name.setter
    def black_name(self, new_name: Callable) -> None:
        """

        Parameters
        ----------
        new_name : Callable
        """
        self.black = new_name(depth=self._white.engine.depth)

    @property
    def depths(self) -> Tuple[int, int]:
        """Gets the depth values of the engine(s)

        Returns
        -------
        Tuple[int, int]
        """
        return self._white.engine.depth, self._black.engine.depth

    @depths.setter
    def depths(self, new_depths: Tuple[int, int]) -> None:
        """Sets the engine(s) new depths

        Parameters
        ----------
        new_depths : Tuple[int, int]
        """
        self._white.depth, self._black.depth = new_depths

    @property
    def depth(self) -> int:
        """Gets the depth value of the engine at play per the board state

        Returns
        -------
        int
        """
        return self._white.engine.depth if self._white.engine.board.turn else self._black.engine.depth

    @depth.setter
    def depth(self, new_depth: int) -> None:
        """Sets the depth value of the engine at play per the board state

        Parameters
        ----------
        new_depth : int
        """
        if self._white.engine.board.turn:
            self._white.depth = int(new_depth)
        else:
            self._black.depth = int(new_depth)

    def flip(self) -> None:
        """Flips the board sides the engine is playing."""
        self._black, self._white = self._white, self._black

    def push(self, pick: Pick) -> None:
        """Push a Pick's Move to the shared board.

        Parameters
        ----------
        pick : Pick
        """
        self._white.push(move=pick.move)

    def __getitem__(self, chess_lib_index: object) -> CMHMEngine2PoolExecutor:
        """Gets the engine corresponding to the python-chess lib COLOR_NAMES index

        Parameters
        ----------
        chess_lib_index : object
        """
        return self._white if chess_lib_index else self._black

    def __setitem__(self, chess_lib_index: Union[bool, int], new_engine: CMHMEngine2PoolExecutor) -> None:
        """Sets the engine corresponding to the python-chess lib COLOR_NAMES index

        Parameters
        ----------
        chess_lib_index : Union[bool, int]
        new_engine : CMHMEngine
        """
        if chess_lib_index:
            self.white = new_engine
        else:
            self.black = new_engine

    def __len__(self) -> int:
        """The length of the container is the number of unique engine instances

        Returns
        -------
        int
        """
        return sum(self.to_set())

    def __contains__(self, item: object) -> bool:
        """

        Parameters
        ----------
        item : object

        Returns
        -------
        bool

        """
        return item is self._white or item is self._black

    def __iter__(self) -> Iterator[CMHMEngine2PoolExecutor]:
        """
        Returns
        -------
        Iterator[CMHMEngine]

        """
        engine: CMHMEngine2PoolExecutor
        for engine in self.to_set():
            yield engine

    def to_set(self) -> Set[CMHMEngine2PoolExecutor]:
        """

        Returns
        -------
        Set[CMHMEngine]
        """
        return {self._white, self._black}


class PlayDeeperChessApp(Tk, BaseChessTkApp):
    """Play against the CMHMEngine in a GUI."""
    _move_executor: ThreadPoolExecutor
    training: bool
    rewarding: bool
    site: str
    face: str
    game_line: List[Pick]
    engine_pools: EnginePoolContainer
    player: Player = Player()
    training_dir: str = "trainings"
    pgn_dir: str = "pgns"
    depth: int = 1
    fullmove_number: int = 1
    faces: Dict[str, Tuple[str, ...]] = state_faces_within_bmp
    dot_dot: Generator[str, str, str] = cycle(['.', '..', '...'])
    selected_square: Optional[int] = None
    possible_squares: Tuple[int, ...] = tuple()
    start_time: datetime = get_local_time()

    def __init__(
            self,
            engine_type: Callable = EnginePoolContainer,
            depth: int = depth,
            time_limit: Optional[float64] = None,
            player_name: str = player.name,
            player_color_is_black: bool = False,
            site: Optional[str] = None,
            engine_type_2: Optional[Callable] = None,
            depth_2: Optional[int] = None,
            time_limit_2: Optional[float64] = None,

    ) -> None:
        """Initialize the PlayChessApp GUI and configure engines.

        Parameters
        ----------
        engine_type : Callable, optional
            Class or factory for the primary engine (default: CMHMEngine).
        depth : int, optional
            Initial search depth for the primary engine (default: class‐level `depth`).
        time_limit : float64
            Passed to Cmhmey III
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
        time_limit : float64
            Passed to Cmhmey III
        """
        self.updating = True
        self.training = False
        self.rewarding = False
        super().__init__()
        screen_height: int = self.winfo_screenheight()
        screen_width: int = int(screen_height * 0.75)
        self.geometry(f"{screen_width}x{screen_height}")
        self.square_size = screen_width // 8
        self.colors = list(DEFAULT_COLORS)
        self.font = DEFAULT_FONT
        self.player.name = player_name
        self.site = f"{self.player.name}'s place" if site is None else site
        if player_color_is_black:
            self.player.index = 1
        self.onboard_player()
        self.engine_pools = EnginePoolContainer(engine_type, depth, time_limit, engine_type_2, depth_2, time_limit_2)
        self.depth = self.engine_pools.depth
        self.set_title()
        self.create_menu()
        self.canvas = Canvas(self)
        self.canvas.pack(fill="both", expand=True)
        self.current_move_index = -1
        self.highlight_squares = set()
        self.game_line = [Pick(Move.from_uci('a1b8'), float64(None))]  # index0 of the game-line will hold illegal move
        self.face = self.get_smily_face()
        self.update_board()
        self._move_executor = ThreadPoolExecutor(max_workers=1)
        self.bind("<Configure>", self.on_resize)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.canvas.bind('<Button-1>', self.activate_piece)
        self.focus_force()
        self.start_time = get_local_time()
        self.set_title()
        if bool(self.player.index) == self.engine_pools.board.turn:
            pick = self.await_engine_pick()
            self.engine_pools.push(pick)
            self.update_board()
        self.updating = False

    def on_closing(self) -> None:
        """Clean up and confirm exit when the window is closed.

        If the app is idle, prompts the user to confirm quitting, then destroys the window.
        If an update is in progress, retries after a short delay.
        """
        if not self.updating:
            self.updating = True
            if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
                self._move_executor.shutdown()
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
        if not self.updating and not self.training and not self.rewarding:
            self.updating = True
            new_depth: Optional[int] = self.ask_depth()
            if new_depth is not None and new_depth != self.depth:
                self.engine_pools.depths = new_depth, new_depth
                self.depth = new_depth
                self.set_title()
            self.updating = False
        elif self.training:
            messagebox.showerror("Error", "Changing depth is not permitted while the engine is training.")
        elif self.rewarding:
            messagebox.showerror(
                "Error", "Changing depth is not permitted while the engine is applying its rewards function."
            )
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
        self.title(
            f"{self.get_mode()} | Depth = {self.depth} | Date = {self.start_time.strftime('%Y.%m.%d')}"
        )

    def get_mode(self) -> str:
        """Get a descriptive string of the current play configuration.

        Returns
        -------
        mode : str
            One of:
            - "<EngineName> vs <EngineName>" (if training)
            - "<PlayerName> vs <EngineName>" (if player_index==0)
            - "<EngineName> vs <PlayerName>" (if player_index==1)
        """
        if self.rewarding:
            return f"Applying Rewards Function"
        if self.training:
            return f"{self.engine_pools.white_name} vs {self.engine_pools.black_name}"
        if self.player.index:
            return f"{self.engine_pools.white_name} vs {self.player.name}"
        return f"{self.player.name} vs {self.engine_pools.black_name}"

    def new_game(self) -> None:
        """Start a brand‐new game after user confirmation.

        If idle, asks “Are you sure?”, resets engines’ boards, and updates the display.
        Displays an error if a training session is running.
        """
        if not self.updating and not self.training and not self.rewarding:
            self.updating = True
            if messagebox.askyesno("New Game", "Are you sure you want to start a new game?"):
                if messagebox.askyesno(
                        f"You are set to play as {self.player.color.title()}",
                        "Would you line to switch sides before starting a new game?"
                ):
                    self.player.index = int(not self.player.index)
                self.reset_engines_board()
                if bool(self.player.index) == self.engine_pools.board.turn:
                    pick = self.await_engine_pick()
                    self.engine_pools.push(pick)
                    outcome: Outcome = self.engine_pools.board.outcome(claim_draw=True)
                    if outcome is not None:
                        return self.show_game_over(outcome=outcome)
                    self.fullmove_number = self.engine_pools.board.fullmove_number
                    self.game_line.append(pick)
                    self.face = self.get_smily_face()
                    self.highlight_squares = {pick.move.from_square, pick.move.to_square}
                self.update_board()
            self.updating = False
        elif self.training:
            messagebox.showerror("Error", "Starting a new game is not permitted while the engine is training.")
        elif self.rewarding:
            messagebox.showerror(
                "Error", "Starting a new game is not while the engine is applying its rewards function."
            )
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
        self.engine_pools.board = new_board
        self.game_line = self.game_line[0:1] + [Pick(move=m, score=float64(None)) for m in new_board.move_stack]
        self.selected_square = None
        self.highlight_squares = {
            self.game_line[-1].move.to_square, self.game_line[-1].move.from_square
        } if self.game_line[-1] != self.game_line[0] else set()
        self.face = self.get_smily_face()
        self.start_time = get_local_time()
        self.set_title()

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
        if not self.updating and not self.training and not self.rewarding:
            self.training = True
            illegal_pick: Pick = self.game_line[0]
            start_game_index: Optional[int]
            end_game_index: Optional[int]
            start_game_index, end_game_index = self.get_training_game_indexes()
            if start_game_index is None or end_game_index is None or end_game_index <= start_game_index:
                self.training = False
                return messagebox.showerror("Error Starting Training Mode", "Invalid game index(es) submitted.")
            game_index: int
            for game_id in range(start_game_index + 1, end_game_index + 1):
                self.start_time = get_local_time()
                self.set_title()
                # We go out of our way to make sure both engines point to the same board object
                board: Board = self.engine_pools.board
                future: Future = Future()
                # Kick off loop with a done future containing an illegal pick result
                future.set_result(illegal_pick)
                while board.outcome(claim_draw=True) is None:
                    if future.done():
                        pick: Pick = future.result()
                        if pick != illegal_pick:
                            move: Move = pick.move
                            board.push(move)
                            self.fullmove_number = board.fullmove_number
                            self.game_line.append(pick)
                            self.face = self.get_smily_face()
                            self.highlight_squares = {move.from_square, move.to_square}
                        future: Future = self._move_executor.submit(
                            self.engine_pools.white.pick_move
                        ) if board.turn else self._move_executor.submit(self.engine_pools.black.pick_move)
                    self.updating = True
                    self.square_size = min(self.canvas.winfo_width(), self.canvas.winfo_height()) // 8
                    self.update_board()
                    self.updating = False
                outcome: Outcome = board.outcome(claim_draw=True)
                game: Game = Game.from_board(board)
                self.add_eval_comments_to_mainline(game=game)
                game_heads = game.headers
                game_heads["Event"] = self.get_mode()
                game_heads["Site"] = self.site
                game_heads["Round"] = str(game_id)
                set_all_datetime_headers(game_heads=game_heads, local_time=self.start_time)
                game_heads["White"] = self.engine_pools.white_name
                game_heads["Black"] = self.engine_pools.black_name
                game_heads["Termination"] = outcome.termination.name
                game_heads["CMHMEngineDepth"] = str(self.depth)
                file_name: str = path.join(
                    self.pgn_dir,
                    self.training_dir,
                    f"{game_heads['Date']}_{game_heads['Event'].replace(' ', '_')}_{game_heads['Round']}.pgn"
                )
                self.save_to_pgn(file_name=file_name, game=game)
                self.highlight_squares = set()
                self.apply_rewards()
                self.reset_engines_board()
                self.engine_pools.flip()
                self.updating = True
                self.update_board()
                self.updating = False
            self.training = False
        elif self.training:
            messagebox.showerror("Error", "The engine is already training.")
        elif self.rewarding:
            messagebox.showerror("Error", "The engine is currently applying its rewards function.")
        else:
            self.after(100, self.train_engine)

    def add_eval_comments_to_mainline(self, game: Game) -> None:
        """Annotate each half-move in the Game’s main line with the corresponding eval score.

        This mutates `game` in place by setting `node.comment = "{eval=…}"` on
        each `ChildNode` in `game.mainline()`.

        **Alignment requirement**
        It is the caller’s responsibility to ensure that:
          - `self.game_line` has exactly one dummy pick at index 0 (for the “illegal” move), and
          - `len(self.game_line) - 1 == number of half-moves in game.mainline()`.

        If those are out of sync, the for-loop will silently stop at the shorter of the two,
        so any mis-alignment should be checked *before* calling this method.

        Parameters
        ----------
        game : chess.pgn.Game
            A PGN `Game` whose main-line nodes you want to annotate with eval comments sourced from `self.game_line`.
        """
        node: ChildNode
        pick: Pick
        for node, pick in zip(game.mainline(), self.game_line[1:]):
            node.comment = f"eval={pick.score:+.2f}" if not isnan(pick.score) else f"eval=?"

    def get_training_game_indexes(self) -> Tuple[Optional[int], Optional[int]]:
        """Gets the training game start and end index (game id is +1 from game index)

        Returns
        -------
        Tuple[Optional[int], Optional[int]]
            start index, end index
        """

        pgn_dir: str = path.join(self.pgn_dir, self.training_dir)
        file_count: int = 0
        if path.isdir(pgn_dir):
            file_count = len([item for item in Path(pgn_dir).iterdir() if item.is_file()])
        message: str = (
            "Do you want auto-set the start index to match the "
            f"continuation point ({file_count}) of the existing training games?"
        )
        if file_count > 0 and messagebox.askyesno(f"{file_count} Previous Training Games Detected", message):
            # Our start will be ✅ and default end will be ✅+1 (only one prompt)
            # I could make the prompt be for a relative value, but then I'd have to error handle file_count + None case
            return file_count, simpledialog.askinteger(
                "Set Training Run End Index",
                "Set the end index for the training run:",
                initialvalue=file_count + 1
            )
        # Else we prompt for both values (allowing possible overwrites of existing files)
        return simpledialog.askinteger(
            "Set Training Run Start Index",
            "Set the start index for the training run:",
            initialvalue=0
        ), simpledialog.askinteger(
            "Set Training Run End Index",
            "Set the end index for the training run:",
            initialvalue=1
        )

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

    def show_game_over(self, outcome: Optional[Outcome] = None, local_start_time: Optional[datetime] = None) -> None:
        """Display an informational dialog with the game outcome.

        Parameters
        ----------
        outcome : chess.Outcome, optional
            Outcome to display. If None, computes from the current engine board.
        local_start_time : datetime, optional
            The start time of the game in local timezone.
        """
        local_start_time = self.start_time if local_start_time is None else local_start_time
        outcome: Optional[Outcome] = self.engine_pools.board.outcome(
            claim_draw=True
        ) if outcome is None else outcome
        messagebox.showinfo("Game Over", f"Game Over: {outcome}")
        pgn_dir: str = path.join(self.pgn_dir, self.training_dir)
        round_number: int = 1
        if path.isdir(pgn_dir):
            round_number = len([item for item in Path(pgn_dir).iterdir() if item.is_file()]) + 1
        game: Game = Game.from_board(self.engine_pools.board)
        self.add_eval_comments_to_mainline(game=game)
        game_heads = game.headers
        game_heads["Event"] = self.get_mode()
        game_heads["Site"] = self.site
        game_heads["Round"] = str(round_number)
        set_all_datetime_headers(game_heads=game_heads, local_time=local_start_time)
        game_heads["White"] = self.engine_pools.white_name if self.player.index else self.player.name
        game_heads["Black"] = self.player.name if self.player.index else self.engine_pools.black_name
        game_heads["Termination"] = outcome.termination.name
        game_heads["CMHMEngineDepth"] = str(self.depth)
        file_name: str = path.join(
            pgn_dir,
            f"{game_heads['Date']}_{game_heads['Event'].replace(' ', '_')}_{game_heads['Round']}.pgn"
        )
        self.save_to_pgn(file_name=file_name, game=game)
        self.highlight_squares = set()
        self.apply_rewards()
        self.reset_engines_board()

    def apply_rewards(self) -> None:
        """Apply rewards function to supported Engine types"""
        for engine in self.engine_pools:
            if isinstance(engine, CMHMEngine2PoolExecutor):
                self.rewarding = True
                update_future: Future = self._move_executor.submit(engine.engine.update_q_values)
                while not update_future.done():
                    self.updating = True
                    self.update_board()
                    self.updating = False
                self.rewarding = False

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

    def get_smily_face(self) -> str:
        """Gets a random smily face per state of mind the engine is in.

        Returns
        -------
        str
        """
        last_score: float64 = self.game_line[-1].score
        turn: bool = self.engine_pools.board.turn
        key: str = 'draw'
        if (last_score > 127 and turn) or (last_score < -127 and not turn):
            key = 'winning'
        if (last_score < -127 and turn) or (last_score > 127 and not turn):
            key = 'losing'
        return choice(self.faces[key])

    def draw_board(self) -> None:
        """Draw the 8×8 chessboard grid and overlay piece symbols.

        Iterates over all `chess.SQUARES`, draws each square’s background color,
        and if a piece is present, centers its unicode symbol with a background marker.

        Notes
        -----
        - Uses `self.square_size`, `self.colors`, and `self.font` for layout.
        - Flips rank ordering so that white’s back rank appears at the bottom.
        """
        board: Board = self.engine_pools.board
        square: int
        half_square_size: int = self.square_size // 2
        piece_bg: str = "⬤"
        font_size = int(self.square_size * 0.6)
        bg_size = font_size + 25
        game_line_font: Tuple[str, int] = (self.font, font_size // 5)
        at_ply_name: str
        at_ply_color: str

        if self.rewarding:
            top_text: str = f"{self.face} Applying Rewards Function{next(self.dot_dot)}"
        else:
            if self.training:
                at_ply_name, at_ply_color = (
                    self.engine_pools.white_name, 'White'
                ) if board.turn else (
                    self.engine_pools.black_name, 'Black'
                )
            elif self.player.index:
                at_ply_name, at_ply_color = (
                    self.engine_pools.white_name, 'White'
                ) if board.turn else (
                    self.player.name, 'Black'
                )
            else:
                at_ply_name, at_ply_color = (
                    self.player.name, 'White'
                ) if board.turn else (
                    self.engine_pools.black_name, 'Black'
                )
            top_text = (
                f"{self.face} {at_ply_name} ({at_ply_color}) is picking Move #{self.fullmove_number}"
                f" (Pick #{len(self.game_line)}){next(self.dot_dot)}"
            )
        self.canvas.create_text(
            half_square_size // 8, (half_square_size // 4),
            anchor='w',
            text=top_text,
            font=game_line_font
        )
        game_line_text: str = ' ⬅ '.join([f"{p:+.2f}" for p in self.game_line[-1:0:-1]])
        self.canvas.create_text(
            half_square_size // 8, (half_square_size // 4) * 3,
            anchor='w',
            text=f"Previous Picks: {game_line_text}",
            font=game_line_font,
        )
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
            y0, y1 = y0 + half_square_size, y1 + half_square_size
            color: str = self.colors[(row_flipped + col) % 2]
            boarder_width: int = 1
            outline_color: str = "black"
            if square in self.highlight_squares:
                boarder_width = 3
                if board.turn:
                    outline_color = "yellow"
                else:
                    outline_color = "blue"
            offset: int = boarder_width // 2
            self.canvas.create_rectangle(
                x0 + offset, y0 + offset, x1 - offset, y1 - offset,
                fill=color, outline=outline_color, width=boarder_width
            )
            piece: Optional[Piece] = board.piece_at(square)
            if piece is not None:
                piece_x = x0 + half_square_size
                piece_y = y0 + half_square_size
                self.canvas.create_text(
                    piece_x,
                    piece_y,
                    text=piece_bg,
                    font=(self.font, bg_size),
                    fill="light green" if square == self.selected_square else "white" if piece.color else "black"
                )
                self.canvas.create_text(
                    piece_x,
                    piece_y,
                    text=piece.unicode_symbol(),
                    font=(self.font, font_size),
                    fill="blue" if piece.color else "yellow"
                )

    def coord_to_square(self, x: int, y: int) -> Optional[int]:
        """
        Convert canvas coordinates to a 0–63 square index,
        or return None if the click is off‐board.
        """
        s = self.square_size
        y_offset = s // 2
        y_rel = y - y_offset

        if not (0 <= x < 8 * s and 0 <= y_rel < 8 * s):
            return None

        col = x // s
        row_flipped = y_rel // s
        row = 7 - row_flipped

        # either:
        return row * 8 + col

    def activate_piece(self, event: Event) -> None:
        """Responds to click events on the Canvas to see if they correspond to playing a move

        Parameters
        ----------
        event : Event
        """
        if not self.updating and not self.training and not self.rewarding and self.engine_pools.board.turn == bool(
                self.player
        ):
            if self.selected_square is None:
                square: Optional[int] = self.coord_to_square(event.x, event.y)
                legal_moves: List[Move] = list(self.engine_pools.board.legal_moves)
                from_to_map: Dict[int:Tuple[int, ...]] = {
                    m.from_square: tuple(
                        mt.to_square for mt in legal_moves if mt.from_square == m.from_square
                    ) for m in legal_moves
                }
                if square is None or square not in from_to_map:
                    return
                self.selected_square = square
                self.possible_squares = from_to_map[square]
            else:
                square: Optional[int] = self.coord_to_square(event.x, event.y)
                if square == self.selected_square:
                    self.selected_square = None
                    self.possible_squares = tuple()
                    self.updating = True
                    self.update_board()
                    self.updating = False
                    return
                if square is None or square not in self.possible_squares:
                    return
                player_move: Move = Move(from_square=self.selected_square, to_square=square)
                if not self.engine_pools.board.is_legal(move=player_move):
                    # The only way end up here is by pawn promotions.
                    promotions: List[Move] = [
                        m for m in self.engine_pools.board.legal_moves if
                        m.promotion is not None and m.to_square == player_move.to_square
                    ]
                    # Prompt player to pick the promotion move from promotions
                    promotion_move: Optional[Move] = get_promotion_choice(
                        promotions=promotions,
                        board=self.engine_pools.board
                    )
                    if player_move is None:
                        return
                    player_move = promotion_move
                pick: Pick = Pick(
                    move=player_move,
                    # Since we don't run the pick_move method on human moves, we default to the line's last pick score.
                    score=self.game_line[-1].score
                )
                self.engine_pools.push(pick)
                self.game_line.append(pick)
                self.face = self.get_smily_face()
                self.highlight_squares = {pick.move.from_square, pick.move.to_square}
                self.selected_square = None
                self.possible_squares = tuple()
                outcome: Outcome = self.engine_pools.board.outcome(claim_draw=True)
                if outcome is not None:
                    return self.show_game_over(outcome=outcome)
                self.fullmove_number = self.engine_pools.board.fullmove_number
                self.updating = True
                self.update_board()
                self.updating = False
                pick = self.await_engine_pick()
                self.engine_pools.push(pick)
                outcome: Outcome = self.engine_pools.board.outcome(claim_draw=True)
                if outcome is not None:
                    return self.show_game_over(outcome=outcome)
                self.fullmove_number = self.engine_pools.board.fullmove_number
                self.game_line.append(pick)
                self.face = self.get_smily_face()
                self.highlight_squares = {pick.move.from_square, pick.move.to_square}
                # TODO: Add autosave point here (or via self.after) once save system is developed
            self.updating = True
            self.update_board()
            self.updating = False

    def await_engine_pick(self) -> Pick:
        """Awaits the engine's pick for the next move.

        Returns
        -------
        Pick
        """
        future: Future = self._move_executor.submit(
            self.engine_pools.white.pick_move
        ) if self.engine_pools.board.turn else self._move_executor.submit(self.engine_pools.black.pick_move)
        while not future.done():
            self.updating = True
            self.square_size = min(self.canvas.winfo_width(), self.canvas.winfo_height()) // 8
            self.update_board()
            self.updating = False
        return future.result()

    def onboard_player(self) -> None:
        """Onboards player at app start"""
        dlg: Toplevel = Toplevel(self)
        dlg.title("Enter Player Info")
        dlg.transient(self)
        dlg.grab_set()
        name_var: StringVar = StringVar(value=self.player.name or "")
        site_var: StringVar = StringVar(value=self.site or "")
        index_var: IntVar = IntVar(value=self.player.index or 0)
        # Player Name
        Label(dlg, text="Player Name:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        name_entry: Entry = Entry(dlg, textvariable=name_var)
        name_entry.grid(row=0, column=1, padx=5, pady=5)
        # Site
        Label(dlg, text="Site:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        site_entry: Entry = Entry(dlg, textvariable=site_var)
        site_entry.grid(row=1, column=1, padx=5, pady=5)
        # Index/(Color) choice
        Label(dlg, text="Starting Side:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        color_frame: Frame = Frame(dlg)
        color_frame.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        Radiobutton(color_frame, text="White", variable=index_var, value=0).pack(side="left")
        Radiobutton(color_frame, text="Black", variable=index_var, value=1).pack(side="left")

        def submit() -> None:
            """updates the app with player info"""
            # assign back to your app state
            self.player.name = name_var.get().strip()
            self.player.index = index_var.get()
            self.site = site_var.get().strip()
            dlg.destroy()

        submit_btn: Button = Button(dlg, text="Submit", command=submit)
        submit_btn.grid(row=3, column=0, columnspan=2, pady=(10, 5))
        dlg.bind("<Return>", lambda e: submit())
        self.update_idletasks()
        dlg.update_idletasks()
        parent_x: int = self.winfo_rootx()
        parent_y: int = self.winfo_rooty()
        parent_w: int = self.winfo_width()
        parent_h: int = self.winfo_height()
        dlg_w: int = dlg.winfo_reqwidth()
        dlg_h: int = dlg.winfo_reqheight()
        x: int = parent_x + (parent_w // 2) - (dlg_w // 2)
        y: int = parent_y + (parent_h // 2) - (dlg_h // 2)
        dlg.geometry(f"{dlg_w}x{dlg_h}+{x}+{y}")
        name_entry.focus_set()
        self.wait_window(dlg)


if __name__ == "__main__":
    app = PlayDeeperChessApp(engine_type=CMHMEngine2PoolExecutor)
    app.mainloop()
