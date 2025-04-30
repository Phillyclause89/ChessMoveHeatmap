"""Play against the CMHMEngine in a GUI."""
from tkinter import Canvas, Event, Menu, Tk, messagebox
from typing import Callable, Optional

from chess import Board, Move, Outcome, Piece, SQUARES

from chmengine import CMHMEngine, CMHMEngine2
from chmutils import BaseChessTkApp, DEFAULT_COLORS, DEFAULT_FONT


class PlayChessApp(Tk, BaseChessTkApp):
    """Play against the CMHMEngine in a GUI."""
    engine: CMHMEngine
    training: bool

    def __init__(self, engine_type: Callable = CMHMEngine, depth: int = 1) -> None:
        """Initialize the PlayChessApp.

        Parameters
        ----------
        engine_type : typing.Callable
        depth : int
        """
        self.updating = True
        self.training = False
        super().__init__()
        self.engine = engine_type(depth=depth)
        self.depth = self.engine.depth
        self.title(f"Play Against {self.engine.__class__.__name__} | Depth = {depth}")
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
        self.bind("<Configure>", self.on_resize)
        self.focus_force()
        self.updating = False

    def on_closing(self) -> None:
        """Clean up resources before closing the application."""
        if not self.updating:
            self.updating = True
            if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
                # TODO Add save game method call here.
                self.destroy()
            self.updating = False
        else:
            self.after(100, self.on_closing)

    def set_depth(self) -> None:
        """Prompt the user to set a new recursion depth for the engine."""
        if not self.updating and not self.training:
            self.updating = True
            new_depth: Optional[int] = self.ask_depth()
            if new_depth is not None and new_depth != self.depth:
                self.engine.depth = new_depth
                self.depth = self.engine.depth
                self.set_title()
            self.updating = False
        elif self.training:
            messagebox.showerror("Error", "Changing depth is not permitted while the engine is training.")
        else:
            self.after(100, self.set_depth)

    def create_menu(self) -> None:
        """Construct the app menu."""
        menu_bar: Menu = Menu(self)
        file_menu: Menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Game", command=self.new_game)
        # TODO: Replace `command=labmda:...` w/ hook to training function.
        file_menu.add_command(label="New Training Session", command=lambda: print('training!'))
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.add_options(menu_bar)
        self.config(menu=menu_bar)

    def set_title(self) -> None:
        """Set the app window title."""
        self.title(f"Play Against {self.engine.__class__.__name__} | Depth = {self.engine.depth}")

    def new_game(self) -> None:
        """Start a new game."""
        if not self.updating:
            self.updating = True
            if messagebox.askyesno("New Game", "Are you sure you want to start a new game?"):
                self.engine.board = Board()
                self.update_board()
            self.updating = False
        else:
            self.after(100, self.new_game)

    def play_move(self, move: Move) -> None:
        """Play a move and let the engine respond.

        Parameters
        ----------
        move : chess.Move
        """
        if move in self.engine.board.legal_moves:
            self.engine.board.push(move)
            self.update_board()
            outcome: Optional[Outcome] = self.engine.board.outcome(claim_draw=True)
            if outcome is None:
                engine_move: Move
                engine_move, _ = self.engine.pick_move()
                self.engine.board.push(engine_move)
                self.update_board()
            else:
                self.show_game_over(outcome=outcome)

    def show_game_over(self, outcome: Optional[Outcome] = None) -> None:
        """Display the game outcome.

        Parameters
        ----------
        outcome : typing.Optional[chess.Outcome]
        """
        outcome: Optional[Outcome] = self.engine.board.outcome(claim_draw=True) if outcome is None else outcome
        messagebox.showinfo("Game Over", f"Game Over: {outcome}")

    def update_board(self) -> None:
        """Update the chessboard display."""
        self.clear_board()
        self.draw_board()

    def clear_board(self) -> None:
        """Clear the canvas."""
        self.canvas.delete("all")

    def draw_board(self) -> None:
        """Draw the chessboard and pieces."""
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
            color: str = self.colors[(row + col) % 2]
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
            piece: Optional[Piece] = self.engine.board.piece_at(square)
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
