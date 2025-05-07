"""Tkinter Dialog for pawn promotion."""
from tkinter import Button, Label, Misc, Tk, X, simpledialog
from typing import Iterator, List, Optional

from chess import Board, Move


class PromotionDialog(simpledialog.Dialog):
    """Prompts user for promotion choice."""
    selected_promotion: Optional[Move]
    promotions: Iterator[Move]
    board: Board

    def __init__(self, parent: Tk, promotions: Iterator[Move], board: Board) -> None:
        """PromotionDialog.__init__

        Parameters
        ----------
        parent : Tk
        promotions : Iterator[Move]
        board : Board
        """
        self.promotions, self.board, self.selected_promotion = promotions, board, None
        super().__init__(parent, title='Pawn Promotion!')

    def buttonbox(self):
        """Overrides default buttons."""

    def body(self, master: Misc) -> None:
        """
        Parameters
        ----------
        master : Misc
        """
        Label(master, text="Choose a promotion piece:").pack(pady=10)
        move: Move
        for move in self.promotions:
            btn: Button = Button(
                master,
                text=self.board.san(move),
                command=lambda m=move: self.on_select(m)
            )
            btn.pack(fill=X, padx=10, pady=5)
        return None  # No initial focus

    def on_select(self, move: Move) -> None:
        """Update selected_promotion field and close.

        Parameters
        ----------
        move : Move
        """
        self.selected_promotion = move
        self.destroy()


def get_promotion_choice(promotions: List[Move], board: Board) -> Optional[Move]:
    """Gets the player's choice of possible promotion moves.

    Parameters
    ----------
    promotions : List[Move]
        List of legal promotion moves available to the player.
    board : Board
        The current chess board instance.

    Returns
    -------
    Union[Move, None]
        The move selected by the player, or None if no selection is made.
    """
    root: Tk = Tk()
    root.withdraw()
    dialog: PromotionDialog = PromotionDialog(parent=root, promotions=promotions, board=board)
    root.destroy()
    return dialog.selected_promotion
