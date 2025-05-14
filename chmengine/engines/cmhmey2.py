"""Cmhmey Jr."""
from random import choice
from typing import List, Optional

from chess import Board, Move
from numpy import float64

from chmengine.engines.cmhmey1 import CMHMEngine
from chmengine.engines.quartney import Quartney
from chmengine.utils import Pick, calculate_better_white_minus_black_score, format_picks

__all__ = ['CMHMEngine2']


class CMHMEngine2(CMHMEngine, Quartney):
    """Cmhmey Jr., the love-child of CMHMEngine and Quartney"""

    def __init__(self, board: Optional[Board] = None, depth: int = 1) -> None:
        """Initialize the CMHMEngine2 instance.

        This constructor extends the parent CMHMEngine by initializing the Q-table database.
        It sets up the engine with an optional initial board and a recursion depth for heatmap
        calculations. Upon initialization, it creates or verifies the Q-table database structure.

        Parameters
        ----------
        board : Optional[chess.Board]
            The initial board state. If None, a standard starting board is used.
        depth : int, default: 1
            The recursion depth for heatmap calculations, which controls the number of future
            moves considered during evaluation.

        Examples
        --------
        >>> import os
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> len(os.listdir(path=engine.cache_dir))
        31
        """
        super().__init__(board=board, depth=depth)
        self._init_qdb()  # List to store moves made during the game as (fen, move_uci) pairs.

    def update_q_values(self, debug: bool = False) -> None:
        """Back-propagate game outcome through the Q-table.

        Pops all moves from the current board history and adjusts each
        stored Q-value in the database based on the final result
        (win/lose/draw).

        Parameters
        ----------
        debug : bool, default=False
            If True, print diagnostics for each back-step.

        Notes
        -----
        Updates the SQLite Q-table entries for every move in the game.

        Examples
        --------
        >>> from io import StringIO
        >>> from chess import pgn
        >>> from chmengine import CMHMEngine2
        >>> pgn_buffer = StringIO(
        ...    '''
        ...    1. f3 e5 2. g4 Qh4# 0-1
        ...
        ...
        ...    '''
        ... )
        >>> game = pgn.read_game(pgn_buffer)
        >>> board = game.board()
        >>> for move in game.mainline_moves():
        ...     board.push(move)
        >>> engine = CMHMEngine2(board=board)
        >>> engine.fen()
        'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3'
        >>> engine.update_q_values()
        >>> engine.fen()
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        """
        while len(self.board.move_stack) > 0:
            last_move: Move = self.board.move_stack[-1]
            current_q: Optional[float64] = self.get_q_value(board=self.board)
            self.board.pop()
            new_move: Move
            new_score: float64
            new_move, new_score = self.pick_move(debug=debug)  # Call pick_move to back-pop the updated
            if debug:
                print(
                    f"Game Pick & Score: ({last_move.uci()}, {current_q:.2f}) --> "
                    f"New Pick & Score: ({new_move.uci()}, {new_score:.2f})\n"
                )

    def pick_move(
            self,
            *args,
            board: Optional[Board] = None,
            debug: bool = False,
            **kwargs
    ) -> Pick:
        """Select a Pick based on heatmap evaluations and Q-table integration.

        This overridden method combines heatmap evaluations with Q-value updates. It evaluates all
        legal moves by calculating their scores from the perspective of the current player (with positive
        scores indicating favorable moves and negative scores indicating unfavorable moves). The evaluation
        score is always from the mover’s perspective. The method then picks one move at random from the
        top-scoring moves, updates the Q-value for that move, and returns the selected move along with its
        score.

        Parameters
        ----------
        board : Optional[chess.Board], default: None
            Pick from a given board instead of intstance board
        debug : bool, default: False
            Allows for a print call showing the current evals of each move choice during anaylsis.

        Returns
        -------
        Pick
            An unpackable tuple-like containing the chosen move and its associated evaluation score.
            The evaluation score is expressed from the perspective of the player making the move—positive
            values indicate favorable outcomes and negative values indicate unfavorable outcomes.

        Raises
        ------
        ValueError
            If Current Board has no legal moves.

        Examples
        --------
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> move, score = engine.pick_move()
        """
        board = self.board if board is None else board
        current_picks: List[Pick] = self.current_picks_list(board=board)
        # PlayCMHMEngine.play() needs a ValueError to detect game termination.
        if len(current_picks) == 0:
            raise ValueError(f"Current Board has no legal moves: {self.fen(board=board)}")
        current_pick: Pick
        for current_pick in current_picks:
            self._update_current_move_choices_(current_pick=current_pick, board=board)
        # noinspection PyTypeChecker,PydanticTypeChecker
        current_picks = sorted(current_picks, reverse=True)
        # Final pick is a random choice of all moves equal to the highest scoring move
        if debug:
            print(
                f"All {len(current_picks)} moves ranked:",
                format_picks(picks=current_picks)
            )
        chosen_pick: Pick = choice([p for p in current_picks if p == current_picks[0 if board.turn else -1]])
        self.set_q_value(value=chosen_pick.score, board=board)
        return chosen_pick

    # pylint: disable=too-many-arguments
    def _update_current_move_choices_(
            self,
            board: Board,
            current_pick: Pick
    ) -> None:
        """Evaluate a candidate Pick and update the list of best Pick choices.

        This method evaluates a move by pushing it to a new board, calculating the response moves
        from the opponent, and assigning the move a score based on the worst-case opponent response.
        If no responses are available (e.g., checkmate or stalemate), it falls back to Q-table values
        or a static evaluation.

        Scores follow the classical convention:
        - Positive values favor White.
        - Negative values favor Black.

        Parameters
        ----------
        board : chess.Board
            The board to apply the current_pick's move to
        current_pick : Pick
            The Pick being evaluated.

        Returns
        -------
        None
        """
        new_board: Board = self.board_copy_pushed(move=current_pick.move, board=board)
        response_moves: List[Pick]
        response_moves = self._get_or_calculate_responses_(
            new_board=new_board,
            go_deeper=True
        )
        if len(response_moves) == 0:
            initial_q_val: Optional[float64] = self.get_q_value(board=new_board)
            if initial_q_val is None:
                current_pick.score = calculate_better_white_minus_black_score(board=new_board, depth=self.depth)
                self.set_q_value(value=current_pick.score, board=new_board)
            else:
                current_pick.score = initial_q_val
        else:
            current_pick.score = response_moves[0 if new_board.turn else -1].score
            self.set_q_value(value=current_pick.score, board=new_board)

    def _get_or_calculate_responses_(
            self,
            new_board: Board,
            go_deeper: bool
    ) -> List[Pick]:
        """Retrieve the opponent's evaluated response picks.

        This method computes heatmap-based evaluation scores for each legal response move from the new board
        state (after a candidate move is applied). The scores are ordered from worst to best from the perspective
        of the current player.

        Parameters
        ----------
        new_board : chess.Board
            The board state after a candidate move is applied.
        go_deeper : bool
            If True, allows one extra ply of recursive scoring on checks or captures.

        Returns
        -------
        List[Pick]
            A list of response Picks (moves and their evaluation scores.)
        """
        # However, that is not our Final score,
        # the score after finding the best response to our move should be the final score.
        next_picks: List[Pick] = self.current_picks_list(board=new_board)
        next_pick: Pick
        for next_pick in next_picks:
            self._get_or_calc_response_move_scores_(
                next_pick=next_pick,
                new_board=new_board,
                go_deeper=go_deeper
            )
        # noinspection PyTypeChecker,PydanticTypeChecker
        return sorted(next_picks, reverse=True)

    def _get_or_calc_response_move_scores_(
            self,
            next_pick: Pick,
            new_board: Board,
            go_deeper: bool
    ) -> None:
        """Evaluate one opponent response move and update the next_pick score.

        This method simulates `next_pick.move` on `new_board`, then:

            1. **Deepens** one ply (only on checks or captures) if `go_deeper` is True and there are legal replies.
            2. Otherwise, **fetches** the cached Q-value for that position if available.
            3. Otherwise, **falls back** to a static heatmap-based evaluation.

        The score is always from White-positive/Black-negative perspective.

        Parameters
        ----------
        next_pick : Pick
            The opponent’s Pick to simulate and evaluate.
        new_board : chess.Board
            Position after the candidate Pick was applied; it’s now the opponent’s turn.
        go_deeper : bool
            If True, allows one extra ply of recursive scoring on checks or captures.
        """
        next_board: Board = self.board_copy_pushed(move=next_pick.move, board=new_board)
        if go_deeper and (
                next_board.is_check() or new_board.piece_at(next_pick.move.to_square)
        ) and next_board.legal_moves.count() > 0:
            next_response_moves: List[Pick] = self._get_or_calculate_responses_(
                new_board=next_board,
                go_deeper=False
            )
            next_pick.score = next_response_moves[-1 if next_board.turn else 0].score
            self.set_q_value(value=next_pick.score, board=next_board)
        else:
            next_q_val: Optional[float64] = self.get_q_value(board=next_board)
            if next_q_val is not None:
                next_pick.score = next_q_val
            else:
                next_pick.score = calculate_better_white_minus_black_score(board=next_board, depth=self.depth)
                self.set_q_value(value=next_pick.score, board=next_board)
