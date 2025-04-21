"""Cmhmey Jr."""
from random import choice
from typing import List, Optional, Tuple

from chess import Board, Move
from numpy import float64

from chmengine.engines.cmhmey1 import CMHMEngine
from chmengine.engines.quartney import Quartney
from chmengine.utils import (
    calculate_white_minus_black_score,
    format_moves,
    insert_choice_into_current_moves,
    null_target_moves
)


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
        """Update the Q-table values after game termination.

        This method back-propagates the outcome of the game by iteratively popping moves off the
        board stack and adjusting their Q-values based on the game outcome. For draws, scores are
        converged toward zero; for wins/losses, scores are bumped or penalized accordingly.

        Parameters
        ----------
        debug : bool

        Side Effects
        ------------
        Updates the Q-values in the Q-table database for each board fen in the move history.

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
            fen: str = self.fen()
            current_q: Optional[float64] = self.get_q_value(fen=fen, board=self.board)
            self.board.pop()
            new_move: Move
            new_score: float64
            new_move, new_score = self.pick_move(debug=debug)  # Call pick_move to back-pop the updated
            if debug:
                print(
                    f"Game Pick & Score: ({last_move.uci()}, {current_q}) --> "
                    f"New Pick & Score: ({new_move.uci()}, {new_score})\n"
                )

    def pick_move(
            self,
            pick_by: str = "",
            board: Optional[Board] = None,
            debug: bool = False
    ) -> Tuple[Move, float64]:
        """Select a move based on heatmap evaluations and Q-table integration.

        This overridden method combines heatmap evaluations with Q-value updates. It evaluates all
        legal moves by calculating their scores from the perspective of the current player (with positive
        scores indicating favorable moves and negative scores indicating unfavorable moves). The evaluation
        score is always from the mover’s perspective. The method then picks one move at random from the
        top-scoring moves, updates the Q-value for that move, and returns the selected move along with its
        score.

        Parameters
        ----------
        pick_by : str, default: ""
            Legecy param from parent class pick_move method. Args are ignored by this classe's overide.
        board : Optional[chess.Board], default: None
            Pick from a given board instead of intstance board
        debug : bool, default: False
            Allows for a print call showing the current evals of each move choice during anaylsis.

        Returns
        -------
        Tuple[chess.Move, numpy.float64]
            A tuple containing the chosen move and its associated evaluation score. The evaluation score is
            expressed from the perspective of the player making the move—positive values indicate favorable
            outcomes and negative values indicate unfavorable outcomes.

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
        current_moves: List[Move] = self.current_moves_list(board=board)
        # PlayCMHMEngine.play() needs a ValueError to detect game termination.
        if len(current_moves) == 0:
            raise ValueError(f"Current Board has no legal moves: {self.fen(board=board)}")
        # moves will be current moves ordered by engine's score best to worst (from white's perspective)
        current_move_choices_ordered: List[Tuple[Optional[Move], Optional[float64]]]
        current_move_choices_ordered, = null_target_moves(number=1)
        current_move: Move
        for current_move in current_moves:
            current_move_choices_ordered = self._update_current_move_choices_ordered_(
                current_move_choices_ordered=current_move_choices_ordered,
                current_move=current_move,
                board=board
            )
        # Final pick is a random choice of all moves equal to the highest scoring move
        if debug:
            print(
                f"All {len(current_move_choices_ordered)} moves ranked:",
                format_moves(moves=current_move_choices_ordered)
            )
        pick_score: float64 = current_move_choices_ordered[0][1] if board.turn else current_move_choices_ordered[-1][1]
        chosen_move: Move
        chosen_q: float64
        chosen_move, chosen_q = choice([(m, s) for m, s in current_move_choices_ordered if s == pick_score])
        self.set_q_value(value=float64(chosen_q), board=board)
        return chosen_move, chosen_q

    def _update_current_move_choices_ordered_(
            self,
            current_move_choices_ordered: List[Tuple[Optional[Move], Optional[float64]]],
            current_move: Move,
            board: Board
    ) -> List[Tuple[Optional[Move], Optional[float64]]]:
        new_board: Board = self.board_copy_pushed(move=current_move, board=board)
        # new_fen represents the boards for the only q-values that we will make updates to
        new_fen: str = new_board.fen()
        # No longer look at q-score at this level, only back prop q's into the calculation
        current_move_choices_ordered = self._update_current_move_choices_(
            current_move_choices_ordered=current_move_choices_ordered, new_board=new_board,
            current_move=current_move, new_fen=new_fen
        )
        return current_move_choices_ordered

    # pylint: disable=too-many-arguments
    def _update_current_move_choices_(
            self,
            current_move_choices_ordered: List[Tuple[Optional[Move], Optional[float64]]],
            new_board: Board,
            current_move: Move,
            new_fen: str
    ) -> List[Tuple[Move, float64]]:
        """Update the ordered list of candidate moves based on a newly calculated heatmap score.

        This method calculates an initial move score using the transposed heatmap data and king box areas,
        then retrieves response moves to further refine the score. The final score is determined as the worst
        (response) outcome for the current player. The candidate list is updated (ordered best to worst) accordingly.

        Parameters
        ----------
        current_move_choices_ordered : List[Tuple[Optional[Move], Optional[numpy.float64]]]
            The current ordered list of candidate moves and their scores.
        new_board : chess.Board
            The board state after the candidate move is applied.
        current_move : chess.Move
            The candidate move being evaluated.
        new_fen : str
            The FEN string representing the new board state.

        Returns
        -------
        List[Tuple[Move, numpy.float64]]
            The updated ordered list of candidate moves with their evaluation scores.
        """
        response_moves: List[Tuple[Optional[Move], Optional[float64]]]
        response_moves = self._get_or_calculate_responses_(
            new_board=new_board,
            go_deeper=True
        )
        # Once all responses to a move reviewed, final move score is the worst outcome to current player.
        best_response_score: Optional[float64] = response_moves[0][1] if new_board.turn else response_moves[-1][1]
        if best_response_score is None:
            initial_q_val: Optional[float64] = self.get_q_value(fen=new_fen, board=new_board)
            if initial_q_val is None:
                final_move_score: float64 = calculate_white_minus_black_score(board=new_board, depth=self.depth)
            else:
                final_move_score = initial_q_val
        else:
            final_move_score: float64 = best_response_score
        self.set_q_value(value=final_move_score, fen=new_fen, board=new_board)
        return insert_choice_into_current_moves(
            choices_ordered_best_to_worst=current_move_choices_ordered,
            move=current_move,
            score=final_move_score
        )

    def _get_or_calculate_responses_(
            self,
            new_board: Board,
            go_deeper: bool
    ) -> List[Tuple[Optional[Move], Optional[float64]]]:
        """Retrieve the opponent's response moves and evaluate them.

        This method computes heatmap-based evaluation scores for each legal response move from the new board
        state (after a candidate move is applied). The scores are ordered from worst to best from the perspective
        of the current player.

        Parameters
        ----------
        new_board : chess.Board
            The board state after a candidate move is applied.
        go_deeper : bool

        Returns
        -------
        List[Tuple[Optional[Move], Optional[numpy.float64]]]
            A list of response moves and their evaluation scores.
        """
        # However, that is not our Final score,
        # the score after finding the best response to our move should be the final score.
        next_moves: List[Move] = self.current_moves_list(board=new_board)
        response_moves: List[Tuple[Optional[Move], Optional[float64]]]
        response_moves, = null_target_moves(number=1)
        next_move: Move
        for next_move in next_moves:
            response_moves = self._get_or_calc_response_move_scores_(
                next_move=next_move,
                response_moves=response_moves,
                new_board=new_board,
                go_deeper=go_deeper
            )
        return response_moves

    def _get_or_calc_response_move_scores_(
            self,
            next_move: Move,
            response_moves: List[Tuple[Optional[Move], Optional[float64]]],
            new_board: Board,
            go_deeper: bool
    ) -> List[Tuple[Move, float64]]:
        """Calculate the evaluation score for a given opponent response move.

        This method computes the Q-score for the next move in the perspective of the current player.
        If no Q-value is found in the database, it calculates a new score using the heatmap data.
        The resulting score is then inserted into the response moves list in order.

        Parameters
        ----------
        next_move : chess.Move
            The opponent's move to evaluate.
        response_moves : List[Tuple[Optional[Move], Optional[numpy.float64]]]
            The current list of evaluated response moves.
        new_board : chess.Board
            The board state after the candidate move is applied.

        Returns
        -------
        List[Tuple[chess.Move, numpy.float64]]
            The updated list of response moves with their evaluation scores.
        """
        # next_move score calculations stay in the perspective of current player
        next_board: Board = self.board_copy_pushed(move=next_move, board=new_board)
        next_fen: str = next_board.fen()
        next_q_val: Optional[float64] = self.get_q_value(fen=next_fen, board=next_board)
        if go_deeper and (
                next_board.is_check() or new_board.piece_at(next_move.to_square)
        ) and next_board.legal_moves.count() > 0:
            next_response_moves = self._get_or_calculate_responses_(
                new_board=next_board,
                go_deeper=False
            )
            next_move_score: Optional[float64]
            next_move_score = next_response_moves[-1][1] if next_board.turn else next_response_moves[0][1]
        elif next_q_val is not None:
            next_move_score = next_q_val
        else:
            next_move_score: float64 = calculate_white_minus_black_score(board=next_board, depth=self.depth)
        self.set_q_value(value=next_move_score, fen=next_fen, board=next_board)
        response_moves = insert_choice_into_current_moves(
            choices_ordered_best_to_worst=response_moves, move=next_move, score=next_move_score
        )
        return response_moves
