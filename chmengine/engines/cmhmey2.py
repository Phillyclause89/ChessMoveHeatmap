"""Cmhmey Jr."""
from random import choice
from typing import List, Optional, Tuple

from chess import Board, Move, Outcome
from numpy import float64
from numpy.typing import NDArray

from chmengine.engines.cmhmey1 import CMHMEngine
from chmengine.engines.quartney import Quartney
from chmengine.utils import (
    insert_choice_into_current_moves,
    calculate_score,
    format_moves,
    insert_ordered_worst_to_best,
)
from chmutils import calculate_chess_move_heatmap_with_better_discount
from heatmaps import ChessMoveHeatmap


class CMHMEngine2(CMHMEngine, Quartney):
    """The Baby of CMHMEngine and Quartney"""

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
            new_q: Optional[float64] = self.get_q_value(fen=fen)
            if debug:
                print(
                    f"Game Pick & Score: ({last_move.uci()}, {current_q}) --> "
                    f"Game Pick & New Score: ({last_move.uci()}, {new_q}) --> "
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
        # Debated if I want to do any updates to the q-value for self.board.fen() if there is one...
        # If I did update it then it would be the negative of the q-value for the new_board
        # No Don't do this! Why not? Because we haven't explored all other branches that board could have lead to
        current_moves: List[Move] = self.current_moves_list(board=board)
        # PlayCMHMEngine.play() needs a ValueError to detect game termination.
        if len(current_moves) == 0:
            raise ValueError(f"Current Board has no legal moves: {self.fen(board=board)}")
        # Index of heatmap colors is opposite the chess lib's color mapping
        current_index: int = self.current_player_heatmap_index(board=board)
        # moves will be current moves ordered by engine's score best to worst
        current_move_choices_ordered: List[Tuple[Optional[Move], Optional[float64]]]
        current_move_choices_ordered, = self.null_target_moves(number=1)
        current_move: Move
        for current_move in current_moves:
            current_move_choices_ordered = self._update_current_move_choices_ordered_(
                current_move_choices_ordered=current_move_choices_ordered,
                current_move=current_move, current_index=current_index,
                board=board
            )
        # Final pick is a random choice of all moves equal to the highest scoring move
        if debug:
            print(
                f"All {len(current_move_choices_ordered)} moves ranked:",
                format_moves(moves=current_move_choices_ordered)
            )
        picks: List[Tuple[Move, float64]] = [
            (m, s) for m, s in current_move_choices_ordered if s == current_move_choices_ordered[0][1]
        ]
        chosen_move: Move
        chosen_q: float64
        chosen_move, chosen_q = choice(picks)
        # The initial board state is a choice for the other player, thus it's q is the negative of the picked q.
        self.set_q_value(value=float64(-chosen_q), board=board)
        return chosen_move, chosen_q

    def _update_current_move_choices_ordered_(
            self,
            current_move_choices_ordered: List[Tuple[Optional[Move], Optional[float64]]],
            current_move: Move,
            current_index: int,
            board: Board
    ) -> List[Tuple[Optional[Move], Optional[float64]]]:
        # Phil, I know you keep thinking you don't need to calculate a score here,
        # but you do sometimes (keep that in mind.)
        new_board: Board = self.board_copy_pushed(move=current_move, board=board)
        # new_fen represents the boards for the only q-values that we will make updates to
        new_fen: str = new_board.fen()
        # No longer look at q-score at this level, only back prop q's into the calculation
        new_current_king_box: List[int]
        new_other_king_box: List[int]
        new_current_king_box, new_other_king_box = self.get_king_boxes(board=new_board)
        current_move_choices_ordered = self._update_current_move_choices_(
            current_move_choices_ordered=current_move_choices_ordered, new_board=new_board,
            current_move=current_move, current_index=current_index, new_current_king_box=new_current_king_box,
            new_other_king_box=new_other_king_box, new_fen=new_fen
        )
        return current_move_choices_ordered

    # pylint: disable=too-many-arguments
    def _update_current_move_choices_(
            self,
            current_move_choices_ordered: List[Tuple[Optional[Move], Optional[float64]]],
            new_board: Board,
            current_move: Move,
            current_index: int,
            new_current_king_box: List[int],
            new_other_king_box: List[int],
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
        current_index : int
            The index corresponding to the current player's heatmap data.
        new_current_king_box : List[int]
            A list of squares representing the area around the current king.
        new_other_king_box : List[int]
            A list of squares representing the area around the opponent's king.
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
            current_index=current_index
        )
        # Once all responses to a move reviewed, final move score is the worst outcome to current player.
        best_response_score: Optional[float64] = response_moves[0][1]
        if best_response_score is None:
            initial_q_val: Optional[float64] = self.get_q_value(fen=new_fen, board=new_board)
            if initial_q_val is None:
                new_outcome: Optional[Outcome] = new_board.outcome(claim_draw=True)
                if new_outcome is not None:
                    is_mate: bool = new_outcome.winner is not None
                    new_heatmap: ChessMoveHeatmap = ChessMoveHeatmap()
                else:
                    is_mate = False
                    # It was fun building up a giant db of heatmaps, but we saw how that turned out in training
                    new_heatmap = calculate_chess_move_heatmap_with_better_discount(
                        board=new_board, depth=self.depth
                    )
                new_heatmap_transposed: NDArray[float64] = new_heatmap.data.transpose()
                # I wanted to rely on the heatmap as much as possible, but any game termination state win or draw
                # results in a zeros heatmap.
                if is_mate:
                    self._update_heatmap_transposed_with_mate_values_(
                        heatmap_transposed=new_heatmap_transposed,
                        player_index=current_index,
                        board=new_board
                    )
                final_move_score: float64 = calculate_score(
                    current_index=current_index, new_heatmap_transposed=new_heatmap_transposed,
                    new_current_king_box=new_current_king_box, new_other_king_box=new_other_king_box
                )
            else:
                final_move_score = initial_q_val
        else:
            final_move_score: float64 = best_response_score

        self.set_q_value(value=final_move_score, fen=new_fen, board=new_board)
        return insert_choice_into_current_moves(
            current_move_choices_ordered=current_move_choices_ordered,
            current_move=current_move,
            final_move_score=final_move_score
        )

    def _get_or_calculate_responses_(
            self,
            new_board: Board,
            current_index: int
    ) -> List[Tuple[Optional[Move], Optional[float64]]]:
        """Retrieve the opponent's response moves and evaluate them.

        This method computes heatmap-based evaluation scores for each legal response move from the new board
        state (after a candidate move is applied). The scores are ordered from worst to best from the perspective
        of the current player.

        Parameters
        ----------
        new_board : chess.Board
            The board state after a candidate move is applied.
        current_index : int
            The index corresponding to the current player in the heatmap.

        Returns
        -------
        List[Tuple[Optional[Move], Optional[numpy.float64]]]
            A list of response moves and their evaluation scores.
        """
        # However, that is not our Final score,
        # the score after finding the best response to our move should be the final score.
        next_moves: List[Move] = self.current_moves_list(board=new_board)
        response_moves: List[Tuple[Optional[Move], Optional[float64]]]
        response_moves, = self.null_target_moves(number=1)
        next_move: Move
        for next_move in next_moves:
            response_moves = self._get_or_calc_response_move_scores_(
                next_move=next_move, response_moves=response_moves, new_board=new_board, current_index=current_index
            )
        return response_moves

    def _get_or_calc_response_move_scores_(
            self,
            next_move: Move,
            response_moves: List[Tuple[Optional[Move], Optional[float64]]],
            new_board: Board,
            current_index: int
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
        current_index : int
            The index for the current player's data in the heatmap.

        Returns
        -------
        List[Tuple[chess.Move, numpy.float64]]
            The updated list of response moves with their evaluation scores.
        """
        # next_move score calculations stay in the perspective of current player
        next_board: Board = self.board_copy_pushed(move=next_move, board=new_board)
        next_fen: str = next_board.fen()
        # next_q_val should be the negative of the fetched q, but can't do that yet, must check for None first
        next_q_val: Optional[float64] = self.get_q_value(fen=next_fen, board=next_board)
        next_outcome: Optional[Outcome] = next_board.outcome(claim_draw=True)
        not_draw: bool = next_outcome is None or next_outcome.winner is not None
        null_q: bool = next_q_val is None
        if null_q and not_draw:
            next_move_score: float64 = self._calculate_next_move_score_(
                next_board=next_board, current_index=current_index
            )
            set_value: float64 = float64(-next_move_score) if (
                    self.current_player_heatmap_index(board=next_board) == self.current_player_heatmap_index()
            ) else float64(next_move_score)
            self.set_q_value(value=set_value, fen=next_fen, board=next_board)
        elif null_q:
            next_move_score: float64 = float64(0.0)
            self.set_q_value(value=next_move_score, fen=next_fen, board=next_board)
        else:
            # Here is where we can safely make next_q_val negative to match the current player's perspective
            next_move_score = float64(-next_q_val) if (  # pylint: disable=invalid-unary-operand-type
                    self.current_player_heatmap_index(board=next_board) == self.current_player_heatmap_index()
            ) else float64(next_q_val)
        if response_moves[0][0] is None:
            response_moves = [(next_move, next_move_score)]
        else:
            insert_ordered_worst_to_best(response_moves, next_move, next_move_score)
        return response_moves

    def _calculate_next_move_score_(
            self,
            next_board: Board,
            current_index: int
    ) -> float64:
        """Calculate the evaluation score for a potential next move.

        This method computes the score for a board state resulting from an opponent's move. It uses
        heatmap data (including a transposed heatmap), king box calculations, and the provided multiplier
        to determine the move's evaluation from the perspective of the current player.

        Parameters
        ----------
        next_board : chess.Board
            The board state after the opponent's move.
        current_index : int
            The index corresponding to the current player in the heatmap.

        Returns
        -------
        numpy.float64
            The evaluation score for the next move.
        """
        next_current_king_box: List[int]
        next_other_king_box: List[int]
        next_current_king_box, next_other_king_box = self.get_king_boxes(board=next_board)
        next_heatmap: ChessMoveHeatmap
        next_heatmap = calculate_chess_move_heatmap_with_better_discount(
            board=next_board, depth=self.depth
        )
        next_heatmap_transposed: NDArray[float64] = next_heatmap.data.transpose()
        if next_board.is_checkmate():
            next_board_copy = next_board.copy()
            next_board_copy.pop()
            # No early exit here as this is a bad is_checkmate result :(
            self._update_heatmap_transposed_with_mate_values_(
                heatmap_transposed=next_heatmap_transposed,
                player_index=self.current_player_heatmap_index(board=next_board_copy),
                board=next_board
            )
        elif next_board.is_check():
            # It will be interesting if we hit inf recursion here, that would be a position of inf counter checks?
            check_responses = self._get_or_calculate_responses_(
                new_board=next_board,
                current_index=current_index
            )
            if check_responses[-1][1] is not None:
                return check_responses[-1][1]
        next_move_score: float64 = calculate_score(
            current_index=current_index, new_heatmap_transposed=next_heatmap_transposed,
            new_current_king_box=next_current_king_box, new_other_king_box=next_other_king_box
        )
        return next_move_score

    def _update_heatmap_transposed_with_mate_values_(
            self,
            heatmap_transposed: NDArray[float64],
            player_index: int,
            board: Board
    ) -> None:
        """Update the transposed heatmap data with mate values for checkmate scenarios.

        This method modifies the heatmap in place to reflect a checkmate situation. The heatmap
        values for the given player index are set to a high value calculated based on the number of
        pieces remaining and the engine's depth, ensuring that checkmate positions receive a score
        that surpasses typical heatmap evaluations.

        Parameters
        ----------
        heatmap_transposed : NDArray[numpy.float64]
            The transposed heatmap data array to be updated.
        player_index : int
            The index (current or opponent) for which to update the mate value.
        board : chess.Board
            The board state where checkmate has been detected.
        """
        heatmap_transposed[player_index] = self.get_mate_value(board)

    def get_mate_value(self, board: Board) -> float64:
        """Gets checkmate value for a square

        Parameters
        ----------
        board : chess.Board

        Returns
        -------
        numpy.float64

        Examples
        --------
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.get_mate_value(board=engine.board)
        64.0
        """
        return float64(self.pieces_count(board=board) * (self.depth + 1))
