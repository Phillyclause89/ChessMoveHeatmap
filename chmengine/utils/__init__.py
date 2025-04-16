"""utilities for the engines"""
from typing import List, Optional, Tuple
from _bisect import bisect_left

from chess import Move
from numpy import float64
from numpy.typing import NDArray


def format_moves(
        moves: List[Tuple[Optional[Move], Optional[float64]]]
) -> List[Optional[Tuple[str, str]]]:
    """Generate a formatted list of moves and their scores for display.

    This static method converts a list of move-score tuples into a list of tuples containing the move
    in UCI format and the score formatted as a string with two decimal places.

    Parameters
    ----------
    moves : List[Tuple[Optional[Move], Optional[numpy.float64]]]
        The list of moves with their evaluation scores.

    Returns
    -------
    List[Optional[Tuple[str, str]]]
        A list of formatted move representations (UCI, score) suitable for printing or logging.
    """
    return [(m.uci(), f"{s:.2f}") for m, s in moves if m is not None]


def calculate_score(
        current_index: int,
        new_heatmap_transposed: NDArray[float64],
        new_current_king_box: List[int],
        new_other_king_box: List[int]
) -> float64:
    """Calculate the evaluation score for a move based on heatmap data and king safety.

    This static method computes the move score as the sum of two components:
    - The difference between the current player's total heatmap intensity and the opponent's.
    - A weighted difference based on the intensity values within the "king box" areas.
    The final score reflects the overall benefit of the move from the perspective of the current player.

    Parameters
    ----------
    current_index : int
        The index for the current player's heatmap data.
    new_heatmap_transposed : NDArray[numpy.float64]
        The transposed heatmap data array.
    new_current_king_box : List[int]
        The list of squares surrounding the current king.
    new_other_king_box : List[int]
        The list of squares surrounding the opponent's king.

    Returns
    -------
    numpy.float64
        The computed evaluation score for the move.
    """
    # Calculating score at this level is only needed in corner-case scenarios
    # where every possible move results in game termination.
    # score is initially, the delta of the sums of each player's heatmap.data values.
    other_index: int = int(not current_index)
    initial_move_score: float64 = sum(
        new_heatmap_transposed[current_index]
    ) - sum(
        new_heatmap_transposed[other_index]
    )
    # king box score adds weights to the scores of squares around the kings.
    initial_king_box_score: float64 = sum(new_heatmap_transposed[current_index][new_other_king_box])
    initial_king_box_score -= sum(
        new_heatmap_transposed[other_index][new_current_king_box]
    )
    # Final score is the agg of both above.
    return float64(initial_move_score + initial_king_box_score)


def insert_ordered_worst_to_best(
        ordered_moves: List[Tuple[Move, float64]],
        move: Move,
        score: float64
) -> None:
    """Insert a move and its score into an ordered list (from worst to best).

    This static method uses a binary insertion (via bisect_left) to insert the new move into
    the list such that the list remains ordered from the lowest score to highest, suitable for response
    move evaluations from the perspective of the current player.

    Parameters
    ----------
    ordered_moves : List[Tuple[chess.Move, numpy.float64]]
        The current list of moves and their scores, ordered from worst to best.
    move : chess.Move
        The move to be inserted.
    score : numpy.float64
        The evaluation score for the move.
    """
    # response moves are inserted to form worst scores to best order (perspective of current player)
    ordered_index: int = bisect_left([x[1] for x in ordered_moves], score)
    ordered_moves.insert(ordered_index, (move, score))


def insert_ordered_best_to_worst(
        ordered_moves: List[Tuple[Move, float64]],
        move: Move,
        score: float64
) -> None:
    """Insert a move and its score into an ordered list (from best to worst).

    This static method uses a binary insertion (via bisect_left) to insert the new move into
    the list such that the list remains ordered from the highest score to lowest.

    Parameters
    ----------
    ordered_moves : List[Tuple[chess.Move, numpy.float64]]
        The current list of moves and their scores, ordered from best to worst.
    move : chess.Move
        The move to be inserted.
    score : numpy.float64
        The evaluation score for the move.
    """
    # current moves are inserted into our moves list in order of best scores to worst
    ordered_index: int = bisect_left([-x[1] for x in ordered_moves], -score)
    ordered_moves.insert(ordered_index, (move, score))


def pieces_count_from_fen(fen: str) -> int:
    """Get pieces count from fen string

    Parameters
    ----------
    fen : str

    Returns
    -------
    int
    """
    _c: str
    return len([_c for _c in fen.split()[0] if _c.isalpha()])


def insert_choice_into_current_moves(
        current_move_choices_ordered: List[Tuple[Optional[Move], Optional[float64]]],
        current_move: Move,
        final_move_score: float64
) -> List[Tuple[Move, float64]]:
    """

    Parameters
    ----------
    current_move_choices_ordered : List[Tuple[Optional[Move], Optional[float64]]]
    current_move : Move
    final_move_score : float64

    Returns
    -------
    List[Tuple[Move, float64]]
    """
    if current_move_choices_ordered[0][0] is None:
        current_move_choices_ordered = [(current_move, final_move_score)]
    else:
        insert_ordered_best_to_worst(
            ordered_moves=current_move_choices_ordered, move=current_move, score=final_move_score
        )
    return current_move_choices_ordered
