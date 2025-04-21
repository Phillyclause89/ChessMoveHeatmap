"""utilities for the engines"""
from typing import List, Optional, Tuple
from _bisect import bisect_left

import chess
from chess import Board, Move, Outcome, square_distance
from numpy import float64
from numpy.typing import NDArray

from chmutils import calculate_chess_move_heatmap_with_better_discount
from heatmaps import ChessMoveHeatmap


def format_moves(
        moves: List[Tuple[Optional[Move], Optional[float64]]]
) -> List[Optional[Tuple[str, str]]]:
    """Generate a formatted list of moves and their scores for display.

    This static method converts a list of move-score tuples into a list of tuples containing the move
    in UCI format and the score formatted as a string with two decimal places.

    Parameters
    ----------
    moves : list
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


def is_draw(winner: Optional[bool]) -> bool:
    """

    Parameters
    ----------
    winner : Optional[bool]

    Returns
    -------
    bool

    Examples
    --------
    >>> from chmengine.utils import is_draw
    >>> from chess import Board
    >>> mate_board = Board('8/2p2p2/4p3/2k5/8/6q1/2K5/1r1q4 w - - 2 59')
    >>> is_draw(mate_board.outcome(claim_draw=True).winner)
    False
    >>> draw_board = Board('6R1/7p/2p2p1k/p1P2Q2/P7/6K1/5P2/8 b - - 0 52')
    >>> is_draw(draw_board.outcome(claim_draw=True).winner)
    True
    """
    return winner is None


def calculate_white_minus_black_score(
        board: Board,
        depth: int,
) -> float64:
    """Calculates a white minus black score (more traditional scoring perspective...)

    Parameters
    ----------
    board : chess.Board
    depth : int

    Returns
    -------
    numpy.float64

    Examples
    --------
    >>> from chmengine.utils import is_draw
    >>> from chess import Board, Move
    >>> default_board, d = Board(), 1
    >>> calculate_white_minus_black_score(board=default_board, depth=d)
    0.0
    >>> default_board.push(Move.from_uci('e2e4'))
    >>> calculate_white_minus_black_score(board=default_board, depth=d)
    10.0
    >>> default_board.push(Move.from_uci('e7e5'))
    >>> calculate_white_minus_black_score(board=default_board, depth=d)
    0.20689655172414234
    >>> default_board.push(Move.from_uci('g1f3'))
    >>> calculate_white_minus_black_score(board=default_board, depth=d)
    -2.1379310344827616
    >>> default_board.push(Move.from_uci('b8c6'))
    >>> calculate_white_minus_black_score(board=default_board, depth=d)
    -3.925925925925924
    >>> default_board.push(Move.from_uci('f1b5'))
    >>> calculate_white_minus_black_score(board=default_board, depth=d)
    2.133333333333335
    >>> mate_board = Board('8/2p2p2/4p3/2k5/8/6q1/2K5/1r1q4 w - - 2 59')
    >>> calculate_white_minus_black_score(board=mate_board, depth=d)
    -1024.0
    >>> draw_board = Board('6R1/7p/2p2p1k/p1P2Q2/P7/6K1/5P2/8 b - - 0 52')
    >>> calculate_white_minus_black_score(board=draw_board, depth=d)
    0.0
    """
    outcome: Optional[Outcome] = board.outcome(claim_draw=True)
    is_terminated: bool = outcome is not None
    if is_terminated and is_draw(outcome.winner):
        return float64(0)
    if is_terminated:
        return checkmate_score(board, depth)
    heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap_with_better_discount(board=board, depth=depth)
    heatmap_transposed: NDArray[float64] = heatmap.data.transpose()
    transposed_white: NDArray[float64] = heatmap_transposed[0]
    transposed_black: NDArray[float64] = heatmap_transposed[1]
    general_move_score: float64 = sum(transposed_white) - sum(transposed_black)
    king_box_white: List[int]
    king_box_black: List[int]
    king_box_white, king_box_black = get_white_and_black_king_boxes(board=board)
    king_box_score: float64 = sum(transposed_white[king_box_black]) - sum(transposed_black[king_box_white])
    return float64(general_move_score + king_box_score)


def checkmate_score(board: Board, depth: int) -> float64:
    """checkmate_score from board and depth

    Parameters
    ----------
    board : chess.Board
    depth : int

    Returns
    -------
    numpy.float64

    Examples
    --------
    >>> from chmengine.utils import is_draw
    >>> from chess import Board
    >>> blk_win_board = Board('8/2p2p2/4p3/2k5/8/6q1/2K5/1r1q4 w - - 2 59')
    >>> checkmate_score(board=blk_win_board, depth=1)
    -1024.0
    """
    mate_score_abs = float64(pieces_count_from_fen(fen=board.fen()) * (depth + 1) * 64)
    return mate_score_abs if not board.turn else float64(-mate_score_abs)


def get_white_and_black_king_boxes(board: Board) -> Tuple[List[int], List[int]]:
    """Compute the bounding boxes for the kings on the board.

    For both the current and opponent kings, this method calculates a "box" (a list of square
    indices) representing the king's immediate surroundings.

    Parameters
    ----------
    board : chess.Board
        The board to use.

    Returns
    -------
    Tuple[List[int], List[int]]
        A tuple containing two lists: the first is the box for the current king, and the second is
        the box for the opponent king. (white_king_box, black_king_box)

    Examples
    --------
    >>> from chmengine.utils import is_draw
    >>> from chess import Board
    >>> white_kb, black_kb = get_white_and_black_king_boxes(board=Board())
    >>> sorted(white_kb), sorted(black_kb)
    ([3, 4, 5, 11, 12, 13], [51, 52, 53, 59, 60, 61])
    """
    white_king_square: int = board.king(True)
    black_king_square: int = board.king(False)
    white_king_box: List[int] = [white_king_square]
    black_king_box: List[int] = [black_king_square]
    long: int
    for long in (-1, 0, 1):
        lat: int
        for lat in (-8, 0, +8):
            wks_box_id: int = white_king_square + long + lat
            bks_box_id: int = black_king_square + long + lat
            if is_valid_king_box_square(wks_box_id, white_king_square):
                white_king_box.append(wks_box_id)
            if is_valid_king_box_square(bks_box_id, black_king_square):
                black_king_box.append(bks_box_id)
    return white_king_box, black_king_box


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
    if score is None:
        raise ValueError()
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

    Examples
    --------
    >>> from chmengine.utils import is_draw
    >>> from chess import Board
    >>> mate_board = Board('8/2p2p2/4p3/2k5/8/6q1/2K5/1r1q4 w - - 2 59')
    >>> pieces_count_from_fen(fen=mate_board.fen())
    8
    >>> pieces_count_from_fen(Board().fen())
    32
    """
    _c: str
    return len([_c for _c in fen.split()[0] if _c.isalpha()])


def insert_choice_into_current_moves(
        choices_ordered_best_to_worst: List[Tuple[Optional[Move], Optional[float64]]],
        move: Move,
        score: float64
) -> List[Tuple[Move, float64]]:
    """inserts ordered best to worst...

    Parameters
    ----------
    choices_ordered_best_to_worst : List[Tuple[Optional[Move], Optional[float64]]]
    move : Move
    score : float64

    Returns
    -------
    List[Tuple[Move, float64]]
    """
    if choices_ordered_best_to_worst[0][0] is None:
        choices_ordered_best_to_worst = [(move, score)]
    else:
        insert_ordered_best_to_worst(
            ordered_moves=choices_ordered_best_to_worst, move=move, score=score
        )
    return choices_ordered_best_to_worst


def insert_choice_into_response_moves(
        choices_ordered_worst_to_best: List[Tuple[Optional[Move], Optional[float64]]],
        move: Move,
        score: float64
) -> List[Tuple[Move, float64]]:
    """inserts ordered worst to best..

    Parameters
    ----------
    choices_ordered_worst_to_best : List[Tuple[Optional[Move], Optional[float64]]]
    move : Move
    score : float64

    Returns
    -------
    List[Tuple[Move, float64]]
    """
    if choices_ordered_worst_to_best[0][0] is None:
        choices_ordered_worst_to_best = [(move, score)]
    else:
        insert_ordered_worst_to_best(
            ordered_moves=choices_ordered_worst_to_best, move=move, score=score
        )
    return choices_ordered_worst_to_best


def is_valid_king_box_square(square_id: int, king_square: int) -> bool:
    """Determine if a square is a valid part of a king's bounding box.

    A square is valid if it is within the board limits, adjacent to the king (distance of 1),
    and either empty or occupied by a piece of the same color as the king.

    Parameters
    ----------
    square_id : int
        The index of the square to check.
    king_square : int
        The square where the king is located.

    Returns
    -------
    bool
        True if the square is valid for inclusion in the king's box; otherwise, False.

    """
    return 0 <= square_id <= 63 and square_distance(king_square, square_id) == 1


def null_target_moves(
        number: int = 6
) -> Tuple[List[Tuple[Optional[Move], Optional[float64]]], ...]:
    """Initialize a tuple of target move lists with null entries.

    This helper method creates a tuple containing 'number' lists, each initialized with a single
    tuple (None, None). These lists serve as starting placeholders for candidate moves and their scores.

    Parameters
    ----------
    number : int, default: 6
        The number of target move lists to create.

    Returns
    -------
    Tuple[List[Tuple[Optional[chess.Move], Optional[numpy.float64]]], ...]
        A tuple of lists, each initially containing one tuple (None, None).
    """
    return tuple([(None, None)] for _ in range(number))
