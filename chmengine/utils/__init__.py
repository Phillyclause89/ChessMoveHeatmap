"""Utilities for engine evaluation and scoring logic."""
from _bisect import bisect_left
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from chess import Board, Move, Outcome, Piece, pgn, square_distance
from numpy import float64
from numpy.typing import NDArray

from chmengine.utils import pick
from chmengine.utils.pick import Pick
from chmutils import calculate_chess_move_heatmap_with_better_discount
from heatmaps import ChessMoveHeatmap

# Python 3.8+ has bit_count(); otherwise count the '1's in the binary repr
try:
    _bit_count: Callable[[int], int] = int.bit_count
except AttributeError:
    def _bit_count(occ: int) -> int:
        return bin(occ).count('1')

__all__ = [
    # Mods
    'pick',
    # Classes
    'Pick',
    # Functions
    'format_moves',
    'calculate_score',
    'is_draw',
    'calculate_white_minus_black_score',
    'calculate_better_white_minus_black_score',
    'get_static_value',
    'get_static_delta_score',
    'better_checkmate_score',
    'checkmate_score',
    'get_white_and_black_king_boxes',
    'insert_ordered_worst_to_best',
    'insert_ordered_best_to_worst',
    'pieces_count_from_fen',
    'pieces_count_from_board',
    'insert_choice_into_current_moves',
    'null_target_moves',
    'is_valid_king_box_square',
    'set_all_datetime_headers',
    'set_utc_headers',
    # Mappings
    'max_moves_map',
]


def format_moves(picks: List[Pick]) -> List[Tuple[str, str]]:
    """Format a list of (move, score) tuples into UCI strings and formatted scores.

    Parameters
    ----------
    picks : List[Pick]
        The list of Picks (moves and their evaluation scores.)

    Returns
    -------
    List[Tuple[str, str]
        Formatted list where each tuple contains the move in UCI format
        and the score rounded to two decimal places.
        Entries with `None` moves are excluded.
    """
    return [(m.uci(), f"{s:.2f}") for m, s in picks if len(picks)]


def calculate_score(
        current_index: int,
        new_heatmap_transposed: NDArray[float64],
        new_current_king_box: List[int],
        new_other_king_box: List[int]
) -> float64:
    """Compute a score for a board state based on heatmap control and king box pressure.

    (Deprecated legacy score calc function)

    Parameters
    ----------
    current_index : int
        Index of the current player in the heatmap data (0 for White, 1 for Black).
    new_heatmap_transposed : numpy.typing.NDArray[numpy.float64]
        A transposed 2x64 array of heatmap intensities (player x squares).
    new_current_king_box : list of int
        Indices of squares surrounding the current player's king.
    new_other_king_box : list of int
        Indices of squares surrounding the opponent's king.

    Returns
    -------
    numpy.float64
        Total score computed by summing general control delta and weighted king box pressure.
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
    """Check if a game result is a draw based on the winner field.

    Parameters
    ----------
    winner : bool or None
        Result of Board.outcome(...).winner. True for White win, False for Black win, None for draw.

    Returns
    -------
    bool
        True if the game is a draw, False otherwise.

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
    """Evaluate the board from a White-minus-Black perspective using heatmap evaluation.

    This function returns a net evaluation score. Positive values favor White, negative values favor Black.
    Terminal game states return fixed scores. Otherwise, the score is derived from:
    - Heatmap intensity differences
    - Control over king box areas

    Parameters
    ----------
    board : chess.Board
        The chess board to evaluate.
    depth : int
        Depth of future move evaluation.

    Returns
    -------
    numpy.float64
        Net evaluation score (White - Black). Terminal states return extreme values.

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
    # Early exit if Game Over.
    outcome: Optional[Outcome] = board.outcome(claim_draw=True)
    is_terminated: bool = outcome is not None
    if is_terminated and is_draw(outcome.winner):
        # Draws are easy to score: zero
        return float64(0)
    if is_terminated:
        # Checkmate score is an unrealistically high upperbound in possible moves (all pieces can move to every square.)
        return checkmate_score(board, depth)
    # See docs on time-complexity of calculate_chess_move_heatmap_with_better_discount
    heatmap_transposed: NDArray[float64] = calculate_chess_move_heatmap_with_better_discount(
        board=board,
        depth=depth
    ).data.transpose()
    transposed_white: NDArray[float64] = heatmap_transposed[0]
    transposed_black: NDArray[float64] = heatmap_transposed[1]
    # General move score is the delta in possible moves for White and Black
    general_move_score: float64 = sum(transposed_white) - sum(transposed_black)
    # King-box move score is the delta in attacking moves on the squares at and around the kings
    king_box_white: List[int]
    king_box_black: List[int]
    king_box_white, king_box_black = get_white_and_black_king_boxes(board=board)
    king_box_score: float64 = sum(transposed_white[king_box_black]) - sum(transposed_black[king_box_white])
    # Final score is the sum of both
    return float64(general_move_score + king_box_score)


max_moves_map: Dict[Piece, int] = {
    Piece.from_symbol('P'): 4,
    Piece.from_symbol('N'): 8,
    Piece.from_symbol('B'): 13,
    Piece.from_symbol('R'): 14,
    Piece.from_symbol('Q'): 27,
    Piece.from_symbol('K'): 8,
    Piece.from_symbol('p'): 4,
    Piece.from_symbol('n'): 8,
    Piece.from_symbol('b'): 13,
    Piece.from_symbol('r'): 14,
    Piece.from_symbol('q'): 27,
    Piece.from_symbol('k'): 8,
}


def get_static_value(piece: Piece) -> int:
    """Retrieve the static movement-based value for a given chess piece.

    This function returns a fixed integer value representing the maximum number of
    moves that the specified piece can make on an empty board. These values serve
    as a simple heuristic for piece mobility.

    Parameters
    ----------
    piece : chess.Piece
        The chess piece for which to look up the static move value.

    Returns
    -------
    int
        The static movement value for the piece, or 0 if the piece is not in the map.
    """
    return max_moves_map.get(piece, 0)


def get_static_delta_score(heatmap: ChessMoveHeatmap) -> float64:
    """Compute the static piece-based delta score from a ChessMoveHeatmap.

    This function multiplies each piece’s move count (from the heatmap’s `piece_counts`)
    by its static movement value (from `get_static_value`) and returns the difference
    between White’s total and Black’s total. It serves as a simple material–mobility heuristic.

    Parameters
    ----------
    heatmap : heatmaps.ChessMoveHeatmap
        The heatmap object containing per-piece move counts for each square.

    Returns
    -------
    numpy.float64
        The signed delta score: (White’s static total) − (Black’s static total).
    """
    piece_map: NDArray[Dict[Piece, float64]] = heatmap.piece_counts
    white_sum: float64 = float64(0.0)
    black_sum: float64 = float64(0.0)
    piece_dict: Dict[Piece, float64]
    i: int
    for i, piece_dict in enumerate(piece_map):
        piece: Piece
        count: float64
        for piece, count in piece_dict.items():
            if piece.color:
                white_sum += count * get_static_value(piece=piece)
            else:
                black_sum += count * get_static_value(piece=piece)
    return float64(white_sum - black_sum)


def calculate_better_white_minus_black_score(
        board: Board,
        depth: int = 1,
) -> float64:
    """Compute an enhanced White-minus-Black evaluation for a given board position.

    This function combines three components into a single signed score:

    1. **Game termination**:
        - If the position is a draw, returns 0.
        - If the position is checkmate, returns a high upper-bound value via `checkmate_score()`.
    2. **Heatmap mobility delta**:
        Difference in total possible moves between White and Black, computed by
        `calculate_chess_move_heatmap_with_better_discount()`.
    3. **King-box mobility delta**:
        Difference in potential moves targeting the opponent’s king region and defending
        the own king region.
    4. **Static delta**:
        Material-mobility delta from `get_static_delta_score()`.

    The final score is the sum of the mobility delta, king-box delta, and static delta,
    representing White’s advantage minus Black’s.

    Parameters
    ----------
    board : chess.Board
        The chess board position to evaluate.
    depth : int, default: 1
        Recursion depth for the heatmap calculation. Higher values yield more thorough
        mobility estimates at the cost of increased computation time.

    Returns
    -------
    numpy.float64
        The signed evaluation: positive values favor White, negative values favor Black.

    Notes
    -----
    - Time complexity is O(b^d) for the heatmap portion (where b ≈ 35 is the branching factor).
    - Checkmate scores use an upper-bound heuristic: all remaining pieces can move to every square,
        scaled by (depth + 1).
    """
    # Early exit if Game Over.
    outcome: Optional[Outcome] = board.outcome(claim_draw=True)
    is_terminated: bool = outcome is not None
    if is_terminated and is_draw(outcome.winner):
        # Draws are easy to score: zero
        return float64(0)
    if is_terminated:
        # Checkmate score is an unrealistically high upperbound in possible moves (all pieces can move to every square.)
        return better_checkmate_score(board, depth)
    # See docs on time-complexity of calculate_chess_move_heatmap_with_better_discount
    heatmap: ChessMoveHeatmap = calculate_chess_move_heatmap_with_better_discount(
        board=board,
        depth=depth
    )
    heatmap_transposed: NDArray[float64] = heatmap.data.transpose()
    transposed_white: NDArray[float64] = heatmap_transposed[0]
    transposed_black: NDArray[float64] = heatmap_transposed[1]
    # General move score is the delta in possible moves for White and Black
    general_move_score: float64 = sum(transposed_white) - sum(transposed_black)
    # King-box move score is the delta in attacking moves on the squares at and around the kings
    king_box_white: List[int]
    king_box_black: List[int]
    king_box_white, king_box_black = get_white_and_black_king_boxes(board=board)
    king_box_score: float64 = sum(transposed_white[king_box_black]) - sum(transposed_black[king_box_white])
    # static piece delta score
    static_delta_score = get_static_delta_score(heatmap=heatmap)
    # Final score is the sum of all
    return float64(general_move_score + king_box_score + static_delta_score)


def better_checkmate_score(board: Board, depth: int) -> float64:
    """Compute an enhanced checkmate evaluation score.

    This function returns a large-magnitude score to represent a checkmate
    outcome, scaled by the number of pieces remaining, the search depth, and
    an additional safety margin multiplier. By multiplying by 64 (squares)
    and 27 (maximum queen moves), we ensure that non-checkmate positional
    scores remain bounded in comparison.

    The sign encodes which side is to move:
    - If it is White’s turn (i.e., White has just been checkmated), returns a
        large negative score (bad for White).
    - If it is Black’s turn (i.e., Black has just been checkmated), returns
        a large positive score (good for White).

    Parameters
    ----------
    board : chess.Board
        The board position where checkmate has occurred.
    depth : int
        The recursion depth used in the evaluation, included to amplify the score
        further for deeper searches.

    Returns
    -------
    numpy.float64
        A signed checkmate score with large magnitude, where positive values
        favor White and negative values favor Black.
    """
    mate_score_abs: float64 = float64(pieces_count_from_board(board=board) * (depth + 1) * 64 * 27)
    return float64(-mate_score_abs) if board.turn else mate_score_abs


def checkmate_score(board: Board, depth: int) -> float64:
    """Return a large signed score for checkmate results.

    The score is scaled by number of remaining pieces and depth.
    Negative if the current player is mated, positive if they deliver mate.

    Parameters
    ----------
    board : chess.Board
        Board state assumed to be in a terminal position.
    depth : int
        Search depth used, for scaling the final score.

    Returns
    -------
    numpy.float64
        Large positive or negative score depending on the outcome.

    Examples
    --------
    >>> from chmengine.utils import is_draw
    >>> from chess import Board
    >>> blk_win_board = Board('8/2p2p2/4p3/2k5/8/6q1/2K5/1r1q4 w - - 2 59')
    >>> checkmate_score(board=blk_win_board, depth=1)
    -1024.0
    """
    mate_score_abs: float64 = float64(pieces_count_from_board(board=board) * (depth + 1) * 64)
    return float64(-mate_score_abs) if board.turn else mate_score_abs


def get_white_and_black_king_boxes(board: Board) -> Tuple[List[int], List[int]]:
    """Compute the list of squares surrounding each king on the board.

    These "king boxes" are used for evaluating positional pressure around the kings.

    Parameters
    ----------
    board : Board
        The board from which to extract king square surroundings.

    Returns
    -------
    Tuple[List[int], List[int]]
        Tuple containing white king box and black king box square indices.
        (white_king_box, black_king_box)

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
        ordered_picks: List[Pick],
        pick: Pick
) -> None:
    """Insert a move into a list of moves sorted from worst to best.

    Parameters
    ----------
    ordered_picks : List[Pick]
        Existing list sorted in ascending order of score.
    pick : Pick
    """
    # response moves are inserted to form worst scores to best order (perspective of current player)
    ordered_index: int = bisect_left([p.score for p in ordered_picks], pick.score)
    ordered_picks.insert(ordered_index, pick)


def insert_ordered_best_to_worst(
        ordered_picks: List[Pick],
        pick: Pick
) -> None:
    """Insert a move into a list of moves sorted from best to worst.

    Parameters
    ----------
    ordered_picks : List[Pick]
        Existing list sorted in descending order of score.
    pick : Pick
    """
    # current moves are inserted into our moves list in order of best scores to worst
    ordered_index: int = bisect_left([-p.score for p in ordered_picks], -pick.score)
    ordered_picks.insert(ordered_index, pick)


def pieces_count_from_fen(fen: str) -> int:
    """Return the number of pieces on the board represented by `fen`.

    This function converts the FEN string into a `Board` object, then uses the internal
    bitboard to count occupied squares in O(1) time. On Python ≥ 3.8, it calls `int.bit_count()`;
    on Python 3.7, it falls back to `bin(...).count('1')` for compatibility.

    Notes
    -----
    For most use cases, especially when you already have a `Board` object,
    prefer using `pieces_count_from_board(board)` instead. This avoids the overhead
    of FEN parsing and achieves the same result more efficiently.

    Parameters
    ----------
    fen : str
        A full FEN string representing a chess position.

    Returns
    -------
    int
        The count of non‑empty squares (i.e. total pieces) on the board.

    Examples
    --------
    >>> from chess import Board
    >>> pieces_count_from_fen('8/2p2p2/4p3/2k5/8/6q1/2K5/1r1q4 w - - 2 59')
    8
    >>> pieces_count_from_fen(Board().fen())
    32
    """
    return _bit_count(Board(fen).occupied)


def pieces_count_from_board(board: Board) -> int:
    """Return the number of pieces on the board

    This uses the internal bitboard to count occupied squares in O(1) time.
    On Python ≥ 3.8 it calls `int.bit_count()`. On Python 3.7 it falls back
    to `bin(...).count('1')` for compatibility.

    Parameters
    ----------
    board : chess.Board
        A board object to count pieces from

    Returns
    -------
    int
        Number of pieces on the board.

    Examples
    --------
    >>> from chess import Board
    >>> mate_board = Board('8/2p2p2/4p3/2k5/8/6q1/2K5/1r1q4 w - - 2 59')
    >>> pieces_count_from_board(mate_board)
    8
    >>> pieces_count_from_board(Board())
    32
    """
    return _bit_count(board.occupied)


def insert_choice_into_current_moves(
        choices_ordered_best_to_worst: List[Pick],
        pick: Pick
) -> List[Pick]:
    """Insert a new candidate move into the current player's move list (best to worst).

    Parameters
    ----------
    choices_ordered_best_to_worst : List[Pick]
        Current ordered list of move choices.
    pick : Pick

    Returns
    -------
    List[Pick]
        Updated list with the move inserted.
    """
    if len(choices_ordered_best_to_worst) == 0:
        return [pick]
    insert_ordered_best_to_worst(
        ordered_picks=choices_ordered_best_to_worst, pick=pick
    )
    return choices_ordered_best_to_worst


def insert_choice_into_response_moves(
        choices_ordered_worst_to_best: List[Pick],
        pick: Pick
) -> List[Pick]:
    """Insert a new candidate move into the opponent's response list (worst to best).

    Parameters
    ----------
    choices_ordered_worst_to_best : List[Pick]
        Opponent's candidate responses, sorted from the lowest score to highest.
    pick : Pick

    Returns
    -------
    List[Pick]
        Updated response list.
    """
    if len(choices_ordered_worst_to_best) == 0:
        return [pick]
    insert_ordered_worst_to_best(
        ordered_picks=choices_ordered_worst_to_best, pick=pick
    )
    return choices_ordered_worst_to_best


def is_valid_king_box_square(square_id: int, king_square: int) -> bool:
    """Check if a square is valid for inclusion in a king's bounding box.

    A square is valid if:
    - It lies on the board (0–63)
    - It is adjacent to the king's square (distance ≤ 1)

    Parameters
    ----------
    square_id : int
        Index of the square to evaluate.
    king_square : int
        Index of the king's square.

    Returns
    -------
    bool
        True if the square should be included in the king box.

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


def set_all_datetime_headers(game_heads: pgn.Headers, local_time: datetime) -> None:
    """Sets all datetime related game headers for the pgn file.

    Parameters
    ----------
    game_heads : chess.pgn.Headers
    local_time : datetime.datetime
    """
    game_heads["Date"] = local_time.strftime("%Y.%m.%d")
    game_heads["Timezone"] = str(local_time.tzinfo)
    set_utc_headers(game_heads, local_time)


def set_utc_headers(game_heads: pgn.Headers, local_time: datetime) -> None:
    """Sets UTC header info of pgn file data from local timestamp

    Parameters
    ----------
    game_heads : chess.pgn.Headers
    local_time : datetime.datetime
    """
    game_heads["UTCDate"] = local_time.astimezone(timezone.utc).strftime("%Y.%m.%d")
    game_heads["UTCTime"] = local_time.astimezone(timezone.utc).strftime("%H:%M:%S")
