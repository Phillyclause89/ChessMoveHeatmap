"""A silly chess engine that picks moves using heatmaps"""
from chmengine.engines import CMHMEngine, CMHMEngine2
from chmengine.play import PlayCMHMEngine
from chmengine.utils import (
    Pick,
    calculate_white_minus_black_score,
    checkmate_score,
    format_moves,
    get_white_and_black_king_boxes,
    insert_ordered_best_to_worst,
    insert_ordered_worst_to_best,
    is_draw,
    pieces_count_from_board,
    pieces_count_from_fen,
    set_all_datetime_headers,
    set_utc_headers
)

__all__ = [
    # Classes
    'CMHMEngine',
    'CMHMEngine2',
    'PlayCMHMEngine',
    'Pick',
    # functions
    'format_moves',
    'calculate_white_minus_black_score',
    'checkmate_score',
    'is_draw',
    'get_white_and_black_king_boxes',
    'insert_ordered_worst_to_best',
    'insert_ordered_best_to_worst',
    'pieces_count_from_fen',
    'pieces_count_from_board',
    'set_all_datetime_headers',
    'set_utc_headers',
]
