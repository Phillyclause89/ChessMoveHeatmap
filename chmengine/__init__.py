"""A silly chess engine that picks moves using heatmaps"""
from chmengine import engines, play, utils
from chmengine.engines import CMHMEngine, CMHMEngine2, Quartney, CMHMEngine2PoolExecutor, CMHMEngine3
from chmengine.play import PlayCMHMEngine
from chmengine.utils import (Pick, better_checkmate_score, calculate_better_white_minus_black_score,
                             calculate_white_minus_black_score, checkmate_score, format_moves, format_picks,
                             get_static_delta_score, get_static_value, get_white_and_black_king_boxes,
                             insert_ordered_best_to_worst, insert_ordered_worst_to_best, is_draw, max_moves_map,
                             pieces_count_from_board, pieces_count_from_fen, set_all_datetime_headers, set_utc_headers)

__all__ = [
    # Mods
    'engines',
    'play',
    'utils',
    # Classes
    'CMHMEngine',
    'CMHMEngine2',
    'Quartney',
    'PlayCMHMEngine',
    'Pick',
    'CMHMEngine2PoolExecutor',
    'CMHMEngine3',
    # functions
    'format_moves',
    'format_picks',
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
    'calculate_better_white_minus_black_score',
    'get_static_value',
    'get_static_delta_score',
    'better_checkmate_score',
    # Mappings
    'max_moves_map',
]
