"""Cmhmey Jr.'s Mad Scientist Uncle Who Likes to make clones of Cmhmey Jr."""

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

from chess import Board, Move
from numpy import float64, isnan

from chmengine.engines.cmhmey2 import CMHMEngine2
from chmengine.utils.pick import Pick

NAN64 = float64(None)

BoardCacheEntry = Dict[str, Union[Board, float64, Pick]]
BoardCache = Dict[Move, BoardCacheEntry]


# Evaluate uncached moves in parallel
def evaluate_move(board: Board, depth: int = 1, debug: bool = False) -> Pick:
    """Offloads eval calculations to another CMHMEngine2 instance.

    Parameters
    ----------
    board : Board
    depth : int
    debug : bool

    Returns
    -------
    Pick
    """
    return CMHMEngine2(board=board, depth=depth).pick_move(debug=debug)


class CMHMEngine2PoolExecutor:
    """CMHMEngine2PoolExecutor"""
    engine: CMHMEngine2
    max_workers: Optional[int]
    depth: int
    board: Board

    def __init__(self, board: Optional[Board] = None, depth: int = 1, max_workers: Optional[int] = None) -> None:
        """Initialize the CMHMEngine2PoolExecutor.

        Parameters
        ----------
        board : Optional[chess.Board]
            The initial board state. If None, a standard starting board is used.
        depth : int, default: 1
            The recursion depth for move evaluations.
        max_workers : Optional[int], default: None
            The maximum number of worker threads for parallel execution.
        """
        self.board = board if board else Board()
        self.depth = depth
        self.max_workers = max_workers
        self.engine = CMHMEngine2(board=self.board, depth=self.depth)

    def pick_move(self, debug: bool = False) -> Pick:
        """
        Select the best move using conditional multithreaded evaluations.

        Parameters
        ----------
        debug : bool, default: False
            If True, prints debug information about the evaluations.

        Returns
        -------
        Pick
            The best move and its associated evaluation score.
        """

        # Initialize board_cache
        _move: Move
        board_cache: BoardCache = {
            move: {
                'board': self.engine.board_copy_pushed(move=move, board=self.board),
                'cached_score': NAN64,  # Initialize cached_score as NaN
                'pick': Pick(move=move, score=NAN64),  # Initialize Pick's score as NaN
            } for move in self.engine.current_moves_list(board=self.board)
        }

        if len(board_cache) == 0:
            raise ValueError("No legal moves available")

        # Populate cached_score from the Q-Table
        uncached_moves: List[Move] = []
        cached_moves: List[Move] = []
        _cache: BoardCacheEntry
        for _move, _cache in board_cache.items():
            cached_score: float64 = float64(self.engine.get_q_value(board=_cache['board']))
            _cache['cached_score'] = cached_score
            if isnan(cached_score):
                uncached_moves.append(_move)
            else:
                cached_moves.append(_move)

        for _move in cached_moves:
            _cache = board_cache[_move]
            self.engine.board = _cache['board']
            _cache['pick'] = self.engine.pick_move(debug=debug)
        self.engine.board = self.board

        if len(uncached_moves) > 0:
            executor: ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_move: Dict[Future, Move] = {
                    executor.submit(
                        evaluate_move, board=board_cache[_move]['board'], depth=self.depth, debug=debug
                    ): _move for _move in uncached_moves
                }
                _future: Future[Pick]
                for _future in as_completed(future_to_move):
                    _move = future_to_move[_future]
                    try:
                        _cache = board_cache[_move]
                        _cache['pick'] = _future.result()
                        if debug:
                            print(f"Evaluated: {_cache}")
                    except Exception as e:
                        if debug:
                            print(f"Error evaluating move {_move}: {e}")

        final_pick: Pick = sorted(
            (_cache['pick'] for _cache in board_cache.values()), reverse=True
        )[0 if self.board.turn else -1]

        # Final learning step: Update Q-Table with the chosen move's score
        self.engine.set_q_value(value=final_pick.score, board=self.board)
        return final_pick


if __name__ == '__main__':
    cmhmey2_executor = CMHMEngine2PoolExecutor()
    pick = cmhmey2_executor.pick_move(debug=True)
    print(f'{pick:+.2f}')
