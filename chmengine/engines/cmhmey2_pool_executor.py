"""Cmhmey Jr.'s Mad Scientist Uncle Who Likes to make clones of Cmhmey Jr."""

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

from chess import Board, Move
from numpy import float64, isnan

from chmengine.engines.cmhmey2 import CMHMEngine2
from chmengine.utils.pick import Pick

NAN64 = float64(None)

BoardCacheEntry = Dict[str, Union[Board, float64, Pick]]
BoardCache = Dict[Move, BoardCacheEntry]


# TODO: Move this helper to `chmengine.utils.__init__.py` (if we can, might have circular import errors?)
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
    # TODO: Property-afy these fields as we finalize them
    depth: int
    board: Board
    engine: CMHMEngine2
    # TODO: Refactor `max_workers` into a dict of options to be passed to `concurrent.futures` things as needed
    max_workers: Optional[int]

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
            _move: {
                'board': self.engine.board_copy_pushed(move=_move, board=self.board),
                'cached_score': NAN64,  # Initialize cached_score as NaN
            } for _move in self.engine.current_moves_list(board=self.board)
        }
        # Throw ValueError if there are no moves to evaluate.
        if len(board_cache) == 0:
            raise ValueError(
                f"No legal moves available from board: {self.board.fen()}"
            ) from self.engine.pick_move(debug=True)

        # Populate cached_score from the Q-Table
        uncached_moves: List[Move] = []
        _cache: BoardCacheEntry
        # TODO: A-B test if this is actually better or if we should just offload all calls of `pick_move`.
        for _move, _cache in board_cache.items():
            # Since CMHMEngine2 writes preliminary scores for deeper positions it has yet to fully explore,
            # we must actually check the game tree nodes 1-2 half-moves deeper for a Q-Score.
            cache_board: Board = _cache['board']
            check_move: Move = next(iter(cache_board.legal_moves))
            check_board: Board = self.engine.board_copy_pushed(
                move=check_move, board=cache_board
            )
            # 2 half-moves deeper in the event there is check or a capture at the first half-move position.
            if (
                    check_board.is_check() or check_board.piece_at(check_move.to_square)
            ) and check_board.legal_moves.count() > 0:
                check_move: Move = next(iter(check_board.legal_moves))
                check_board: Board = self.engine.board_copy_pushed(
                    move=check_move, board=check_board
                )
            # If this deeper Q-Score turns up null then we know we got a new position on our hands
            if isnan(float64(self.engine.get_q_value(board=check_board))):
                uncached_moves.append(_move)
            # Else we just call `pick_move` on the main thread/proc
            # as it will just be fast lookups and a few back-pop updates.
            else:
                self.engine.board = cache_board
                # We just need the deeper calls to update the q-table, don't actually need their pick here: `_`.
                _ = self.engine.pick_move(debug=debug)
                if debug:
                    print(f"Evaluated on Main Thread: {_cache}")
        # Just reset the self.engine.board in case hit our else condition in the loop above.
        self.engine.board = self.board
        # If there are any unexplored positions then we offload those to ProcessPoolExecutor
        if len(uncached_moves) > 0:
            executor: ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
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
                        # We just need the deeper calls to update the q-table, don't actually need their pick here: `_`.
                        _ = _future.result().score
                        if debug:
                            print(f"Evaluated in Process Pool: {_cache}")
                    except Exception as e:
                        if debug:
                            print(f"Error evaluating move in Process Pool:{_move}: {e}")
        # The final self.engine.pick_move call will be superfast now
        # as the previous `pick_move` calls ensured all our positions have scores.
        return self.engine.pick_move(debug=debug)

    def push(self, move: Move) -> None:
        """Helper method to update the internal board state with a pushed move.

        Parameters
        ----------
        move : Move
        """
        self.engine.board.push(move=move)
        # self.board could be the same instance, but we reassign just to be safe.
        self.board = self.engine.board


if __name__ == '__main__':
    cmhmey2_executor: CMHMEngine2PoolExecutor = CMHMEngine2PoolExecutor()
    print(cmhmey2_executor.board)
    # On first visit both `pick_move` calls below will take a while...
    pick: Pick = cmhmey2_executor.pick_move(debug=True)
    print(f'{pick:+.2f}')
    cmhmey2_executor.push(pick.move)
    print(cmhmey2_executor.board)
    #  On revisits, only the second `pick_move` call can take a while
    #  should the back population change the outcome of the first call to lead to an unseen position.
    pick = cmhmey2_executor.pick_move(debug=True)
    print(f'{pick:+.2f}')
    cmhmey2_executor.push(pick.move)
    print(cmhmey2_executor.board)
