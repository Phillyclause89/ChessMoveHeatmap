"""Cmhmey Jr.'s Mad Scientist Uncle Who Likes to make clones of Cmhmey Jr."""

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from sqlite3 import OperationalError
from typing import Dict, Optional, Union

from chess import Board, Move
from numpy import float64

from chmengine.engines.cmhmey2 import CMHMEngine2
from chmengine.utils.pick import Pick

NAN64 = float64(None)

BoardCacheEntry = Dict[str, Union[Board, float64, Pick]]
BoardCache = Dict[Move, BoardCacheEntry]


# TODO: Move this helper to `chmengine.utils.__init__.py` (if we can, might have circular import errors?)
def evaluate_move(board: Board, depth: int = 1, debug: bool = False, cache_dir: str = CMHMEngine2.cache_dir) -> Pick:
    """Offloads eval calculations to another CMHMEngine2 instance.

    Parameters
    ----------
    board : Board
        The chess board state to evaluate.
    depth : int
        The search depth for the evaluation.
    debug : bool
        Whether to enable debug output.
    cache_dir : str
        The cache directory for the engine.

    Returns
    -------
    Pick
        The best move and its associated evaluation score.

    Raises
    ------
    OperationalError
        If the database remains locked after exhausting retries or for unexpected operational errors.
    """
    CMHMEngine2.cache_dir = cache_dir
    try:
        return CMHMEngine2(board=board, depth=depth).pick_move(debug=debug)
    except OperationalError as error_o:
        if "database is locked" in str(error_o):
            try:
                return evaluate_move(board=board, depth=depth, debug=debug, cache_dir=cache_dir)
            except RecursionError:
                pass
        raise OperationalError('Unexpected Operational Error: {str(error_o)}') from error_o


class CMHMEngine2PoolExecutor:
    """CMHMEngine2PoolExecutor"""
    cache_dir: str = CMHMEngine2.cache_dir
    engine: CMHMEngine2
    # TODO: Refactor `max_workers` into a dict of options to be passed to `concurrent.futures` things as needed.
    executor: ProcessPoolExecutor

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
        CMHMEngine2.cache_dir = self.cache_dir
        self.engine = CMHMEngine2(board=board if board else Board(), depth=depth)
        self.executor = ProcessPoolExecutor(max_workers=max_workers)

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
                'board': self.engine.board_copy_pushed(move=_move),
            } for _move in self.engine.current_moves_list()
        }
        # Throw ValueError if there are no moves to evaluate.
        if len(board_cache) == 0:
            # Try/Catch root value error that would have happened had we made it to the end of the method call.
            try:
                return self.engine.pick_move(debug=True)
            except ValueError as root_error:
                raise ValueError(
                    f"No legal moves available from board: {self.engine.fen()}"
                ) from root_error
        future_to_move: Dict[Future, Move] = {
            self.executor.submit(
                evaluate_move,
                board=board_cache[_move]['board'],
                depth=self.engine.depth,
                debug=debug,
                cache_dir=self.cache_dir
            ): _move for _move in board_cache
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

    def shutdown(self) -> None:
        """Shut down the ProcessPoolExecutor."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown()

    def __del__(self):
        """Ensure the ProcessPoolExecutor is shut down when the instance is deleted."""
        self.shutdown()


if __name__ == '__main__':
    cmhmey2_executor: CMHMEngine2PoolExecutor = CMHMEngine2PoolExecutor()
    print(cmhmey2_executor.engine.board)
    # On first visit both `pick_move` calls below will take a while...
    pick: Pick = cmhmey2_executor.pick_move(debug=True)
    print(f'{pick:+.2f}')
    cmhmey2_executor.push(pick.move)
    print(cmhmey2_executor.engine.board)
    #  On revisits, only the second `pick_move` call can take a while
    #  should the back population change the outcome of the first call to lead to an unseen position.
    pick = cmhmey2_executor.pick_move(debug=True)
    print(f'{pick:+.2f}')
    cmhmey2_executor.push(pick.move)
    print(cmhmey2_executor.engine.board)
