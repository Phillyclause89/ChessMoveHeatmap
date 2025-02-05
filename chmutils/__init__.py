from typing import Optional, Tuple
import chess
from chess import Move, Board
from heatmaps import GradientHeatmap


def calculate_heatmap(board: Board, depth: int = 1,
                      heatmap: Optional[GradientHeatmap] = None,
                      discount: int = 1) -> GradientHeatmap:
    """
    Recursively computes a gradient heatmap for a given chess board position.

    This function traverses the legal moves of the board recursively to build
    a heatmap that represents the "intensity" of move activity for each square.
    The intensity is accumulated with a discount factor to account for the branching
    factor at each move level. If no heatmap is provided, a new one is initialized.

    Parameters
    ----------
    board : chess.Board
        The chess board position for which to calculate the heatmap.
        Please refer to the `Notes` section for performance warnings related to high depth values.
    depth : int, optional
        The recursion depth to explore legal moves. A higher depth results in a more
        comprehensive heatmap, but increases computation time. The default is 1.
        Please refer to the `Notes` section for performance warnings related to high depth values.
    heatmap : Optional[GradientHeatmap], optional
        An existing `GradientHeatmap` instance to which the move intensities will be added.
        This parameter is primarily used internally during recursive calls and should
        be left as its default value (None) when initially calling the function.
    discount : int, optional
        A multiplier used to discount the contribution of moves as the recursion deepens.
        This parameter is intended for internal use and should typically not be set by the user.
        The default is 1.

    Returns
    -------
    GradientHeatmap
        The computed `GradientHeatmap` containing accumulated move intensities for each square
        on the board, considering the specified recursion depth and discounting.

    Notes
    -----
    - The `heatmap` and `discount` parameters are reserved for internal recursive processing.
      Users should not provide values for these parameters unless they need to override default behavior.
    - The time complexity of this function is **O(b^d)**, where **b ≈ 35** is the branching factor of chess,
      and **d** is the recursion depth. Please see performance warnings below regarding high depths.
    - **Warning:** This function does not implement safeguards to limit excessive recursion depth.
      Very high `depth` values can lead to significant performance degradation and may hit Python's
      recursion depth limitations. It is recommended to avoid setting depth too high, especially
      with complex board positions, as the time complexity grows exponentially.
    - The `depth` parameter controls how many layers of future moves are explored:
      - **depth 0** calculates results from the current player's current moves only.
      - **depth 1** calculates both the current player's current moves
                    and the opponent's possible future moves in their upcoming turn.
      - **depth 2** continues this pattern into the current player's future moves
                    but stops short of the opponent's future moves in their turn thereafter.
    - In theory, **only odd depths** will give an equalized representation of both players,
      while **even depths** will be biased toward the current player.

    Examples
    --------
    >>> import chess
    >>> from heatmaps import GradientHeatmap
    >>> from chmutils import calculate_heatmap
    >>> brd = chess.Board()
    >>> # Calculate a heatmap with a recursion depth of 1.
    >>> depth1_hmap = calculate_heatmap(brd, depth=1)
    >>> print(depth1_hmap.colors)
    >>> # Calculate a heatmap with a recursion depth of 2.
    >>> depth2_hmap = calculate_heatmap(brd, depth=2)
    >>> print(depth2_hmap.colors)
    """
    if heatmap is None:
        heatmap = GradientHeatmap()

    moves: Tuple[Move] = tuple(board.legal_moves)
    num_moves: int = len(moves)

    if num_moves == 0:
        return heatmap

    color_index: int = int(not board.turn)
    move: Move

    for move in moves:
        target_square: int = move.to_square
        heatmap[target_square][color_index] += (1.0 / discount)

        if depth > 0:
            new_board: Board = board.copy()
            new_board.push(move)
            heatmap = calculate_heatmap(new_board, depth - 1, heatmap, discount * num_moves)
    return heatmap


if __name__ == "__main__":
    from timeit import timeit

    b = chess.Board()

    for d in range(6):
        hmap = timeit("calculate_heatmap(board=b, depth=d)", number=1, globals=globals())
        print(f"depth={d}", hmap)
