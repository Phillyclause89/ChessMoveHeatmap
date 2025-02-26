"""CHMUtils"""
import os
import sqlite3
from sqlite3 import Connection, Cursor
from typing import Dict, List, Optional, Tuple, Union
import chess
from chess import Move, Board, Piece
from numpy import float_
import numpy as np
import heatmaps


def calculate_heatmap(
        board: Board, depth: int = 1,
        heatmap: Optional[heatmaps.GradientHeatmap] = None,
        discount: int = 1
) -> heatmaps.GradientHeatmap:
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
    >>> print(depth1_hmap.colors[16])
    #affefe
    >>> # Calculate a heatmap with a recursion depth of 2.
    >>> depth2_hmap = calculate_heatmap(brd, depth=2)
    >>> print(depth2_hmap.colors[16])
    #afffff
    """
    if heatmap is None:
        heatmap = heatmaps.GradientHeatmap()

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


def calculate_chess_move_heatmap(
        board: Board, depth: int = 1,
        heatmap: Optional[heatmaps.ChessMoveHeatmap] = None,
        discount: int = 1) -> heatmaps.ChessMoveHeatmap:
    """
    Recursively computes a chess move heatmap that tracks both move intensities and piece counts.

    This function extends the standard gradient heatmap calculation by additionally updating,
    for each move, a per-square dictionary of chess piece counts. For each legal move, the target
    square’s intensity is incremented (as in calculate_heatmap), and the count corresponding to
    the moving piece is also incremented.

    Parameters
    ----------
    board : chess.Board
        The chess board position for which to calculate the move heatmap.
    depth : int, optional
        The recursion depth to explore legal moves. Higher depth yields more comprehensive data,
        but at the cost of exponentially increased computation time.
    heatmap : Optional[heatmaps.ChessMoveHeatmap], optional
        An existing ChessMoveHeatmap instance to which both move intensities and piece counts
        will be added. For initial calls, this should be left as None.
    discount : int, optional
        A multiplier used to discount contributions of moves as the recursion deepens.
        It is updated recursively to moderate the exponential growth in contributions.
        Default is 1.

    Returns
    -------
    heatmaps.ChessMoveHeatmap
        The computed ChessMoveHeatmap containing:
            - A gradient heatmap of move intensities per square.
            - A corresponding array of piece count dictionaries per square.

    Notes
    -----
    - The function updates both the standard heatmap data (for move intensity) and, in addition,
        the `piece_counts` attribute.
    - As with `calculate_heatmap`, the parameters `heatmap` and `discount` are intended for internal use.
    - The recursion depth controls the number of future move layers considered. Only odd depths tend to provide
        balanced data between both players.
    - The time complexity is approximately O(b^d),
        where b ≈ 35 (the average branching factor in chess) and d is the depth.

    Examples
    --------
    >>> import chess
    >>> from heatmaps import ChessMoveHeatmap
    >>> from chmutils import calculate_chess_move_heatmap
    >>> brd = chess.Board()
    >>> depth1_cmhm = calculate_chess_move_heatmap(brd, depth=1)
    >>> print(", ".join([f"{p.unicode_symbol()}: {cnt}" for p, cnt in depth1_cmhm.piece_counts[16].items() if cnt]))
    ♙: 1.0, ♘: 1.0
    >>> depth2_cmhm = calculate_chess_move_heatmap(brd, depth=3)
    >>> print(", ".join([f"{p.unicode_symbol()}: {cnt}" for p, cnt in depth2_cmhm.piece_counts[16].items() if cnt]))
    ♙: 1.849999999999993, ♘: 1.849999999999984, ♗: 0.10000000000000005, ♖: 0.05000000000000001, ♝: 0.09067002690459958

    """
    if heatmap is None:
        heatmap = heatmaps.ChessMoveHeatmap()

    moves: Tuple[Move] = tuple(board.legal_moves)
    num_moves: int = len(moves)

    if num_moves == 0:
        return heatmap

    color_index: int = int(not board.turn)
    move: Move

    for move in moves:
        target_square: int = move.to_square
        heatmap[target_square][color_index] += (1.0 / discount)
        from_square: int = move.from_square
        piece: Piece = board.piece_at(from_square)
        heatmap.piece_counts[target_square][piece] += (1.0 / discount)

        if depth > 0:
            new_board: Board = board.copy()
            new_board.push(move)
            heatmap = calculate_chess_move_heatmap(new_board, depth - 1, heatmap, discount * num_moves)
    return heatmap


def calculate_chess_move_heatmap_with_better_discount(
        board: Board, depth: int = 1,
        depth_map: Optional[Tuple[List[Union[int, heatmaps.ChessMoveHeatmap]]]] = None) -> heatmaps.ChessMoveHeatmap:
    """
    Recursively computes a chess move heatmap for a given board state while applying a discount
    to moves based on the branching factor at each level of recursion. This discounting approach
    reduces the weight of moves that occur in positions with many legal alternatives, thereby
    "better discounting" moves from deeper, more complex branches.

    The function explores all legal moves from the current board state up to the specified depth.
    For each move, it updates a depth-specific accumulator (the ``depth_map``) that stores:
        - A count of the total number of legal moves (branches) encountered at that depth.
        - A ``ChessMoveHeatmap`` that aggregates move data, including:
            - Incrementing the value at the target square (indexed by the opponent’s color).
            - Tracking piece counts for moves arriving at each square.

    After the recursive exploration is complete (i.e., at the exit depth), the function aggregates the
    heatmaps from each depth level. Although the code uses a reversed iteration over ``depth_map``,
    due to the way recursion updates the accumulator (using ``depth_map[depth]``), the reversed order
    actually corresponds to: processing from the shallowest (initial board state) to the deepest level.
    At each level, the accumulated values are divided by a discount factor determined by the number of
    branches at that level, ensuring that moves from positions with more alternatives contribute proportionally
    less to the final heatmap.

    Parameters
    ----------
    board : chess.Board
        The current chess board state from which legal moves are generated.
    depth : int, optional
        The depth (or ply) to which moves are recursively explored. A depth of 1 considers only the
        immediate legal moves, while higher depths recursively analyze subsequent moves. Default is 1.
    depth_map : Optional[Tuple[List[Union[int, heatmaps.ChessMoveHeatmap]]]], optional
        An internal accumulator used during recursion. Each element in the tuple corresponds to a
        depth level and is a list containing:
            - An integer count of legal move branches encountered at that depth.
            - A ``ChessMoveHeatmap`` instance that aggregates move and piece count data for that depth.
        If not provided, it is initialized automatically. External callers should typically omit this parameter.

    Returns
    -------
    heatmaps.ChessMoveHeatmap
        A composite chess move heatmap where each square’s value reflects the discounted cumulative
        influence of moves leading to that square. The heatmap also maintains aggregated piece movement
        counts.

    Notes
    -----
    - ``depth_map`` is not supplied, it is initialized as a tuple of pairs for each depth level (from 0 to ``depth``),
        with each pair consisting of a branch count (initialized to 0) and a new ``ChessMoveHeatmap``.
    - During recursion, for each legal move:
        * The total branch count for the current depth is incremented by the number of legal moves.
        * The target square’s value in the current depth’s heatmap is increased by 1.0 for the opponent’s color.
        * The piece count for the target square is incremented based on the piece moving from the source square.
    - Once the deepest level is reached (i.e., in the original call where ``depth == len(depth_map) - 1``),
        the function aggregates the heatmaps from all depth levels. Although the iteration uses ``depth_map[::-1]``,
        this reversed order corresponds to: processing from the shallowest level (the original board state)
        to the deepest level, due to how updates are applied using ``depth_map[depth]`` during recursion.
    - At each aggregated level, the heatmap values (both move counts and piece counts) are divided by a discount factor,
        which is updated based on the number of branches at that level. This mechanism weights moves from positions with
        fewer alternatives more heavily.

    Examples
    --------
    >>> import chess
    >>> from chmutils import calculate_chess_move_heatmap_with_better_discount
    >>> brd = chess.Board()
    >>> depth1_cmhm = calculate_chess_move_heatmap_with_better_discount(brd, depth=1)
    >>> print(", ".join([f"{p.unicode_symbol()}: {cnt}" for p, cnt in depth1_cmhm.piece_counts[16].items() if cnt]))
    ♙: 1.0, ♘: 1.0
    >>> depth2_cmhm = calculate_chess_move_heatmap_with_better_discount(brd, depth=3)
    >>> print(", ".join([f"{p.unicode_symbol()}: {cnt}" for p, cnt in depth2_cmhm.piece_counts[16].items() if cnt]))
    ♙: 1.85, ♘: 1.85, ♗: 0.1, ♖: 0.05, ♝: 0.09110312289373175
    """
    if depth_map is None:
        # If depth=1, the map will look like:
        # ([0, <heatmaps.ChessMoveHeatmap object at 0x99A7630>], [0, <heatmaps.ChessMoveHeatmap object at 0x74465C0>])
        # Isolate counts at each depth. The int at index 0 needs to keep track of the number of branches at each depth
        depth_map = tuple([0, heatmaps.ChessMoveHeatmap()] for _ in range(depth + 1))

    moves: Tuple[Move] = tuple(board.legal_moves)
    num_moves: int = len(moves)
    # Each sub-depth's total branches will be counted here
    depth_map[depth][0] += num_moves

    color_index: int = int(not board.turn)
    move: Move

    for move in moves:
        target_square: int = move.to_square
        depth_map[depth][1][target_square][color_index] += np.float64(1.0)
        from_square: int = move.from_square
        piece: Piece = board.piece_at(from_square)
        depth_map[depth][1].piece_counts[target_square][piece] += np.float64(1.0)

        if depth > 0:
            new_board: Board = board.copy()
            new_board.push(move)
            depth_map = calculate_chess_move_heatmap_with_better_discount(
                new_board,
                depth - 1,
                depth_map
            )
    # exit depth will always equal len(depth_map) - 1
    if depth == len(depth_map) - 1:
        # return object
        heatmap: heatmaps.ChessMoveHeatmap = heatmaps.ChessMoveHeatmap()
        # discount is the total number parrallel branches at each depth
        discount: int = 1
        branches: int
        heatmap_at_depth: heatmaps.ChessMoveHeatmap
        # Need to loop through depth_map in backwards order to update discount correctly
        for branches, heatmap_at_depth in depth_map[::-1]:
            square: int
            for square in chess.SQUARES:
                heatmap_at_depth[square] /= discount
                p_counts: Dict[Piece, np.float64] = heatmap_at_depth.piece_counts[square]
                for piece in p_counts:
                    p_counts[piece] /= discount
            # TODO: Make a div method for ChessMoveHeatmap to replace the above with the below `/ prev_branches`
            heatmap += heatmap_at_depth  # / prev_branches
            # Update discount to apply to the next heatmap being added in the next loop
            discount = branches
        # Final return
        return heatmap
    # Recursive return
    return depth_map


# Assume PIECE_KEYS is defined (e.g., from chess.UNICODE_PIECE_SYMBOLS.values())
PIECE_KEYS: Tuple[Piece] = heatmaps.PIECES


def flatten_heatmap(heatmap: heatmaps.ChessMoveHeatmap) -> Dict[str, np.float64]:
    """Flatten a ChessMoveHeatmap into a dictionary of primitive values.

    This function converts a ChessMoveHeatmap into a flat dictionary where each key
    represents a specific attribute for a given square. The keys are constructed in the
    following format:

    - "sq{square}_white": Move intensity for White on that square.
    - "sq{square}_black": Move intensity for Black on that square.
    - "sq{square}_piece_{symbol}": The count (or intensity) for a specific piece (identified by its Unicode symbol)
        on that square.

    Parameters
    ----------
    heatmap : heatmaps.ChessMoveHeatmap
        The ChessMoveHeatmap object to be flattened.

    Returns
    -------
    Dict[str, numpy.float64]
        A dictionary with keys for each square's move intensities and piece counts.
    """
    flat: Dict[str, np.float64] = {}
    square: int
    for square in range(64):
        flat[f"sq{square}_white"] = heatmap.data[square][0]
        flat[f"sq{square}_black"] = heatmap.data[square][1]
        # For each piece key, store its count from piece_counts.
        key: Piece
        for key in PIECE_KEYS:
            flat[f"sq{square}_piece_{key.unicode_symbol()}"] = heatmap.piece_counts[square][key]
    return flat


def inflate_heatmap(data: Dict[str, float]) -> heatmaps.ChessMoveHeatmap:
    """Inflate a flat dictionary of primitive values back into a ChessMoveHeatmap.

    This function reconstructs a ChessMoveHeatmap from a dictionary that was
    produced by `flatten_heatmap()`. It assumes that the dictionary contains keys in
    the format "sq{square}_white", "sq{square}_black", and "sq{square}_piece_{symbol}" for
    each square (0-63).

    Parameters
    ----------
    data : Dict[str, float]
        A flat dictionary of primitive values representing a ChessMoveHeatmap.

    Returns
    -------
    heatmaps.ChessMoveHeatmap
        The reconstituted ChessMoveHeatmap object.
    """
    # Create a new heatmap object.
    heatmap: heatmaps.ChessMoveHeatmap = heatmaps.ChessMoveHeatmap()
    square: int
    for square in range(64):
        heatmap.data[square][0] = data[f"sq{square}_white"]
        heatmap.data[square][1] = data[f"sq{square}_black"]
        key: Piece
        for key in PIECE_KEYS:
            heatmap.piece_counts[square][key] = data[f"sq{square}_piece_{key.unicode_symbol()}"]
    return heatmap


CACHE_DIR: str = "SQLite3Caches"


class HeatmapCache:
    """A caching mechanism for ChessMoveHeatmap objects using SQLite.

    This class stores flattened heatmap data in a SQLite database, indexed by a unique
    key derived from the board's FEN string. If a cached heatmap exists for a given board
    configuration and recursion depth, it is returned. Otherwise, the heatmap is computed,
    stored in the database, and then returned.

    Attributes
    ----------
    depth : int
        The recursion depth associated with the heatmap calculations.
    board : chess.Board
        The chess board whose heatmap is being cached.
    db_path : str
        The relative file path to the SQLite database used for caching.
    """
    depth: int
    board: chess.Board
    db_path: str
    cache_dir: str = CACHE_DIR

    def __init__(self, board: chess.Board, depth: int) -> None:
        """Initialize the HeatmapCache.

        Parameters
        ----------
        board : chess.Board
            The chess board for which the heatmap cache is maintained.
        depth : int
            The recursion depth associated with the heatmap calculations.
        """
        self.depth = depth
        self.db_path = os.path.join(self.cache_dir, f"heatmap_cache_depth_{self.depth}.db")
        self.board = board
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create the cache table in the SQLite database if it does not exist.

        This method sets up the database schema with columns for each square's 
        move intensities and piece counts, and creates an index on the cache key 
        for faster lookups.
        """
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        conn: Connection
        with sqlite3.connect(self.db_path) as conn:
            cur: Cursor = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            # Construct column definitions for each square.
            col_defs: List[str] = []
            square: int
            for square in range(64):
                col_defs.append(f"sq{square}_white REAL")
                col_defs.append(f"sq{square}_black REAL")
                key: Piece
                for key in PIECE_KEYS:
                    col_defs.append(f"sq{square}_piece_{key.unicode_symbol()} REAL")

            cols: str = ", ".join(col_defs)
            create_stmt: str = f"CREATE TABLE IF NOT EXISTS heatmap_cache (cache_key TEXT PRIMARY KEY, {cols})"
            cur.execute(create_stmt)

            # Add an index for faster lookups (though PRIMARY KEY already has one)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON heatmap_cache(cache_key);")

    @property
    def _cache_key(self) -> str:
        """Build a unique cache key based on the board's FEN string.

        Returns
        -------
        str
            A string key uniquely representing the current board configuration.
        """
        return f"{self.board.fen()}"

    def get_cached_heatmap(self) -> Optional[heatmaps.ChessMoveHeatmap]:
        """Retrieve a cached ChessMoveHeatmap for the given board and depth, if available.

        Returns
        -------
        Union[heatmaps.ChessMoveHeatmap, None]
            The cached ChessMoveHeatmap if found; otherwise, None.
        """
        key: str = self._cache_key
        conn: Connection
        with sqlite3.connect(self.db_path) as conn:
            cur: Cursor = conn.cursor()
            cur.execute("SELECT * FROM heatmap_cache WHERE cache_key = ?", (key,))
            row: Optional[Tuple[float, ...]] = cur.fetchone()
            if row is None:
                return row
            col_names: List[str] = [description[0] for description in cur.description]
        data: Dict[str, float] = dict(zip(col_names, row))
        # Remove the cache_key from the dict
        data.pop("cache_key", None)
        return inflate_heatmap(data)

    def store_heatmap(self, cmhm: heatmaps.ChessMoveHeatmap) -> None:
        """Store a ChessMoveHeatmap in the cache.

        The given heatmap is flattened into a dictionary of primitive values and inserted
        into the SQLite database. If an entry with the same cache key already exists, it is replaced.

        Parameters
        ----------
        cmhm : heatmaps.ChessMoveHeatmap
            The ChessMoveHeatmap object to be stored.
        """
        key: str = self._cache_key
        flat: Dict[str, float_] = flatten_heatmap(cmhm)

        # Prepare the column names and placeholders.
        # We'll assume the schema matches the flattened dict keys exactly.
        columns: List[str] = ["cache_key"] + list(flat.keys())
        placeholders: str = ", ".join(["?"] * len(columns))
        values: List[str, float_] = [key] + list(flat.values())

        conn: Connection
        with sqlite3.connect(self.db_path) as conn:
            cur: Cursor = conn.cursor()
            # Use INSERT OR REPLACE to update existing cache if needed.
            sql: str = f"INSERT OR REPLACE INTO heatmap_cache ({', '.join(columns)}) VALUES ({placeholders})"
            cur.execute(sql, values)


def get_or_compute_heatmap(board: chess.Board, depth: int) -> heatmaps.ChessMoveHeatmap:
    """Retrieve a ChessMoveHeatmap from the cache, or compute and cache it if not available.

    This function first attempts to retrieve a cached heatmap based on the board's FEN and the
    specified recursion depth. If the cached heatmap is not found, it computes the heatmap,
    stores it in the cache, and returns the newly computed object.

    Parameters
    ----------
    board : chess.Board
        The chess board position for which the heatmap is computed.
    depth : int
        The recursion depth used for calculating the heatmap.

    Returns
    -------
    heatmaps.ChessMoveHeatmap
        The ChessMoveHeatmap corresponding to the board and depth.
    """
    cache: HeatmapCache = HeatmapCache(board, depth)
    cached: Optional[heatmaps.ChessMoveHeatmap] = cache.get_cached_heatmap()
    if cached is not None:
        return cached

    # Not cached; compute the heatmap.
    cmhm: heatmaps.ChessMoveHeatmap = calculate_chess_move_heatmap(board, depth=depth)
    cache.store_heatmap(cmhm)
    return cmhm


def get_or_compute_heatmap_with_better_discounts(board: chess.Board, depth: int) -> heatmaps.ChessMoveHeatmap:
    """Retrieve a ChessMoveHeatmap from the cache, or compute and cache it if not available.

    This function first attempts to retrieve a cached heatmap based on the board's FEN and the
    specified recursion depth. If the cached heatmap is not found, it computes the heatmap,
    stores it in the cache, and returns the newly computed object.

    Parameters
    ----------
    board : chess.Board
        The chess board position for which the heatmap is computed.
    depth : int
        The recursion depth used for calculating the heatmap.

    Returns
    -------
    heatmaps.ChessMoveHeatmap
        The ChessMoveHeatmap corresponding to the board and depth.
    """
    cache: HeatmapCache = HeatmapCache(board, depth)
    cached: Optional[heatmaps.ChessMoveHeatmap] = cache.get_cached_heatmap()
    if cached is not None:
        return cached

    # Not cached; compute the heatmap.
    cmhm: heatmaps.ChessMoveHeatmap = calculate_chess_move_heatmap_with_better_discount(board, depth=depth)
    cache.store_heatmap(cmhm)
    return cmhm
