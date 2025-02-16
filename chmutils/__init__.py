import pickle
import sqlite3
from typing import Any, Dict, Optional, Tuple
import chess
from chess import Move, Board, Piece
import heatmaps


def calculate_heatmap(board: Board, depth: int = 1,
                      heatmap: Optional[heatmaps.GradientHeatmap] = None,
                      discount: int = 1) -> heatmaps.GradientHeatmap:
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
      the `piece_counts` attribute. For a move from square s₁ to s₂ with moving piece P, the update is:

      - Move intensity:
        $$ H[s_2, c] \mathrel{+}= \frac{1}{\text{discount}} $$
      - Piece count:
        $$ \text{piece\_counts}[s_2][P] \mathrel{+}= \frac{1}{\text{discount}} $$

    - As with `calculate_heatmap`, the parameters `heatmap` and `discount` are intended for internal use.
    - The recursion depth controls the number of future move layers considered. Only odd depths tend to provide
      balanced data between both players.
    - The time complexity is approximately O(b^d), where b ≈ 35 (the average branching factor in chess) and d is the depth.

    Examples
    --------
    >>> import chess
    >>> from heatmaps import ChessMoveHeatmap
    >>> from chmutils import calculate_chess_move_heatmap
    >>> brd = chess.Board()
    >>> depth1_cmhm = calculate_chess_move_heatmap(brd, depth=1)
    >>> print(depth1_cmhm.colors)
    >>> depth2_cmhm = calculate_chess_move_heatmap(brd, depth=2)
    >>> print(depth2_cmhm.colors)
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


def get_cached_chess_move_heatmap(board: Board, depth: int = 1) -> heatmaps.ChessMoveHeatmap:
    """
    Retrieve or compute a ChessMoveHeatmap for the given board and recursion depth.

    This function first constructs a cache key based on the board's FEN and the specified depth.
    It then checks a SQLite database to see if the heatmap primitives have been stored. If so,
    it deserializes the stored data and initializes a new ChessMoveHeatmap object with it.
    If not, it calls calculate_chess_move_heatmap to compute the heatmap, caches the primitive values,
    and returns the new ChessMoveHeatmap.

    Parameters
    ----------
    board : chess.Board
        The chess board position for which to calculate the move heatmap.
    depth : int, optional
        The recursion depth for heatmap calculation. Higher values yield more detailed heatmaps,
        but at the cost of increased computation time. Default is 1.

    Returns
    -------
    ChessMoveHeatmap
        The computed (or cached) chess move heatmap containing both move intensities and per-square
        piece count dictionaries.
    """
    # Construct a unique key from the board's FEN and the depth
    key = f"{board.fen()}_{depth}"

    # Connect to (or create) the SQLite database
    conn = sqlite3.connect("heatmap_cache.db")
    cur = conn.cursor()

    # Create the cache table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS heatmap_cache (
            key TEXT PRIMARY KEY,
            data BLOB,
            piece_counts BLOB
        )
    """)
    conn.commit()

    # Try to fetch cached data
    cur.execute("SELECT data, piece_counts FROM heatmap_cache WHERE key = ?", (key,))
    row = cur.fetchone()
    if row is not None:
        data_blob, piece_counts_blob = row
        data = pickle.loads(data_blob)
        piece_counts = pickle.loads(piece_counts_blob)
        # Create a new ChessMoveHeatmap object and initialize its internal attributes.
        # (Assuming that the constructor of ChessMoveHeatmap accepts a 'data' parameter,
        # and that you can set the protected attribute _piece_counts directly.)
        heatmap = heatmaps.ChessMoveHeatmap(data=data)
        heatmap._piece_counts = piece_counts  # bypass the read-only property if necessary
        conn.close()
        return heatmap
    else:
        # No cached data; compute the heatmap.
        heatmap = calculate_chess_move_heatmap(board, depth=depth)
        # Serialize the primitives.
        data_blob = pickle.dumps(heatmap.data)
        piece_counts_blob = pickle.dumps(heatmap.piece_counts)
        # Insert into the database.
        cur.execute("INSERT INTO heatmap_cache (key, data, piece_counts) VALUES (?, ?, ?)",
                    (key, data_blob, piece_counts_blob))
        conn.commit()
        conn.close()
        return heatmap


# Assume PIECE_KEYS is defined (e.g., from chess.UNICODE_PIECE_SYMBOLS.values())
PIECE_KEYS = tuple(chess.UNICODE_PIECE_SYMBOLS.values())


def flatten_heatmap(cmhm: heatmaps.ChessMoveHeatmap) -> Dict[str, Any]:
    """
    Flatten a ChessMoveHeatmap into a dictionary of primitive values.
    Keys will be strings like "sq0_white", "sq0_black", and
    "sq0_piece_<symbol>" for each square (0-63).
    """
    flat = {}
    for square in range(64):
        flat[f"sq{square}_white"] = float(cmhm.data[square][0])
        flat[f"sq{square}_black"] = float(cmhm.data[square][1])
        # For each piece key, store its count from piece_counts.
        for key in PIECE_KEYS:
            flat[f"sq{square}_piece_{key}"] = float(cmhm.piece_counts[square].get(key, 0))
    return flat


def inflate_heatmap(data: Dict[str, Any]) -> heatmaps.ChessMoveHeatmap:
    """
    Inflate a flat dictionary of primitive values back into a ChessMoveHeatmap.
    Assumes the dictionary has keys in the format defined in flatten_heatmap.
    """
    # Create a new heatmap object.
    cmhm = heatmaps.ChessMoveHeatmap()
    for square in range(64):
        # Update intensity values.
        cmhm.data[square][0] = data.get(f"sq{square}_white", 0.0)
        cmhm.data[square][1] = data.get(f"sq{square}_black", 0.0)
        # Reconstruct the piece_counts dictionary.
        counts = {}
        for key in PIECE_KEYS:
            counts[key] = data.get(f"sq{square}_piece_{key}", 0.0)
        # Assuming piece_counts is writable (or using a protected attribute)
        cmhm._piece_counts[square] = counts
    return cmhm


class HeatmapCache:
    """
    A caching mechanism for ChessMoveHeatmap objects using SQLite.

    This class stores flattened heatmap data in a SQLite database, indexed by a key
    constructed from the board's FEN string and the recursion depth (e.g., "fen_depth").
    If the key exists in the database, the cached heatmap is returned; otherwise, the heatmap
    is computed, stored in the database, and then returned.
    """
    depth: int
    board: chess.Board
    db_path: str

    def __init__(self, board: chess.Board, depth: int):
        self.depth = depth
        self.db_path = f"heatmap_cache_depth_{self.depth}.db"
        self.board = board
        self._initialize_db()

    def _initialize_db(self) -> None:
        """
        Create the cache table if it does not exist.
        The schema here uses one row per heatmap; each row contains many columns corresponding
        to each flattened primitive value.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Construct column definitions for each square.
        # For each square we store white and black intensities and piece counts for each key.
        col_defs = []
        for square in range(64):
            col_defs.append(f"sq{square}_white REAL")
            col_defs.append(f"sq{square}_black REAL")
            for key in PIECE_KEYS:
                safe_key = key.replace(" ", "_")
                col_defs.append(f"sq{square}_piece_{safe_key} REAL")

        # Join all column definitions.
        cols = ", ".join(col_defs)
        create_stmt = f"CREATE TABLE IF NOT EXISTS heatmap_cache (cache_key TEXT PRIMARY KEY, {cols})"
        cur.execute(create_stmt)
        conn.commit()
        conn.close()

    @property
    def _cache_key(self) -> str:
        """Build a unique cache key from the board FEN and depth."""
        return f"{self.board.fen()}"

    def get_cached_heatmap(self) -> Optional[heatmaps.ChessMoveHeatmap]:
        """Retrieve a cached ChessMoveHeatmap for the given board and depth, if available."""
        key = self._cache_key
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM heatmap_cache WHERE cache_key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        if row:
            # Convert the row (tuple) into a dict where column names are keys.
            # For simplicity, assume the column order is known.
            col_names = [description[0] for description in cur.description]
            data = dict(zip(col_names, row))
            # Remove the cache_key from the dict
            data.pop("cache_key", None)
            return inflate_heatmap(data)
        return None

    def store_heatmap(self, cmhm: heatmaps.ChessMoveHeatmap) -> None:
        """
        Store a ChessMoveHeatmap in the cache.

        The heatmap is flattened into a dictionary of primitive values and inserted into the database.
        """
        key = self._cache_key
        flat = flatten_heatmap(cmhm)

        # Prepare the column names and placeholders.
        # We'll assume the schema matches the flattened dict keys exactly.
        columns = ["cache_key"] + list(flat.keys())
        placeholders = ", ".join(["?"] * len(columns))
        values = [key] + [flat[col] for col in flat]

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        # Use INSERT OR REPLACE to update existing cache if needed.
        sql = f"INSERT OR REPLACE INTO heatmap_cache ({', '.join(columns)}) VALUES ({placeholders})"
        cur.execute(sql, values)
        conn.commit()
        conn.close()

    def get_or_compute_heatmap(self) -> heatmaps.ChessMoveHeatmap:
        """
        Retrieve the ChessMoveHeatmap from the cache if available; otherwise, compute it,
        store it in the cache, and return the new heatmap.
        """
        cached = self.get_cached_heatmap()
        if cached is not None:
            return cached

        # Not cached; compute the heatmap.
        cmhm = calculate_chess_move_heatmap(board, depth=self.depth)
        self.store_heatmap(cmhm)
        return cmhm


# Example usage:
if __name__ == "__main__":
    import chess

    board = chess.Board()
    cmhm1 = HeatmapCache(board, depth=1).get_or_compute_heatmap()
    print(cmhm1.data[16])
    cmhm2 = HeatmapCache(board, depth=3).get_or_compute_heatmap()
    print(cmhm2.data[16])

    # while True:
    #     moves = list(board.legal_moves)
    #     [print(i, m) for i, m in enumerate(moves)]
    #     move = input("Pick move by index")
    #     try:
    #         i = int(move)
    #         board.push(moves[i])
    #         cmhm = cache.get_or_compute_heatmap(board, depth=1)
    #         print(cmhm.colors)
    #     except IndexError as e:
    #         print(e)
# # Example usage:
# if __name__ == "__main__":
#     import chess
#
#     board = chess.Board()
#     cmhm0 = get_cached_chess_move_heatmap(board, depth=1)
#     print(cmhm0.colors)
#     cmhm1 = get_cached_chess_move_heatmap(board, depth=1)
#     print(cmhm1.colors)

# if __name__ == "__main__":
#     from timeit import timeit
#
#     b = chess.Board()
#
#     for d in range(4):
#         hmap_t = timeit("calculate_heatmap(board=b, depth=d)", number=1, globals=globals())
#         print(d, hmap_t)
#
#     hmap = calculate_heatmap(board=b, depth=1)
#     print(hmap)
#
#     for d in range(4):
#         cmhmap_t = timeit("calculate_chess_move_heatmap(board=b, depth=d)", number=1, globals=globals())
#         print(d, cmhmap_t)
#
#     cmhmap = calculate_chess_move_heatmap(board=b, depth=1)
#     print(cmhmap)
