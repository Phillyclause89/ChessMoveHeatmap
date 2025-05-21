"""CHMUtils"""
from datetime import datetime
from os import makedirs, path
from sqlite3 import Connection, Cursor, connect
from typing import Dict, List, Optional, Tuple, Union

from chess import Board, Move, Piece
from numpy import float64

from chmutils import base_chess_tk_app, concurrent, game_builder
from chmutils.base_chess_tk_app import (
    BaseChessTkApp,
    DARK_SQUARE_COLOR_PROMPT,
    DEFAULT_COLORS,
    DEFAULT_FONT,
    LIGHT_SQUARE_COLOR_PROMPT
)
from chmutils.concurrent import PPExecutor
from chmutils.game_builder import GBuilder
from chmutils.player import Player
from chmutils.promotion_dialog import PromotionDialog, get_promotion_choice
from heatmaps import ChessMoveHeatmap, GradientHeatmap, PIECES

__all__ = [
    # Mods
    'base_chess_tk_app',
    'concurrent',
    'game_builder',
    # Classes
    'Player',
    'PPExecutor',
    'GBuilder',
    'BaseChessTkApp',
    'HeatmapCache',
    'BetterHeatmapCache',
    'PromotionDialog',
    # Functions
    'calculate_heatmap',
    'calculate_chess_move_heatmap',
    'calculate_chess_move_heatmap_with_better_discount',
    'fill_depth_map_with_counts',
    'flatten_heatmap',
    'inflate_heatmap',
    'get_or_compute_heatmap',
    'get_or_compute_heatmap_with_better_discounts',
    'get_local_time',
    'is_within_bmp',
    'get_promotion_choice',
    # Constants
    'PIECE_KEYS',
    'CACHE_DIR',
    'DARK_SQUARE_COLOR_PROMPT',
    'LIGHT_SQUARE_COLOR_PROMPT',
    'DEFAULT_COLORS',
    'DEFAULT_FONT',
    # Mappings
    'state_faces',
    'state_faces_within_bmp'
]


def calculate_heatmap(
        board: Board, depth: int = 1,
        heatmap: Optional[GradientHeatmap] = None,
        discount: int = 1
) -> GradientHeatmap:
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


def calculate_chess_move_heatmap(
        board: Board, depth: int = 1,
        heatmap: Optional[ChessMoveHeatmap] = None,
        discount: int = 1) -> ChessMoveHeatmap:
    # noinspection PyUnresolvedReferences
    """Recursively computes a chess move heatmap that tracks both move intensities and piece counts.

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
    >>> from chess import Board
    >>> from chmutils import calculate_chess_move_heatmap
    >>> brd = Board()
    >>> depth1_cmhm = calculate_chess_move_heatmap(brd, depth=1)
    >>> print(", ".join([f"{p.unicode_symbol()}: {cnt}" for p, cnt in depth1_cmhm.piece_counts[16].items() if cnt]))
    ♙: 1.0, ♘: 1.0
    >>> depth2_cmhm = calculate_chess_move_heatmap(brd, depth=3)
    >>> print(", ".join([f"{p.unicode_symbol()}: {cnt}" for p, cnt in depth2_cmhm.piece_counts[16].items() if cnt]))
    ♙: 1.849999999999993, ♘: 1.849999999999984, ♗: 0.10000000000000005, ♖: 0.05000000000000001, ♝: 0.09067002690459958
    """
    if heatmap is None:
        heatmap = ChessMoveHeatmap()

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


def calculate_chess_move_heatmap_with_better_discount(board: Board, depth: int = 1) -> ChessMoveHeatmap:
    # noinspection PyUnresolvedReferences
    """Recursively computes a chess move heatmap for a given board state while applying a discount
    to moves based on the branching factor at each level of recursion. This discounting approach
    reduces the weight of moves that occur in positions with many legal alternatives, thereby
    emphasizing moves from branches with fewer alternatives.

    The function first constructs a depth-specific accumulator (``depth_map``) that holds, for each depth level:
        - An integer count of the total number of legal moves (branches) encountered.
        - A ``ChessMoveHeatmap`` instance that aggregates move data:
            - Increments the target square’s value (indexed by the opponent’s color) for each move.
            - Tracks piece counts for moves arriving at each square.

    The accumulator is populated by the helper function ``fill_depth_map_with_counts``. Once all moves are
    processed up to the specified depth, the function aggregates the heatmaps from each depth level.
    Although the code iterates over ``depth_map[::-1]``, due to how recursion updates the accumulator via
    ``depth_map[depth]``, this reversed order corresponds to 'processing from the shallowest (initial board state)
    to the deepest level.' At each level, the heatmap values (both move counts and piece counts) are divided by a
    discount factor determined by the branch count, ensuring that moves from positions with many alternatives
    contribute proportionally less to the final heatmap.

    Parameters
    ----------
    board : chess.Board
        The current chess board state from which legal moves are generated.
    depth : int, optional
        The depth (or ply) to which moves are recursively explored. A depth of 1 considers only immediate legal
        moves, while higher depths recursively analyze subsequent moves. Default is 1.

    Returns
    -------
    heatmaps.ChessMoveHeatmap
        A composite chess move heatmap where each square’s value reflects the discounted cumulative influence
        of moves leading to that square. The heatmap also maintains aggregated piece movement counts.

    Examples
    --------
    >>> from chess import Board
    >>> from chmutils import calculate_chess_move_heatmap_with_better_discount
    >>> brd = Board()
    >>> depth1_cmhm = calculate_chess_move_heatmap_with_better_discount(brd, depth=1)
    >>> print(", ".join([f"{p.unicode_symbol()}: {cnt}" for p, cnt in depth1_cmhm.piece_counts[16].items() if cnt]))
    ♙: 1.0, ♘: 1.0
    >>> depth2_cmhm = calculate_chess_move_heatmap_with_better_discount(brd, depth=3)
    >>> print(", ".join([f"{p.unicode_symbol()}: {cnt}" for p, cnt in depth2_cmhm.piece_counts[16].items() if cnt]))
    ♙: 1.85, ♘: 1.85, ♗: 0.1, ♖: 0.05, ♝: 0.09110312289373175
    """
    # Initialize depth_map as a tuple of [branch_count, ChessMoveHeatmap] pairs for each depth level.
    depth_map: Tuple[List[Union[int, ChessMoveHeatmap]], ...] = tuple(
        [0, ChessMoveHeatmap()] for _ in range(depth + 1)
    )
    # Recursively populate the depth_map with move counts and heatmap data
    depth_map = fill_depth_map_with_counts(board, depth, depth_map)
    heatmap: ChessMoveHeatmap = ChessMoveHeatmap()
    # Aggregate the heatmaps from each depth level with appropriate discounting.
    # discount is the total number branches at each depth, initially this is always 1.
    discount: int = 1
    branches: int
    heatmap_at_depth: ChessMoveHeatmap
    # Although iterating over depth_map[::-1] might suggest processing from deepest to shallowest,
    # the recursion updates (via depth_map[depth]) mean that the reversed order processes from the shallowest
    # (initial board state) to the deepest level.
    for branches, heatmap_at_depth in depth_map[::-1]:
        heatmap += heatmap_at_depth / discount
        # Update the discount factor based on the branch count at this level.
        discount = branches if branches > 0 else discount
    return heatmap


def fill_depth_map_with_counts(
        board: Board,
        depth: int,
        depth_map: Tuple[List[Union[int, ChessMoveHeatmap]], ...]
) -> Tuple[List[Union[int, ChessMoveHeatmap]], ...]:
    """Recursively populates the depth_map accumulator with move counts and heatmap data for each depth level.

    This function explores all legal moves from the given board state and updates the accumulator at the
    specified depth. For each legal move, it performs the following:

    - Increments the branch count at the current depth level by the number of legal moves.
    - Updates the current depth’s ``ChessMoveHeatmap``:
        - Increases the count for the target square (indexed by the opponent’s color) by 1.0.
        - Increments the piece count for the target square based on the piece moving from the source square.

    If the current depth is greater than zero, the function recursively processes the move by creating a copy
    of the board, applying the move, and invoking itself with a decremented depth. The accumulator (``depth_map``)
    is updated in place, ensuring that each depth level tracks the cumulative counts and heatmap data.

    Parameters
    ----------
    board : chess.Board
        The current chess board state from which legal moves are generated.
    depth : int
        The remaining depth (or ply) to which moves are recursively explored.
    depth_map : Tuple[List[Union[int, heatmaps.ChessMoveHeatmap]], ...]
        An accumulator that tracks, for each depth level, both the branch count and the aggregated move data.
        Each element is a list where the first item is an integer representing the number of legal move branches,
        and the second item is a ``ChessMoveHeatmap`` that aggregates move and piece count data.

    Returns
    -------
    Tuple[List[Union[int, heatmaps.ChessMoveHeatmap]], ...]
        The updated depth_map accumulator, populated with move counts and heatmap data for all levels processed
        during recursion.

    Notes
    -----
    - This function is intended for internal use by ``calculate_chess_move_heatmap_with_better_discount``.
    - The branch counts accumulated at each depth level are later used to compute discount factors during the
        heatmap aggregation phase.
    """
    moves: Tuple[Move] = tuple(board.legal_moves)
    num_moves: int = len(moves)
    # Update the branch count for the current depth.
    depth_map[depth][0] += num_moves
    color_index: int = int(not board.turn)
    move: Move
    for move in moves:
        target_square: int = move.to_square
        # Update at the target square, the move's player's move count.
        depth_map[depth][1][target_square][color_index] += float64(1.0)
        from_square: int = move.from_square
        piece: Piece = board.piece_at(from_square)
        # Update at the target square, the piece count for move's piece.
        depth_map[depth][1].piece_counts[target_square][piece] += float64(1.0)

        if depth > 0:
            new_board: Board = board.copy()
            new_board.push(move)
            depth_map = fill_depth_map_with_counts(
                new_board,
                depth - 1,
                depth_map
            )
    return depth_map


# Assume PIECE_KEYS is defined (e.g., from chess.UNICODE_PIECE_SYMBOLS.values())
PIECE_KEYS: Tuple[Piece] = PIECES


def flatten_heatmap(heatmap: ChessMoveHeatmap) -> Dict[str, float64]:
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
    flat: Dict[str, float64] = {}
    square: int
    for square in range(64):
        flat[f"sq{square}_white"] = heatmap.data[square][0]
        flat[f"sq{square}_black"] = heatmap.data[square][1]
        # For each piece key, store its count from piece_counts.
        key: Piece
        for key in PIECE_KEYS:
            flat[f"sq{square}_piece_{key.unicode_symbol()}"] = heatmap.piece_counts[square][key]
    return flat


def inflate_heatmap(data: Dict[str, float]) -> ChessMoveHeatmap:
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
    heatmap: ChessMoveHeatmap = ChessMoveHeatmap()
    square: int
    for square in range(64):
        heatmap.data[square][0] = float64(data[f"sq{square}_white"])
        heatmap.data[square][1] = float64(data[f"sq{square}_black"])
        key: Piece
        for key in PIECE_KEYS:
            heatmap.piece_counts[square][key] = float64(data[f"sq{square}_piece_{key.unicode_symbol()}"])
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
    board: Board
    db_path: str
    cache_dir: str = path.join(".", CACHE_DIR, "Faster")

    def __init__(self, board: Board, depth: int) -> None:
        """Initialize the HeatmapCache.

        Parameters
        ----------
        board : chess.Board
            The chess board for which the heatmap cache is maintained.
        depth : int
            The recursion depth associated with the heatmap calculations.
        """
        self.depth = depth
        self.db_path = path.join(self.cache_dir, f"heatmap_cache_depth_{self.depth}.db")
        self.board = board
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create the cache table in the SQLite database if it does not exist.

        This method sets up the database schema with columns for each square's 
        move intensities and piece counts, and creates an index on the cache key 
        for faster lookups.
        """
        if not path.isdir(self.cache_dir):
            makedirs(self.cache_dir)
        conn: Connection
        with connect(self.db_path) as conn:
            cur: Cursor = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            # Construct column definitions for each square.
            col_definitions: List[str] = []
            square: int
            for square in range(64):
                col_definitions.append(f"sq{square}_white REAL")
                col_definitions.append(f"sq{square}_black REAL")
                key: Piece
                for key in PIECE_KEYS:
                    col_definitions.append(f"sq{square}_piece_{key.unicode_symbol()} REAL")

            cols: str = ", ".join(col_definitions)
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

    def get_cached_heatmap(self) -> Optional[ChessMoveHeatmap]:
        """Retrieve a cached ChessMoveHeatmap for the given board and depth, if available.

        Returns
        -------
        Union[heatmaps.ChessMoveHeatmap, None]
            The cached ChessMoveHeatmap if found; otherwise, None.
        """
        key: str = self._cache_key
        conn: Connection
        with connect(self.db_path) as conn:
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

    def store_heatmap(self, cmhm: ChessMoveHeatmap) -> None:
        """Store a ChessMoveHeatmap in the cache.

        The given heatmap is flattened into a dictionary of primitive values and inserted
        into the SQLite database. If an entry with the same cache key already exists, it is replaced.

        Parameters
        ----------
        cmhm : heatmaps.ChessMoveHeatmap
            The ChessMoveHeatmap object to be stored.
        """
        key: str = self._cache_key
        flat: Dict[str, float64] = flatten_heatmap(cmhm)

        # Prepare the column names and placeholders.
        # We'll assume the schema matches the flattened dict keys exactly.
        columns: List[str] = ["cache_key"] + list(flat.keys())
        placeholders: str = ", ".join(["?"] * len(columns))
        values: List[str, float64] = [key] + list(flat.values())

        conn: Connection
        with connect(self.db_path) as conn:
            cur: Cursor = conn.cursor()
            # Use INSERT OR REPLACE to update existing cache if needed.
            sql: str = f"INSERT OR REPLACE INTO heatmap_cache ({', '.join(columns)}) VALUES ({placeholders})"
            cur.execute(sql, values)


def get_or_compute_heatmap(board: Board, depth: int) -> ChessMoveHeatmap:
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
    cached: Optional[ChessMoveHeatmap] = cache.get_cached_heatmap()
    if cached is not None:
        return cached

    # Not cached; compute the heatmap.
    cmhm: ChessMoveHeatmap = calculate_chess_move_heatmap(board, depth=depth)
    cache.store_heatmap(cmhm)
    return cmhm


class BetterHeatmapCache(HeatmapCache):
    """Overrides cache_dir"""
    cache_dir: str = path.join(".", CACHE_DIR, "Better")


def get_or_compute_heatmap_with_better_discounts(board: Board, depth: int) -> ChessMoveHeatmap:
    """Retrieve a ChessMoveHeatmap from the Better cache, or compute and cache Better if not available.

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
    cache: BetterHeatmapCache = BetterHeatmapCache(board, depth)
    cached: Optional[ChessMoveHeatmap] = cache.get_cached_heatmap()
    if cached is not None:
        return cached

    # Not cached; compute the heatmap.
    cmhm: ChessMoveHeatmap = calculate_chess_move_heatmap_with_better_discount(board, depth=depth)
    cache.store_heatmap(cmhm)
    return cmhm


def get_local_time() -> datetime:
    """Gets time in local system time

    Returns
    -------
    datetime.datetime
    """
    return datetime.now(datetime.now().astimezone().tzinfo)


state_faces: Dict[str, Tuple[str, ...]] = {
    'winning': (
        '☺',
        '✨',
        '╰(*°▽°*)╯',
        '(❁´◡`❁)',
        "(●'◡'●)",
        '(⌐■_■)',
        '(☞ﾟヮﾟ)☞',
        '(¬‿¬)',
        '❤',
        ';)',
        ':)',
        ':-)',
        ':-D',
        ';-)',
        ';D',
        '（￣︶￣）↗　',
        '(～￣▽￣)～φ',
        '(゜▽゜*)',
        '♪(´▽`ʃ♡ƪ)',
        '╰(*°▽°*)╯o',
        '(*^▽^*)┛o',
        '(*￣▽￣*)ブ',
        '♪(^∇^*)',
        '(oﾟvﾟ)ノ',
        '(/≧▽≦)/',
        '(((o(*ﾟ▽ﾟ*)o)))',
        '♪(´▽｀)',
        r'\(￣︶￣*\))',
        '||ヽ(*￣▽￣*)ノミ',
        '|Юo(*°▽°*)o',
        '(ﾉ◕ヮ◕)ﾉ*:･ﾟ',
        '✧(o゜▽゜)o☆',
        'o(*￣▽￣*)oヽ',
        '(✿ﾟ▽ﾟ)ノ',
        'b(￣▽￣)d',
        'o(^▽^)o',
        '(⌐■‿■)',
        '(★‿★)',
        '( ͡• ͜ʖ ͡• )',
        '( ͡° ͜ʖ ͡°)',
        '( ͡~ ͜ʖ ͡°)',
        '(￣▽￣)',
        '（︶♡︶）',
        '༼ つ ◕͜◕ ༽つ',
        r'\(^o^)/',
        '≧◡≦',
        '(＾▽＾)',
        '☆*:.｡.o(≧▽≦)o.｡.:*☆',
        '☀',
        '✩',
        r'\(^o^)/',
        r'\(^ω^\)',
        '(ﾉ^∇^)ﾉ',
        '(ﾉ^o^)ﾉ',
        '^o^',
        '^ω^',
        '^v^',
        '(≧◡≦)',
        '(◕‿◕)',
        '(◠‿◠✿)',
        '(✿◠‿◠)',
        '(＾▽＾)',
        '(⌒▽⌒)☆',
        '(★‿★)',
        '☆*:.｡.o(≧▽≦)o.｡.:*☆',
        '☀',
        '☁',
        '☂',
        '✩',
        '✪',
    ),

    'losing': (
        '☹',
        '☠',
        '（；´д｀）ゞ',
        '＞﹏＜',
        '{{{(>_<)}}}',
        'ಥ_ಥ',
        '(っ °Д °;)っ',
        '(ง •_•)ง',
        '＼（〇_ｏ）／',
        '(•ˋ _ ˊ•)',
        '(⊙_◎)',
        '(●__●)',
        '(((φ(◎ロ◎;)φ)))',
        '(◎﹏◎)',
        '○|￣|_╮',
        '（╯＿╰）',
        '╭┌( ´_ゝ` )┐',
        '－_－b',
        '( ╯□╰ )',
        '(x_x)_〆',
        '(´Д｀ )',
        '(￣_￣|||)',
        '"╮(╯-╰)╭...',
        '( ＿ ＿)ノ｜',
        '(。﹏。)',
        '(。>︿<)_θ',
        '(+_+)?',
        '(＠_＠;)',
        '(⊙_⊙…)?',
        '(*￣０￣)ノ',
        '(＃°Д°)',
        '( ˘︹˘ )',
        '(* ￣︿￣)ヽ',
        '（≧□≦）ノ',
        '╚(•⌂•)╝',
        '○|￣|_ =3',
        '(╯‵□′)╯︵┻━┻',
        '(╯°□°）╯︵ ┻━┻┻━┻ ︵',
        '＼( °□° )／ ︵ ┻━┻',
        'o(一︿一+)o', '(￣﹏￣；)',
        '(ㆆ_ㆆ)',
        '<(＿　＿)>',
        '.·´¯`(>▂<)´¯`·.',
        '⊙﹏⊙',
        '∥┗( T﹏T )┛',
        '༼ つ ◕︵◕ ༽つ',
        'T_T',
        'Q_Q',
        ';_;',
        '._.',
        '(╥﹏╥)',
        '(ಥ﹏ಥ)',
        '(⋟﹏⋞)',
        '(╯︵╰,)',
        '(╯︵╰ )',
        '(╯_╰)',
        '(ಥ_ʖಥ)',
        '☁',
        '⚡',
        '⚡_⚡',
        '☂',
        '☁☁',
    ),

    'draw': (
        '^_^',
        r"¯\_(ツ)_/¯",
        '(¬_¬ )',
        '^_~',
        '^_____^',
        ':/',
        '+_+',
        '*_*',
        '=_=',
        '-_-',
        '⚆_⚆',
        '(⊙_◎)',
        '(●__●)',
        '◉_◉',
        '♨_♨',
        '←_←',
        '→_→',
        '<@_@>',
        '┑(￣Д ￣)┍',
        '(￣_,￣ )',
        '╮(╯-╰)╭',
        '……]((o_ _)彡',
        '☆ㄟ( ▔, ▔ )ㄏ',
        '(。_。)',
        '(>* - *<)',
        '(°°)～',
        '(‧‧)nnn≡',
        '[。。]≡',
        r'--\(˙<>˙)/--',
        '(ʘ ͟ʖ ʘ)',
        '( ͠° ͟ʖ ͡°)',
        r'¯\_( ͡° ͜ʖ ͡°)_/¯',
        r'¯\_(ツ)_/¯',
        '༼ つ ◕_◕ ༽つ',
        '(_　_)。゜zｚＺ',
        '(￣o￣) . z Z',
        'ヾ(^▽^*)))',
        '(～﹃～)~zZ',
        '(・_・)',
        '(･_･)',
        '(¬､¬)',
        '(–_–)',
        '(━_━)',
        '-.-',
        '._.',
        '(´-ω-` )',
        '(―_―) zzZ',
        '(¬_¬ )',
        '(ʅ( ‾⊖◝)ʃ)',
    )
}


def is_within_bmp(s: str) -> bool:
    """
    Check whether all characters in the string `s` have Unicode code points
    in the range U+0000 through U+FFFF (inclusive).

    :param s: the input string to test
    :return: True if every character is <= U+FFFF, False otherwise
    """
    return all(ord(ch) <= 0xFFFF for ch in s)


state_faces_within_bmp: Dict[str, Tuple[str, ...]] = {
    state: tuple(face for face in state_faces[state] if is_within_bmp(s=face)) for state in state_faces
}
