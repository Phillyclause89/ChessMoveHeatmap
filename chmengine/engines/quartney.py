"""Cmhmey Jr's Mom, Quartney"""
from abc import ABCMeta, abstractmethod
from os import makedirs, path
from sqlite3 import Connection, Cursor, connect
from typing import Optional, Tuple, Union

from chess import Board, Move
from numpy import float64

from chmengine.utils import pieces_count_from_fen
from chmutils import CACHE_DIR


class Quartney(metaclass=ABCMeta):
    """Mother class of Cmhmey Jr., managing Q‑table persistence and move selection."""
    _depth: int
    board: Optional[Board]
    cache_dir: str = path.join(".", CACHE_DIR, "QTables")

    def qtable_filename(
            self,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            pieces_count: Optional[Union[int, str]] = None
    ) -> str:
        """Build the Q‑table filename based on depth and piece count.

        Parameters
        ----------
        fen : str, optional
            FEN string of the position. **Deprecated**—pass `board` instead.
        board : chess.Board, optional
            Board object to derive FEN (and piece count) if `fen` is not given.
        pieces_count : int or str, optional
            Explicit piece count to use. If provided, skips recomputing from board/FEN.

        Returns
        -------
        str
            File name of the form
            `"qtable_depth_{depth}_piece_count_{pieces_count}.db"`.

        Examples
        --------
        >>> import os
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.qtable_filename() in os.listdir(path=engine.cache_dir)
        True
        """
        return (
            f"qtable_depth_{self.depth}_"
            f"piece_count_{self.pieces_count(board=board, fen=fen) if pieces_count is None else pieces_count}.db"
        )

    def pieces_count(self, board: Optional[Board] = None, fen: Optional[str] = None) -> int:
        """Return the number of pieces on the board in O(1) time.

        This uses `Board.occupied.bit_count()` when given a `board`.
        The `fen` argument is supported for backward compatibility **but deprecated**.

        Parameters
        ----------
        board : chess.Board, optional
            Board whose pieces to count. Preferred parameter.
        fen : str, optional
            FEN string of position (fallback; deprecated).

        Returns
        -------
        int
            Total number of pieces on the board.

        Examples
        --------
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.pieces_count()
        32
        """
        return pieces_count_from_fen(
            fen=self.fen(board=board)
        ) if fen is None else pieces_count_from_fen(fen=fen)

    def qdb_path(
            self,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            pieces_count: Optional[Union[int, str]] = None,
    ) -> str:
        """Get the full path to the Q‑table database file.

        Parameters
        ----------
        fen : str, optional
            FEN string for file naming (deprecated).
        board : chess.Board, optional
            Board object for file naming.
        pieces_count : int or str, optional
            Explicit piece count override.

        Returns
        -------
        str
            Relative path: `<cache_dir>/<qtable_filename(...)>`

        Examples
        --------
        >>> import os
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.qdb_path() == os.path.join(engine.cache_dir, engine.qtable_filename())
        True
        """
        return path.join(self.cache_dir, self.qtable_filename(fen=fen, board=board, pieces_count=pieces_count))

    def _init_qdb(self) -> None:
        """Ensure the cache directory exists and initialize all Q‑table DBs.

        For each possible piece count from 2 up to 32, this creates a SQLite file
        (if missing) containing a single table `q_table(fen, q_value)`.

        Side Effects
        ------------
        - Creates `self.cache_dir` if not already present.
        - Creates or updates SQLite files under that directory.
        """
        if not path.isdir(self.cache_dir):
            makedirs(self.cache_dir)
        pieces_count: int
        for pieces_count in range(2, 33):  # We are using dbs for 2-32 pieces
            qdb_path: str = self.qdb_path(pieces_count=pieces_count)
            q_conn: Connection
            with connect(qdb_path) as q_conn:
                q_cursor: Cursor = q_conn.cursor()
                q_cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS q_table (
                        fen TEXT,
                        q_value REAL,
                        PRIMARY KEY (fen)
                    )
                    """
                )

    def get_q_value(
            self,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            pieces_count: Optional[int] = None
    ) -> Optional[float64]:
        """Retrieve a cached Q‑value for a position, or None if uncached.

        Parameters
        ----------
        fen : str, optional
            FEN of the position (deprecated; use `board`).
        board : chess.Board, optional
            Board object for filename lookup.
        pieces_count : int, optional
            Override piece count for selecting DB file.

        Returns
        -------
        float64 or None
            Stored Q‑value, or `None` if no entry exists.

        Examples
        --------
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> q = engine.get_q_value()
        """
        fen = self.fen(board=board) if fen is None else fen
        q_conn: Connection
        with connect(self.qdb_path(fen=fen, board=board, pieces_count=pieces_count)) as q_conn:
            q_cursor: Cursor = q_conn.cursor()
            q_cursor.execute(
                "SELECT q_value FROM q_table WHERE fen = ?",
                (fen,)
            )
            row: Optional[Tuple[float]] = q_cursor.fetchone()
            return float64(row[0]) if row is not None else None

    def set_q_value(
            self,
            value: float,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            pieces_count: Optional[int] = None
    ) -> None:
        """Insert or update the Q‑value for a given position in the DB.

        Parameters
        ----------
        value : float
            Q‑value to store.
        fen : str, optional
            Position FEN (deprecated; prefer `board`).
        board : chess.Board, optional
            Board object for filename lookup.
        pieces_count : int, optional
            Override piece count for selecting DB file.

        Examples
        --------
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.set_q_value(0.0, '1k6/8/8/8/8/3K4/8/8 w - - 0 1')
        """
        if fen is None:
            fen = self.fen(board=board)
        with connect(self.qdb_path(fen=fen, board=board, pieces_count=pieces_count)) as q_conn:
            q_cursor = q_conn.cursor()
            q_cursor.execute(
                "INSERT OR REPLACE INTO q_table (fen, q_value) VALUES (?, ?)",
                (fen, value)
            )

    @abstractmethod
    def fen(self, board: Board) -> str:
        """Return the FEN string representing `board`.

        Parameters
        ----------
        board : chess.Board
            Board to serialize.

        Returns
        -------
        str
            FEN string of `board`.
        """

    @abstractmethod
    def update_q_values(self, debug: bool = False) -> None:
        """Back‑propagate game outcome through the Q‑table.

        Pops moves from the current board history and adjusts each
        stored Q‑value in the database based on the final result
        (win/lose/draw).

        Parameters
        ----------
        debug : bool, default=False
            If `True`, print diagnostics for each back‑step.

        Side Effects
        ------------
        Updates the SQLite Q‑table entries for every move in the game.

        Examples
        --------
        >>> from io import StringIO
        >>> from chess import pgn
        >>> from chmengine import CMHMEngine2
        >>> pgn_buffer = StringIO(
        ...    '''
        ...    1. f3 e5 2. g4 Qh4# 0-1
        ...
        ...
        ...    '''
        ... )
        >>> game = pgn.read_game(pgn_buffer)
        >>> board = game.board()
        >>> for move in game.mainline_moves():
        ...     board.push(move)
        >>> engine = CMHMEngine2(board=board)
        >>> engine.fen()
        'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3'
        >>> engine.update_q_values()
        >>> engine.fen()
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        """

    @abstractmethod
    def pick_move(
            self,
            pick_by: str = "",
            board: Optional[Board] = None,
            debug: bool = False
    ) -> Tuple[Move, float64]:
        """Select a move based on heatmap scores and update its Q‑value.

        This method evaluates all legal moves on `board` (or on
        the engine’s current board if `board` is None), picks one of
        the top‑scoring moves at random, writes the new Q‑value to
        the database, and returns `(move, score)`.

        Parameters
        ----------
        pick_by : str, default=""
            Legacy parameter (ignored).
        board : chess.Board, optional
            Board to pick from; defaults to `self.board`.
        debug : bool, default=False
            If `True`, print the full move‑score table.

        Returns
        -------
        (chess.Move, numpy.float64)
            The chosen move and its score.

        Raises
        ------
        ValueError
            If there are no legal moves.

        Examples
        --------
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> move, score = engine.pick_move()
        """

    @property
    def depth(self) -> int:
        """Current search depth used for heatmap and Q‑table lookups.

        Returns
        -------
        int
            Depth ≥ 0.

        Examples
        --------
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.depth
        1
        """
        return self._depth

    @depth.setter
    def depth(self, new_depth: int):
        """Set the recursion depth and reinitialize Q‑tables.

        Parameters
        ----------
        new_depth : int
            New depth value (must be >=0)

        Raises
        ------
        ValueError
            If `new_depth < 0`.

        Side Effects
        ------------
        Updates `self._depth` and recreates the Q‑table databases
        under `self.cache_dir`.

        Examples
        --------
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.depth = 3
        >>> engine.depth
        3
        """
        if new_depth < 0:
            raise ValueError(f"Invalid depth, value must be greater than or equal to 0, got {new_depth}")
        self._depth = int(new_depth)
        self._init_qdb()
