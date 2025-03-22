"""A once off script to migrate data from old schema to new schema"""
import argparse
import sqlite3
from argparse import ArgumentParser, Namespace
from sqlite3 import Connection, Cursor
from os import path
from typing import List, Optional, TextIO, Tuple
import numpy as np
from numpy.typing import NDArray
from chess import Board, Outcome


class WTF(Exception):
    r"""A WTF Exception, what more can I say ¯\_(ツ)_/¯"""

    def __init__(self, row: Optional[Tuple[str, float]], piece_count: int, outcome: Optional[Outcome] = None) -> None:
        self.row = row
        self.piece_count = piece_count
        # Customize the message as an Easter egg surprise
        if outcome is None:
            message = f"WTF?! Unexpected piece count: {piece_count} for row: {row}"
        elif isinstance(outcome, Outcome):
            message = f"WTF?! Unexpected Outcome: {outcome.termination.name} "
            message += f"for piece count: {piece_count} and row: {row}"
        else:
            message = f"WTF?! Unexpected Outcome: {type(outcome)} for piece count: {piece_count} and row: {row}"
        super().__init__(message)


def piece_count_and_outcome_from_fen(fen: str) -> Tuple[int, Outcome]:
    """

    Parameters
    ----------
    fen : str

    Returns
    -------
    int

    """
    board: Board = Board(fen)
    return len(board.piece_map()), board.outcome(claim_draw=True)


class OutcomeCounter:
    """Counts Outcome attributes"""
    _black_wins: int
    _white_wins: int
    _insufficient_material_count: int
    _stalemate_count: int
    _checkmate_count: int

    def __init__(self) -> None:
        self._checkmate_count = 0
        self._stalemate_count = 0
        self._insufficient_material_count = 0
        self._white_wins = 0
        self._black_wins = 0

    def update(self, outcome: Optional[Outcome]) -> None:
        r"""Inplace update of the OutcomeCounter via outcome

        Parameters
        ----------
        outcome : Optional[chess.Outcome]
            Object returned by chess.Board.outcome(claim_draw=True)

        Raises
        ------
        TypeError
            If outcome argument is not Optional[chess.Outcome]
        WTF
            Should never raise this ¯\_(ツ)_/¯

        """
        if isinstance(outcome, Outcome):
            if outcome.termination.name == Outcome.termination.CHECKMATE.name:
                self._checkmate_count += 1
                if outcome.winner:
                    self._white_wins += 1
                else:
                    self._black_wins += 1
            elif outcome.termination.name == Outcome.termination.STALEMATE.name:
                self._stalemate_count += 1
            elif outcome.termination.name == Outcome.termination.INSUFFICIENT_MATERIAL.name:
                self._insufficient_material_count += 1
            else:
                raise WTF(row=None, piece_count=-1, outcome=outcome)
        elif outcome is not None:
            raise TypeError(f"Argument for outcome must be type None or chess.Outcome, got {type(Outcome)}")

    @property
    def checkmate_count(self) -> int:
        """

        Returns
        -------
        int
        """
        return self._checkmate_count

    @property
    def stalemate_count(self) -> int:
        """

        Returns
        -------
        int
        """
        return self._stalemate_count

    @property
    def insufficient_material_count(self) -> int:
        """

        Returns
        -------
        int
        """
        return self._insufficient_material_count

    @property
    def black_wins(self) -> int:
        """

        Returns
        -------
        int
        """
        return self._black_wins

    @property
    def white_wins(self) -> int:
        """

        Returns
        -------
        int
        """
        return self._white_wins


def log_db_stats(
        cursor: Cursor,
        log_file_path: str,
        percentiles: Tuple[int, ...] = (0, 1, 5, 10, 20, 40, 50, 60, 80, 90, 95, 99, 100),
        cursor_statement: str = "SELECT state_fen, q_value FROM q_table"
) -> None:
    r"""Logs statistics about the database connected by the cursor argument.

    Parameters
    ----------
    cursor : sqlite3.Cursor
    log_file_path : str
    percentiles : Tuple[int, ...]
    cursor_statement : str

    Raises
    ------
    ValueError
        If the database file connected by *cursor* has no rows
    WTF
        Should never raise this ¯\_(ツ)_/¯

    """
    # Fetch all data from the old database
    cursor.execute(cursor_statement)
    rows: List[Optional[Tuple[str, float]]] = cursor.fetchall()
    row_count = len(rows)
    if row_count <= 0:
        raise ValueError(f"The database file connected by {cursor} fetched no rows from {cursor_statement}!")
    outcome_counter: OutcomeCounter
    piece_counts: List[Optional[int]]
    q_values: List[Optional[float]]
    outcome_counter, piece_counts, q_values = process_rows(rows)
    # Convert to numpy arrays for easier calculations
    q_values_np: NDArray[np.float64] = np.array(q_values, dtype=np.float64)
    piece_counts_np: NDArray[int] = np.array(piece_counts, dtype=int)
    # Calculate statistics
    min_q: NDArray[np.float64] = np.min(q_values_np)
    max_q: NDArray[np.float64] = np.max(q_values_np)
    mean_q: NDArray[np.float64] = np.mean(q_values_np)
    perc_q: NDArray[np.float64] = np.percentile(q_values_np, percentiles)
    min_piece_count: NDArray[int] = np.min(piece_counts_np)
    max_piece_count: NDArray[int] = np.max(piece_counts_np)
    mean_piece_count: NDArray[np.float64] = np.mean(piece_counts_np)
    perc_piece_count: NDArray[np.float64] = np.percentile(piece_counts_np, percentiles)
    # Log the statistics
    log: TextIO
    with open(log_file_path, 'w', encoding='UTF-8') as log:
        log.write("Database Statistics:\n")
        log.write(f"Total Positions: {row_count}\n")
        percentage: float = (outcome_counter.checkmate_count / row_count) * 100
        log.write(
            f"Total Checkmates: {outcome_counter.checkmate_count} ({percentage:.2f}%)"
        )
        percentage = (outcome_counter.white_wins / row_count) * 100
        log.write(
            f"Total White Wins: {outcome_counter.white_wins} ({percentage:.2f}%)"
        )
        percentage = (outcome_counter.black_wins / row_count) * 100
        log.write(
            f"Total Black Wins: {outcome_counter.black_wins} ({percentage:.2f}%)"
        )
        percentage = (outcome_counter.stalemate_count / row_count) * 100
        log.write(
            f"Total Stalemates: {outcome_counter.stalemate_count} ({percentage:.2f}%)"
        )
        percentage = (outcome_counter.insufficient_material_count / row_count) * 100
        log.write(
            f"Total Insufficient Materials: {outcome_counter.insufficient_material_count} ({percentage:.2f}%)"
        )
        log.write(f"Min q_value: {min_q}\n")
        log.write(f"Max q_value: {max_q}\n")
        log.write(f"Mean q_value: {mean_q}\n")
        log.write(f"q_value Percentiles:\n{perc_q}\n")
        log.write(f"Min Piece Count: {min_piece_count}\n")
        log.write(f"Max Piece Count: {max_piece_count}\n")
        log.write(f"Mean Piece Count: {mean_piece_count}\n")
        log.write(f"Piece Count Percentiles:\n{perc_piece_count}\n")
        log.write("\nPiece Count Distribution:\n")
        # Count piece count occurrences
        piece_count: NDArray[int]
        count: NDArray[int]
        for piece_count, count in zip(*np.unique(piece_counts_np, return_counts=True)):
            log.write(f"Piece Count {piece_count}: {count} positions\n")


def process_rows(rows: List[Tuple[str, float]]) -> Tuple[OutcomeCounter, List[Optional[int]], List[Optional[float]]]:
    """

    Parameters
    ----------
    rows : List[Tuple[str, float]]

    Returns
    -------
    Tuple[OutcomeCounter, List[Optional[int]], List[Optional[float]]]
    """
    # Initialize lists to hold the q_values and piece counts
    q_values: List[Optional[float]] = []
    piece_counts: List[Optional[int]] = []
    outcomes: List[Optional[Outcome]] = []
    outcome_counter: OutcomeCounter = OutcomeCounter()
    row: Tuple[str, float]
    for row in rows:
        state_fen: str
        q_value: float
        state_fen, q_value = row
        q_values.append(q_value)
        p_c: int
        outcome: Outcome
        p_c, outcome = piece_count_and_outcome_from_fen(state_fen)
        piece_counts.append(p_c)
        outcomes.append(outcome)
        outcome_counter.update(outcome)
    return outcome_counter, piece_counts, q_values


def new_qtable_filename(depth: int, piece_count: int) -> str:
    """returns f"qtable_depth_{`depth`}_piece_count_{`piece_count`}.db"

    Parameters
    ----------
    depth : int
    piece_count : int

    Returns
    -------
    str

    """
    return f"qtable_depth_{depth}_piece_count_{piece_count}.db"


def create_new_db_files(folder: str, depth: int = 1) -> None:
    """sets up our new q-table schema for migration.

    Creates upto 31 new db files:
    -----------------------------
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_2.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_3.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_4.db")

    *cont...*

    os.path.join(folder, f"qtable_depth_{depth}_piece_count_30.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_31.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_32.db")

    Parameters
    ----------
    folder : str
    depth : int

    """
    piece_count: int
    for piece_count in range(2, 33):  # We are using dbs for 2-32 pieces
        db_path: str = path.join(folder, new_qtable_filename(depth, piece_count))
        conn: Connection = sqlite3.connect(db_path)
        cursor: Cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS q_table (
                state_fen TEXT,
                q_value REAL,
                PRIMARY KEY (state_fen)
            )
        """)
        conn.commit()
        conn.close()


def migrate_data(old_cursor: Cursor, folder: str, depth: int = 1) -> None:
    """Migrates data from old_cursor (expected to be in old schema) to new db schema.

    Warning:
    --------
    **create_new_db_files(folder: str, depth: int)**  should be called to create the new db files
    **before** calling this function!

    Parameters
    ----------
    old_cursor : sqlite3.Cursor
    folder : str
    depth : int

    """
    # Fetch all data from the old database
    old_cursor.execute("SELECT state_fen, q_value FROM q_table")
    rows: List[Optional[Tuple[str, float]]] = old_cursor.fetchall()

    # For each row, calculate the piece count and insert into the appropriate new database
    row: Optional[Tuple[str, float]]
    for row in rows:
        state_fen: str
        q_value: float
        state_fen, q_value = row
        piece_count: int = piece_count_and_outcome_from_fen(state_fen)
        if 2 <= piece_count <= 32:
            # Insert into the correct new database
            db_path: str = path.join(folder, new_qtable_filename(depth, piece_count))
            conn: Connection
            with sqlite3.connect(db_path) as conn:
                cursor: Cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO q_table (state_fen, q_value)
                        VALUES (?, ?)
                    """, (state_fen, q_value))
                    conn.commit()
                except sqlite3.DatabaseError as sql_error:
                    raise WTF(row, piece_count) from sql_error
        else:
            raise WTF(row, piece_count)


if __name__ == "__main__":
    # Set up argument parser
    parser: ArgumentParser = argparse.ArgumentParser(
        description="Migrate data from the old database to the new schema.")
    parser.add_argument(
        "depth",
        type=int,
        nargs="?",  # Makes it optional
        default=1,  # Default value if no argument is provided
        help="The depth of the source database file (e.g., 1 for qtable_depth_1.db)"
    )
    parser.add_argument(
        "root",
        type=str,
        nargs="?",  # Makes it optional
        default=".",  # Default value if no argument is provided
        help="Root dir of the source database file (e.g. '.' for location of '__main__')"
    )
    parser.add_argument(
        "caches_dir",
        type=str,
        nargs="?",  # Makes it optional
        default="SQLite3Caches",  # Default value if no argument is provided
        help="dir within root\\ of the source database file (e.g. '.' for location of '__main__')"
    )
    parser.add_argument(
        "qtable_dir",
        type=str,
        nargs="?",  # Makes it optional
        default="QTables",  # Default value if no argument is provided
        help="dir within root\\caches_dir\\ of the source database file"
    )
    args: Namespace = parser.parse_args()
    # Path to the databases:
    db_folder: str = path.join(args.root, args.caches_dir, args.qtable_dir)
    old_db_path: str = path.join(db_folder, f"qtable_depth_{args.depth}.db")
    print(f"Old db path: {old_db_path}")
    log_file: str = path.join(db_folder, f"migration_qtable_depth_{args.depth}.log")
    print(f"Log file path: {log_file}")

    # Initialize connection to old database
    old_conn: Connection
    with sqlite3.connect(old_db_path) as old_conn:
        old_cur: Cursor = old_conn.cursor()
        # Log the statistics about the original database
        log_db_stats(cursor=old_cur, log_file_path=log_file)
        # Create the new database files
        create_new_db_files(folder=db_folder, depth=args.depth)
        # Migrate the data
        migrate_data(old_cursor=old_cur, folder=db_folder, depth=args.depth)

    print(f"Migration complete! Check {log_file} for stats.")
