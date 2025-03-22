"""A once off script to migrate data from old schema to new schema"""
import argparse
import sqlite3
from argparse import ArgumentParser, Namespace
from sqlite3 import Connection, Cursor
from os import path
from typing import Any, List, Optional, TextIO, Tuple, Union
import numpy as np  # For calculating statistics
from numpy.typing import NDArray
from chess import Board
from numpy import ndarray


# Create a custom exception class for Easter egg "WTF" errors
class WTF(Exception):
    r"""A WTF Exception, what more can I say ¯\_(ツ)_/¯"""

    def __init__(self, row: Any, piece_count: int) -> None:
        self.row = row
        self.piece_count = piece_count
        # Customize the message as an Easter egg surprise
        message = f"WTF?! Unexpected piece count {piece_count} for row {row}"
        super().__init__(message)


# Function to get the piece count from a FEN string
def piece_count_from_fen(fen: str) -> int:
    """

    Parameters
    ----------
    fen : str

    Returns
    -------
    int

    """
    board = Board(fen)
    return len(board.piece_map())


# Function to log stats about the original database
def log_db_stats(cursor: Cursor, log_file_path: str) -> None:
    """Logs statistics about the old database.

    Parameters
    ----------
    cursor : sqlite3.Cursor
    log_file_path : str
    """
    # Fetch all data from the old database
    cursor.execute("SELECT state_fen, q_value FROM q_table")
    rows: List[Optional[Tuple[str, float]]] = cursor.fetchall()

    # Initialize lists to hold the q_values and piece counts
    q_values: List[Optional[float]] = []
    piece_counts: List[Optional[int]] = []

    row: Tuple[str, float]
    for row in rows:
        state_fen: str
        q_value: float
        state_fen, q_value = row
        q_values.append(q_value)
        piece_counts.append(piece_count_from_fen(state_fen))

    # Convert to numpy arrays for easier calculations
    q_values_np: NDArray[np.float64] = np.array(q_values, dtype=np.float64)
    piece_counts_np: NDArray[int] = np.array(piece_counts, dtype=int)

    # Calculate statistics
    min_q: NDArray[np.float64] = np.min(q_values_np)
    max_q: NDArray[np.float64] = np.max(q_values_np)
    mean_q: NDArray[np.float64] = np.mean(q_values_np)
    min_piece_count: NDArray[int] = np.min(piece_counts_np)
    max_piece_count: NDArray[int] = np.max(piece_counts_np)
    mean_piece_count: NDArray[np.float64] = np.mean(piece_counts_np)

    # Log the statistics
    log: TextIO
    with open(log_file_path, 'w', encoding='UTF-8') as log:
        log.write("Database Statistics:\n")
        log.write(f"Total Positions: {len(rows)}\n")
        log.write(f"Min q_value: {min_q}\n")
        log.write(f"Max q_value: {max_q}\n")
        log.write(f"Mean q_value: {mean_q}\n")
        log.write(f"Min Piece Count: {min_piece_count}\n")
        log.write(f"Max Piece Count: {max_piece_count}\n")
        log.write(f"Mean Piece Count: {mean_piece_count}\n")
        log.write("\nPiece Count Distribution:\n")

        # Count piece count occurrences
        piece_count: Any
        count: Any
        for piece_count, count in zip(*np.unique(piece_counts_np, return_counts=True)):
            log.write(f"Piece Count {piece_count}: {count} positions\n")


def new_qtable_filename(depth: int, piece_count: int) -> str:
    """

    Parameters
    ----------
    depth : int
    piece_count : int

    Returns
    -------
    str

    """
    return f"qtable_depth_{depth}_piece_count_{piece_count}.db"


# Create connections and tables for the new database files
def create_new_db_files(folder: str, depth: int = 1) -> None:
    """sets up our new q-table schema for , 31 db files total:

    os.path.join(folder, f"qtable_depth_{depth}_piece_count_2.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_3.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_4.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_5.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_6.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_7.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_8.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_9.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_10.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_11.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_12.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_13.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_14.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_15.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_16.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_17.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_18.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_19.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_20.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_21.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_22.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_23.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_24.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_25.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_26.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_27.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_28.db")
    os.path.join(folder, f"qtable_depth_{depth}_piece_count_29.db")
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


# Function to migrate data from old database to new schema based on piece count
def migrate_data(old_cursor: Cursor, folder: str, depth: int = 1) -> None:
    """Migrates data from old db schema to new schema"""
    # Fetch all data from the old database
    old_cursor.execute("SELECT state_fen, q_value FROM q_table")
    rows: List[Any] = old_cursor.fetchall()

    # For each row, calculate the piece count and insert into the appropriate new database
    row: Any
    for row in rows:
        state_fen: Any
        q_value: Any
        state_fen, q_value = row
        piece_count: int = piece_count_from_fen(state_fen)
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
    args: Namespace = parser.parse_args()
    # Paths to the old and new databases
    db_folder: str = path.join("chmengine", "SQLite3Caches", "QTables")
    old_db_path: str = path.join(db_folder, f"qtable_depth_{args.depth}.db")  # Your original DB path
    print(f"Old db path: {old_db_path}")
    log_file: str = path.join(db_folder, f"migration_qtable_depth_{args.depth}.log")  # Log file for statistics
    print(f"Log file path: {log_file}")

    # Initialize connection to old database
    old_conn: Connection
    with sqlite3.connect(old_db_path) as old_conn:
        old_cur: Cursor = old_conn.cursor()
        # Log the statistics about the original database
        log_db_stats(old_cur, log_file)
        # Create the new database files
        create_new_db_files(db_folder, depth=args.depth)
        # Migrate the data
        migrate_data(old_cur, db_folder, depth=args.depth)

    print(f"Migration complete! Check {log_file} for stats.")
