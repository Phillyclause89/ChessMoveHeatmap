"""A once off script to migrate data from old schema to new schema"""

import sqlite3
from sqlite3 import Connection, Cursor
from os import path
from typing import Any, List

from chess import Board


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


# Create connections and tables for the new database files
def create_new_db_files(folder, depth=1) -> None:
    """sets up our new q-table schema"""
    piece_count: int
    for piece_count in range(2, 33):  # We are using dbs for 2-32 pieces
        db_path: str = path.join(folder, f"qtable_depth_{depth}_piece_count_{piece_count}.db")
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
def migrate_data(old_cursor) -> None:
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
            db_path: str = path.join(db_folder, f"qtable_piece_count_{piece_count}.db")
            conn: Connection = sqlite3.connect(db_path)
            cursor: Cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO q_table (state_fen, q_value)
                    VALUES (?, ?)
                """, (state_fen, q_value))
                conn.commit()
            except sqlite3.IntegrityError as sql_error:
                raise WTF(row, piece_count) from sql_error
            finally:
                conn.close()
        else:
            raise WTF(row, piece_count)


if __name__ == "__main__":
    # Paths to the old and new databases
    db_folder: str = path.join(".", "SQLite3Caches", "QTables")
    old_db_path: str = path.join(db_folder, "qtable_depth_1.db")  # Your original DB path
    # Where the new DB files will be stored

    # Initialize connections to old and new databases
    old_conn: Connection = sqlite3.connect(old_db_path)
    old_cur: Cursor = old_conn.cursor()

    # Create the new database files
    create_new_db_files(db_folder)

    # Migrate the data
    migrate_data(old_cur)

    # Close the old connection
    old_conn.close()

    print("Migration complete!")
