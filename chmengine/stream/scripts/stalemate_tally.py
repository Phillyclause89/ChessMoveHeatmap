"""Count stalemate positions in PGN files grouped by ratings buckets.
This script reads PGN files, processes the games, and counts the number of stalemate positions
for each Elo rating bucket. It uses the chess library to handle PGN parsing and stalemate detection.

Context Request: 
    I have a list of chess games downloaded from the lichess db
    Like this some billion games
    I want to study all these games individually to see how many games out of how many games ended in a stalemate!
    The games which ended in a stalemate will be grouped under these rating categories of players:
        1000-1200, 1200-1500, 1500-1700, 1700-1900, 1900-2100, 2100-2500
    Only games which will be considered is when a game ended in a sudden stalemate by a blunder by a player who was
    in a completely winning position and lost the game due to stalemate.
    It generally happens in time pressure.
    I want to make a study how many players commit these mistakes in that rating ranges.
    Can anybody help me with that?
"""
from argparse import ArgumentParser, Namespace
from os import listdir, path
from typing import Dict, List, Optional, TextIO, Union

from chess import Board, Move
from chess.pgn import Game, read_game

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        description="Count stalemate positions in PGN files grouped by ratings buckets.")
    parser.add_argument(
        "--pgn_dir",
        type=str,
        default=path.join('.', "pgns"),
        help="Directory to PGN files to process"
    )
    parser.add_argument(
        '--elo_bucket_thresholds',
        nargs='+',
        type=List[int],
        default=[1200, 1500, 1700, 1900, 2100],
        help="Thresholds of the Elo rating buckets to use for grouping"
    )
    parser.add_argument(
        '--min_elo',
        type=Optional[int],
        default=None,
        help="Minimum Elo rating to consider"
    )
    parser.add_argument(
        '--max_elo',
        type=Optional[int],
        default=None,
        help="Maximum Elo rating to consider"
    )
    args: Namespace = parser.parse_args()
    pgn_dir: Union[str, bytes] = args.pgn_dir
    elo_bucket_thresholds: List[int] = args.elo_bucket_thresholds
    min_elo: Optional[int] = args.min_elo
    max_elo: Optional[int] = args.max_elo

    # Ensure elo_bucket_thresholds is sorted
    elo_bucket_thresholds.sort()
    if path.exists(pgn_dir):
        print(f"PGN directory: {pgn_dir}")
    else:
        print(f"PGN directory does not exist: {pgn_dir}")
        exit(1)
    f: Union[str, bytes]
    pgs: List[Union[str, bytes]] = [f for f in listdir(pgn_dir) if str(f).lower().endswith(".pgn")]
    if len(pgs) == 0:
        print(f"No PGN files found in directory: {pgn_dir}")
        exit(1)
    # Initialize the stalemate count for each Elo bucket
    elo_buckets: Dict[int, Union[Dict[str, Union[int, str]], Dict[str, Union[int, None, str]]]] = {}
    last_bucket: Optional[int] = None
    i: int
    k: int
    for i, k in enumerate(elo_bucket_thresholds):
        if i == 0:
            if min_elo is not None:
                elo_buckets[k] = {"count": 0, 'min': min_elo, 'max': k, 'name': f"{min_elo}-{k}"}
            else:
                elo_buckets[k] = {"count": 0, 'min': None, 'max': k, 'name': f"{k}-down"}
        else:
            elo_buckets[k] = {"count": 0, 'min': last_bucket, 'max': k, 'name': f"{last_bucket}-{k}"}
        last_bucket = k
    if max_elo is not None:
        elo_buckets[max_elo] = {"count": 0, 'min': last_bucket, 'max': max_elo, 'name': f"{last_bucket}-{max_elo}"}
    else:
        elo_buckets[last_bucket + 1] = {"count": 0, 'min': last_bucket, 'max': None, 'name': f"{last_bucket}+up"}

    # Process each PGN file
    pgn: Union[str, bytes]
    for pgn in pgs:
        print(f"Processing PGN file: {pgn}")
        file: TextIO
        with open(path.join(pgn_dir, pgn), "r") as file:
            game: Optional[Game] = read_game(file)
            if game is None or len(game.errors) > 0:
                print(f"Skipping file '{pgn}' due to read errors.")
                continue
            if game is not None:
                # Get the players' ratings
                white_elo_s: Optional[str] = game.headers.get("WhiteElo")
                black_elo_s: Optional[str] = game.headers.get("BlackElo")
                if white_elo_s is None or black_elo_s is None:
                    print(f"Skipping game due to missing Elo ratings: {game.headers}")
                    continue
                try:
                    white_elo: int = int(white_elo_s)
                    black_elo: int = int(black_elo_s)
                except ValueError:
                    print(f"Invalid Elo ratings: {white_elo}, {black_elo}")
                    continue

                # Check if the game ended in stalemate
                board: Board = Board()
                move: Move
                for move in game.mainline_moves():
                    try:
                        board.push(move)
                    except Exception:
                        print(f"Invalid move in game: {move}")
                        break
                if board.is_stalemate():
                    elo: int
                    for elo in (white_elo, black_elo):
                        bucket: Union[Dict[str, Union[int, str]], Dict[str, Union[int, None, str]]]
                        for bucket in elo_buckets.values():
                            if bucket['min'] is not None and bucket['max'] is not None:
                                if bucket['min'] <= elo < bucket['max']:
                                    bucket['count'] += 1
                                    continue
                            if bucket['min'] is None:
                                if elo < bucket['max']:
                                    bucket['count'] += 1
                                    continue
                            if bucket['max'] is None:
                                if elo > bucket['min']:
                                    bucket['count'] += 1
                                    continue

    # Print the results
    print("Stalemate counts by Elo bucket:")
    for value in elo_buckets.values():
        print(f"{value['name']}: {value['count']} Stalemates")
