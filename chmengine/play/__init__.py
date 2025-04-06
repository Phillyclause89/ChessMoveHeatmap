"""Play or Train the engine(s)"""
import datetime
from os import makedirs, path
from typing import Callable, List, Optional, Tuple
import chess
import numpy
from chess import Move, Outcome, pgn
from chmengine.engines.cmhmey1 import CMHMEngine
from chmengine.engines.cmhmey2 import CMHMEngine2

__all__ = ['PlayCMHMEngine']


class PlayCMHMEngine:
    """Play a game against the engine."""
    player_name: str = "Unknown"
    cpu_name: str = "chmengine.CMHMEngine"
    site: str = player_name
    game_round: int = 0
    round_results: List[Optional[chess.pgn.Game]]
    player_color: str = chess.COLOR_NAMES[1]
    player_index: int = 0
    cpu_color: str = chess.COLOR_NAMES[0]
    cpu_index: int = 1
    engine: CMHMEngine
    pgn_dir: str = "pgns"
    training_dir: str = "trainings"

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            player_color: str = player_color,
            player_index: int = player_index,  # Inverse of chess libs mapping of colors
            depth: int = 1,
            board: Optional[chess.Board] = None,
            player_name: str = player_name,
            site: str = site,
            game_round: int = game_round,
            engine: Optional[Callable] = None
    ) -> None:
        """

        Parameters
        ----------
        player_color : str
        player_index : int
        depth : int
        board : Optional[chess.Board]
        player_name : str
        site : str
        game_round : int

        Examples
        --------
        >>> from chmengine import PlayCMHMEngine
        >>> game = PlayCMHMEngine()
        >>> type(game.engine), game.game_round, game.site
        (<class 'chmengine.CMHMEngine'>, 0, 'Unknown')
        >>> game.cpu_name, game.cpu_color, game.cpu_index
        ('chmengine.CMHMEngine(depth=1)', 'black', 1)
        >>> game.player_name, game.player_color, game.player_index
        ('Unknown', 'white', 0)
        >>> game.engine.board
        Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        >>> game2 = PlayCMHMEngine(depth=3, player_name='Phillyclause89', player_color='black',site="Phil's Place")
        >>> game2.site
        "Phil's Place"
        >>> game2.cpu_name, game2.cpu_color, game2.cpu_index
        ('chmengine.CMHMEngine(depth=3)', 'white', 0)
        >>> game2.player_name, game2.player_color, game2.player_index
        ('Phillyclause89', 'black', 1)
        """
        self.pgn_dir = path.join(".", self.pgn_dir)
        self.training_dir = path.join(self.pgn_dir, self.training_dir)
        self.player_name = str(player_name)
        self.site = str(site)
        self.game_round = int(game_round)
        self.round_results = []
        self.engine = CMHMEngine(depth=depth, board=board) if engine is None else engine(depth=depth, board=board)
        self.cpu_name = f"""{str(self.engine.__class__).split("'")[1]}(depth={self.engine.depth})"""
        if (
                player_color != self.player_color and player_color.lower() in chess.COLOR_NAMES
        ) or (
                player_index != self.player_index and isinstance(player_index, int)
        ):
            self.player_index, self.cpu_index = self.cpu_index, self.player_index
            self.player_color, self.cpu_color = self.cpu_color, self.player_color

    def play(self, pick_by: str = "delta") -> None:
        """Play a game against the engine"""
        self.game_round += 1
        local_time = self.get_local_time()
        print(f"Round: {self.game_round} | Time: {str(local_time)}\n{self.engine.board}")
        other_moves: List[chess.Move] = list(self.engine.board.legal_moves)
        print(f"All legal moves: {', '.join([m.uci() for m in other_moves])}\nCalculating move scores...")
        my_move_choice: Tuple[chess.Move, numpy.float64] = self.engine.pick_move(pick_by=pick_by)
        print(f"My recommended move has a {pick_by} score of {my_move_choice[1]:.2f}: {my_move_choice[0]}")
        move_number: int = 1
        while other_moves:
            if self.engine.board.turn:
                if int(not self.engine.board.turn) == self.cpu_index:
                    print("I'm white, and thus it's my move!")
                    print(f"I'll play {move_number}. {my_move_choice[0]}")
                    self.engine.board.push(my_move_choice[0])
                else:
                    print(self.engine.board)
                    player_move_choice = chess.Move.from_uci(
                        input(f"You're white, and thus it's your move: {move_number}. ")
                    )
                    if player_move_choice in other_moves:
                        self.engine.board.push(player_move_choice)
            else:
                if int(not self.engine.board.turn) == self.cpu_index:
                    print("I'm black, and thus it's my move!")
                    print(f"I'll play {move_number}. {self.engine.board.move_stack[-1]} {my_move_choice[0]}")
                    self.engine.board.push(my_move_choice[0])
                    move_number += 1
                else:
                    print(self.engine.board)
                    player_move_choice = chess.Move.from_uci(
                        input(
                            f"You're black, and thus it's your move: {move_number}. {self.engine.board.move_stack[-1]} "
                        )
                    )
                    if player_move_choice in other_moves:
                        self.engine.board.push(player_move_choice)
                        move_number += 1
            print(self.engine.board)
            try:
                other_moves = list(self.engine.board.legal_moves)
                print(f"All legal moves: {', '.join([m.uci() for m in other_moves])}")
                my_move_choice = self.engine.pick_move(pick_by=pick_by)
                print(my_move_choice)
                print(f"My recommended move has a {pick_by} score of {my_move_choice[1]:.2f}: {my_move_choice[0]}")
            except ValueError:
                outcome = self.engine.board.outcome()
                game = pgn.Game.from_board(self.engine.board)
                if isinstance(self.engine, CMHMEngine2):
                    self.engine.update_q_values()
                print(f"Game Over: {outcome}\n{self.engine.board}")
                game_heads = game.headers
                game_heads["Event"] = f"{self.player_name} vs {self.cpu_name}" if (
                    self.cpu_index
                ) else f"{self.cpu_name} vs {self.player_name}"
                game_heads["Site"] = self.site
                game_heads["Round"] = str(self.game_round)
                self.set_all_datetime_headers(game_heads, local_time)
                game_heads["White"] = self.player_name if self.cpu_index else self.cpu_name
                game_heads["Black"] = self.cpu_name if self.cpu_index else self.player_name
                game_heads["Termination"] = outcome.termination.name
                game_heads["CMHMEngineMode"] = f"pick_by='{pick_by}'"
                game_heads["CMHMEngineDepth"] = str(self.engine.depth)
                file_name: str = path.join(
                    self.pgn_dir,
                    f"{game_heads['Date']}_{game_heads['Event'].replace(' ', '_')}_{game_heads['Round']}.pgn"
                )
                self.save_to_pgn(file_name, game)
                self.round_results.append(game)
                self.engine.board = chess.Board()
                break

    def save_to_pgn(self, file_name: str, game: pgn.Game) -> None:
        """

        Parameters
        ----------
        file_name : str
        game

        """
        if not path.isdir(self.pgn_dir):
            makedirs(self.pgn_dir)
        if not path.isdir(self.training_dir):
            makedirs(self.training_dir)
        with open(file_name, "w", encoding="utf-8") as file:
            print(game, file=file, end="\n\n")

    def train_cmhmey_jr(self, training_games: int = 1000, training_games_start: int = 0) -> None:
        """Trains engine. CMHMEngine2 specifically

        Parameters
        ----------
        training_games : int
        training_games_start : int
        """
        if not isinstance(self.engine, CMHMEngine2):
            raise TypeError(f"Current engine is not type chmengine.CMHMEngine2: {type(self.engine)}")
        for i in range(training_games_start, training_games):
            game_n: int = i + 1
            print(f"Game {game_n}")
            local_time: datetime = self.get_local_time()
            print(local_time)
            move_number: int = 0
            last_move: str = ""
            while self.engine.board.outcome() is None and not self.engine.board.can_claim_draw():
                move_number += 1
                print(self.engine.board)
                move: Move
                score: numpy.float64
                move, score = self.engine.pick_move()
                s_str: str = f"{score:.2f}"
                white_move_text: str = f"{move_number}. {move.uci()}: {s_str}"
                if len(self.engine.board.move_stack) > 0:
                    last_move = self.engine.board.move_stack[-1].uci()
                black_move_text: str = f"{move_number}. {last_move} {move.uci()}: {s_str}"
                print(white_move_text if self.engine.board.turn else black_move_text)
                self.engine.board.push(move)
            outcome: Outcome = self.engine.board.outcome(claim_draw=True)
            game = pgn.Game.from_board(self.engine.board)
            self.engine.update_q_values()
            game_heads = game.headers
            game_heads["Event"] = "CMHMEngine2 vs CMHMEngine2"
            game_heads["Site"] = "Kingdom of Phil"
            game_heads["Round"] = str(i + 1)
            self.set_all_datetime_headers(game_heads, local_time)
            game_heads["White"] = "CMHMEngine2"
            game_heads["Black"] = "CMHMEngine2"
            game_heads["Termination"] = outcome.termination.name
            game_heads["CMHMEngineDepth"] = str(self.engine.depth)
            file_name: str = path.join(
                self.training_dir,
                f"{game_heads['Date']}_{game_heads['Event'].replace(' ', '_')}_{game_heads['Round']}.pgn"
            )
            self.save_to_pgn(file_name, game)
            print(game)
            self.engine.board = chess.Board()

    def set_all_datetime_headers(self, game_heads: pgn.Headers, local_time: datetime) -> None:
        """

        Parameters
        ----------
        game_heads : chess.pgn.Headers
        local_time : datetime.datetime
        """
        game_heads["Date"] = local_time.strftime("%Y.%m.%d")
        game_heads["Timezone"] = str(local_time.tzinfo)
        self.set_utc_headers(game_heads, local_time)

    @staticmethod
    def get_local_time() -> datetime:
        """

        Returns
        -------
        datetime.datetime

        """
        return datetime.datetime.now(datetime.datetime.now().astimezone().tzinfo)

    @staticmethod
    def set_utc_headers(game_heads: pgn.Headers, local_time: datetime) -> None:
        """Sets UTC header info of pgn file data from local timestamp

        Parameters
        ----------
        game_heads : chess.pgn.Headers
        local_time : datetime.datetime
        """
        game_heads["UTCDate"] = local_time.astimezone(datetime.timezone.utc).strftime("%Y.%m.%d")
        game_heads["UTCTime"] = local_time.astimezone(datetime.timezone.utc).strftime("%H:%M:%S")
