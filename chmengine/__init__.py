"""A silly chess engine that picks moves using heatmaps"""
import datetime
import random
from os import path
from typing import Dict, List, Optional, Tuple

import chess
from chess import pgn
import numpy

import heatmaps
import chmutils


class CMHMEngine:
    """A silly chess engine that picks moves using heatmaps"""
    _board: chess.Board
    _depth: int

    def __init__(self, board: Optional[chess.Board] = None, depth: int = 1) -> None:
        """

        Parameters
        ----------
        board : Optional[chess.Board]
        depth : int

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.board
        Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        >>> engine.depth
        1
        """
        try:
            self.board = board
        except TypeError as type_error:
            if board is not None:
                raise TypeError(type_error) from type_error
            self._board = chess.Board()
        self.depth = depth

    @property
    def depth(self) -> int:
        """Gets curret depth setting

        Returns
        -------
        int

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.depth
        1
        >>> engine.depth = 3
        >>> engine.depth
        3
        """
        return self._depth

    @depth.setter
    def depth(self, new_depth: int):
        """Sets curret depth setting if >= 0"""
        if new_depth < 0:
            raise ValueError(f"depth must be greater than or equal to 0, got {new_depth}")
        self._depth = new_depth

    @property
    def board(self) -> chess.Board:
        """Gets curret board object

        Returns
        -------
        chess.Board

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.board
        Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        >>> engine.board.push(chess.Move(8, 16))
        >>> engine.board
        Board('rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1')
        """
        return self._board

    @board.setter
    def board(self, board: chess.Board):
        try:
            assert board.is_valid()
            self._board = board
        except AssertionError as assertion_error:
            raise ValueError(f"The board is invalid: {board}") from assertion_error
        except AttributeError as attribute_error:
            raise TypeError(f"The board is of the wrong type: {type(board)}") from attribute_error

    def board_copy_pushed(self, move: chess.Move) -> chess.Board:
        """

        Parameters
        ----------
        move

        Returns
        -------
        chess.Board

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> new_board = engine.board_copy_pushed(chess.Move(8, 16))
        >>> new_board
        Board('rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1')
        >>> engine.board
        Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        >>> engine.board = new_board
        >>> engine.board
        Board('rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1')
        """
        board_copy: chess.Board = self.board.copy()
        board_copy.push(move)
        return board_copy

    def pick_move(self) -> Tuple[chess.Move, numpy.float64]:
        """
        Returns
        -------
        Tuple[chess.Move, numpy.float64]

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.pick_move()
        (Move.from_uci('e2e4'), 10.0)
        """
        color_index: int = int(not self.board.turn)
        other_index: int = int(self.board.turn)
        move_maps_items: List[Tuple[chess.Move, heatmaps.GradientHeatmap]] = self.get_or_calc_move_maps_list()
        target_moves: List[Tuple[Optional[chess.Move], Optional[numpy.float64]]] = [(None, None)]
        for move, heatmap in move_maps_items:
            transposed_map: numpy.ndarray = heatmap.data.transpose()
            current_player_sum: numpy.float64 = sum(transposed_map[color_index])
            other_player_sum: numpy.float64 = sum(transposed_map[other_index])
            delta: numpy.float64 = numpy.float64(current_player_sum - other_player_sum)
            current_best_detla: numpy.float64 = target_moves[0][1]
            if current_best_detla is None or delta > current_best_detla:
                target_moves = [(move, delta)]
            elif delta == current_best_detla:
                target_moves.append((move, delta))
        return random.choice(target_moves)

    def get_or_calc_move_maps_list(self) -> List[Tuple[chess.Move, heatmaps.GradientHeatmap]]:
        """

        Returns
        -------
        List[Tuple[chess.Move, heatmaps.GradientHeatmap]]

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.get_or_calc_move_maps_list()[0][0]
        Move.from_uci('g1h3')
        """
        return list(self.get_or_calc_move_maps().items())

    def get_or_calc_move_maps(self) -> Dict[chess.Move, heatmaps.GradientHeatmap]:
        """Gets or Calcs the heatmaps produced from pushing each move.

        Returns
        -------
        dict

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> move_maps = engine.get_or_calc_move_maps()
        >>> move_maps_items = list(move_maps.items())
        >>> move_maps_items[0][0]
        Move.from_uci('g1h3')
        >>> type(move_maps_items[0][1])
        <class 'heatmaps.ChessMoveHeatmap'>
        """
        moves: List[chess.Move] = list(self.board.legal_moves)
        if len(moves) > 0:
            return {
                move: chmutils.get_or_compute_heatmap_with_better_discounts(
                    board=self.board_copy_pushed(move),
                    depth=self.depth
                ) for move in moves
            }
        raise ValueError("Current Board has no legal moves.")


# pylint: disable=too-few-public-methods
class PlayCMHMEngine:
    """Play a game aginst the engine."""
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

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            player_color: str = player_color,
            player_index: int = player_index,  # Inverse of chess libs mapping of colors
            depth: int = 1,
            board: Optional[chess.Board] = None,
            player_name: str = player_name,
            site: str = site,
            game_round: int = game_round
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
        self.player_name = str(player_name)
        self.site = str(site)
        self.game_round = int(game_round)
        self.round_results = []
        self.engine = CMHMEngine(depth=depth, board=board)
        self.cpu_name = f"""{str(self.engine.__class__).split("'")[1]}(depth={self.engine.depth})"""
        if (
                player_color != self.player_color and player_color.lower() in chess.COLOR_NAMES
        ) or (
                player_index != self.player_index and isinstance(player_index, int)
        ):
            self.player_index, self.cpu_index = self.cpu_index, self.player_index
            self.player_color, self.cpu_color = self.cpu_color, self.player_color

    def play(self) -> None:
        """Play a game agianst the engine"""
        self.game_round += 1
        local_time = datetime.datetime.now(datetime.datetime.now().astimezone().tzinfo)
        print(f"Round: {self.game_round} | Time: {str(local_time)}")
        print(self.engine.board)
        other_moves: List[chess.Move] = list(self.engine.board.legal_moves)
        print(f"All legal moves: {', '.join([m.uci() for m in other_moves])}")
        my_move_choice: Tuple[chess.Move, numpy.float64] = self.engine.pick_move()
        print(f"My recommended move has a score of {my_move_choice[1]:.2f}: {my_move_choice[0]}")
        while other_moves:
            if self.engine.board.turn:
                if int(not self.engine.board.turn) == self.cpu_index:
                    print("I'm white, and thus it's my move!")
                    print(f"I'll play {my_move_choice[0]}")
                    self.engine.board.push(my_move_choice[0])
                else:
                    player_move_choice = chess.Move.from_uci(input("You're white, and thus it's your move: "))
                    if player_move_choice in other_moves:
                        self.engine.board.push(player_move_choice)
            else:
                if int(not self.engine.board.turn) == self.cpu_index:
                    print("I'm black, and thus it's my move!")
                    print(f"I'll play {my_move_choice[0]}")
                    self.engine.board.push(my_move_choice[0])
                else:
                    player_move_choice = chess.Move.from_uci(input("You're black, and thus it's your move: "))
                    if player_move_choice in other_moves:
                        self.engine.board.push(player_move_choice)
            print(self.engine.board)
            try:
                other_moves = list(self.engine.board.legal_moves)
                print(f"All legal moves: {', '.join([m.uci() for m in other_moves])}")
                my_move_choice = self.engine.pick_move()
                print(f"My recommended move has a score of {my_move_choice[1]:.2f}: {my_move_choice[0]}")
            except ValueError:
                print("Game Over:")
                print(self.engine.board)
                outcome = self.engine.board.outcome()
                game = pgn.Game.from_board(self.engine.board)
                game_heads = game.headers
                game_heads["Event"] = f"{self.player_name} vs {self.cpu_name}" if (
                    self.cpu_index
                ) else f"{self.cpu_name} vs {self.player_name}"
                game_heads["Site"] = self.site
                game_heads["Round"] = str(self.game_round)
                game_heads["Date"] = local_time.strftime("%Y.%m.%d")
                game_heads["Timezone"] = str(local_time.tzinfo)
                game_heads["UTCDate"] = local_time.astimezone(datetime.timezone.utc).strftime("%Y.%m.%d")
                game_heads["UTCTime"] = local_time.astimezone(datetime.timezone.utc).strftime("%H:%M:%S")
                game_heads["White"] = self.player_name if self.cpu_index else self.cpu_name
                game_heads["Black"] = self.cpu_name if self.cpu_index else self.player_name
                game_heads["Termination"] = outcome.termination.name
                fname = path.join(
                    ".",
                    "pgns",
                    f"{game_heads['Date']}_{game_heads['Event'].replace(' ', '_')}_{game_heads['Round']}.pgn"
                )
                with open(fname, "w", encoding="utf-8") as file:
                    print(game, file=file, end="\n\n")
                self.round_results.append(game)
                self.engine.board = chess.Board()
                break
