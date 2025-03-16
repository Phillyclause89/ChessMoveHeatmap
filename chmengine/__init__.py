"""A silly chess engine that picks moves using heatmaps"""
import random
import sqlite3
from bisect import bisect_left
import datetime
from os import makedirs, path
from typing import Any, Callable, Dict, List, Optional, Tuple
import chess
from chess import Board, Move, Outcome, Piece, pgn
import numpy
from numpy import float_
from numpy.typing import NDArray
import heatmaps
import chmutils
from heatmaps import GradientHeatmap


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
            self.depth = int(depth)
            self.board = board
        except TypeError as type_error:
            if board is not None:
                raise TypeError(type_error) from type_error
            self.depth = int(depth)
            self._board = chess.Board()

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
        """Sets current depth setting if >= 0"""
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

    def pick_move(self, pick_by: str = "all-delta") -> Tuple[chess.Move, numpy.float64]:
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
        >>> engine.pick_move(pick_by="all-max")
        (Move.from_uci('e2e4'), 30.0)
        >>> engine.pick_move(pick_by="all-min")[1]
        20.0
        >>> engine.pick_move(pick_by="all-delta")
        (Move.from_uci('e2e4'), 10.0)
        >>> engine.pick_move(pick_by="king-atk")[1]
        0.0
        >>> engine.pick_move(pick_by="king-def")[1]
        0.0
        >>> engine.pick_move(pick_by="king-delta")[1]
        0.0
        """
        current_index: int = self.current_player_heatmap_index
        other_index: int = self.other_player_heatmap_index
        move_maps_items: List[Tuple[chess.Move, heatmaps.GradientHeatmap]] = self.get_or_calc_move_maps_list()
        (
            target_moves_by_delta,
            target_moves_by_king_delta,
            target_moves_by_max_current,
            target_moves_by_max_other_king,
            target_moves_by_min_current_king,
            target_moves_by_min_other
        ) = self.null_target_moves()
        current_king_box: List[int]
        other_king_box: List[int]
        current_king_box, other_king_box = self.get_king_boxes()

        move: Move
        heatmap: GradientHeatmap
        for move, heatmap in move_maps_items:
            if self.heatmap_data_is_zeros(heatmap):
                return move, numpy.nan
            # target_moves_by_max_other_king == "king-atk"
            other_king_sum: numpy.float64
            (
                other_king_sum,
                target_moves_by_max_other_king
            ) = self.update_target_moves_by_max_other_king(
                target_moves_by_max_other_king,
                heatmap, move, current_index,
                other_king_box
            )
            # target_moves_by_min_current_king == "king-def"
            current_king_min: numpy.float64
            (
                current_king_min,
                target_moves_by_min_current_king
            ) = self.update_target_moves_by_min_current_king(
                target_moves_by_min_current_king,
                heatmap, move, other_index,
                current_king_box
            )
            # target_moves_by_king_delta == "king-delta"
            target_moves_by_king_delta = self.update_target_moves_by_king_delta(
                target_moves_by_king_delta, move,
                current_king_min, other_king_sum
            )

            transposed_map: NDArray[numpy.float64] = heatmap.data.transpose()
            # target_moves_by_max_current == "max"
            current_player_sum: numpy.float64
            (
                current_player_sum,
                target_moves_by_max_current
            ) = self.update_target_moves_by_max_current(
                target_moves_by_max_current,
                transposed_map, move,
                current_index
            )
            # target_moves_by_min_other == "min"
            other_player_sum: numpy.float64
            (
                other_player_sum,
                target_moves_by_min_other
            ) = self.update_target_moves_by_min_other(
                target_moves_by_min_other, transposed_map,
                move, other_index
            )
            # target_moves_by_delta == "delta"
            target_moves_by_delta = self.update_target_moves_by_delta(
                target_moves_by_delta, current_player_sum,
                other_player_sum, move
            )
        target_moves: List[Any] = []
        if "all-delta" in pick_by.lower():
            target_moves += target_moves_by_delta
        if "all-min" in pick_by.lower():
            target_moves += target_moves_by_min_other
        if "all-max" in pick_by.lower():
            target_moves += target_moves_by_max_current
        if "king-atk" in pick_by.lower():
            target_moves += target_moves_by_max_other_king
        if "king-def" in pick_by.lower():
            target_moves += target_moves_by_min_current_king
        if "king-delta" in pick_by.lower():
            target_moves += target_moves_by_king_delta

        return random.choice(target_moves)

    @staticmethod
    def null_target_moves(
            number: int = 6
    ) -> Tuple[List[Tuple[Optional[chess.Move], Optional[numpy.float64]]], ...]:
        """

        Parameters
        ----------
        number : int

        Returns
        -------
        Tuple[List[Tuple[Optional[chess.Move], Optional[numpy.float64]]]]

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.null_target_moves()
        ([(None, None)], [(None, None)], [(None, None)], [(None, None)], [(None, None)], [(None, None)])
        >>> engine.null_target_moves(1)
        ([(None, None)],)
        >>> a, b = engine.null_target_moves(2)
        >>> a, b
        ([(None, None)], [(None, None)])
        >>> a.append((None, None))
        >>> a, b
        ([(None, None), (None, None)], [(None, None)])
        """
        return tuple([(None, None)] for _ in range(number))

    @property
    def other_player_heatmap_index(self) -> int:
        """Get the heatmap index of the inactive player per the board state

        Returns
        -------
        int

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> from chess import Move
        >>> engine = CMHMEngine()
        >>> engine.other_player_heatmap_index
        1
        >>> engine.board.push(Move.from_uci("e2e4"))
        >>> engine.other_player_heatmap_index
        0
        """
        return int(self.board.turn)

    @property
    def current_player_heatmap_index(self) -> int:
        """Get the heatmap index of the active player per the board state

        Returns
        -------
        int

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> from chess import Move
        >>> engine = CMHMEngine()
        >>> engine.current_player_heatmap_index
        0
        >>> engine.board.push(Move.from_uci("e2e4"))
        >>> engine.current_player_heatmap_index
        1
        """
        return int(not self.board.turn)

    @staticmethod
    def update_target_moves_by_delta(
            target_moves_by_delta: List[Tuple[Optional[chess.Move], Optional[numpy.float64]]],
            current_player_sum: numpy.float64,
            other_player_sum: numpy.float64,
            move: chess.Move
    ) -> List[Tuple[chess.Move, numpy.float64]]:
        """

        Parameters
        ----------
        target_moves_by_delta
        current_player_sum
        other_player_sum
        move

        Returns
        -------

        """
        delta: numpy.float64 = numpy.float64(current_player_sum - other_player_sum)
        current_best_delta: numpy.float64 = target_moves_by_delta[0][1]
        if current_best_delta is None or delta > current_best_delta:
            target_moves_by_delta = [(move, delta)]
        elif delta == current_best_delta:
            target_moves_by_delta.append((move, delta))
        return target_moves_by_delta

    @staticmethod
    def update_target_moves_by_min_other(
            target_moves_by_min_other: List[Tuple[Optional[chess.Move], Optional[numpy.float64]]],
            transposed_map: numpy.ndarray,
            move: chess.Move,
            other_index: int
    ) -> Tuple[numpy.float64, List[Tuple[chess.Move, numpy.float64]]]:
        """

        Parameters
        ----------
        target_moves_by_min_other
        transposed_map
        move
        other_index

        Returns
        -------

        """
        other_player_sum: numpy.float64 = sum(transposed_map[other_index])
        current_best_min: numpy.float64 = target_moves_by_min_other[0][1]
        if current_best_min is None or current_best_min > other_player_sum:
            target_moves_by_min_other = [(move, other_player_sum)]
        elif current_best_min == other_player_sum:
            target_moves_by_min_other.append((move, other_player_sum))
        return other_player_sum, target_moves_by_min_other

    @staticmethod
    def update_target_moves_by_max_current(
            target_moves_by_max_current: List[Tuple[Optional[chess.Move], Optional[numpy.float64]]],
            transposed_map: numpy.ndarray,
            move: chess.Move,
            color_index: int
    ) -> Tuple[numpy.float64, List[Tuple[chess.Move, numpy.float64]]]:
        """

        Parameters
        ----------
        target_moves_by_max_current
        transposed_map
        move
        color_index

        Returns
        -------

        """
        current_player_sum: numpy.float64 = sum(transposed_map[color_index])
        current_best_max: numpy.float64 = target_moves_by_max_current[0][1]
        if current_best_max is None or current_player_sum > current_best_max:
            target_moves_by_max_current = [(move, current_player_sum)]
        elif current_best_max == current_player_sum:
            target_moves_by_max_current.append((move, current_player_sum))
        return current_player_sum, target_moves_by_max_current

    @staticmethod
    def update_target_moves_by_king_delta(
            target_moves_by_king_delta: List[Tuple[Optional[chess.Move], Optional[numpy.float64]]],
            move: chess.Move,
            current_king_min: numpy.float64,
            other_king_sum: numpy.float64
    ) -> List[Tuple[chess.Move, numpy.float64]]:
        """

        Parameters
        ----------
        target_moves_by_king_delta
        move
        current_king_min
        other_king_sum

        Returns
        -------

        """
        current_king_delta: numpy.float64 = numpy.float64(other_king_sum - current_king_min)
        current_best_king_delta: numpy.float64 = target_moves_by_king_delta[0][1]
        if current_best_king_delta is None or current_king_delta > current_best_king_delta:
            target_moves_by_king_delta = [(move, current_king_delta)]
        elif current_king_delta == current_best_king_delta:
            target_moves_by_king_delta.append((move, current_king_delta))
        return target_moves_by_king_delta

    @staticmethod
    def update_target_moves_by_min_current_king(
            target_moves_by_min_current_king: List[Tuple[Optional[chess.Move], Optional[numpy.float64]]],
            heatmap: heatmaps.GradientHeatmap,
            move: chess.Move,
            other_index: int,
            current_king_box: List[int]
    ) -> Tuple[numpy.float64, List[Tuple[chess.Move, numpy.float64]]]:
        """

        Parameters
        ----------
        target_moves_by_min_current_king
        heatmap
        move
        other_index
        current_king_box

        Returns
        -------

        """
        current_king_map = heatmap[current_king_box].transpose()[other_index]
        current_king_min = sum(current_king_map)
        current_best_king_min = target_moves_by_min_current_king[0][1]
        if current_best_king_min is None or current_best_king_min > current_king_min:
            target_moves_by_min_current_king = [(move, current_king_min)]
        elif current_king_min == current_best_king_min:
            target_moves_by_min_current_king.append((move, current_king_min))
        return current_king_min, target_moves_by_min_current_king

    @staticmethod
    def update_target_moves_by_max_other_king(
            target_moves_by_max_other_king: List[Tuple[Optional[chess.Move], Optional[numpy.float64]]],
            heatmap: heatmaps.GradientHeatmap,
            move: chess.Move,
            color_index: int,
            other_king_box: List[int]
    ) -> Tuple[numpy.float64, List[Tuple[chess.Move, numpy.float64]]]:
        """

        Parameters
        ----------
        target_moves_by_max_other_king
        heatmap
        move
        color_index
        other_king_box

        Returns
        -------

        """
        other_king_map = heatmap[other_king_box].transpose()[color_index]
        other_king_sum = sum(other_king_map)
        current_best_king_max = target_moves_by_max_other_king[0][1]
        if current_best_king_max is None or other_king_sum > current_best_king_max:
            target_moves_by_max_other_king = [(move, other_king_sum)]
        elif other_king_sum == current_best_king_max:
            target_moves_by_max_other_king.append((move, other_king_sum))
        return other_king_sum, target_moves_by_max_other_king

    @staticmethod
    def heatmap_data_is_zeros(heatmap: heatmaps.GradientHeatmap) -> bool:
        """

        Parameters
        ----------
        heatmap : heatmaps.GradientHeatmap

        Returns
        -------
        bool

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> from heatmaps import GradientHeatmap
        >>> hmap = GradientHeatmap()
        >>> CMHMEngine.heatmap_data_is_zeros(hmap)
        True
        >>> hmap[32][1] = 1.0
        >>> CMHMEngine.heatmap_data_is_zeros(hmap)
        False
        """
        return (heatmap.data == numpy.float64(0.0)).all()

    def get_king_boxes(self, board: Optional[chess.Board] = None) -> Tuple[List[int], List[int]]:
        """current_king_box, other_king_box

        Parameters
        ----------
        board : Optional[chess.Board]

        Returns
        -------
        Tuple[List[int], List[int]]
        """
        other_king_square: int = self.board.king(not self.board.turn) if board is None else board.king(not board.turn)
        current_king_square: int = self.board.king(self.board.turn) if board is None else board.king(board.turn)
        other_king_box: List[int] = [other_king_square]
        current_king_box: List[int] = [current_king_square]
        long: int
        for long in (-1, 0, 1):
            lat: int
            for lat in (-8, 0, +8):
                oks_box_id: int = other_king_square + long + lat
                cks_box_id: int = current_king_square + long + lat
                if self.is_valid_king_box_square(board, oks_box_id, other_king_square):
                    other_king_box.append(oks_box_id)
                if self.is_valid_king_box_square(board, cks_box_id, current_king_square):
                    current_king_box.append(cks_box_id)
        return current_king_box, other_king_box

    @staticmethod
    def is_valid_king_box_square(board: chess.Board, square_id: int, king_square: int) -> bool:
        """

        Parameters
        ----------
        board : chess.Board
        square_id : int
        king_square : int

        Returns
        -------
        bool
        """
        if square_id < 0 or 63 < square_id or chess.square_distance(king_square, square_id) != 1:
            return False
        piece_at_square: Optional[Piece] = board.piece_at(square_id)
        if isinstance(piece_at_square, chess.Piece) and piece_at_square.color != board.piece_at(king_square).color:
            return False
        return True

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

    def get_or_calc_move_maps(self, depth: Optional[int] = None) -> Dict[chess.Move, heatmaps.GradientHeatmap]:
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
        moves = self.current_moves_list()
        if len(moves) > 0:
            return {
                move: chmutils.get_or_compute_heatmap_with_better_discounts(
                    board=self.board_copy_pushed(move),
                    depth=self.depth if depth is None else depth
                ) for move in moves
            }
        raise ValueError("Current Board has no legal moves.")

    def current_moves_list(self, board: Optional[chess.Board] = None) -> List[chess.Move]:
        """

        Parameters
        ----------
        board : Optional[chess.Board]

        Returns
        -------
        List[chess.Move]

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> from chess import Move
        >>> engine = CMHMEngine()
        >>> engine.current_moves_list()[15]
        Move.from_uci('e2e4')
        >>> engine.current_moves_list(engine.board_copy_pushed(Move.from_uci('e2e4')))[15]
        Move.from_uci('e7e5')
        >>> engine.current_moves_list()[15]
        Move.from_uci('e2e4')
        >>> engine.board.push(Move.from_uci('e2e4'))
        >>> engine.current_moves_list()[15]
        Move.from_uci('e7e5')
        """
        return list(self.board.legal_moves) if board is None else list(board.legal_moves)


class CMHMEngine2(CMHMEngine):
    """Overrides CMHMEngine.pick_move"""
    qdb_path: str
    cache_dir: str = path.join(".", chmutils.CACHE_DIR, "QTables")

    def __init__(self, board: Optional[chess.Board] = None, depth: int = 1) -> None:
        super().__init__(board, depth)
        self.qdb_path = path.join(self.cache_dir, f"qtable_depth_{self.depth}.db")
        self._init_qdb()  # List to store moves made during the game as (state_fen, move_uci) pairs.

    def _init_qdb(self) -> None:
        """Initializes the Q-table database and creates the table if it does not exist."""
        if not path.isdir(self.cache_dir):
            makedirs(self.cache_dir)
        with sqlite3.connect(self.qdb_path) as q_conn:
            q_cursor = q_conn.cursor()
            q_cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS q_table (
                    state_fen TEXT,
                    q_value REAL,
                    PRIMARY KEY (state_fen)
                )
                """
            )

    def get_q_value(self, state_fen: str) -> Optional[float]:
        """Returns the Q-value for a given state and move, or None if not found."""
        with sqlite3.connect(self.qdb_path) as q_conn:
            q_cursor = q_conn.cursor()
            q_cursor.execute(
                "SELECT q_value FROM q_table WHERE state_fen = ?",
                (state_fen,)
            )
            row: Optional[Tuple[float, ...]] = q_cursor.fetchone()
            return row[0] if row is not None else None

    def set_q_value(self, state_fen: str, value: float) -> None:
        """Sets or updates the Q-value for a given state and move."""
        with sqlite3.connect(self.qdb_path) as q_conn:
            q_cursor = q_conn.cursor()
            q_cursor.execute(
                "INSERT OR REPLACE INTO q_table (state_fen, q_value) VALUES (?, ?)",
                (state_fen, value)
            )

    def update_q_values(self) -> None:
        """Update Q Table after game termination

        pops all moves out of the board object

        """
        outcome: Optional[Outcome] = self.board.outcome(claim_draw=True)
        while len(self.board.move_stack) > 0:
            state: str = self.board.fen()
            current_q: Optional[float] = self.get_q_value(state)
            if current_q is None:
                self.board.pop()
                continue
            # The q score of a board fen is relative to the player who just moved
            if outcome.winner is None:  # Not checking for draws first causes bugs in training
                # In draw state we converge all scores on zero by 20%
                if current_q < 0:
                    new_q = current_q + abs(current_q * 0.2)
                elif current_q > 0:
                    new_q = current_q - abs(current_q * 0.2)
                else:
                    new_q = current_q
            # In checkmate states, the board turn is equal to the loser (this also caused me bugs at first)
            elif self.board.turn != outcome.winner:
                # Winner gets a 20% bump to their chosen move's scores (now with a +0.1 bump to brake past 0 if needed)
                new_q = current_q + abs(current_q * 0.2) + numpy.float64(0.1)
            elif self.board.turn == outcome.winner:
                # Loser gets a -20% anti-bump  to their chosen move's scores (now with a -0.1 anti-bump to brake past 0)
                new_q = current_q - abs(current_q * 0.2) - numpy.float64(0.1)
            else:
                raise ValueError(f"How did we get here? outcome:{outcome} board turn: {self.board.turn}")
            self.set_q_value(state, new_q)
            self.board.pop()

    def pick_move(
            self,
            pick_by: str = "",
            early_exit: bool = False,
            king_box_multiplier: int = 1
    ) -> Tuple[chess.Move, numpy.float64]:
        """Overides parents pick_move method

        Parameters
        ----------
        pick_by : str
        early_exit : bool
        king_box_multiplier : int

        Returns
        -------
        Tuple[chess.Move, numpy.float64]

        Examples
        --------
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.pick_move()
        (Move.from_uci('e2e4'), 10.0)
        """
        current_moves: List[Move] = self.current_moves_list()
        # PlayCMHMEngine.play() needs a ValueError to detect game termination.
        if len(current_moves) == 0:
            raise ValueError(f"Current Board has no legal moves: {self.board.fen()}")
        # Index of heatmap colors is opposite the chess lib's color mapping
        current_index: int = self.current_player_heatmap_index
        other_index: int = self.other_player_heatmap_index
        # moves will be current moves ordered by engine's score best to worst
        moves: List[Tuple[Optional[Move], Optional[float_]]]
        moves, = self.null_target_moves(1)
        move: Move
        for move in current_moves:
            move_uci: str = move.uci()
            new_board: Board = self.board_copy_pushed(move)
            state_fen = new_board.fen()
            q_val = self.get_q_value(state_fen)
            if q_val is None:
                print(f"Calculating score for move: {move_uci}")
                new_current_king_box: List[int]
                new_other_king_box: List[int]
                new_current_king_box, new_other_king_box = self.get_king_boxes(new_board)
                new_heatmap: heatmaps.ChessMoveHeatmap = chmutils.get_or_compute_heatmap_with_better_discounts(
                    new_board, depth=self.depth
                )
                new_heatmap_transposed: NDArray[numpy.float64] = new_heatmap.data.transpose()
                # I wanted to rely on the heatmap as much as possible, but any game termination state win or draw
                # results in a zeros heatmap. Thus, we cheat with new_board.is_checkmate here. (draw scores stay zero.)
                if self.heatmap_data_is_zeros(new_heatmap) and new_board.is_checkmate():
                    # Checkmate moves get scored as all the surviving pieces (opponent's pieces are POWs now)
                    # being able to move to every square (the war is over, pieces can go hangout on the same square
                    # with their surviving buddies). Anyway, this should just about guarantee checkmates get
                    # a score higher than what a heatmap can produce from a position.
                    new_heatmap_transposed[current_index] = numpy.float64(
                        len(new_board.piece_map()) * (self.depth + 1)
                    )
                    if early_exit:
                        return move, sum(new_heatmap_transposed[current_index])
                initial_move_score = self.calculate_score(
                    current_index, other_index, new_heatmap_transposed,
                    king_box_multiplier, new_current_king_box, new_other_king_box
                )
                # However, that is not our Final score,
                # the score after finding the best response to our move should be the final score.
                next_moves: List[Move] = self.current_moves_list(new_board)
                response_moves: List[Tuple[Optional[Move], Optional[numpy.float64]]]
                response_moves, = self.null_target_moves(1)
                next_move: Move
                for next_move in next_moves:
                    # next_move score calculations stay in the perspective of current player
                    next_board: Board = new_board.copy()
                    next_board.push(next_move)
                    next_current_king_box: List[int]
                    next_other_king_box: List[int]
                    next_current_king_box, next_other_king_box = self.get_king_boxes(next_board)
                    next_heatmap: heatmaps.ChessMoveHeatmap = chmutils.get_or_compute_heatmap_with_better_discounts(
                        next_board, depth=self.depth
                    )
                    next_heatmap_transposed: NDArray[numpy.float64] = next_heatmap.data.transpose()
                    if self.heatmap_data_is_zeros(next_heatmap) and next_board.is_checkmate():
                        # No early exit here as this is a bad is_checkmate result :(
                        next_heatmap_transposed[other_index] = numpy.float64(
                            len(next_board.piece_map()) * (self.depth + 1)
                        )
                    next_move_score: numpy.float64 = self.calculate_score(
                        current_index, other_index, next_heatmap_transposed,
                        king_box_multiplier, next_current_king_box, next_other_king_box
                    )
                    if response_moves[0][0] is None:
                        response_moves = [(next_move, next_move_score)]
                    else:
                        # response moves are inserted to form worst scores to best order (perspective of current player)
                        pos: int = bisect_left([x[1] for x in response_moves], next_move_score)
                        response_moves.insert(pos, (next_move, next_move_score))
                print(f"{move_uci} initial score: {initial_move_score:.2f} ->", self.formatted_moves(response_moves))
                # Once all responses to a move reviewed, final move score is the worst outcome to current player.
                best_response_score: Optional[numpy.float64] = response_moves[0][1]
            else:
                print("Fetching Score From Q table...")
                initial_move_score, best_response_score = numpy.float64(q_val), numpy.float64(q_val)
            move_score: numpy.float64 = best_response_score if best_response_score is not None else initial_move_score
            if q_val is None:
                self.set_q_value(state_fen, move_score)
            if moves[0][0] is None:
                moves = [(move, move_score)]
            else:
                # current moves are inserted into our moves list in order of best scores to worst
                pos = bisect_left([-x[1] for x in moves], -move_score)
                moves.insert(pos, (move, move_score))
            print(f"{move_uci} final score: {move_score:.2f}")
        # Final pick is a random choice of all moves equal to the highest scoring move
        print("All moves ranked:", self.formatted_moves(moves))
        picks = [(m, s) for m, s in moves if s == moves[0][1]]
        print("Engine moves:", self.formatted_moves(picks))
        chosen_move, chosen_q = random.choice(picks)
        return chosen_move, chosen_q

    @staticmethod
    def calculate_score(
            current_index: int,
            other_index: int,
            new_heatmap_transposed: NDArray[numpy.float64],
            king_box_multiplier: int,
            new_current_king_box: List[int],
            new_other_king_box: List[int]
    ) -> numpy.float64:
        """

        Parameters
        ----------
        current_index : int
        other_index : int
        new_heatmap_transposed : NDArray[numpy.float64]
        king_box_multiplier : int
        new_current_king_box : List[int]
        new_other_king_box : List[int]

        Returns
        -------
        numpy.float64

        """
        # Calculating score at this level is only needed in corner-case scenarios
        # where every possible move results in game termination.
        # score is initially, the delta of the sums of each player's heatmap.data values.
        initial_move_score: numpy.float64 = sum(
            new_heatmap_transposed[current_index]
        ) - sum(
            new_heatmap_transposed[other_index]
        )
        # king box score adds weights to the scores of squares around the kings.
        initial_king_box_score: numpy.float64 = sum(new_heatmap_transposed[current_index][new_other_king_box]) - (
                sum(new_heatmap_transposed[other_index][new_current_king_box]) * king_box_multiplier
        )
        # Final score is the agg of both above.
        return numpy.float64(initial_move_score + initial_king_box_score)

    @staticmethod
    def formatted_moves(moves: List[Tuple[Optional[Move], Optional[numpy.float64]]]) -> List[Optional[Tuple[str, str]]]:
        """

        Parameters
        ----------
        moves : List[Tuple[Optional[Move], Optional[numpy.float64]]]

        Returns
        -------
        List[Optional[Tuple[str, str]]]
        """
        return [(m.uci(), f"{s:.2f}") for m, s in moves if m is not None]


# pylint: disable=too-few-public-methods
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
            outcome: Outcome = self.engine.board.outcome()
            if outcome is None:
                outcome = self.engine.board.outcome(claim_draw=True)
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
