"""Cmhmey Sr."""
import random
from typing import Any, Dict, List, Optional, Tuple
import chess
import numpy
from chess import Move, Piece
from numpy.typing import NDArray
import chmutils
import heatmaps
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

    def board_copy_pushed(self, move: chess.Move, board: Optional[chess.Board] = None) -> chess.Board:
        """

        Parameters
        ----------
        move : chess.Move
        board : Optional[chess.Board]

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
        board_copy: chess.Board = self.board.copy() if board is None else board.copy()
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
