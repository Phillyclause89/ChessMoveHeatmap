"""Cmhmey Sr."""
from random import choice
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

from chess import Board, Move
from numpy import nan, float64, ndarray
from numpy.typing import NDArray

from chmengine.utils import is_valid_king_box_square, null_target_moves
from chmutils import get_or_compute_heatmap_with_better_discounts
from heatmaps import GradientHeatmap


class CMHMEngine:
    """A silly chess engine that picks moves using heatmaps"""
    _board: Board
    _depth: int

    def __init__(self, board: Optional[Board] = None, depth: int = 1) -> None:
        """Initialize the CMHMEngine (Cmhmey Sr.) instance.

        Parameters
        ----------
        board : Optional[chess.Board]
            The initial chess board state. If None, a standard starting board is used.
        depth : int, default: 1
            The recursion depth used for heatmap calculations. This value controls how many layers
            of future moves are considered when evaluating positions.

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
            self._board = Board()

    @property
    def depth(self) -> int:
        """Get the current recursion depth setting.

        Returns
        -------
        int
            The current recursion depth used for heatmap calculations.

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.depth
        1
        """
        return self._depth

    @depth.setter
    def depth(self, new_depth: int):
        """Set the current recursion depth setting.

        Parameters
        ----------
        new_depth : int
            The new recursion depth. Must be greater than or equal to 0.

        Raises
        ------
        ValueError
            If new_depth is less than 0.

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.depth = 3
        >>> engine.depth
        3
        """
        if new_depth < 0:
            raise ValueError(f"depth must be greater than or equal to 0, got {new_depth}")
        self._depth = int(new_depth)

    @property
    def board(self) -> Board:
        """Get the current chess board.

        Returns
        -------
        chess.Board
            The current board state.

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.board
        Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        """
        return self._board

    @board.setter
    def board(self, board: Board):
        """Set the chess board state.

        Parameters
        ----------
        board : chess.Board
            A valid chess.Board object. The board is copied internally.

        Raises
        ------
        ValueError
            If the provided board is not valid.
        TypeError
            If the provided board is not of type chess.Board.
        """
        try:
            assert board.is_valid()
            self._board = board.copy()
        except AssertionError as assertion_error:
            raise ValueError(f"The board is invalid: {board}") from assertion_error
        except AttributeError as attribute_error:
            raise TypeError(f"The board is of the wrong type: {type(board)}") from attribute_error

    def board_copy_pushed(self, move: Move, board: Optional[Board] = None) -> Board:
        """Return a copy of the board with the given move applied.

        Parameters
        ----------
        move : chess.Move
            The move to push onto the board.
        board : Optional[chess.Board], default: None
            The board to copy; if None, uses the engine's current board.

        Returns
        -------
        chess.Board
            A new board instance with the move applied.

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> from chess import Move
        >>> engine = CMHMEngine()
        >>> new_board = engine.board_copy_pushed(Move.from_uci('e2e4'))
        >>> new_board
        Board('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')
        """
        board_copy: Board = self.board.copy() if board is None else board.copy()
        board_copy.push(move)
        return board_copy

    def pick_move(
            self,
            pick_by: str = "all-delta",
            board: Optional[Board] = None
    ) -> Tuple[Move, float64]:
        """Select a move based on various heatmap-derived criteria.

        The method evaluates candidate moves using multiple heuristics (such as delta, maximum,
        minimum, king attack/defense, etc.) and returns one move chosen at random from the set
        of the best candidate moves according to the specified criteria.

        The evaluation score is provided from the perspective of the player making the move:
            a positive score indicates a move that is considered beneficial for the mover,
            while a negative score indicates a move that is considered detrimental.
        This scoring convention is different from many traditional chess engines,
        where scores are often expressed as positive for White advatage and negative for Black.

        Parameters
        ----------
        pick_by : str, default: "all-delta"
            A string indicating the selection heuristic. Supported options include "all-delta",
            "all-max", "all-min", "king-atk", "king-def", and "king-delta".
        board : chess.Board, default: None
            Used by child classes; pick_move method

        Returns
        -------
        Tuple[chess.Move, numpy.float64]
            A tuple containing the chosen move and its associated evaluation score, where the score is
            from the perspective of the player making the move (i.e., positive values indicate favorable
            outcomes for that player, and negative values indicate unfavorable outcomes).

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> picked_move, eval_score = engine.pick_move()
        >>> picked_move
        Move.from_uci('e2e4')
        >>> # A positive score indicates a move leading to a good position for the mover,
        >>> # whereas a negative score indicates a leading to a bad position.
        >>> eval_score
        10.0
        """
        if board is not None:
            warn(
                UserWarning("CMHMEngine does not use board argument. Use CMHMEngine2 to pick moves from other boards.")
            )
        current_index: int = self.current_player_heatmap_index()
        other_index: int = self.other_player_heatmap_index()
        move_maps_items: List[Tuple[Move, GradientHeatmap]] = self.get_or_calc_move_maps_list()
        (
            target_moves_by_delta,
            target_moves_by_king_delta,
            target_moves_by_max_current,
            target_moves_by_max_other_king,
            target_moves_by_min_current_king,
            target_moves_by_min_other
        ) = null_target_moves()
        current_king_box: List[int]
        other_king_box: List[int]
        current_king_box, other_king_box = self.get_king_boxes(board=self.board)

        move: Move
        heatmap: GradientHeatmap
        for move, heatmap in move_maps_items:
            if self.heatmap_data_is_zeros(heatmap):
                return move, nan
            # target_moves_by_max_other_king == "king-atk"
            other_king_sum: float64
            (
                other_king_sum,
                target_moves_by_max_other_king
            ) = self.update_target_moves_by_max_other_king(
                target_moves_by_max_other_king,
                heatmap, move, current_index,
                other_king_box
            )
            # target_moves_by_min_current_king == "king-def"
            current_king_min: float64
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

            transposed_map: NDArray[float64] = heatmap.data.transpose()
            # target_moves_by_max_current == "max"
            current_player_sum: float64
            (
                current_player_sum,
                target_moves_by_max_current
            ) = self.update_target_moves_by_max_current(
                target_moves_by_max_current,
                transposed_map, move,
                current_index
            )
            # target_moves_by_min_other == "min"
            other_player_sum: float64
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

        return choice(target_moves)

    def other_player_heatmap_index(self, board: Optional[Board] = None) -> int:
        """Get the heatmap index corresponding to the inactive (other) player.

        Returns
        -------
        int
            The index for the other player. Typically, if the current turn is White (index 0),
            this returns 1, and vice versa.

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.other_player_heatmap_index()
        1
        """
        return int((self.board.turn if board is None else board.turn))

    def current_player_heatmap_index(self, board: Optional[Board] = None) -> int:
        """Get the heatmap index corresponding to the active (current) player.

        Parameters
        ----------
        board : Optional[chess.Board]

        Returns
        -------
        int
            The index for the current player. Typically, if the current turn is White (index 0),
            this returns 0, and vice versa.

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> engine.current_player_heatmap_index()
        0
        """
        return int(not (self.board.turn if board is None else board.turn))

    @staticmethod
    def update_target_moves_by_delta(
            target_moves_by_delta: List[Tuple[Optional[Move], Optional[float64]]],
            current_player_sum: float64,
            other_player_sum: float64,
            move: Move
    ) -> List[Tuple[Move, float64]]:
        """Update the candidate moves based on the delta between current and other player's sums.

        The delta is computed as the difference between the current player's sum and the other player's sum.
        If the calculated delta is greater than the current best, the candidate list is replaced; if equal,
        the move is appended.

        Parameters
        ----------
        target_moves_by_delta : List[Tuple[Optional[chess.Move], Optional[numpy.float64]]]
            The current list of candidate moves and their delta scores.
        current_player_sum : numpy.float64
            The sum of move intensities for the current player.
        other_player_sum : numpy.float64
            The sum of move intensities for the other player.
        move : chess.Move
            The move being evaluated.

        Returns
        -------
        List[Tuple[chess.Move, numpy.float64]]
            The updated list of candidate moves with their delta scores.
        """
        delta: float64 = float64(current_player_sum - other_player_sum)
        current_best_delta: float64 = target_moves_by_delta[0][1]
        if current_best_delta is None or delta > current_best_delta:
            target_moves_by_delta = [(move, delta)]
        elif delta == current_best_delta:
            target_moves_by_delta.append((move, delta))
        return target_moves_by_delta

    @staticmethod
    def update_target_moves_by_min_other(
            target_moves_by_min_other: List[Tuple[Optional[Move], Optional[float64]]],
            transposed_map: ndarray,
            move: Move,
            other_index: int
    ) -> Tuple[float64, List[Tuple[Move, float64]]]:
        """Update the candidate moves for minimizing the opponent's sum.

        Calculates the opponent's total move intensity from the transposed heatmap, and updates
        the candidate list if the new sum is lower than the current best.

        Parameters
        ----------
        target_moves_by_min_other : List[Tuple[Optional[chess.Move], Optional[numpy.float64]]]
            The current candidate moves for minimizing the opponent's intensity.
        transposed_map : numpy.ndarray
            The transposed heatmap data array.
        move : chess.Move
            The move being evaluated.
        other_index : int
            The index corresponding to the opponent.

        Returns
        -------
        Tuple[numpy.float64, List[Tuple[chess.Move, numpy.float64]]]
            A tuple containing the opponent's sum and the updated candidate list.
        """
        other_player_sum: float64 = sum(transposed_map[other_index])
        current_best_min: float64 = target_moves_by_min_other[0][1]
        if current_best_min is None or current_best_min > other_player_sum:
            target_moves_by_min_other = [(move, other_player_sum)]
        elif current_best_min == other_player_sum:
            target_moves_by_min_other.append((move, other_player_sum))
        return other_player_sum, target_moves_by_min_other

    @staticmethod
    def update_target_moves_by_max_current(
            target_moves_by_max_current: List[Tuple[Optional[Move], Optional[float64]]],
            transposed_map: ndarray,
            move: Move,
            color_index: int
    ) -> Tuple[float64, List[Tuple[Move, float64]]]:
        """Update the candidate moves for maximizing the current player's intensity.

        Computes the current player's total intensity from the transposed heatmap and updates the candidate
        list if the new sum is greater than the current best.

        Parameters
        ----------
        target_moves_by_max_current : List[Tuple[Optional[chess.Move], Optional[numpy.float64]]]
            The current candidate moves for maximizing the current player's intensity.
        transposed_map : numpy.ndarray
            The transposed heatmap data array.
        move : chess.Move
            The move being evaluated.
        color_index : int
            The index corresponding to the current player.

        Returns
        -------
        Tuple[numpy.float64, List[Tuple[chess.Move, numpy.float64]]]
            A tuple containing the current player's sum and the updated candidate list.
        """
        current_player_sum: float64 = sum(transposed_map[color_index])
        current_best_max: float64 = target_moves_by_max_current[0][1]
        if current_best_max is None or current_player_sum > current_best_max:
            target_moves_by_max_current = [(move, current_player_sum)]
        elif current_best_max == current_player_sum:
            target_moves_by_max_current.append((move, current_player_sum))
        return current_player_sum, target_moves_by_max_current

    @staticmethod
    def update_target_moves_by_king_delta(
            target_moves_by_king_delta: List[Tuple[Optional[Move], Optional[float64]]],
            move: Move,
            current_king_min: float64,
            other_king_sum: float64
    ) -> List[Tuple[Move, float64]]:
        """Update candidate moves based on the king's delta value.

        Calculates the delta between the opponent's king intensity and the current king's minimum intensity,
        updating the candidate list if this delta is greater than the current best.

        Parameters
        ----------
        target_moves_by_king_delta : List[Tuple[Optional[chess.Move], Optional[numpy.float64]]]
            The current candidate moves for the king delta criterion.
        move : chess.Move
            The move being evaluated.
        current_king_min : numpy.float64
            The minimum intensity value for the current king.
        other_king_sum : numpy.float64
            The total intensity value for the opponent's king.

        Returns
        -------
        List[Tuple[chess.Move, numpy.float64]]
            The updated list of candidate moves based on king delta.
        """
        current_king_delta: float64 = float64(other_king_sum - current_king_min)
        current_best_king_delta: float64 = target_moves_by_king_delta[0][1]
        if current_best_king_delta is None or current_king_delta > current_best_king_delta:
            target_moves_by_king_delta = [(move, current_king_delta)]
        elif current_king_delta == current_best_king_delta:
            target_moves_by_king_delta.append((move, current_king_delta))
        return target_moves_by_king_delta

    @staticmethod
    def update_target_moves_by_min_current_king(
            target_moves_by_min_current_king: List[Tuple[Optional[Move], Optional[float64]]],
            heatmap: GradientHeatmap,
            move: Move,
            other_index: int,
            current_king_box: List[int]
    ) -> Tuple[float64, List[Tuple[Move, float64]]]:
        """Update candidate moves for minimizing the current king's intensity.

        Extracts the intensity values for the current king from the heatmap and updates the candidate
        list if a lower intensity sum is found.

        Parameters
        ----------
        target_moves_by_min_current_king : List[Tuple[Optional[chess.Move], Optional[numpy.float64]]]
            The candidate moves list for minimizing current king's intensity.
        heatmap : heatmaps.GradientHeatmap
            The heatmap object containing move intensities.
        move : chess.Move
            The move being evaluated.
        other_index : int
            The index corresponding to the opponent.
        current_king_box : List[int]
            A list of board squares representing the area around the current king.

        Returns
        -------
        Tuple[numpy.float64, List[Tuple[chess.Move, numpy.float64]]]
            A tuple containing the current king's sum and the updated candidate list.
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
            target_moves_by_max_other_king: List[Tuple[Optional[Move], Optional[float64]]],
            heatmap: GradientHeatmap,
            move: Move,
            color_index: int,
            other_king_box: List[int]
    ) -> Tuple[float64, List[Tuple[Move, float64]]]:
        """Update candidate moves for maximizing the opponent king's intensity.

        Calculates the opponent king's total intensity from the heatmap over the specified area and
        updates the candidate list if a higher intensity sum is found.

        Parameters
        ----------
        target_moves_by_max_other_king : List[Tuple[Optional[chess.Move], Optional[numpy.float64]]]
            The candidate moves list for maximizing opponent king's intensity.
        heatmap : heatmaps.GradientHeatmap
            The heatmap object containing move intensities.
        move : chess.Move
            The move being evaluated.
        color_index : int
            The index corresponding to the opponent.
        other_king_box : List[int]
            A list of board squares representing the area around the opponent's king.

        Returns
        -------
        Tuple[numpy.float64, List[Tuple[chess.Move, numpy.float64]]]
            A tuple containing the opponent king's sum and the updated candidate list.
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
    def heatmap_data_is_zeros(heatmap: GradientHeatmap) -> bool:
        """Check if the heatmap data is entirely zero.

        Parameters
        ----------
        heatmap : heatmaps.GradientHeatmap
            The heatmap object to check.

        Returns
        -------
        bool
            True if all values in the heatmap data are zero; otherwise, False.

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
        return (heatmap.data == float64(0.0)).all()

    def get_king_boxes(self, board: Optional[Board] = None) -> Tuple[List[int], List[int]]:
        """Compute the bounding boxes for the kings on the board.

        For both the current and opponent kings, this method calculates a "box" (a list of square
        indices) representing the king's immediate surroundings.

        Parameters
        ----------
        board : Optional[chess.Board], default: None
            The board to use; if None, the engine's current board is used.

        Returns
        -------
        Tuple[List[int], List[int]]
            A tuple containing two lists: the first is the box for the current king, and the second is
            the box for the opponent king.
        """
        board = self.board if board is None else board
        other_king_square: int = board.king(not board.turn)
        current_king_square: int = board.king(board.turn)
        other_king_box: List[int] = [other_king_square]
        current_king_box: List[int] = [current_king_square]
        long: int
        for long in (-1, 0, 1):
            lat: int
            for lat in (-8, 0, +8):
                oks_box_id: int = other_king_square + long + lat
                cks_box_id: int = current_king_square + long + lat
                if is_valid_king_box_square(oks_box_id, other_king_square):
                    other_king_box.append(oks_box_id)
                if is_valid_king_box_square(cks_box_id, current_king_square):
                    current_king_box.append(cks_box_id)
        return current_king_box, other_king_box

    def get_or_calc_move_maps_list(self) -> List[Tuple[Move, GradientHeatmap]]:
        """Retrieve a list of move-to-heatmap mappings.

        This method converts the dictionary returned by `get_or_calc_move_maps()` into a list of
        tuples for easier iteration, where each tuple contains a move and its corresponding heatmap.

        Returns
        -------
        List[Tuple[chess.Move, heatmaps.GradientHeatmap]]
            A list of (move, heatmap) tuples.

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> first_move = engine.get_or_calc_move_maps_list()[0][0]
        >>> first_move
        Move.from_uci('g1h3')
        """
        return list(self.get_or_calc_move_maps().items())

    def get_or_calc_move_maps(self, depth: Optional[int] = None) -> Dict[Move, GradientHeatmap]:
        """Compute or retrieve precomputed heatmaps for all legal moves from the current board.

        For each legal move from the current board, this method generates a corresponding heatmap by
        applying the move and evaluating the resulting position with a given recursion depth.

        Parameters
        ----------
        depth : Optional[int], default: None
            The recursion depth for the heatmap calculation. If None, the engine's current depth is used.

        Returns
        -------
        Dict[chess.Move, heatmaps.GradientHeatmap]
            A dictionary mapping each legal move to its corresponding heatmap.

        Raises
        ------
        ValueError
            If the current board has no legal moves.

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> move_maps = engine.get_or_calc_move_maps()
        >>> some_move = list(move_maps.keys())[0]
        >>> type(move_maps[some_move])
        <class 'heatmaps.ChessMoveHeatmap'>
        """
        moves = self.current_moves_list()
        if len(moves) > 0:
            return {
                move: get_or_compute_heatmap_with_better_discounts(
                    board=self.board_copy_pushed(move),
                    depth=self.depth if depth is None else depth
                ) for move in moves
            }
        raise ValueError("Current Board has no legal moves.")

    def current_moves_list(self, board: Optional[Board] = None) -> List[Move]:
        """Retrieve the list of legal moves for the current board position.

        Parameters
        ----------
        board : Optional[chess.Board], default: None
            The board for which to get legal moves. If None, the engine's current board is used.

        Returns
        -------
        List[chess.Move]
            A list of legal moves available on the board.

        Examples
        --------
        >>> from chmengine import CMHMEngine
        >>> engine = CMHMEngine()
        >>> moves = engine.current_moves_list()
        >>> moves[15]
        Move.from_uci('e2e4')
        """
        return list(self.board.legal_moves) if board is None else list(board.legal_moves)

    def fen(self, board: Optional[Board] = None) -> str:
        """Obtain the FEN string for a given board state.
        If no board is provided, the engine's current board is used.

        Parameters
        ----------
        board : Optional[Board]
            The board for which to retrieve the FEN string.

        Returns
        -------
        str
            The FEN string representing the board state.

        Examples
        --------
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.fen()
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        """
        return self.board.fen() if board is None else board.fen()
