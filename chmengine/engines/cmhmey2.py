"""Cmhmey Jr."""
import random
import sqlite3
from os import makedirs, path
from sqlite3 import Connection, Cursor
from typing import List, Optional, Tuple
from _bisect import bisect_left
import chess
import numpy
from chess import Board, Move, Outcome
from numpy import float_
from numpy.typing import NDArray
import chmutils
import heatmaps
from chmengine.engines.cmhmey1 import CMHMEngine


class CMHMEngine2(CMHMEngine):
    """Overrides CMHMEngine.pick_move"""
    cache_dir: str = path.join(".", chmutils.CACHE_DIR, "QTables")

    def __init__(self, board: Optional[chess.Board] = None, depth: int = 1) -> None:
        super().__init__(board, depth)
        self._init_qdb()  # List to store moves made during the game as (state_fen, move_uci) pairs.

    def qtable_filename(
            self,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            piece_count: Optional[int] = None
    ) -> str:
        """returns f"qtable_depth_{`self.depth`}_piece_count_{`piece_count`}.db"

        Parameters
        ----------
        fen : Optional[str]
        board : chess.Board
        piece_count : Optional[int]

        Returns
        -------
        str

        """
        if piece_count is None:
            if board is None:
                if fen is None:
                    piece_count = len([c for c in self.board.fen().split()[0] if c.isalpha()])
                else:
                    piece_count = len([c for c in fen.split()[0] if c.isalpha()])
            else:
                piece_count = len([c for c in board.fen().split()[0] if c.isalpha()])
        return f"qtable_depth_{self.depth}_piece_count_{piece_count}.db"

    def qdb_path(
            self,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            piece_count: Optional[int] = None,
    ) -> str:
        """returns os.path.join(self.cache_dir, self.qtable_filename(fen, board, piece_count))

        Parameters
        ----------
        fen: Optional[str]
        board : chess.Board
        piece_count: Optional[int]

        Returns
        -------
        str
        """
        return path.join(self.cache_dir, self.qtable_filename(fen=fen, board=board, piece_count=piece_count))

    def _init_qdb(self) -> None:
        """Initializes the Q-table database and creates the table if it does not exist."""
        if not path.isdir(self.cache_dir):
            makedirs(self.cache_dir)
        piece_count: int
        for piece_count in range(2, 33):  # We are using dbs for 2-32 pieces
            qdb_path: str = self.qdb_path(piece_count=piece_count)
            q_conn: Connection
            with sqlite3.connect(qdb_path) as q_conn:
                q_cursor: Cursor = q_conn.cursor()
                q_cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS q_table (
                        state_fen TEXT,
                        q_value REAL,
                        PRIMARY KEY (state_fen)
                    )
                    """
                )

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
        self._init_qdb()

    def state_fen(self, board: Optional[Board] = None) -> str:
        """FEN string of board.fen() or self.board.fen() if board is None

        Parameters
        ----------
        board : Optional[Board]

        Returns
        -------
        str
        """
        return self.board.fen() if board is None else board.fen()

    def get_q_value(
            self,
            state_fen: Optional[str] = None,
            board: Optional[Board] = None,
            piece_count: Optional[int] = None
    ) -> Optional[numpy.float64]:
        """Returns the Q-value for a given state and move, or None if not found."""
        if state_fen is None:
            state_fen = self.state_fen(board=board)
        q_conn: Connection
        with sqlite3.connect(self.qdb_path(fen=state_fen, board=board, piece_count=piece_count)) as q_conn:
            q_cursor: Cursor = q_conn.cursor()
            q_cursor.execute(
                "SELECT q_value FROM q_table WHERE state_fen = ?",
                (state_fen,)
            )
            row: Optional[Tuple[float]] = q_cursor.fetchone()
            return numpy.float64(row[0]) if row is not None else None

    def set_q_value(
            self,
            value: float,
            state_fen: Optional[str] = None,
            board: Optional[Board] = None,
            piece_count: Optional[int] = None
    ) -> None:
        """Sets or updates the Q-value for a given state and move."""
        if state_fen is None:
            state_fen = self.state_fen(board)
        with sqlite3.connect(self.qdb_path(fen=state_fen, board=board, piece_count=piece_count)) as q_conn:
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
            current_q: Optional[float] = self.get_q_value(state_fen=state, board=self.board)
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
            # TODO: Maybe cap these by the checkmate scores, but then I would need to calc the checkmate score for each
            elif self.board.turn != outcome.winner:
                # Winner gets a 20% bump to their chosen move's scores (now with a +0.1 bump to brake past 0 if needed)
                new_q = current_q + abs(current_q * 0.2) + numpy.float64(0.1)
            elif self.board.turn == outcome.winner:
                # Loser gets a -20% anti-bump  to their chosen move's scores (now with a -0.1 anti-bump to brake past 0)
                new_q = current_q - abs(current_q * 0.2) - numpy.float64(0.1)
            else:
                raise ValueError(f"How did we get here? outcome:{outcome} board turn: {self.board.turn}")
            self.set_q_value(value=new_q, state_fen=state, board=self.board)
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
        """
        # Debated if I want to do any updates to the q-value for self.board.fen() if there is one...
        # If I did update it then it would be the negative of the q-value for the new_board
        # No Don't do this! Why not? Because we haven't explored all other branches that board could have lead to
        current_moves: List[Move] = self.current_moves_list()
        # PlayCMHMEngine.play() needs a ValueError to detect game termination.
        if len(current_moves) == 0:
            raise ValueError(f"Current Board has no legal moves: {self.board.fen()}")
        # Index of heatmap colors is opposite the chess lib's color mapping
        current_index: int = self.current_player_heatmap_index
        other_index: int = self.other_player_heatmap_index
        # moves will be current moves ordered by engine's score best to worst
        current_move_choices_ordered: List[Tuple[Optional[Move], Optional[float_]]]
        current_move_choices_ordered, = self.null_target_moves(1)
        current_move: Move
        for current_move in current_moves:
            # Phil, I know you keep thinking you don't need to calculate a score here,
            # but you do sometimes (keep that in mind.)
            current_move_uci: str = current_move.uci()
            new_board: Board = self.board_copy_pushed(current_move)
            # new_state_fen represents the boards for the only q-values that we will make updates to
            new_state_fen = new_board.fen()
            # No longer look at q-score at this level, only back prop q's into the calculation
            print(f"Calculating score for move: {current_move_uci}")
            new_current_king_box: List[int]
            new_other_king_box: List[int]
            new_current_king_box, new_other_king_box = self.get_king_boxes(new_board)
            # It was fun building up a giant db of heatmaps, but we saw how that turned out in training
            new_heatmap: heatmaps.ChessMoveHeatmap = chmutils.calculate_chess_move_heatmap_with_better_discount(
                new_board, depth=self.depth
            )
            new_heatmap_transposed: NDArray[numpy.float64] = new_heatmap.data.transpose()
            # I wanted to rely on the heatmap as much as possible, but any game termination state win or draw
            # results in a zeros heatmap. Thus, we cheat with new_board.is_checkmate here. (draw scores stay zero.)
            if self.heatmap_data_is_zeros(new_heatmap) and new_board.is_checkmate():
                self.update_heatmap_transposed_with_mate_values(new_heatmap_transposed, current_index, new_board)
                if early_exit:
                    # TODO: Weigh the costs of this early exit feature that you are not actually using right now
                    mate_score = numpy.float64(sum(new_heatmap_transposed[current_index]))
                    self.set_q_value(value=numpy.float64(-mate_score))
                    return current_move, mate_score
            current_move_choices_ordered = self.update_current_move_choices(
                current_move_choices_ordered, new_board,
                current_move, new_heatmap_transposed,
                current_index, other_index,
                new_current_king_box, new_other_king_box,
                king_box_multiplier, new_state_fen,
                current_move_uci
            )
        # Final pick is a random choice of all moves equal to the highest scoring move
        print("All moves ranked:", self.formatted_moves(current_move_choices_ordered))
        picks = [(m, s) for m, s in current_move_choices_ordered if s == current_move_choices_ordered[0][1]]
        print("Engine moves:", self.formatted_moves(picks))
        chosen_move, chosen_q = random.choice(picks)
        self.set_q_value(value=numpy.float64(-chosen_q))
        return chosen_move, chosen_q

    def update_current_move_choices(
            self,
            current_move_choices_ordered: List[Tuple[Optional[Move], Optional[numpy.float64]]],
            new_board: chess.Board,
            current_move: chess.Move,
            new_heatmap_transposed: NDArray[numpy.float64],
            current_index: int,
            other_index: int,
            new_current_king_box: List[int],
            new_other_king_box: List[int],
            king_box_multiplier: int,
            new_state_fen: str,
            current_move_uci: str
    ) -> List[Tuple[Move, numpy.float64]]:
        """

        Parameters
        ----------
        current_move_choices_ordered : List[Tuple[Optional[Move], Optional[numpy.float64]]]
        new_board : chess.Board
        current_move : chess.Move
        new_heatmap_transposed : NDArray[numpy.float64]
        current_index : int
        other_index : int
        new_current_king_box : List[int]
        new_other_king_box : List[int]
        king_box_multiplier : int
        new_state_fen : str
        current_move_uci : str

        Returns
        -------
        List[Tuple[Move, numpy.float64]]
        """
        initial_move_score: numpy.float64 = self.calculate_score(
            current_index, other_index, new_heatmap_transposed,
            king_box_multiplier, new_current_king_box, new_other_king_box
        )
        response_moves: List[Tuple[Optional[Move], Optional[numpy.float64]]]
        response_moves = self.get_or_calculate_responses(new_board, other_index, current_index, king_box_multiplier)
        print(
            f"{current_move_uci} initial score: {initial_move_score:.2f} ->",
            self.formatted_moves(response_moves)
        )
        # Once all responses to a move reviewed, final move score is the worst outcome to current player.
        best_response_score: Optional[numpy.float64] = response_moves[0][1]
        final_move_score: numpy.float64
        final_move_score = best_response_score if best_response_score is not None else initial_move_score
        self.set_q_value(value=final_move_score, state_fen=new_state_fen, board=new_board)
        if current_move_choices_ordered[0][0] is None:
            current_move_choices_ordered = [(current_move, final_move_score)]
        else:
            self.insert_ordered_best_to_worst(current_move_choices_ordered, current_move, final_move_score)
        print(f"{current_move_uci} final score: {final_move_score:.2f}")
        return current_move_choices_ordered

    def get_or_calculate_responses(
            self,
            new_board: chess.Board,
            other_index: int,
            current_index: int,
            king_box_multiplier: int
    ) -> List[Tuple[Optional[Move], Optional[numpy.float64]]]:
        """Gets next player's response moves (ordered worst to best in current player's perspective)

        Parameters
        ----------
        new_board : chess.Board
        other_index : int
        current_index : int
        king_box_multiplier : int

        Returns
        -------
        List[Tuple[Optional[Move], Optional[numpy.float64]]]
        """
        # However, that is not our Final score,
        # the score after finding the best response to our move should be the final score.
        next_moves: List[Move] = self.current_moves_list(new_board)
        response_moves: List[Tuple[Optional[Move], Optional[numpy.float64]]]
        response_moves, = self.null_target_moves(1)
        next_move: Move
        for next_move in next_moves:
            response_moves = self.get_or_calc_next_move_score(
                next_move, response_moves, new_board, current_index,
                other_index, king_box_multiplier
            )
        return response_moves

    def get_or_calc_next_move_score(
            self,
            next_move: chess.Move,
            response_moves: List[Tuple[Optional[Move], Optional[numpy.float64]]],
            new_board: chess.Board,
            current_index: int,
            other_index: int,
            king_box_multiplier: int
    ) -> List[Tuple[chess.Move, numpy.float64]]:
        """Gets or calculates the next (opponent's) heatmap q-score in the perspective of the current player.


        Parameters
        ----------
        next_move : chess.Move
        response_moves : List[Tuple[Optional[Move], Optional[numpy.float64]]]
        new_board : chess.Board
        current_index : int
        other_index : int
        king_box_multiplier : int

        Returns
        -------
        List[Tuple[chess.Move, numpy.float64]]
        """
        # next_move score calculations stay in the perspective of current player
        next_board: Board = self.board_copy_pushed(next_move, new_board)
        next_state_fen: str = next_board.fen()
        # next_q_val should be the negative of the fetched q, but can't do that yet, must check for None first
        next_q_val: Optional[numpy.float64] = self.get_q_value(state_fen=next_state_fen, board=next_board)
        if next_q_val is None:
            next_move_score = self.calculate_next_move_score(
                next_board, current_index,
                other_index, king_box_multiplier
            )
        else:
            # Here is where we can safely make next_q_val negative to match the current player's perspective
            next_move_score = numpy.float64(-next_q_val)
        if response_moves[0][0] is None:
            response_moves = [(next_move, next_move_score)]
        else:
            self.insert_ordered_worst_to_best(response_moves, next_move, next_move_score)
        return response_moves

    def calculate_next_move_score(
            self,
            next_board: chess.Board,
            current_index: int,
            other_index: int,
            king_box_multiplier: int
    ) -> numpy.float64:
        """

        Parameters
        ----------
        next_board : chess.Board
        current_index : int
        other_index : int
        king_box_multiplier : int

        Returns
        -------
        numpy.float64
        """
        next_current_king_box: List[int]
        next_other_king_box: List[int]
        next_current_king_box, next_other_king_box = self.get_king_boxes(next_board)
        next_heatmap: heatmaps.ChessMoveHeatmap
        next_heatmap = chmutils.calculate_chess_move_heatmap_with_better_discount(
            next_board, depth=self.depth
        )
        next_heatmap_transposed: NDArray[numpy.float64] = next_heatmap.data.transpose()
        if self.heatmap_data_is_zeros(next_heatmap) and next_board.is_checkmate():
            # No early exit here as this is a bad is_checkmate result :(
            self.update_heatmap_transposed_with_mate_values(
                next_heatmap_transposed, other_index, next_board
            )
        next_move_score: numpy.float64 = self.calculate_score(
            current_index, other_index, next_heatmap_transposed,
            king_box_multiplier, next_current_king_box, next_other_king_box
        )
        return next_move_score

    def update_heatmap_transposed_with_mate_values(
            self,
            heatmap_transposed: NDArray[numpy.float64],
            player_index: int,
            board: chess.Board
    ) -> None:
        """Inplace updates to the transposed heatmap creating an upper/lower bounds of the score system

        # Checkmate moves get scored as all the surviving pieces (opponent's pieces are POWs now)
        # being able to move to every square (the war is over, pieces can go hangout on the same square
        # with their surviving buddies). Anyway, this should just about guarantee checkmates get
        # a score higher than what a heatmap can produce from a position.

        Parameters
        ----------
        heatmap_transposed : NDArray[numpy.float64]
        player_index : int
        board : chess.Board
        """
        heatmap_transposed[player_index] = numpy.float64(
            len(board.piece_map()) * (self.depth + 1)
        )

    @staticmethod
    def insert_ordered_best_to_worst(
            ordered_moves: List[Tuple[chess.Move, numpy.float64]],
            move: chess.Move,
            score: numpy.float64
    ) -> None:
        """Inplace ordered insertion of move and score; ordered high to low

        Parameters
        ----------
        ordered_moves : List[Tuple[chess.Move, numpy.float64]]
        move : chess.Move
        score : numpy.float64
        """
        # current moves are inserted into our moves list in order of best scores to worst
        ordered_index = bisect_left([-x[1] for x in ordered_moves], -score)
        ordered_moves.insert(ordered_index, (move, score))

    @staticmethod
    def insert_ordered_worst_to_best(
            ordered_moves: List[Tuple[chess.Move, numpy.float64]],
            move: chess.Move,
            score: numpy.float64
    ) -> None:
        """Inplace ordered insertion of move and score; ordered low to high

        Parameters
        ----------
        ordered_moves : List[Tuple[chess.Move, numpy.float64]]
        move : chess.Move
        score : numpy.float64
        """
        # response moves are inserted to form worst scores to best order (perspective of current player)
        ordered_index: int = bisect_left([x[1] for x in ordered_moves], score)
        ordered_moves.insert(ordered_index, (move, score))

    @staticmethod
    def calculate_score(
            current_index: int,
            other_index: int,
            new_heatmap_transposed: NDArray[numpy.float64],
            king_box_safety_multiplier: int,
            new_current_king_box: List[int],
            new_other_king_box: List[int]
    ) -> numpy.float64:
        """

        Parameters
        ----------
        current_index : int
        other_index : int
        new_heatmap_transposed : NDArray[numpy.float64]
        king_box_safety_multiplier : int
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
        initial_king_box_score: numpy.float64 = sum(new_heatmap_transposed[current_index][new_other_king_box])
        initial_king_box_score -= sum(
            new_heatmap_transposed[other_index][new_current_king_box]
        ) * king_box_safety_multiplier
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
