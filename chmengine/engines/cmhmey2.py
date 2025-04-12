"""Cmhmey Jr."""
import random
import sqlite3
from os import makedirs, path
from sqlite3 import Connection, Cursor
from typing import List, Optional, Tuple, Union
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
        """Initialize the CMHMEngine2 instance.

        This constructor extends the parent CMHMEngine by initializing the Q-table database.
        It sets up the engine with an optional initial board and a recursion depth for heatmap
        calculations. Upon initialization, it creates or verifies the Q-table database structure.

        Parameters
        ----------
        board : Optional[chess.Board]
            The initial board state. If None, a standard starting board is used.
        depth : int, default: 1
            The recursion depth for heatmap calculations, which controls the number of future
            moves considered during evaluation.

        Examples
        --------
        >>> import os
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> len(os.listdir(path=engine.cache_dir))
        31
        """
        super().__init__(board=board, depth=depth)
        self._init_qdb()  # List to store moves made during the game as (fen, move_uci) pairs.

    def qtable_filename(
            self,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            pieces_count: Optional[Union[int, str]] = None
    ) -> str:
        """Generate the filename for the Q-table database.

        The filename is constructed based on the current recursion depth and the number of pieces
        present on the board. If no piece count is provided, it is determined from the FEN string.

        Parameters
        ----------
        fen : Union[Optional[str], ]
            A FEN string to use for calculating the piece count. If None, the engine's current board
            FEN is used.
        board : Union[Optional[Board], ]
            A board object from which to determine the piece count if fen is not provided.
        pieces_count : Union[Optional[Union[int, str]], ]
            The number of pieces on the board. If not provided, it is calculated from the FEN.

        Returns
        -------
        str
            A filename string in the format: "qtable_depth_{depth}_piece_count_{piece_count}.db"

        Examples
        --------
        >>> import os
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.qtable_filename() in os.listdir(path=engine.cache_dir)
        True
        """
        return (
            f"qtable_depth_{self.depth}_"
            f"piece_count_{self.pieces_count(board=board, fen=fen) if pieces_count is None else pieces_count}.db"
        )

    def pieces_count(self, board: Optional[chess.Board] = None, fen: Optional[str] = None) -> int:
        """Get pieces count from board state.

        Parameters
        ----------
        board : Optional[chess.Board]
        fen : Optional[str]

        Returns
        -------
        int

        Examples
        --------
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.pieces_count()
        32
        """
        return self._pieces_count_from_fen_(
            fen=self.fen(board=board)
        ) if fen is None else self._pieces_count_from_fen_(fen=fen)

    @staticmethod
    def _pieces_count_from_fen_(fen: str) -> int:
        """Get pieces count from fen string

        Parameters
        ----------
        fen : str

        Returns
        -------
        int
        """
        _c: str
        return len([_c for _c in fen.split()[0] if _c.isalpha()])

    def qdb_path(
            self,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            pieces_count: Optional[Union[int, str]] = None,
    ) -> str:
        """Generate the full file path to the Q-table database.

        This method joins the cache directory with the Q-table filename generated by `qtable_filename()`,
        resulting in the full relative path where the database is stored.

        Parameters
        ----------
        fen : Union[Optional[str],]
            A FEN string to use for filename generation.
        board : Union[Optional[Board],]
            A board object to use for filename generation if fen is not provided.
        pieces_count : Union[Optional[Union[int, str]],]
            The piece count to use for filename generation.

        Returns
        -------
        str
            The full file path to the Q-table database.

        Examples
        --------
        >>> import os
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.qdb_path() == os.path.join(engine.cache_dir, engine.qtable_filename())
        True
        """
        return path.join(self.cache_dir, self.qtable_filename(fen=fen, board=board, pieces_count=pieces_count))

    def _init_qdb(self) -> None:
        """Initialize the Q-table database.

        This method ensures that the cache directory exists and creates the Q-table database
        for various piece count values (from 2 to 32 pieces). For each piece count, it creates
        a table (if not already present) with columns corresponding to move intensities for each
        square.

        Side Effects
        ------------
        Creates the directory "SQLite3Caches" if it does not exist and sets up the required
        tables and indexes in each Q-table database.
        """
        if not path.isdir(self.cache_dir):
            makedirs(self.cache_dir)
        pieces_count: int
        for pieces_count in range(2, 33):  # We are using dbs for 2-32 pieces
            qdb_path: str = self.qdb_path(pieces_count=pieces_count)
            q_conn: Connection
            with sqlite3.connect(qdb_path) as q_conn:
                q_cursor: Cursor = q_conn.cursor()
                q_cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS q_table (
                        fen TEXT,
                        q_value REAL,
                        PRIMARY KEY (fen)
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
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
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
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.depth = 3
        >>> engine.depth
        3
        """
        if new_depth < 0:
            raise ValueError(f"Invalid depth, value must be greater than or equal to 0, got {new_depth}")
        self._depth = int(new_depth)
        self._init_qdb()

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

    def get_q_value(
            self,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            pieces_count: Optional[int] = None
    ) -> Optional[numpy.float64]:
        """Retrieve the Q-value for a given board state from the Q-table database.

        If the state is not found in the database, returns None.

        Parameters
        ----------
        fen : str
            The FEN string of the board state. If None, the engine's current board FEN is used.
        board : Board
            The board corresponding to the FEN. Used for determining the file path if needed.
        pieces_count : Optional[int]
            The piece count to use for determining the database file.

        Returns
        -------
        Union[numpy.float64, None]
            The Q-value associated with the board state, or None if the state is not cached.

        Examples
        --------
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> q = engine.get_q_value()
        """
        fen = self.fen(board=board) if fen is None else fen
        q_conn: Connection
        with sqlite3.connect(self.qdb_path(fen=fen, board=board, pieces_count=pieces_count)) as q_conn:
            q_cursor: Cursor = q_conn.cursor()
            q_cursor.execute(
                "SELECT q_value FROM q_table WHERE fen = ?",
                (fen,)
            )
            row: Optional[Tuple[float]] = q_cursor.fetchone()
            return numpy.float64(row[0]) if row is not None else None

    def set_q_value(
            self,
            value: float,
            fen: Optional[str] = None,
            board: Optional[Board] = None,
            pieces_count: Optional[int] = None
    ) -> None:
        """Set or update the Q-value for a given board state in the Q-table database.

        Parameters
        ----------
        value : float
            The Q-value to store for the state.
        fen : Optional[str]
            The FEN string of the board state. If None, the engine's current board FEN is used.
        board : Optional[Board]
            The board used for determining the file path if fen is None.
        pieces_count : Optional[int]
            The piece count to use for determining the database file.

        Examples
        --------
        >>> from chmengine import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> engine.set_q_value(0.0, '1k6/8/8/8/8/3K4/8/8 w - - 0 1')
        """
        if fen is None:
            fen = self.fen(board=board)
        with sqlite3.connect(self.qdb_path(fen=fen, board=board, pieces_count=pieces_count)) as q_conn:
            q_cursor = q_conn.cursor()
            q_cursor.execute(
                "INSERT OR REPLACE INTO q_table (fen, q_value) VALUES (?, ?)",
                (fen, value)
            )

    def update_q_values(self, debug: bool = False) -> None:
        """Update the Q-table values after game termination.

        This method back-propagates the outcome of the game by iteratively popping moves off the
        board stack and adjusting their Q-values based on the game outcome. For draws, scores are
        converged toward zero; for wins/losses, scores are bumped or penalized accordingly.

        Parameters
        ----------
        debug : bool

        Side Effects
        ------------
        Updates the Q-values in the Q-table database for each board fen in the move history.

        Examples
        --------
        >>> from io import StringIO
        >>> from chess import pgn
        >>> from chmengine import CMHMEngine2
        >>> pgn_buffer = StringIO(
        ...    '''
        ...    1. f3 e5 2. g4 Qh4# 0-1
        ...
        ...
        ...    '''
        ... )
        >>> game = pgn.read_game(pgn_buffer)
        >>> board = game.board()
        >>> for move in game.mainline_moves():
        ...     board.push(move)
        >>> engine = CMHMEngine2(board=board)
        >>> engine.fen()
        'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3'
        >>> engine.update_q_values()
        >>> engine.fen()
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        """
        while len(self.board.move_stack) > 0:
            last_move: Move = self.board.move_stack[-1]
            fen: str = self.fen()
            current_q: Optional[numpy.float64] = self.get_q_value(fen=fen, board=self.board)
            self.board.pop()
            new_move: Move
            new_score: numpy.float64
            new_move, new_score = self.pick_move(debug=debug)  # Call pick_move to back-pop the updated
            new_q: Optional[numpy.float64] = self.get_q_value(fen=fen)
            if debug:
                print(
                    f"Game Pick & Score: ({last_move.uci()}, {current_q}) --> "
                    f"Game Pick & New Score: ({last_move.uci()}, {new_q}) --> "
                    f"New Pick & Score: ({new_move.uci()}, {new_score})\n"
                )

    def pick_move(
            self,
            pick_by: str = "",
            board: Optional[chess.Board] = None,
            debug: bool = False
    ) -> Tuple[chess.Move, numpy.float64]:
        """Select a move based on heatmap evaluations and Q-table integration.

        This overridden method combines heatmap evaluations with Q-value updates. It evaluates all
        legal moves by calculating their scores from the perspective of the current player (with positive
        scores indicating favorable moves and negative scores indicating unfavorable moves). The evaluation
        score is always from the mover’s perspective. The method then picks one move at random from the
        top-scoring moves, updates the Q-value for that move, and returns the selected move along with its
        score.

        Parameters
        ----------
        pick_by : str, default: ""
            Legecy param from parent class pick_move method. Args are ignored by this classe's overide.
        board : Optional[chess.Board], default: None
            Pick from a given board instead of intstance board
        debug : bool, default: False
            Allows for a print call showing the current evals of each move choice during anaylsis.

        Returns
        -------
        Tuple[chess.Move, numpy.float64]
            A tuple containing the chosen move and its associated evaluation score. The evaluation score is
            expressed from the perspective of the player making the move—positive values indicate favorable
            outcomes and negative values indicate unfavorable outcomes.

        Raises
        ------
        ValueError
            If Current Board has no legal moves.

        Examples
        --------
        >>> from chmengine.engines.cmhmey2 import CMHMEngine2
        >>> engine = CMHMEngine2()
        >>> move, score = engine.pick_move()
        """
        # Debated if I want to do any updates to the q-value for self.board.fen() if there is one...
        # If I did update it then it would be the negative of the q-value for the new_board
        # No Don't do this! Why not? Because we haven't explored all other branches that board could have lead to
        current_moves: List[Move] = self.current_moves_list(board=board)
        # PlayCMHMEngine.play() needs a ValueError to detect game termination.
        if len(current_moves) == 0:
            raise ValueError(f"Current Board has no legal moves: {self.fen(board=board)}")
        # Index of heatmap colors is opposite the chess lib's color mapping
        current_index: int = self.current_player_heatmap_index(board=board)
        other_index: int = self.other_player_heatmap_index(board=board)
        # moves will be current moves ordered by engine's score best to worst
        current_move_choices_ordered: List[Tuple[Optional[Move], Optional[float_]]]
        current_move_choices_ordered, = self.null_target_moves(number=1)
        current_move: Move
        for current_move in current_moves:
            # Phil, I know you keep thinking you don't need to calculate a score here,
            # but you do sometimes (keep that in mind.)
            new_board: Board = self.board_copy_pushed(move=current_move, board=board)
            # new_fen represents the boards for the only q-values that we will make updates to
            new_fen = new_board.fen()
            # No longer look at q-score at this level, only back prop q's into the calculation
            new_current_king_box: List[int]
            new_other_king_box: List[int]
            new_current_king_box, new_other_king_box = self.get_king_boxes(board=new_board)
            # It was fun building up a giant db of heatmaps, but we saw how that turned out in training
            new_heatmap: heatmaps.ChessMoveHeatmap = chmutils.calculate_chess_move_heatmap_with_better_discount(
                board=new_board, depth=self.depth
            )
            new_heatmap_transposed: NDArray[numpy.float64] = new_heatmap.data.transpose()
            # I wanted to rely on the heatmap as much as possible, but any game termination state win or draw
            # results in a zeros heatmap. Thus, we cheat with new_board.is_checkmate here. (draw scores stay zero.)
            if self.heatmap_data_is_zeros(heatmap=new_heatmap) and new_board.is_checkmate():
                self._update_heatmap_transposed_with_mate_values_(
                    heatmap_transposed=new_heatmap_transposed,
                    player_index=current_index,
                    board=new_board
                )
            current_move_choices_ordered = self._update_current_move_choices_(
                current_move_choices_ordered=current_move_choices_ordered, new_board=new_board,
                current_move=current_move, new_heatmap_transposed=new_heatmap_transposed,
                current_index=current_index, other_index=other_index, new_current_king_box=new_current_king_box,
                new_other_king_box=new_other_king_box, new_fen=new_fen
            )
        # Final pick is a random choice of all moves equal to the highest scoring move
        if debug:
            print("All moves ranked:", self._formatted_moves_(moves=current_move_choices_ordered))
        picks: List[Tuple[Move, numpy.float64]] = [
            (m, s) for m, s in current_move_choices_ordered if s == current_move_choices_ordered[0][1]
        ]
        chosen_move: Move
        chosen_q: numpy.float64
        chosen_move, chosen_q = random.choice(picks)
        self.set_q_value(value=numpy.float64(-chosen_q), board=board)
        return chosen_move, chosen_q

    def _update_current_move_choices_(
            self,
            current_move_choices_ordered: List[Tuple[Optional[Move], Optional[numpy.float64]]],
            new_board: chess.Board,
            current_move: chess.Move,
            new_heatmap_transposed: NDArray[numpy.float64],
            current_index: int,
            other_index: int,
            new_current_king_box: List[int],
            new_other_king_box: List[int],
            new_fen: str
    ) -> List[Tuple[Move, numpy.float64]]:
        """Update the ordered list of candidate moves based on a newly calculated heatmap score.

        This method calculates an initial move score using the transposed heatmap data and king box areas,
        then retrieves response moves to further refine the score. The final score is determined as the worst
        (response) outcome for the current player. The candidate list is updated (ordered best to worst) accordingly.

        Parameters
        ----------
        current_move_choices_ordered : List[Tuple[Optional[Move], Optional[numpy.float64]]]
            The current ordered list of candidate moves and their scores.
        new_board : chess.Board
            The board state after the candidate move is applied.
        current_move : chess.Move
            The candidate move being evaluated.
        new_heatmap_transposed : NDArray[numpy.float64]
            The transposed heatmap data for the new board state.
        current_index : int
            The index corresponding to the current player's heatmap data.
        other_index : int
            The index corresponding to the opponent's heatmap data.
        new_current_king_box : List[int]
            A list of squares representing the area around the current king.
        new_other_king_box : List[int]
            A list of squares representing the area around the opponent's king.
        new_fen : str
            The FEN string representing the new board state.

        Returns
        -------
        List[Tuple[Move, numpy.float64]]
            The updated ordered list of candidate moves with their evaluation scores.
        """
        initial_q_val: Optional[numpy.float64] = self.get_q_value(fen=new_fen, board=new_board)
        if initial_q_val is None:
            initial_move_score: numpy.float64 = self._calculate_score_(
                current_index=current_index, other_index=other_index, new_heatmap_transposed=new_heatmap_transposed,
                new_current_king_box=new_current_king_box, new_other_king_box=new_other_king_box
            )
        else:
            initial_move_score = initial_q_val
        response_moves: List[Tuple[Optional[Move], Optional[numpy.float64]]]
        response_moves = self._get_or_calculate_responses_(
            new_board=new_board,
            other_index=other_index,
            current_index=current_index,
            check_check=False
        )
        # Once all responses to a move reviewed, final move score is the worst outcome to current player.
        best_response_score: Optional[numpy.float64] = response_moves[0][1]
        final_move_score: numpy.float64
        final_move_score = best_response_score if best_response_score is not None else initial_move_score
        self.set_q_value(value=final_move_score, fen=new_fen, board=new_board)
        if current_move_choices_ordered[0][0] is None:
            current_move_choices_ordered = [(current_move, final_move_score)]
        else:
            self._insert_ordered_best_to_worst_(
                ordered_moves=current_move_choices_ordered, move=current_move, score=final_move_score
            )
        return current_move_choices_ordered

    def _get_or_calculate_responses_(
            self,
            new_board: chess.Board,
            other_index: int,
            current_index: int,
            check_check: bool
    ) -> List[Tuple[Optional[Move], Optional[numpy.float64]]]:
        """Retrieve the opponent's response moves and evaluate them.

        This method computes heatmap-based evaluation scores for each legal response move from the new board
        state (after a candidate move is applied). The scores are ordered from worst to best from the perspective
        of the current player.

        Parameters
        ----------
        new_board : chess.Board
            The board state after a candidate move is applied.
        other_index : int
            The index corresponding to the opponent in the heatmap.
        current_index : int
            The index corresponding to the current player in the heatmap.

        Returns
        -------
        List[Tuple[Optional[Move], Optional[numpy.float64]]]
            A list of response moves and their evaluation scores.
        """
        # However, that is not our Final score,
        # the score after finding the best response to our move should be the final score.
        next_moves: List[Move] = self.current_moves_list(board=new_board)
        response_moves: List[Tuple[Optional[Move], Optional[numpy.float64]]]
        response_moves, = self.null_target_moves(number=1)
        next_move: Move
        for next_move in next_moves:
            response_moves = self._get_or_calc_next_move_score_(
                next_move=next_move, response_moves=response_moves, new_board=new_board, current_index=current_index,
                other_index=other_index, check_check=check_check
            )
        return response_moves

    def _get_or_calc_next_move_score_(
            self,
            next_move: chess.Move,
            response_moves: List[Tuple[Optional[Move], Optional[numpy.float64]]],
            new_board: chess.Board,
            current_index: int,
            other_index: int,
            check_check: bool
    ) -> List[Tuple[chess.Move, numpy.float64]]:
        """Calculate the evaluation score for a given opponent response move.

        This method computes the Q-score for the next move in the perspective of the current player.
        If no Q-value is found in the database, it calculates a new score using the heatmap data.
        The resulting score is then inserted into the response moves list in order.

        Parameters
        ----------
        next_move : chess.Move
            The opponent's move to evaluate.
        response_moves : List[Tuple[Optional[Move], Optional[numpy.float64]]]
            The current list of evaluated response moves.
        new_board : chess.Board
            The board state after the candidate move is applied.
        current_index : int
            The index for the current player's data in the heatmap.
        other_index : int
            The index for the opponent's data in the heatmap.
        check_check : bool

        Returns
        -------
        List[Tuple[chess.Move, numpy.float64]]
            The updated list of response moves with their evaluation scores.
        """
        # next_move score calculations stay in the perspective of current player
        next_board: Board = self.board_copy_pushed(move=next_move, board=new_board)
        next_fen: str = next_board.fen()
        # next_q_val should be the negative of the fetched q, but can't do that yet, must check for None first
        next_q_val: Optional[numpy.float64] = self.get_q_value(fen=next_fen, board=next_board)
        next_outcome: Optional[Outcome] = next_board.outcome(claim_draw=True)
        not_draw: bool = next_outcome is None or next_outcome.winner is not None
        null_q: bool = next_q_val is None
        if null_q and not_draw:
            next_move_score: numpy.float64 = self._calculate_next_move_score_(
                next_board=next_board, current_index=current_index, other_index=other_index
            )
            set_value: numpy.float64 = numpy.float64(-next_move_score) if not check_check else numpy.float64(
                next_move_score
            )
            self.set_q_value(value=set_value, fen=next_fen, board=next_board)
        elif null_q:
            next_move_score: numpy.float64 = numpy.float64(0.0)
            self.set_q_value(value=next_move_score, fen=next_fen, board=next_board)
        else:
            # Here is where we can safely make next_q_val negative to match the current player's perspective
            next_move_score = numpy.float64(-next_q_val) if not check_check else numpy.float64(next_q_val)
        if response_moves[0][0] is None:
            response_moves = [(next_move, next_move_score)]
        else:
            self._insert_ordered_worst_to_best_(response_moves, next_move, next_move_score)
        return response_moves

    def _calculate_next_move_score_(
            self,
            next_board: chess.Board,
            current_index: int,
            other_index: int
    ) -> numpy.float64:
        """Calculate the evaluation score for a potential next move.

        This method computes the score for a board state resulting from an opponent's move. It uses
        heatmap data (including a transposed heatmap), king box calculations, and the provided multiplier
        to determine the move's evaluation from the perspective of the current player.

        Parameters
        ----------
        next_board : chess.Board
            The board state after the opponent's move.
        current_index : int
            The index corresponding to the current player in the heatmap.
        other_index : int
            The index corresponding to the opponent in the heatmap.

        Returns
        -------
        numpy.float64
            The evaluation score for the next move.
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
            self._update_heatmap_transposed_with_mate_values_(
                next_heatmap_transposed, other_index, next_board
            )
        elif next_board.is_check():
            # It will be interesting if we hit inf recursion here, that would be a position of inf counter checks?
            check_responses = self._get_or_calculate_responses_(
                new_board=next_board,
                other_index=other_index,
                current_index=current_index,
                check_check=True
            )
            if check_responses[-1][1] is not None:
                return check_responses[-1][1]
        next_move_score: numpy.float64 = self._calculate_score_(
            current_index, other_index, next_heatmap_transposed,
            next_current_king_box, next_other_king_box
        )
        return next_move_score

    def _update_heatmap_transposed_with_mate_values_(
            self,
            heatmap_transposed: NDArray[numpy.float64],
            player_index: int,
            board: chess.Board
    ) -> None:
        """Update the transposed heatmap data with mate values for checkmate scenarios.

        This method modifies the heatmap in place to reflect a checkmate situation. The heatmap
        values for the given player index are set to a high value calculated based on the number of
        pieces remaining and the engine's depth, ensuring that checkmate positions receive a score
        that surpasses typical heatmap evaluations.

        Parameters
        ----------
        heatmap_transposed : NDArray[numpy.float64]
            The transposed heatmap data array to be updated.
        player_index : int
            The index (current or opponent) for which to update the mate value.
        board : chess.Board
            The board state where checkmate has been detected.
        """
        heatmap_transposed[player_index] = numpy.float64(
            len(board.piece_map()) * (self.depth + 1)
        )

    @staticmethod
    def _insert_ordered_best_to_worst_(
            ordered_moves: List[Tuple[chess.Move, numpy.float64]],
            move: chess.Move,
            score: numpy.float64
    ) -> None:
        """Insert a move and its score into an ordered list (from best to worst).

        This static method uses a binary insertion (via bisect_left) to insert the new move into
        the list such that the list remains ordered from the highest score to lowest.

        Parameters
        ----------
        ordered_moves : List[Tuple[chess.Move, numpy.float64]]
            The current list of moves and their scores, ordered from best to worst.
        move : chess.Move
            The move to be inserted.
        score : numpy.float64
            The evaluation score for the move.
        """
        # current moves are inserted into our moves list in order of best scores to worst
        ordered_index = bisect_left([-x[1] for x in ordered_moves], -score)
        ordered_moves.insert(ordered_index, (move, score))

    @staticmethod
    def _insert_ordered_worst_to_best_(
            ordered_moves: List[Tuple[chess.Move, numpy.float64]],
            move: chess.Move,
            score: numpy.float64
    ) -> None:
        """Insert a move and its score into an ordered list (from worst to best).

        This static method uses a binary insertion (via bisect_left) to insert the new move into
        the list such that the list remains ordered from the lowest score to highest, suitable for response
        move evaluations from the perspective of the current player.

        Parameters
        ----------
        ordered_moves : List[Tuple[chess.Move, numpy.float64]]
            The current list of moves and their scores, ordered from worst to best.
        move : chess.Move
            The move to be inserted.
        score : numpy.float64
            The evaluation score for the move.
        """
        # response moves are inserted to form worst scores to best order (perspective of current player)
        ordered_index: int = bisect_left([x[1] for x in ordered_moves], score)
        ordered_moves.insert(ordered_index, (move, score))

    @staticmethod
    def _calculate_score_(
            current_index: int,
            other_index: int,
            new_heatmap_transposed: NDArray[numpy.float64],
            new_current_king_box: List[int],
            new_other_king_box: List[int]
    ) -> numpy.float64:
        """Calculate the evaluation score for a move based on heatmap data and king safety.

        This static method computes the move score as the sum of two components:
        - The difference between the current player's total heatmap intensity and the opponent's.
        - A weighted difference based on the intensity values within the "king box" areas.
        The final score reflects the overall benefit of the move from the perspective of the current player.

        Parameters
        ----------
        current_index : int
            The index for the current player's heatmap data.
        other_index : int
            The index for the opponent's heatmap data.
        new_heatmap_transposed : NDArray[numpy.float64]
            The transposed heatmap data array.
        new_current_king_box : List[int]
            The list of squares surrounding the current king.
        new_other_king_box : List[int]
            The list of squares surrounding the opponent's king.

        Returns
        -------
        numpy.float64
            The computed evaluation score for the move.
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
        )
        # Final score is the agg of both above.
        return numpy.float64(initial_move_score + initial_king_box_score)

    @staticmethod
    def _formatted_moves_(
            moves: List[Tuple[Optional[Move], Optional[numpy.float64]]]
    ) -> List[Optional[Tuple[str, str]]]:
        """Generate a formatted list of moves and their scores for display.

        This static method converts a list of move-score tuples into a list of tuples containing the move
        in UCI format and the score formatted as a string with two decimal places.

        Parameters
        ----------
        moves : List[Tuple[Optional[Move], Optional[numpy.float64]]]
            The list of moves with their evaluation scores.

        Returns
        -------
        List[Optional[Tuple[str, str]]]
            A list of formatted move representations (UCI, score) suitable for printing or logging.
        """
        return [(m.uci(), f"{s:.2f}") for m, s in moves if m is not None]
