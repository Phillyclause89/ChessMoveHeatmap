"""Cmhmey the 3rd!! Son of Cmhmey Jr. (mother unknown, rumor is this might be a Canaan situation)"""

from time import time
from typing import Iterator, List, Optional, Tuple

from chess import Board, Move
from numpy import float64, inf

from chmengine.engines.cmhmey2 import CMHMEngine2
from chmengine.utils import Pick, calculate_better_white_minus_black_score


class CMHMEngine3(CMHMEngine2):
    """A time-limited, single-process chess engine that extends CMHMEngine2 with
    per-move caching and iterative deepening on the current best move.

    Notes
    -----
    - Builds a snapshot for each legal root move and attaches a Pick(move, score)
        seeded from the Q-table or static eval.
    - Iteratively deepens along the current best move’s branch until the time_limit
        expires, restarting from root whenever a different move becomes best.
    - Backs up improved scores into the Q-table after each deeper evaluation.

    Warnings
    --------
    - The default `time_limit` is positive infinity—by default the engine will search
        until you manually stop it.
    - Even if you interrupt the process (for example via KeyboardInterrupt) before
        `pick_move` returns, all the Q-table updates performed up to that point remain
        committed. That ongoing learning can improve future lookups.
    - To avoid unbounded CPU use, supply an explicit finite `time_limit` either when
        constructing the engine or in each `pick_move` call.
    """
    time_limit: float64 = inf
    overhead: float64 = float64(0)

    def __init__(
            self,
            board: Optional[Board] = None,
            depth: int = 1,
            time_limit: float64 = time_limit,
            overhead: float64 = overhead
    ) -> None:
        """Initialize CMHMEngine3.

        Parameters
        ----------
        board : Optional[Board]
            The initial chess position. If None, defaults to the standard start position.
        depth : int
            The recursion depth parameter used by calculate_better_white_minus_black_score.
        time_limit : float64, default=inf
            Maximum wall-clock seconds spent in pick_move before returning the current best.
            NOTE: The default is infinite—this engine will search until you terminate it.
            To avoid unbounded searches, provide a finite time_limit here or when
            calling pick_move.
        overhead : float64, default=float64(0.43)
            Accounts for overhead when scheduling return time of pick_move via time_limit.
        """
        super().__init__(board=board, depth=depth)
        self.time_limit, self.overhead = float64(time_limit), float64(overhead)

    def pick_move(
            self,
            *args,
            board: Optional[Board] = None,
            time_limit: Optional[float] = None,
            debug: bool = False,
            **kwargs
    ) -> Pick:
        """Choose the best move using iterative deepening on the current best branch.

        If no finite `time_limit` is provided (and the class’s time_limit remains infinite),
        this will search until manually interrupted.

        Otherwise, it:
        1) Takes a snapshot board (defaulting to `self.board` or the provided `board`).
        2) Generates all legal moves and creates a list of (Pick, Board) pairs,
            with `Pick.score` seeded from the Q-table or a static heatmap eval.
        3) Sorts that list to find the best and second-best scores.
        4) While time remains (< deadline), calls _search_current_best_ on the top entry.
            If the second-best score ever exceeds the updated best, it re-roots
            by resorting to the full move list.
        5) At timeout (or if only one move), returns the Pick with the best score
            and writes that score back into the Q-table for the root position.

        Warnings
        --------
        - Without a finite time_limit, `pick_move` will continue indefinitely.
        - You can safely cancel an infinite search at any moment; any Q-table writes
            already performed will persist and benefit later searches.

        Parameters
        ----------
        *args, **kwargs : passed through to super().pick_move for fallback
        board : Optional[Board]
            Position to analyze; defaults to this engine’s internal board.
        time_limit : Optional[float]
            If set, overrides the engine’s time_limit for this call.
        debug : bool
            If True, enables verbose timing and score-change logs.

        Returns
        -------
        Pick
            The chosen move and its evaluation score, from the root’s perspective.
        """
        if time_limit is None and board is None:
            deadline: float64 = float64(time() + self.time_limit - self.overhead)
            board = self.board
        elif time_limit is None:
            deadline: float64 = float64(time() + self.time_limit - self.overhead)
        elif board is None:
            board = self.board
            deadline: float64 = float64(time() + time_limit - self.overhead)
        else:
            deadline: float64 = float64(time() + time_limit - self.overhead)

        pick: Pick
        picked_board: Board
        sorted_picks_n_boards: List[Tuple[Pick, Board]] = sorted(
            self.pick_board_generator(board=board),
            key=lambda x: x[0],
            reverse=True
        )
        if len(sorted_picks_n_boards) == 0:
            raise ValueError(f"Current Board has no legal moves: {self.fen(board=board)}")
        best_index: int
        second_best_index: int
        best_index, second_best_index = 0 if board.turn else -1, 1 if board.turn else -2
        best_pnb: Tuple[Pick, Board] = sorted_picks_n_boards[best_index]
        if len(sorted_picks_n_boards) == 1:
            return best_pnb[0]
        else:
            second_best_score: float64 = sorted_picks_n_boards[second_best_index][0].score
        while time() < deadline:
            sorted_picks_n_boards[best_index] = self._search_current_best_(best_pnb=best_pnb)
            if second_best_score > best_pnb[0].score:
                sorted_picks_n_boards[best_index] = (
                    sorted_picks_n_boards[best_index][0], self.board_copy_pushed(move=best_pnb[0].move, board=board)
                )
                sorted_picks_n_boards = sorted(sorted_picks_n_boards, key=lambda x: x[0], reverse=True)
            best_pnb = sorted_picks_n_boards[best_index]
            second_best_score = sorted_picks_n_boards[second_best_index][0].score
        final_pick: Pick = sorted(sorted_picks_n_boards, key=lambda x: x[0], reverse=True)[best_index][0]
        self.set_q_value(value=final_pick.score, board=board)
        return final_pick

    def _search_current_best_(
            self,
            best_pnb: Tuple[Pick, Board]
    ) -> Tuple[Pick, Board]:
        """Deepen the search one ply along the current best branch.

        Starting from the child board in best_pnb, generates all legal replies,
        picks the strongest reply (by Q-table or static eval), and uses its score
        to update the parent move’s Pick.score. Also writes the new score into
        the Q-table for that child position.

        Parameters
        ----------
        best_pnb : Tuple[Pick, Board]
            A tuple containing the current best Pick (move & score) from the root,
            and the Board reached by making that move.

        Returns
        -------
        Tuple[Pick, Board]
            A tuple of (Pick, Board) where:
            • `Pick.move` is the same as best_pnb[0].move
            • `Pick.score` has been updated to the best child reply’s score
            • Board is the child position after that best reply
        """
        root_pick: Pick
        root_board: Board
        root_pick, root_board = best_pnb
        sub_picks_n_boards: List[Tuple[Pick, Board]] = sorted(
            self.pick_board_generator(board=root_board),
            key=lambda x: x[0],
            reverse=True
        )

        # If this move leads to no legal replies, keep the current score
        if len(sub_picks_n_boards) == 0:
            return best_pnb

        # Get the best reply move and use its score as the evaluation for the current move
        best_response_pick, best_response_board = sub_picks_n_boards[0 if root_board.turn else -1]
        root_pick.score = best_response_pick.score
        # Second q-value set.
        self.set_q_value(value=best_response_pick.score, board=root_board)
        return root_pick, best_response_board

    def pick_board_generator(self, board: Optional[Board] = None) -> Iterator[Tuple[Pick, Board]]:
        """Lazily generate (Pick, Board) pairs for each legal move from `board`.

        For each legal move:
        1) Snapshot the board after the move.
        2) Create Pick(move, score) where score is pulled from the Q-table
            or computed on demand via get_or_calc_score.

        This generator is the entry point for building the per-move cache in pick_move.

        Parameters
        ----------
        board : Union[Board, None]
            The position to branch from; defaults to this engine’s internal board.

        Returns
        -------
        Iterator[Tuple[Pick, Board]]
            An iterator over each legal move’s Pick and the resulting Board.
        """
        board = self.board if board is None else board
        move: Move
        result_board: Board
        return ((Pick(move=move, score=self.get_or_calc_deeper_score(result_board)), result_board) for
                move, result_board in
                ((move, self.board_copy_pushed(move=move, board=board)) for move in board.legal_moves))

    def get_or_calc_deeper_score(self, board: Board) -> float64:
        """

        Parameters
        ----------
        board : Board

        Returns
        -------
        float64
        """
        score: float64 = sorted(
            (
                self.get_or_calc_score(result_board) for result_board in
                (self.board_copy_pushed(move=move, board=board) for move in board.legal_moves)
            ),
            reverse=True
        )[0 if board.turn else -1]
        self.set_q_value(value=score, board=board)
        return score

    def get_or_calc_score(self, board: Board) -> float64:
        """Retrieve a Q-table score for `board`, or compute & cache a static eval if missing.

        This ensures every position’s score is seeded in the Q-table before any search
        deeper into it.

        Parameters
        ----------
        board : Board
            The position for which to fetch or compute a score.

        Returns
        -------
        float64
            The stored or newly calculated evaluation (positive = White-good).
        """
        score: Optional[float64] = self.get_q_value(board=board)
        if score is None:
            score: float64 = calculate_better_white_minus_black_score(board=board, depth=self.depth)
            # First q-value set.
            self.set_q_value(value=score, board=board)
        return score
