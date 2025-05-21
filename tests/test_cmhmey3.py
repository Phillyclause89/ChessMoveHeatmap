"""Tests Cmhmey Jr.'s son Cmhmey the 3rd"""
from os import environ, path
from time import perf_counter
from typing import List, Optional, Tuple
from unittest import TestCase

from chess import Board, Move
from numpy import Inf, float64, mean, percentile
from numpy.typing import NDArray

from chmengine import CMHMEngine2, Pick, Quartney
from chmengine.engines.cmhmey3 import CMHMEngine3
from chmutils import BetterHeatmapCache, HeatmapCache
from tests.utils import CACHE_DIR, clear_test_cache

MATE_IN_ONE_4 = '2k5/Q1p5/2K5/8/8/8/8/8 w - - 0 1'

MATE_IN_ONE_3 = 'kb6/p7/5p2/2RPpp2/2RK4/2BPP3/6B1/8 w - e6 0 2'

MATE_IN_ONE_2 = 'kb6/p3p3/5p2/2RP1p2/2RK4/2BPP3/6B1/8 b - - 0 1'

MATE_IN_ONE_1 = '4k2r/ppp4p/4pBp1/2Q5/4B3/4p1PK/PP1r3P/5R2 w - - 2 32'

HeatmapCache.cache_dir = CACHE_DIR
BetterHeatmapCache.cache_dir = CACHE_DIR
CMHMEngine2.cache_dir = CACHE_DIR
Quartney.cache_dir = CACHE_DIR
CMHMEngine3.cache_dir = CACHE_DIR

pipeline_env: bool = environ.get('GITHUB_ACTIONS', '').lower() == 'true'


class TestCMHMEngine3(TestCase):
    """Tests Cmhmey Jr.'s son Cmhmey the 3rd"""
    starting_board: Board
    engine: CMHMEngine3
    time_limit: float64 = float64(5.0)
    pipeline_env: bool = pipeline_env

    def setUp(self) -> None:
        """Sets up the engine instance and test environment."""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))
        self.engine = CMHMEngine3(time_limit=self.time_limit)
        self.assertIsInstance(self.engine, CMHMEngine3)
        self.assertEqual(self.engine.cache_dir, CACHE_DIR)
        self.starting_board = Board()

    def tearDown(self) -> None:
        """Cleans up the test environment."""
        clear_test_cache()
        self.assertFalse(path.exists(CACHE_DIR))

    def test___init__(self):
        self.assertEqual(self.engine.depth, 1)
        self.assertEqual(self.engine.time_limit, self.time_limit)
        self.assertEqual(self.engine.fen(), self.starting_board.fen())
        self.engine.depth = 100
        self.engine.time_limit = 100
        _board = Board(fen=MATE_IN_ONE_1)
        self.engine.board = Board(fen=MATE_IN_ONE_1)
        self.assertEqual(self.engine.depth, 100)
        self.assertEqual(self.engine.time_limit, 100)
        self.assertEqual(self.engine.fen(), _board.fen())

    def test_pick_move(self, sample_trim_div: int = 1) -> None:
        """Perform a full benchmark of pick_move latency under various scenarios,
        collecting response times to drive our Second-Order Performance Calibration.

        This test exercises three patterns of engine invocation:
        1) The initial pick_move on the root position.
        2) The first-time pick_move after making one legal move.
        3) Repeated pick_move calls on the root position (“revisits”).

        For each category it records:
        • All individual lag times (duration − time_limit)
        • Aggregate statistics: mean, median (50th pct), 10th and 90th percentiles

        These metrics feed into our “Median-Centric Overhead Tuning” loop:
        we assert that each measurement does not exceed its known worst-case cap.
        Whenever a cap is exceeded, the test fails and reports which measurement
        was closest to breaking, prompting manual adjustment of the engine’s
        `overhead` parameter.

        Parameters
        ----------
        sample_trim_div : int, default=1
            Controls how many response moves to sample before revisiting the root.
            Must be ≥1. Increasing this reduces the number of first-time/resume samples.

        Raises
        ------
        AssertionError
            If any measured statistic exceeds its configured cap. All failing
            metrics will be reported together, allowing the user to adjust
            `overhead` once per batch run instead of one error at a time.

        Notes
        -----
        This is the core method in our Second-Order Performance Calibration system.
        It must be rerun repeatedly, with caps reset to –Inf, to iteratively discover
        the minimal `overhead` that keeps the known worst-case median lag at or below zero.
        """
        if sample_trim_div < 1:
            raise ValueError(f"sample_trim_div must be greater than `1`, got `{sample_trim_div}`")
        pick: Pick
        lag_time: float
        pick, lag_time = self.measure_pick(message=1)
        init_w_moves: List[Move] = list(self.engine.board.legal_moves)
        first_time_pick_times: List[float] = [lag_time]  # list of all times a position was seen for the first time.
        init_board_pick_times: List[float] = [lag_time]  # list of all times the first board position was seen.
        revisit_pick_times: List[float] = []  # lists all the times the init board position was revisited.
        i: int
        move: Move
        for i, move in enumerate(init_w_moves[:len(init_w_moves) // sample_trim_div], 1):
            self.engine.board.push(move)
            response_pick: Pick
            response_lag_time: float
            response_pick, response_lag_time = self.measure_pick(move=move, message=2)
            first_time_pick_times.append(response_lag_time)
            self.engine.board.pop()
            new_pick: Pick
            new_lag_time: float
            new_pick, new_lag_time = self.measure_pick(i=i, message=3)
            init_board_pick_times.append(new_lag_time)
            revisit_pick_times.append(new_lag_time)
        avg_lag_time: float = float(mean(init_board_pick_times))
        avg_response: float = float(mean(first_time_pick_times))
        avg_revisit: float = float(mean(revisit_pick_times))
        pre_lag_time: NDArray[float] = percentile(init_board_pick_times, [0, 1, 10, 25, 50, 75, 90, 99, 100])
        pre_response: NDArray[float] = percentile(first_time_pick_times, [0, 1, 10, 25, 50, 75, 90, 99, 100])
        pre_revisit: NDArray[float] = percentile(revisit_pick_times, [0, 1, 10, 25, 50, 75, 90, 99, 100])
        print(
            f"mean pick lag-time: {avg_lag_time:.3f}s\npercentiles(0, 1, 10, 25, 50, 75, 90, 99, 100):\n{pre_lag_time}"
        )
        print(
            f"mean response lag-time: {avg_response:.3f}s\npercentiles"
            f"(0, 1, 10, 25, 50, 75, 90, 99, 100):\n{pre_response}"
        )
        print(
            f"mean revisit lag-time: {avg_revisit:.3f}s\npercentiles(0, 1, 10, 25, 50, 75, 90, 99, 100):\n{pre_revisit}"
        )
        max_delta: float = -Inf
        min_delta: float = Inf
        closest_to_failing_cap: str = ''
        closest_to_failing_floor: str = ''
        # lag-time Assertions (used for `Iterative Second-Order Performance Calibration` of default `overhead` value)
        measurement_name: str
        measurement: float
        cap: float
        floor: float
        cap_failures: List[AssertionError] = []
        floor_failures: List[AssertionError] = []
        # noinspection IncorrectFormatting
        for measurement_name, measurement, floor, cap in sorted(
            (
                ('cap<=0: 10th pct response lag-time', pre_response[2],  -0.5614995999999994, -0.2400717999999955),
                ('cap<=0:          10th pct lag-time', pre_lag_time[2],  -0.5578028999999987, -0.17771110000000334),
                ('cap<=0:   10th pct revisit lag-time', pre_revisit[2],  -0.5579626099999999, -0.17094372999999835),
                ('cap==0:             Median lag-time', pre_lag_time[4], -0.37646469999999965, 0.0443965999999989),
                ('cap~=0:      Median revisit lag-time', pre_revisit[4], -0.3590221499999977,  0.05673685000000006),
                ('cap~=0:      Average response lag-time', avg_response, -0.17482825238095132, 0.05919819999999651),
                ('cap~=0:               Average lag-time', avg_lag_time, -0.26118467142857177, 0.10533138571428632),
                ('cap~=0:        Average revisit lag-time', avg_revisit, -0.24991738500000035, 0.12051575000000066),
                ('cap~=0:    Median response lag-time', pre_response[4], -0.1961125999999993,  0.18266420000000494),
                ('cap>=0:  90th pct response lag-time', pre_response[6],  0.11212199999999939, 0.34710730000000467),
                ('cap>=0:           90th pct lag-time', pre_lag_time[6],  0.16665559999999857, 0.6522717999999941),
                ('cap>=0:    90th pct revisit lag-time', pre_revisit[6],  0.1801250799999993,  0.6534579899999955),
            ),
            key=lambda x: x[1],
        ):
            try:
                delta: float = self.assertMeasurementBelowCap(
                    measurement_name, measurement, Inf if self.pipeline_env else cap
                )
                max_delta, closest_to_failing_cap = (
                    (delta, measurement_name) if max_delta < delta else (max_delta, closest_to_failing_cap)
                )
            except AssertionError as error:
                print(error)
                cap_failures.append(error)
            try:
                delta: float = self.assertMeasurementAboveFloor(
                    measurement_name, measurement, -Inf if self.pipeline_env else floor
                )
                min_delta, closest_to_failing_floor = (
                    (delta, measurement_name) if min_delta > delta else (min_delta, closest_to_failing_floor)
                )
            except AssertionError as error:
                print(error)
                floor_failures.append(error)
        all_fails: List[AssertionError] = cap_failures + floor_failures
        fail_count: int = len(all_fails)
        if fail_count == 0:
            print(
                f"closest_to_failing_cap: '{closest_to_failing_cap}' | max_delta: {max_delta:.3f}s ",
                f"closest_to_failing_floor: '{closest_to_failing_floor}' | min_delta: {min_delta:.3f}",
                sep='\n',
            )
        else:
            nl: str = '\n'
            raise AssertionError(
                f"{fail_count} AssertionError{'s' if fail_count > 1 else ''} encountered during test:{nl}{nl}"
                f"{nl.join((f'Failure #{n}: {f}' for n, f in enumerate(all_fails, start=1)))}"
            ) from all_fails[0]

    def assertMeasurementAboveFloor(self, measurement_name: str, measurement: float, floor: float) -> float:
        """Assert that a single performance measurement does not exceed its historical best case measurements.

        Parameters
        ----------
        measurement_name : str
            A descriptive label for the statistic (e.g. "Median lag-time").
        measurement : float
            The newly observed value of that statistic in seconds.
        floor : float
            The current best-case threshold for this statistic.  Typically set
            to Inf on first tuning round to allow discovery of new minima in each tuning round,
            or to the previously recorded best-case for regression checks.

        Returns
        -------
        float
            The difference (measurement − cap), negative if the floor was broken,
            positive if safely above.  A lower positive value indicates we are closer to failing our
            current `floor` assertion.
        """
        delta: float = measurement - floor
        print(f"{measurement_name}: {measurement:+.3f}s | floor: {floor:+.3f}s | delta: {delta:+.3f}s")
        self.assertGreaterEqual(
            measurement, floor,
            f"New floor found for: '{measurement_name}', measurement too low: {measurement:.3f}s (delta={delta:+.3f}s)"
        )
        return delta

    def assertMeasurementBelowCap(self, measurement_name: str, measurement: float, cap: float) -> float:
        """Assert that a single performance measurement does not exceed its historical worst case measurements.

        Parameters
        ----------
        measurement_name : str
            A descriptive label for the statistic (e.g. "Median lag-time").
        measurement : float
            The newly observed value of that statistic in seconds.
        cap : float
            The current worst-case threshold for this statistic.  Typically set
            to –Inf to allow discovery of new maxima in each tuning round,
            or to the previously recorded worst-case for regression checks.

        Returns
        -------
        float
            The difference (measurement − cap), positive if the cap was exceeded,
            negative if safely below.  A higher negative delta indicates we are closer to failing our
            current `cap` assertion.
        """
        delta: float = measurement - cap
        print(f"{measurement_name}: {measurement:.3f}s | cap: {cap:.3f}s | delta: {delta:.3f}s")
        self.assertLessEqual(
            measurement, cap,
            f"New cap found for: '{measurement_name}', measurement too high: {measurement:.3f}s (delta={delta:+.3f}s)"
        )
        return delta

    def measure_pick(
            self,
            i: Optional[int] = None,
            move: Optional[Move] = None,
            message: int = 0
    ) -> Tuple[Pick, float]:
        """Time a single pick_move invocation and assert its individual lag is within a loose bound.

        This method wraps a single call to engine.pick_move(), measures elapsed time,
        computes lag_time = elapsed − self.time_limit, and verifies it does not
        wildly overshoot (using assertAlmostEqual with low precision).  It prints
        the move, score, duration, and lag_time according to the `message` flag.

        While assertAlmostEqual ensures no extreme stragglers break the run,
        the detailed per-call values feed into the broader statistical analysis
        within test_pick_move for Second-Order Performance Calibration.

        Parameters
        ----------
        i : Optional[int]
            A sequence index for revisit calls, used in the debug print.
        move : Optional[Move]
            The Move just played before this pick_move call, for context in logs.
        message : int
            Controls the verbosity of printed output:
            1 = initial root call, 2 = first-time response call, 3 = revisit call.

        Returns
        -------
        Tuple[Pick, float]
            A tuple of (chosen Pick, lag_time).  The lag_time is the difference
            between actual duration and the configured time_limit, positive if
            the call overran the deadline, negative if it finished early.
        """
        start: float = perf_counter()
        pick: Pick = self.engine.pick_move()
        end: float = perf_counter()
        duration: float = (end - start)
        lag_time: float = duration - self.time_limit
        self.assertAlmostEqual(lag_time, 0, -1)
        if message == 1:
            print(
                f"Start-> {self.engine.fen()} initial pick_move call: "
                f"({pick[0].uci()}, {pick[1]:.2f}) (duration={duration:.3f}s, lag_time={lag_time:.3f}s)"
            )
        elif message == 2:
            if move is not None:
                print(
                    f"{move} -> {self.engine.fen()} initial pick_move call: "
                    f"({pick[0].uci()}, {pick[1]:.2f}) (duration={duration:.3f}s, lag_time={lag_time:.3f}s)"
                )
        elif message == 3:
            if i is not None:
                print(
                    f"Start-> {self.engine.fen()} revisit {i} pick_move call: "
                    f"({pick[0].uci()}, {pick[1]:.2f}) (duration={duration:.3f}s, lag_time={lag_time:.3f}s)"
                )
        return pick, lag_time

    def test__search_current_best_(self):
        """TBD"""

    def test_get_or_calc_score(self):
        """TBD"""

    def test_pick_board_generator(self):
        """TBD"""
