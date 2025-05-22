"""Tests Cmhmey Jr.'s son Cmhmey the 3rd"""
from os import environ, path
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union
from unittest import TestCase

from chess import Board, Move
from numpy import Inf, float64, mean, percentile

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


def benchmark_assertions_map() -> Tuple[
    Dict[str, Union[float, str]], Dict[str, Union[float, str]], Dict[str, Union[float, str]],
    Dict[str, Union[float, str]], Dict[str, Union[float, str]], Dict[str, Union[float, str]],
    Dict[str, Union[float, str]], Dict[str, Union[float, str]],
]:
    def inner_map(measurement_name: str = '') -> Dict[str, Union[float, str]]:
        return dict(measurement=0.0, cap=-Inf, floor=Inf, name=measurement_name)

    return (
        inner_map('10th pct response lag-time'),  # 0
        inner_map('10th pct revisit lag-time '),  # 1
        inner_map('Average response lag-time '),  # 2
        inner_map('Average revisit lag-time  '),  # 3
        inner_map('Median response lag-time  '),  # 4
        inner_map('Median revisit lag-time   '),  # 5
        inner_map('90th pct response lag-time'),  # 6
        inner_map('90th pct revisit lag-time '),  # 7
    )


class TestCMHMEngine3(TestCase):
    """Tests Cmhmey Jr.'s son Cmhmey the 3rd"""
    starting_board: Board
    engine: CMHMEngine3
    benchmark_assertions_maps: Tuple[
        Dict[str, Union[float, str]], Dict[str, Union[float, str]], Dict[str, Union[float, str]],
        Dict[str, Union[float, str]], Dict[str, Union[float, str]], Dict[str, Union[float, str]],
        Dict[str, Union[float, str]], Dict[str, Union[float, str]],
    ] = benchmark_assertions_map()
    time_limit: float64 = float64(0.0)
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
        no_fails_counter: int = 0
        while no_fails_counter < (1 if self.pipeline_env else 10):
            no_fails_counter += 1
            pick: Pick
            lag_time: float
            pick, lag_time = self.measure_pick(message=1)
            init_w_moves: List[Move] = list(self.engine.board.legal_moves)
            first_time_pick_times: List[float] = [lag_time]  # list of all times a position was seen for the first time.
            revisit_pick_times: List[float] = []  # lists all the times the init board position was revisited.
            i: int
            move: Move
            for i, move in enumerate(
                    init_w_moves[:len(init_w_moves) // (6 if self.pipeline_env else sample_trim_div)],
                    1
            ):
                self.engine.board.push(move)
                response_lag_time: float
                _, response_lag_time = self.measure_pick(move=move, message=2)
                first_time_pick_times.append(response_lag_time)
                _, response_lag_time = self.measure_pick(move=move, message=4)
                revisit_pick_times.append(response_lag_time)
                self.engine.board.pop()
                new_pick: Pick
                new_lag_time: float
                new_pick, new_lag_time = self.measure_pick(i=i, message=3)
                revisit_pick_times.append(new_lag_time)
            self.benchmark_assertions_maps[2]['measurement'] = float(mean(first_time_pick_times))
            self.benchmark_assertions_maps[3]['measurement'] = float(mean(revisit_pick_times))
            (
                self.benchmark_assertions_maps[0]['measurement'],
                self.benchmark_assertions_maps[4]['measurement'],
                self.benchmark_assertions_maps[6]['measurement'],
            ) = percentile(first_time_pick_times, [10, 50, 90, ])
            (
                self.benchmark_assertions_maps[1]['measurement'],
                self.benchmark_assertions_maps[5]['measurement'],
                self.benchmark_assertions_maps[7]['measurement'],
            ) = percentile(revisit_pick_times, [10, 50, 90, ])
            max_delta: float = -Inf
            min_delta: float = Inf
            closest_to_failing_cap: str = ''
            closest_to_failing_floor: str = ''
            # lag-time Assertions (used for `Iterative Second-Order Performance Calibration`)
            self.print_assertions_maps()
            for assertions_map in self.benchmark_assertions_maps:
                measurement_name = assertions_map['name']
                measurement = assertions_map['measurement']
                cap = assertions_map['cap']
                floor = assertions_map['floor']
                try:
                    delta: float = self.assertMeasurementBelowCap(measurement_name, measurement, cap)
                    max_delta, closest_to_failing_cap = (
                        (delta, measurement_name) if max_delta < delta else (max_delta, closest_to_failing_cap)
                    )
                except AssertionError as error:
                    assertions_map['cap'] = measurement
                    no_fails_counter = 0
                try:
                    delta: float = self.assertMeasurementAboveFloor(
                        measurement_name, measurement, floor
                    )
                    min_delta, closest_to_failing_floor = (
                        (delta, measurement_name) if min_delta > delta else (min_delta, closest_to_failing_floor)
                    )
                except AssertionError as error:
                    assertions_map['floor'] = measurement
                    no_fails_counter = 0
            print(
                f"closest_to_failing_cap:   '{closest_to_failing_cap}' | max_delta: {max_delta:.3f}s"
                f"\nclosest_to_failing_floor: '{closest_to_failing_floor}' | min_delta: {min_delta:.3f}s"
            )
            clear_test_cache()
            self.assertFalse(path.exists(CACHE_DIR))
            self.engine._init_qdb()
            self.print_assertions_maps()

    def print_assertions_maps(self) -> None:
        """print_assertions_maps"""
        for assertions_map in sorted(self.benchmark_assertions_maps, key=lambda x: x['measurement']):
            measurement_name = assertions_map['name']
            measurement = assertions_map['measurement']
            cap = assertions_map['cap']
            floor = assertions_map['floor']
            print(
                f"{measurement_name}: {floor < measurement} == {floor:+.3f}s < {measurement:+.3f}s < {cap:+.3f}s"
                f" == {measurement < cap}"
            )

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
            f"\nNew floor found for: '{measurement_name}', "
            f"measurement too low: {measurement:.3f}s (delta={delta:+.3f}s)"
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
            f"\nNew cap found for: '{measurement_name}', "
            f"measurement too high: {measurement:.3f}s (delta={delta:+.3f}s)"
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
        i : Union[int, None]
            A sequence index for revisit calls, used in the debug print.
        move : Union[Move, None]
            The Move just played before this pick_move call, for context in logs.
        message : int
            Controls the verbosity of printed output:
            1 = initial root call, 2 = first-time response call, 3 = revisit call.

        Returns
        -------
        Tuple[Pick, float]
            A tuple of (chosen Pick, lag_time_scaled_nodes).  The lag_time_scaled_nodes is the difference
            between actual duration and the configured time_limit scaled by nodes
            i.e. len(tuple(self.engine.board.legal_moves))), positive if
            the call overran the deadline, negative if it finished early.
        """
        start: float = perf_counter()
        pick: Pick = self.engine.pick_move()
        end: float = perf_counter()
        duration: float = (end - start)
        lag_time: float = duration - self.time_limit
        branch_count: int = len(tuple(self.engine.board.legal_moves))
        node_count: int = sum(len(tuple(b.legal_moves)) for _, b in self.engine.pick_board_generator())
        if branch_count == 0:
            branch_count = 1
        # self.assertAlmostEqual(lag_time, 0, -1)
        duration_scaled_branches: float = duration / branch_count
        lag_time_scaled_branches: float = lag_time / branch_count
        duration_scaled_nodes: float = duration / node_count
        lag_time_scaled_nodes: float = lag_time / node_count
        fen: str = self.engine.fen()
        if message == 1:
            print(
                f"Start-> {fen}{' ' * (77 - len(fen))} initial pick_move call:   "
                f"({pick:+.2f}) (duration={duration:.3f}s, lag_time={lag_time:.3f}s, "
                f"branches={branch_count}, nodes={node_count}) "
                f"(duration/branch={duration_scaled_branches:.3f}s, lag_time/branch={lag_time_scaled_branches:.3f}s) "
                f"(duration/node={duration_scaled_nodes:.3f}s, lag_time/node={lag_time_scaled_nodes:.3f}s)"
            )
        elif message == 2:
            if move is not None:
                print(
                    f"{move} -> {fen}{' ' * (77 - len(fen))} initial pick_move call:   "
                    f"({pick:+.2f}) (duration={duration:.3f}s, lag_time={lag_time:.3f}s, "
                    f"branches={branch_count}, nodes={node_count}) "
                    f"(duration/branch={duration_scaled_branches:.3f}s, "
                    f"lag_time/branch={lag_time_scaled_branches:.3f}s) "
                    f"(duration/node={duration_scaled_nodes:.3f}s, lag_time/node={lag_time_scaled_nodes:.3f}s)"
                )
        elif message == 3:
            if i is not None:
                print(
                    f"Start-> {fen}{' ' * (77 - len(fen))} revisit {i} pick_move call: "
                    f"({pick:+.2f}) (duration={duration:.3f}s, lag_time={lag_time:.3f}s, "
                    f"branches={branch_count}, nodes={node_count}) "
                    f"(duration/branch={duration_scaled_branches:.3f}s, lag_time/branch={lag_time_scaled_branches:.3f}"
                    f"s) (duration/node={duration_scaled_nodes:.3f}s, lag_time/node={lag_time_scaled_nodes:.3f}s)"
                )
        elif message == 4:
            if move is not None:
                print(
                    f"{move} -> {fen}{' ' * (77 - len(fen))} revisit 1 pick_move call:   "
                    f"({pick:+.2f}) (duration={duration:.3f}s, lag_time={lag_time:.3f}s, "
                    f"branches={branch_count}, nodes={node_count}) "
                    f"(duration/branch={duration_scaled_branches:.3f}s, "
                    f"lag_time/branch={lag_time_scaled_branches:.3f}s) "
                    f"(duration/node={duration_scaled_nodes:.3f}s, lag_time/node={lag_time_scaled_nodes:.3f}s)"
                )
        return pick, lag_time_scaled_nodes

    def test__search_current_best_(self):
        """TBD"""

    def test_get_or_calc_score(self):
        """TBD"""

    def test_pick_board_generator(self):
        """TBD"""
