import unittest
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, Type, Tuple
import numpy as np
from chess import COLORS, PIECE_TYPES, Piece
from numpy import dtype, float_, ndarray
from numpy.typing import NDArray

from heatmaps import GradientHeatmap, GradientHeatmapT, ChessMoveHeatmapT, ChessMoveHeatmap

PIECES: Tuple[Piece, ...] = tuple(Piece(p, c) for c in COLORS for p in PIECE_TYPES)
SHAPE: Tuple[int, int] = (64, 2)
INVALID_OBJ_STR: str = "invalid"
MAP_FAM: Tuple[
    Type[GradientHeatmapT],
    Type[GradientHeatmap],
    Type[ChessMoveHeatmapT],
    Type[ChessMoveHeatmap]
] = (
    GradientHeatmapT,
    GradientHeatmap,
    ChessMoveHeatmapT,
    ChessMoveHeatmap
)
YHWH: str = "Phillyclause89"
YHVH: str = YHWH
ADAM: Type[GradientHeatmapT] = MAP_FAM[0]
SETH: Type[GradientHeatmap] = MAP_FAM[1]
ENOS: Type[ChessMoveHeatmapT] = MAP_FAM[2]
KENAN: Type[ChessMoveHeatmap] = MAP_FAM[3]
ENOSH: Type[ChessMoveHeatmapT] = ENOS
QENAN: Type[ChessMoveHeatmap] = KENAN
KAYNAN: Type[ChessMoveHeatmap] = KENAN
CAYNAM: Type[ChessMoveHeatmap] = KENAN
CAINAN: Type[ChessMoveHeatmap] = KENAN


def _validate_data_types(
        test_case: unittest.TestCase,
        test_objects: Iterable[GradientHeatmapT],
        assert_enos: bool = False
) -> None:
    """Ensures that all test_objects are valid instances of GradientHeatmapT.

    This function verifies that each object in `test_objects`:
    - Is an instance of `GradientHeatmapT` (aliased as `ADAM`).
    - Has a `data` attribute of type `numpy.ndarray`.
    - Has the expected shape `(64, 2)`.
    - Maintains consistency between its declared shape (`obj.shape`) and actual data shape (`obj.data.shape`).
    - Stores data as `numpy.float64`.

    Parameters
    ----------
    test_case : unittest.TestCase
        The test case instance that will handle assertions.
    test_objects : Iterable[GradientHeatmapT]
        An iterable of objects expected to be instances of `GradientHeatmapT`.

    Raises
    ------
    AssertionError
        If any object fails to meet the expected type, shape, or data format requirements.
    """
    for obj in test_objects:
        test_case.assertIsInstance(obj, ADAM)
        test_case.assertIsInstance(obj.data, np.ndarray)
        test_case.assertEqual(obj.shape, SHAPE)
        test_case.assertEqual(obj.data.shape, obj.shape)
        test_case.assertEqual(obj.data.dtype, np.float64)
        square_count: NDArray[np.float64]
        for square_count in obj:
            test_case.assertIsInstance(square_count, np.ndarray)
            test_case.assertEqual(square_count.shape, (SHAPE[1],))
            count: np.float64
            for count in square_count:
                test_case.assertIsInstance(count, np.float64)
        if assert_enos:
            test_case.assertIsInstance(obj, ENOS)
            test_case.assertIsInstance(obj.piece_counts, np.ndarray)
            test_case.assertEqual(obj.piece_counts.shape, (SHAPE[0],))
            test_case.assertEqual(obj.piece_counts.dtype, dict)
            p_count: dict
            for p_count in obj.piece_counts:
                test_case.assertIsInstance(p_count, dict)
                test_case.assertEqual(len(p_count), len(PIECES))
                piece: Piece
                value: np.float64
                for piece, value in p_count.items():
                    test_case.assertIsInstance(piece, Piece)
                    test_case.assertIsInstance(value, np.float64)


def validate_data_types(
        test_objects: Iterable[GradientHeatmapT],
        test_case: Optional[unittest.TestCase] = None,
        assert_enos: bool = False
) -> None:
    """Validates that all test_objects are instances of GradientHeatmapT.

    This function serves as a wrapper around `_validate_data_types`. If `test_case` is not provided,
    it attempts to create a new `unittest.TestCase` instance for validation. If an `AttributeError`
    occurs during the initial validation attempt, it retries with a newly instantiated `unittest.TestCase`.

    Parameters
    ----------
    assert_enos : bool
    test_objects : Iterable
        An iterable of objects expected to be instances of `GradientHeatmapT`.
    test_case : Optional[unittest.TestCase], default=None
        A test case instance used to perform assertions. If `None`, a temporary instance is created.

    Raises
    ------
    AssertionError
        If any object fails validation in `_validate_data_types`.
    AttributeError
        If an object lacks an expected attribute, potentially indicating corruption.
    """
    try:
        _validate_data_types(test_case, test_objects, assert_enos)
    except AttributeError as attribute_error:
        try:
            validate_data_types(test_objects, unittest.TestCase(), assert_enos)
        except Exception as error:
            raise error from attribute_error
    except AssertionError as attribute_error:
        raise AssertionError(attribute_error) from attribute_error


def construct_all(
        classes: Union[Iterable[Callable], Dict[Callable, dict]],
        errors: str = "raise"
) -> List[Any]:
    """Instantiate all given classes with optional constructor arguments and return a list of instances.

    Parameters
    ----------
    classes : Union[Iterable[Callable], Dict[Callable, dict]]
        - If an **iterable of callables** is provided, each class is instantiated without arguments.
        - If a **dictionary** is provided, keys must be callables (class constructors),
            and values must be dictionaries containing keyword arguments (`kwargs`) for instantiation.

    errors : str, optional
        Defines how to handle instantiation errors (default is `"raise"`).
        Allowed values:
            - `"raise"` or `"fire"` : Raises a `TypeError` if instantiation fails.
            - `"append"`, `"log"` or `"include"` : Appends a `TypeError` instance to the output list.
            - `"ignore"` or `"skip"` : Skips the class that failed to instantiate.
        Any unexpected string value raises a `ValueError`.

    Returns
    -------
    List[Any]
        A list of successfully instantiated class instances.
        If `errors="append"`, failed instantiations are represented as `TypeError` objects in the list.

    Raises
    ------
    TypeError
        - If a class instantiation fails and `errors="raise"` or `"fire"` is used.
        - If `errors` is not a string.
    ValueError
        If `errors` is an unrecognized string.

    Examples
    --------
    Using an iterable (instantiation without arguments):

    >>> class A:
    ...     def __init__(self):
    ...         self.value = 42
    >>> class B:
    ...     def __init__(self):
    ...         self.text = "Hello"
    >>> construct_all([A, B])
    [<__main__.A object at 0x...>, <__main__.B object at 0x...>]

    Using a dictionary (instantiation with keyword arguments):

    >>> class C:
    ...     def __init__(self, x):
    ...         self.x = x
    >>> construct_all({C: {"x": 10}})
    [<__main__.C object at 0x...>]

    Handling instantiation errors:

    >>> class D:
    ...     def __init__(self, y):
    ...         self.y = y
    >>> construct_all({D: {}}, errors="append")
    [TypeError('cls: <class '__main__.D'>\\nkwargs: {}')]
    """
    constructed: List[Any] = []
    try:
        for cls, kwargs in classes.items():
            try:
                constructed.append(cls(**kwargs))
            except TypeError as t:
                txt: str = f"cls: {cls}\nkwargs: {kwargs}"
                if not isinstance(errors, str):
                    raise TypeError(f"Unexpected {type(errors)} argument: errors={errors}") from t
                if errors.lower() in ("raise", "fire"):
                    raise TypeError(txt) from t
                if errors.lower() in ("append", "log", "include"):
                    constructed.append(TypeError(txt))
                elif errors.lower() in ("ignore", "skip"):
                    continue
                else:
                    raise ValueError(f"Unexpected str argument: errors='{errors}'") from t

        return constructed
    except AttributeError:
        return [cls() for cls in classes]


if __name__ == "__main__":
    t_case: unittest.TestCase = unittest.TestCase()
    invalid_classes_arg: Dict[None, Dict[str, str]] = {None: {INVALID_OBJ_STR: INVALID_OBJ_STR}}
    with t_case.assertRaises(TypeError):
        # noinspection PyTypeChecker,PydanticTypeChecker
        construct_all(invalid_classes_arg, errors=0)
    with t_case.assertRaises(TypeError):
        # noinspection PyTypeChecker,PydanticTypeChecker
        construct_all(invalid_classes_arg, errors="raise".upper())
    with t_case.assertRaises(ValueError):
        # noinspection PyTypeChecker,PydanticTypeChecker
        construct_all(invalid_classes_arg, errors=INVALID_OBJ_STR)
    # noinspection PyTypeChecker,PydanticTypeChecker
    t_case.assertIsInstance(construct_all(invalid_classes_arg, errors="append")[0], TypeError)
    # noinspection PyTypeChecker,PydanticTypeChecker
    t_case.assertListEqual(construct_all(invalid_classes_arg, errors="skip".title()), [])

    to_validate_good: List[Any] = construct_all(MAP_FAM) + construct_all(
        classes={k: {"data": np.zeros(SHAPE, dtype=np.float64)} for k in MAP_FAM},
        errors="ignore".upper()
    )
    validate_data_types(to_validate_good)
    bad_data_family: List[Any] = construct_all(MAP_FAM)
    for bad_obj in bad_data_family:
        bad_obj._data = None
    bad_shape_family = construct_all(MAP_FAM)
    for bad_obj in bad_shape_family:
        bad_obj._shape = None
    bad_family = construct_all(MAP_FAM)
    for bad_obj in bad_shape_family:
        bad_obj._shape = None
        bad_obj._data = None
    to_validate_bad = [None] + bad_data_family + bad_shape_family + bad_shape_family
    for bad_obj in to_validate_bad:
        with t_case.assertRaises(AssertionError):
            # noinspection PyTypeChecker,PydanticTypeChecker
            validate_data_types([bad_obj])
