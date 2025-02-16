import unittest
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, Type, Tuple
import numpy as np
from chess import COLORS, PIECE_TYPES, Piece

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


def _validate_data_types(test_case: unittest.TestCase, test_objects: Iterable[GradientHeatmapT]) -> None:
    """Helper function that ensures all test_objects are still valid GradientHeatmapT form

    Parameters
    ----------
    test_case : unittest.TestCase
    test_objects : Iterable[GradientHeatmapT]
    """
    for obj in test_objects:
        test_case.assertIsInstance(obj, ADAM)
        test_case.assertIsInstance(obj.data, np.ndarray)
        test_case.assertEqual(obj.shape, SHAPE)
        test_case.assertEqual(obj.data.shape, obj.shape)
        test_case.assertEqual(obj.data.dtype, np.float64)


def validate_data_types(
        test_objects: Iterable[GradientHeatmapT],
        test_case: Optional[unittest.TestCase] = None,
) -> None:
    """Helper function that ensures all test_objects are still valid GradientHeatmapT form

    Parameters
    ----------
    test_objects : Iterable[GradientHeatmapT]
    test_case : Optional[unittest.TestCase]

    """
    try:
        _validate_data_types(test_case, test_objects)
    except AttributeError as a:
        try:
            validate_data_types(test_objects, unittest.TestCase())
        except Exception as e:
            raise e from a
    except AssertionError as a:
        raise AssertionError(a) from a


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
