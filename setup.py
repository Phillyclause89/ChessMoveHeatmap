"""setup"""
from argparse import ArgumentParser, Namespace
from os import path
from typing import Any, Dict, List, Optional, Tuple

import chess
from Cython.Build import cythonize
from setuptools import setup

BUILD_EXT: str = 'build_ext'
INIT_PY: str = '__init__.py'
CHMENGINE: str = 'chmengine'
CHMUTILS = 'chmutils'
UTILS: str = 'utils'
ENGINES: str = 'engines'
CHESS_PACKAGE: Optional[str] = path.dirname(chess.__file__)
CHESS_SCRIPTS: str = path.join(CHESS_PACKAGE, '*.py')

MAIN_MODULES: Tuple[str, str, str, str, str, str, str, str, str, str, str, str, str] = (
    path.join('heatmaps', INIT_PY),
    path.join(CHMUTILS, 'concurrent.py'),
    path.join(CHMUTILS, 'game_builder.py'),
    path.join(CHMUTILS, 'base_chess_tk_app.py'),
    path.join(CHMUTILS, 'player.py'),
    path.join(CHMUTILS, INIT_PY),
    path.join(CHMENGINE, UTILS, 'pick.py'),
    path.join(CHMENGINE, UTILS, INIT_PY),
    path.join(CHMENGINE, ENGINES, 'cmhmey1.py'),
    path.join(CHMENGINE, ENGINES, 'quartney.py'),
    path.join(CHMENGINE, ENGINES, 'cmhmey2.py'),
    path.join(CHMENGINE, ENGINES, INIT_PY),
    path.join(CHMENGINE, INIT_PY),
)
EXCLUDE: str = path.join(CHESS_PACKAGE, '_interactive.py')


def main(compiler_directives: Dict[str, str]) -> None:
    """Setup function for ChessMoveHeatmap Cython support.

    Parameters
    ----------
    compiler_directives : Dict[str,str]
    """
    args: Namespace = get_script_args()
    both: bool = args.all or (args.chess and args.main)
    force_flag: List[Optional[str]] = ['--force'] if args.force else []
    if not args.chess or both:
        setup_main(compiler_directives, force_flag)
    if not args.main or both:
        setup_chess(compiler_directives, force_flag)


def setup_chess(compiler_directives: Dict[str, str], force_flag: List[Optional[str]]) -> None:
    """Compiles the chess lib (excluding _interactive.py)

    Parameters
    ----------
    compiler_directives : Dict[str,str]
    force_flag : List[Optional[str]]
    """
    ext_modules_chess: Any = cythonize(
        module_list=CHESS_SCRIPTS,
        exclude=EXCLUDE,
        compiler_directives=compiler_directives
    )
    setup(
        ext_modules=ext_modules_chess,
        script_args=[BUILD_EXT, '--build-lib', path.join(CHESS_PACKAGE, '..')] + force_flag
    )


def setup_main(compiler_directives: Dict[str, str], force_flag: List[Optional[str]]) -> None:
    """Compiles the ChessMoveHeatmap lib

    Parameters
    ----------
    compiler_directives : Dict[str,str]
    force_flag : List[Optional[str]]
    """
    ext_modules_main: Any = cythonize(
        module_list=MAIN_MODULES,
        compiler_directives=compiler_directives
    )
    setup(
        ext_modules=ext_modules_main,
        script_args=[BUILD_EXT, '--inplace'] + force_flag
    )


def get_script_args() -> Namespace:
    """Gets script arguments.

    Returns
    -------
    Namespace
    """
    parser: ArgumentParser = ArgumentParser(description="Setup script for ChessMoveHeatmap Cython support.")
    parser.add_argument(
        '--main',
        action='store_true',
        help="Run the setup for the main modules."
    )
    parser.add_argument(
        '--chess',
        action='store_true',
        help="Run the setup for the chess modules."
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Run the setup for both main and chess modules."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force a rebuild of all modules, even if no changes are detected."
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(compiler_directives={'language_level': "3"})
