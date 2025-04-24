"""setup"""
from os import path

import chess
from Cython.Build import cythonize
from setuptools import setup

CHESS_PACKAGE = path.dirname(chess.__file__)
chess_scripts = path.join(CHESS_PACKAGE, "*.py")
module_list = (
    path.join("heatmaps", "__init__.py"),
    path.join("chmutils", "__init__.py"),
    path.join("chmengine", "utils", "pick.py"),
    path.join("chmengine", "utils", "__init__.py"),
    path.join("chmengine", "engines", "cmhmey1.py"),
    path.join("chmengine", "engines", "quartney.py"),
    path.join("chmengine", "engines", "cmhmey2.py"),
    path.join("chmengine", "engines", "__init__.py"),
    path.join("chmengine", "__init__.py"),
)
exclude = path.join(CHESS_PACKAGE, "_interactive.py")
compiler_directives = {'language_level': "3"}
ext_modules_main = cythonize(
    module_list=module_list,
    compiler_directives=compiler_directives
)
setup(
    ext_modules=ext_modules_main,
    script_args=['build_ext', '--inplace']
)

ext_modules_chess = cythonize(
    module_list=chess_scripts,
    exclude=exclude,
    compiler_directives=compiler_directives
)
setup(
    ext_modules=ext_modules_chess,
    script_args=["build_ext", "--build-lib", path.join(CHESS_PACKAGE, "..")]
)
