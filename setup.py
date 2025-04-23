"""setup"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        (
            "heatmaps/__init__.py",
            "chmutils/__init__.py",
            "chmengine/utils/pick.py",
            "chmengine/utils/__init__.py",
            "chmengine/engines/cmhmey1.py",
            "chmengine/engines/quartney.py",
            "chmengine/engines/cmhmey2.py",
            "chmengine/engines/__init__.py",
            "chmengine/play/__init__.py",
            "chmengine/__init__.py",

        ),
        compiler_directives={
            'language_level': "3",
        }
    )
)
