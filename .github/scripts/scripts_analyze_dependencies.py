"""
Identify what unittests to run based on changes to source files.

This script analyzes a given list of changed Python source files and determines
which test files in the `tests/` directory are affected, either directly or
indirectly (transitively), by those changes.

The analysis works by:
1. Walking all `.py` files in selected root directories.
2. Parsing each file's imports using the `ast` module.
3. Building a reverse dependency graph mapping each file to the files that import it.
4. Traversing that graph starting from the changed files to find all affected dependents.
5. Printing out paths of affected test files.

Example usage from the command line:
    (venv) PS ChessMoveHeatmap> python .github/scripts/scripts_analyze_dependencies.py "chmengine/engines/cmhmey2.py"
    tests\test_chmengine.py
    tests\test_chmengine_play.py
    tests\test_cmhmey1.py
    tests\test_cmhmey2.py
    tests\test_engine_utils.py

    Process finished with exit code 0

This will print all test files that should be re-run.
"""

from _ast import AST, Module
from ast import Import, ImportFrom, parse, walk as ast_walk
from os import path, sep, walk as os_walk
from sys import argv
from typing import Any, Dict, List, Optional, Set, TextIO, Union


def list_all_py_files(roots: List[str]) -> List[str]:
    """
    Recursively find all `.py` files under the specified root directories.

    Parameters
    ----------
    roots : List[str]
        A list of root directory paths to search.

    Returns
    -------
    List[str]
        A list of relative file paths to `.py` files found under the root directories.
        Files starting with a dot (hidden files) are excluded.
    """
    py_files: List[str] = []
    root: str
    for root in roots:
        dirpath: str
        filenames: List[str]
        for dirpath, _, filenames in os_walk(root):
            file: str
            for file in filenames:
                if file.endswith('.py') and not file.startswith('.'):
                    py_files.append(path.relpath(path.join(dirpath, file)))
    return py_files


def filepath_to_module(filepath: str) -> str:
    """
    Convert a relative file path to a Python-style module name.

    Parameters
    ----------
    filepath : str
        A relative path like 'foo/bar/baz.py'.

    Returns
    -------
    str
        A module name like 'foo.bar.baz'.
    """
    if filepath.endswith('.py'):
        filepath = filepath[:-3]
    return filepath.replace(sep, ".")


def find_imported_modules(file_path: str) -> Set[str]:
    """
    Parse a Python file and extract the names of all modules it imports.

    Parameters
    ----------
    file_path : str
        The path to the Python file to analyze.

    Returns
    -------
    Set[str]
        A set of module names imported by the file. Returns an empty set if
        the file cannot be parsed due to a syntax error.
    """
    file: TextIO
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree: Union[AST, Module] = parse(file.read(), filename=file_path)
        except SyntaxError:
            return set()  # Skip invalid Python files
    imports: Set[Union[Optional[str], Any]] = set()
    node: AST
    for node in ast_walk(tree):
        if isinstance(node, Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def normalize_module_name(name: str) -> str:
    """
    Normalize a module name by stripping '.__init__' if present.

    This ensures that both 'foo' and 'foo.__init__' are treated as the same module.

    Parameters
    ----------
    name : str
        The module name to normalize.

    Returns
    -------
    str
        The normalized module name.
    """
    return name[:-9] if name.endswith('.__init__') else name


def build_reverse_dependency_graph(py_files: List[str]) -> Dict[str, Set[str]]:
    """
    Build a reverse dependency graph from a list of Python files.

    The graph maps each file to the set of files that import it.

    Parameters
    ----------
    py_files : List[str]
        A list of file paths representing all Python files in the project.

    Returns
    -------
    Dict[str, Set[str]]
        A dictionary where keys are file paths and values are sets of file paths
        that depend on (i.e., import) them.
    """
    file_to_module: Dict[str, str] = {f: filepath_to_module(f) for f in py_files}
    module_to_file: Dict[str, str] = {normalize_module_name(v): k for k, v in file_to_module.items()}
    reverse_deps: Dict[str, Set[str]] = {f: set() for f in py_files}
    py_file: str
    for py_file in py_files:
        imported_modules: Set[str] = find_imported_modules(py_file)
        mod: str
        for mod in imported_modules:
            if mod in module_to_file:
                imported_file: str = module_to_file[mod]
                reverse_deps[imported_file].add(py_file)
    return reverse_deps


def find_all_dependents(
        changed_files: List[str],
        reverse_deps: Dict[str, Set[str]],
) -> Set[str]:
    """
    Find all files that (transitively) depend on the given changed files.

    Parameters
    ----------
    changed_files : List[str]
        A list of changed Python source file paths.

    reverse_deps : Dict[str, Set[str]]
        A reverse dependency graph mapping each file to the set of files that import it.

    Returns
    -------
    Set[str]
        A set of file paths representing all files that directly or indirectly
        depend on any of the changed files.
    """
    dependents: Set[str] = set()
    stack: List[str] = list(changed_files)
    while stack:
        current: str = stack.pop()
        dependent: str
        for dependent in reverse_deps.get(current, []):
            if dependent not in dependents:
                dependents.add(dependent)
                stack.append(dependent)
    return dependents


def main(changed_files: List[str]) -> None:
    """
    Main entry point to identify test files affected by given changed files.

    Parameters
    ----------
    changed_files : List[str]
        A list of changed Python file paths to analyze.

    Side Effects
    ------------
    Prints out paths to test files that should be rerun.
    """
    py_files: List[str] = list_all_py_files(
        ["chmengine", "chmutils", "heatmaps", "tests", "tooltips"]
    ) + ["main.py", "standalone_color_legend.py"]
    reverse_graph: Dict[str, Set[str]] = build_reverse_dependency_graph(py_files)
    changed_files = [path.normpath(f) for f in changed_files]
    affected: Set[str] = set(changed_files)
    affected |= find_all_dependents(changed_files, reverse_graph)
    affected_tests: Set[str] = {f for f in affected if path.normpath(f).startswith("tests" + sep)}
    test_file: Optional[str]
    for test_file in sorted(affected_tests):
        print(test_file)


if __name__ == "__main__":
    # Example usage: python script.py "foo/bar.py tests/test_foo.py"
    input_files: List[str] = argv[1].split()
    main(input_files)
