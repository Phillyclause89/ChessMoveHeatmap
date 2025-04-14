"""Identify what unittests to run"""
import ast
import os
import sys
from typing import Dict, List, Set


def list_all_py_files(roots: List[str]) -> List[str]:
    """Find all .py files under specified root directories."""
    py_files = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith('.py') and not f.startswith('.'):
                    py_files.append(os.path.relpath(os.path.join(dirpath, f)))
    return py_files


def filepath_to_module(filepath: str) -> str:
    """Convert a file path like foo/bar/baz.py to module name foo.bar.baz"""
    if filepath.endswith('.py'):
        filepath = filepath[:-3]
    return filepath.replace(os.sep, ".")


def find_imported_modules(file_path: str) -> Set[str]:
    """Return set of modules imported by this file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read(), filename=file_path)
        except SyntaxError:
            return set()  # Skip invalid Python files
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def normalize_module_name(name: str) -> str:
    """Strip .__init__ from module names if present."""
    return name[:-9] if name.endswith('.__init__') else name


def build_reverse_dependency_graph(py_files: List[str]) -> Dict[str, Set[str]]:
    """Build a reverse dependency graph: file X is imported by set(Y, Z...)"""
    file_to_module = {f: filepath_to_module(f) for f in py_files}
    module_to_file = {normalize_module_name(v): k for k, v in file_to_module.items()}
    reverse_deps: Dict[str, Set[str]] = {f: set() for f in py_files}
    for py_file in py_files:
        imported_modules = find_imported_modules(py_file)
        for mod in imported_modules:
            # Only consider local modules (i.e., modules that exist in this repo)
            if mod in module_to_file:
                imported_file = module_to_file[mod]
                reverse_deps[imported_file].add(py_file)

    return reverse_deps


def find_all_dependents(
        changed_files: List[str],
        reverse_deps: Dict[str, Set[str]],
) -> Set[str]:
    """Find all files that (transitively) depend on any of the changed ones."""
    dependents = set()
    stack = list(changed_files)
    while stack:
        current = stack.pop()
        for dependent in reverse_deps.get(current, []):
            if dependent not in dependents:
                dependents.add(dependent)
                stack.append(dependent)

    return dependents


def main(changed_files: List[str]):
    """main"""
    py_files = list_all_py_files(
        ["chmengine", "chmutils", "heatmaps", "tests", "tooltips"]
    ) + ["main.py", "standalone_color_legend.py"]
    reverse_graph = build_reverse_dependency_graph(py_files)
    # Normalize changed file paths
    changed_files = [os.path.normpath(f) for f in changed_files]
    # INCLUDE THE CHANGED FILES THEMSELVES
    affected = set(changed_files)
    # Add anything that depends on them
    affected |= find_all_dependents(changed_files, reverse_graph)
    # Filter to test files
    affected_tests = {f for f in affected if os.path.normpath(f).startswith("tests" + os.sep)}
    for test_file in sorted(affected_tests):
        print(test_file)


if __name__ == "__main__":
    # Example usage: python script.py "foo/bar.py tests/test_foo.py"
    input_files = sys.argv[1].split()
    main(input_files)
