"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'ChessMoveHeatmap'
# noinspection PyShadowingBuiltins
copyright = '2025, Phillyclause89'
author = 'Phillyclause89'
release = '000.069.420'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
autodoc_mock_imports = ['chmengine.stream']
extensions = [
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary'
]

numpydoc_show_class_members = False

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '*',
    'inherited-members': True,
    'show-inheritance': True,
}

autosummary_generate = True

# Overwrite existing stubs each run (so changes to signatures/docstrings propagate)
autosummary_generate_overwrite = True

# (Optional) if you still see stub-related noise, suppress autosummary warnings
suppress_warnings = ['autosummary']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
