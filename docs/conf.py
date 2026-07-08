"""Sphinx configuration for deephedge."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from deephedge._version import __version__  # noqa: E402

project = "deephedge"
copyright = "2026, deephedge contributors"
author = "deephedge contributors"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "furo"
html_title = f"deephedge {__version__}"

exclude_patterns = ["_build"]
