"""Sphinx configuration for qrc_thresher API docs."""

from __future__ import annotations

import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(ROOT, 'src')

if SRC not in sys.path:
    sys.path.insert(0, SRC)

project = 'qrc_thresher'
author = 'qrc_thresher contributors'
copyright = f'{datetime.utcnow().year}, {author}'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': False,
}

templates_path = ['_templates']
exclude_patterns: list[str] = []

html_theme = 'alabaster'
html_static_path = ['_static']
