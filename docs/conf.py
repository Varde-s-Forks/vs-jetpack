# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path('..').resolve()))

# -- Project information -----------------------------------------------------
project = 'vs-mask'
copyright = '2021, IEW'
author = 'IEW'

# The full version, including alpha/beta/rc tags
meta: Dict[str, Any] = {}
with Path('../vsmask/_metadata.py').resolve().open() as f:
    exec(f.read(), meta)

version = release = meta['__version__']


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx_autodoc_typehints',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/theme_overrides.css']
html_style = 'css/theme_overrides.css'


# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'                       # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_member_order
# Shouldn't affect anything since
# we're using sphinx_autodoc_typehints
autodoc_typehints = 'description'                       # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_typehints
autoclass_content = 'both'                              # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autosummary_generate = True                             # https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html?highlight=autosummary_generate#confval-autosummary_generate  # noqa: E501
autodoc_mock_imports = [
    'vapoursynth', 'vstools'                             # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_mock_imports
]
smartquotes = True                                      # https://www.sphinx-doc.org/en/master/usage/configuration.html?highlight=smartquotes#confval-smartquotes
html_show_sphinx = False                                # https://www.sphinx-doc.org/en/master/usage/configuration.html?highlight=smartquotes#confval-html_show_sphinx
pygments_style = 'sphinx'                               # https://www.sphinx-doc.org/en/master/usage/configuration.html?highlight=pygments_style#confval-pygments_style
autodoc_type_aliases = {                                # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_type_aliases
    'MorphoFunc': 'vsmask.types.MorphoFunc',
    'ZResizer': 'vsmask.types.ZResizer',
}
autodoc_preserve_defaults = True                        # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_preserve_defaults

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
