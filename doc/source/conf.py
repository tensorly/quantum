# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('...'))


# -- Project information -----------------------------------------------------

project = 'TensorLy-Quantum'
from datetime import datetime
year = datetime.now().year
copyright = f'{year}, TensorLy-Quantum developers'
author = 'TensorLy-Quantum developers'

# The full version, including alpha/beta/rc tags
import tlquantum
release = tlquantum.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'numpydoc.numpydoc',
    # "nbsphinx",
    'sphinx_gallery.gen_gallery'
]

sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs' : './examples',
    # path where to save gallery generated examples
    'gallery_dirs'  : 'auto_examples'
}

# Remove the permalinks ("¶" symbols)
# html_permalinks = ""
html_permalinks_icon = ''

# NumPy
numpydoc_class_members_toctree = False
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False

# generate autosummary even if no references
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_default_flags = ['members']

# Napoleon
napoleon_google_docstring = False
napoleon_use_rtype = False

# imgmath/mathjax
imgmath_image_format = 'svg'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'tensorly_sphinx_theme'
html_show_sphinx = False
html_logo = '_static/logos/tensorly-quantum-logo.png'

# "<project> v<release> documentation" by default.
html_title = 'Tensor-Based Quantum Machine Learning'

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = 'TensorLy-Quantum'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'github_url': 'https://github.com/tensorly/quantum',
    'google_analytics' : 'G-0WBR1J14K6',
    'nav_links' : [('Install', 'install'),
                   ('User Guide', 'user_guide/index'),
                   ('API', 'modules/api'),
                   ('Examples', 'auto_examples/index'),
                   ('About Us', 'about')],
    'external_nav_links' : [('TensorLy', 'http://tensorly.org/dev')]
}
