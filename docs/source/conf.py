# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pathlib
import sys

# This path is pointing to project/docs/source
CURRENT_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
RESOKIT_PATH = CURRENT_PATH.parent.parent

sys.path.insert(0, str(RESOKIT_PATH))

import resokit

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ResoKit"
copyright = "2025, Gianuzzi Emmanuel, Cerioni Matías"
author = "Gianuzzi Emmanuel, Cerioni Matías"
release = "0.0.1"

# The full version, including alpha/beta/rc tags
release = resokit.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for nbsphinx output -------------------------------------------------
nbsphinx_prompt_width = "0pt"


# =============================================================================
# INJECT README INTO THE RESTRUCTURED TEXT index.rst
# =============================================================================

import m2r

with open(RESOKIT_PATH / "README.md") as fp:
    readme_md = fp.read().split("<!-- BODY -->")[-1]


README_RST_PATH = CURRENT_PATH / "_dynamic" / "README"


with open(README_RST_PATH, "w") as fp:
    fp.write(".. FILE AUTO GENERATED !! \n")
    fp.write(m2r.convert(readme_md))
    print(f"{README_RST_PATH} regenerated!")
